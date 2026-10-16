"""Calibrate ERA5-Land shortwave radiation with overlapping NSRDB observations."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _read_era5_ssrd(path: str | Path, latitude: float, longitude: float) -> pd.Series:
    """Read ERA5-Land hourly surface solar radiation in J m-2 per hour."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
        time_col = next(
            (column for column in ("valid_time", "time", "DATE", "datetime") if column in raw),
            None,
        )
        if time_col is None:
            raise ValueError("ERA5-Land CSV needs a time column (valid_time, time, DATE, or datetime)")
        value_col = next(
            (column for column in ("ssrd", "surface_solar_radiation_downwards") if column in raw),
            None,
        )
        if value_col is None:
            raise ValueError("ERA5-Land file needs ssrd or surface_solar_radiation_downwards")
        index = pd.to_datetime(raw[time_col], errors="coerce", utc=True)
        series = pd.Series(pd.to_numeric(raw[value_col], errors="coerce").to_numpy(), index=index)
    else:
        import xarray as xr

        sources: list[Path] = []
        temporary_directory = None
        try:
            if zipfile.is_zipfile(path):
                temporary_directory = tempfile.TemporaryDirectory(prefix="samy-era5-")
                with zipfile.ZipFile(path) as archive:
                    archive.extractall(temporary_directory.name)
                sources = sorted(Path(temporary_directory.name).rglob("*.nc"))
                if not sources:
                    raise ValueError(f"ERA5-Land archive contains no NetCDF files: {path}")
            else:
                sources = [path]

            series = None
            errors = []
            for source in sources:
                try:
                    with xr.open_dataset(source) as dataset:
                        variable_name = next(
                            (
                                name
                                for name in ("ssrd", "surface_solar_radiation_downwards")
                                if name in dataset.data_vars
                            ),
                            None,
                        )
                        if variable_name is None:
                            continue
                        variable = dataset[variable_name]
                        spatial_dims = {
                            "latitude": "latitude" if "latitude" in variable.dims else "lat",
                            "longitude": "longitude" if "longitude" in variable.dims else "lon",
                        }
                        for axis, target in (("latitude", latitude), ("longitude", longitude)):
                            dim = spatial_dims[axis]
                            if dim in variable.dims:
                                variable = variable.sel({dim: target}, method="nearest")
                        values = variable.to_series()
                        if isinstance(values.index, pd.MultiIndex):
                            values.index = values.index.get_level_values(-1)
                        series = values
                        break
                except Exception as exc:  # include later source files in multi-file CDS archives
                    errors.append(f"{source.name}: {exc}")
            if series is None:
                detail = f" ({'; '.join(errors)})" if errors else ""
                raise ValueError(f"No readable ERA5-Land solar-radiation variable found in {path}{detail}")
        finally:
            if temporary_directory is not None:
                temporary_directory.cleanup()

        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("ERA5-Land radiation data must have a datetime coordinate")
        index = pd.to_datetime(series.index, errors="coerce", utc=True)
        series = pd.Series(pd.to_numeric(series.to_numpy(), errors="coerce"), index=index)

    valid = ~series.index.isna()
    series = series.loc[valid].sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series.name = "ssrd"
    if series.empty:
        raise ValueError("ERA5-Land file contains no valid radiation timestamps")
    return series


def _load_nsrdb(path: str | Path) -> pd.DataFrame:
    """Load NSRDB GHI/DNI/DHI and aggregate its timestamps to hourly values."""
    raw = pd.read_csv(path)
    required = {"Year", "Month", "Day", "Hour", "GHI", "DNI", "DHI"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"NSRDB file missing required columns: {missing}")
    minute = raw["Minute"].astype(int) if "Minute" in raw else 0
    index = pd.to_datetime(
        {
            "year": raw["Year"].astype(int),
            "month": raw["Month"].astype(int),
            "day": raw["Day"].astype(int),
            "hour": raw["Hour"].astype(int),
            "minute": minute,
        },
        errors="coerce",
    )
    frame = raw[["GHI", "DNI", "DHI"]].apply(pd.to_numeric, errors="coerce")
    frame.index = index
    frame = frame.loc[~frame.index.isna()].sort_index()
    # NSRDB is requested in local standard time by this repository's downloader.
    return frame.resample("h").mean()


def _cos_zenith(latitude: float, longitude: float, utc_index: pd.DatetimeIndex) -> np.ndarray:
    """Approximate cosine of solar zenith angle for UTC timestamps."""
    lat_rad = np.radians(latitude)
    day = utc_index.dayofyear.to_numpy()
    hour = utc_index.hour.to_numpy() + utc_index.minute.to_numpy() / 60.0
    gamma = 2 * np.pi * (day - 1 + (hour - 12) / 24) / 365.0
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    solar_hour = hour + longitude / 15.0
    hour_angle = np.radians((solar_hour - 12.0) * 15.0)
    return (
        np.sin(lat_rad) * np.sin(declination)
        + np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle)
    )


def calibrate_era5_land_solar(
    era5_path: str | Path,
    nsrdb_path: str | Path,
    *,
    latitude: float,
    longitude: float,
    utc_offset_hours: float,
    min_overlap_samples: int = 24,
) -> pd.DataFrame:
    """Return calibrated hourly GHI/DNI/DHI in local standard time.

    ERA5-Land surface solar radiation is an hourly energy amount in J m-2, so
    it is converted to W m-2 before calibration. NSRDB overlap is used to learn
    monthly/hourly GHI scale factors and diffuse fractions. ERA5 UTC timestamps
    are shifted to fixed local standard time to match the NSRDB download format.
    """
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Latitude/longitude are outside valid geographic bounds")
    if not -14 <= utc_offset_hours <= 14:
        raise ValueError("utc_offset_hours must be between -14 and 14")
    if min_overlap_samples < 1:
        raise ValueError("min_overlap_samples must be a positive integer")

    era5_ssrd = _read_era5_ssrd(era5_path, latitude, longitude)
    # Reindex to fixed local standard time (LST), keeping timestamps naive like NSRDB.
    era5_index_lst = (era5_ssrd.index + pd.to_timedelta(utc_offset_hours, unit="h")).tz_localize(None)
    era5_ghi = pd.Series(era5_ssrd.to_numpy(dtype=float) / 3600.0, index=era5_index_lst, name="GHI_era5")
    era5_ghi = era5_ghi.replace([np.inf, -np.inf], np.nan).clip(lower=0)

    nsrdb = _load_nsrdb(nsrdb_path)
    if nsrdb.empty:
        raise ValueError("NSRDB file contains no valid hourly observations")
    overlap = era5_ghi.index.intersection(nsrdb.index)
    era_overlap = era5_ghi.reindex(overlap)
    nsr_overlap = nsrdb["GHI"].reindex(overlap)
    usable = era_overlap.notna() & nsr_overlap.notna() & (era_overlap > 1.0) & (nsr_overlap > 10.0)
    if int(usable.sum()) < min_overlap_samples:
        raise ValueError(
            f"Insufficient daytime ERA5/NSRDB overlap: {int(usable.sum())} samples; "
            f"need at least {min_overlap_samples}"
        )

    ratios = pd.DataFrame({"era5": era_overlap[usable], "nsrdb": nsr_overlap[usable]})
    ratios["month"] = ratios.index.month
    ratios["hour"] = ratios.index.hour
    ratios["ratio"] = (ratios["nsrdb"] / ratios["era5"]).clip(0.1, 5.0)
    global_ratio = float(ratios["ratio"].median())
    monthly_ratio = ratios.groupby("month")["ratio"].median()
    monthly_hourly_ratio = ratios.groupby(["month", "hour"])["ratio"].median()

    factors = np.array(
        [
            monthly_hourly_ratio.get(
                (timestamp.month, timestamp.hour),
                monthly_ratio.get(timestamp.month, global_ratio),
            )
            for timestamp in era5_ghi.index
        ],
        dtype=float,
    )
    ghi = era5_ghi * factors

    # Learn NSRDB diffuse fraction from high-quality daylight records.
    nsrdb_utc_index = nsrdb.index + pd.to_timedelta(utc_offset_hours, unit="h")
    nsrdb_cosz = _cos_zenith(latitude, longitude, nsrdb_utc_index)
    daylight = (nsrdb_cosz > 0.05) & (nsrdb["GHI"].to_numpy() > 10)
    training = nsrdb.loc[daylight].copy()
    training["cosz"] = nsrdb_cosz[daylight]
    training["kd"] = (training["DHI"] / training["GHI"].clip(lower=1)).clip(0, 1)
    closure_error = np.abs(training["GHI"] - training["DNI"] * training["cosz"] - training["DHI"])
    training = training.loc[closure_error <= training["GHI"] * 0.05]
    if training.empty:
        raise ValueError("NSRDB has no daylight rows with valid GHI = DNI*cos(zenith) + DHI closure")
    training["month"] = training.index.month
    training["hour"] = training.index.hour
    kd_global = float(training["kd"].median())
    kd_month_hour = training.groupby(["month", "hour"])["kd"].median()
    kd_month = training.groupby("month")["kd"].median()

    era5_utc_index = pd.DatetimeIndex(ghi.index + pd.to_timedelta(-utc_offset_hours, unit="h"), tz="UTC")
    cosz = _cos_zenith(latitude, longitude, era5_utc_index)
    calibrated_ghi = ghi.to_numpy(dtype=float)
    calibrated_ghi[cosz <= 0] = 0.0
    kd = np.array(
        [
            kd_month_hour.get((ts.month, ts.hour), kd_month.get(ts.month, kd_global))
            for ts in ghi.index
        ],
        dtype=float,
    )
    kd = np.clip(kd, 0.0, 1.0)
    dhi = kd * calibrated_ghi
    dni = np.zeros_like(calibrated_ghi)
    sun_up = cosz > 0.05
    dni[sun_up] = np.minimum((calibrated_ghi[sun_up] - dhi[sun_up]) / cosz[sun_up], 1200.0)
    # Preserve energy closure when the DNI cap applies and at low sun angles.
    dhi = np.maximum(calibrated_ghi - dni * np.maximum(cosz, 0.0), 0.0)

    result = pd.DataFrame(
        {"GHI": calibrated_ghi, "DNI": dni, "DHI": dhi},
        index=pd.DatetimeIndex(ghi.index, name="DATE"),
    )
    return result
