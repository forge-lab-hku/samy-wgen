"""Calibrate ERA5-Land shortwave radiation with overlapping NSRDB observations."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Literal

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
    """Load observations at their recorded instant, including the minute and zenith."""
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
    columns = ["GHI", "DNI", "DHI"]
    if "Solar Zenith Angle" in raw:
        columns.append("Solar Zenith Angle")
    frame = raw[columns].apply(pd.to_numeric, errors="coerce")
    frame.index = index
    frame = frame.loc[~frame.index.isna()].sort_index()
    if frame.index.has_duplicates:
        raise ValueError("NSRDB contains duplicate observation timestamps")
    return frame


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


def _nsrdb_time_basis(
    nsrdb: pd.DataFrame,
    basis: Literal["auto", "utc", "lst"],
    latitude: float,
    longitude: float,
    utc_offset_hours: float,
) -> tuple[str, dict[str, float], dict[str, float], str]:
    """Select UTC or fixed LST only from decisive solar evidence."""
    if basis not in {"auto", "utc", "lst"}:
        raise ValueError("nsrdb_time_basis must be auto, utc, or lst")
    if basis != "auto":
        return basis, {}, {}, "explicit parameter"
    if utc_offset_hours == 0:
        raise ValueError("Cannot infer NSRDB time basis at zero UTC offset; specify nsrdb_time_basis='utc' or 'lst'")
    if "Solar Zenith Angle" not in nsrdb:
        ghi = nsrdb["GHI"].to_numpy(dtype=float)
        bright = np.isfinite(ghi) & (ghi > 20.0)
        bright_times = nsrdb.index[bright]
        if int(bright.sum()) < 1000 or (
            len(bright_times) and bright_times.max() - bright_times.min() < pd.Timedelta(days=180)
        ):
            raise ValueError(
                "Insufficient NSRDB daylight evidence to infer time basis; "
                "specify nsrdb_time_basis='utc' or 'lst'"
            )
        night_fraction = {}
        for candidate, shift in (("utc", 0.0), ("lst", -utc_offset_hours)):
            utc_index = pd.DatetimeIndex(nsrdb.index + pd.to_timedelta(shift, unit="h"), tz="UTC")
            predicted = _cos_zenith(latitude, longitude, utc_index)
            night_fraction[candidate] = float(np.mean(predicted[bright] <= 0))
        best = min(night_fraction, key=night_fraction.get)
        other = "lst" if best == "utc" else "utc"
        if (
            night_fraction[best] > 0.03
            or night_fraction[other] - night_fraction[best] < 0.03
            or night_fraction[other] < 3 * night_fraction[best]
        ):
            raise ValueError(
                f"Ambiguous NSRDB time basis from GHI daylight (UTC night fraction="
                f"{night_fraction['utc']:.3f}, LST={night_fraction['lst']:.3f}); "
                "specify nsrdb_time_basis='utc' or 'lst'"
            )
        return best, {}, night_fraction, "GHI daylight versus calculated solar night"
    measured = np.cos(np.radians(nsrdb["Solar Zenith Angle"].to_numpy(dtype=float)))
    scores = {}
    for candidate, shift in (("utc", 0.0), ("lst", -utc_offset_hours)):
        utc_index = pd.DatetimeIndex(nsrdb.index + pd.to_timedelta(shift, unit="h"), tz="UTC")
        predicted = _cos_zenith(latitude, longitude, utc_index)
        valid = np.isfinite(measured) & (measured > 0.1)
        if int(valid.sum()) < 48:
            raise ValueError("Insufficient NSRDB zenith evidence; specify nsrdb_time_basis='utc' or 'lst'")
        scores[candidate] = float(np.mean(np.abs(measured[valid] - predicted[valid])))
    best = min(scores, key=scores.get)
    other = "lst" if best == "utc" else "utc"
    if scores[best] > 0.08 or scores[other] - scores[best] < 0.04 or scores[other] < 2 * scores[best]:
        raise ValueError(
            f"Ambiguous NSRDB time basis (UTC MAE={scores['utc']:.3f}, "
            f"LST MAE={scores['lst']:.3f}); specify nsrdb_time_basis='utc' or 'lst'"
        )
    return best, scores, {}, "Solar Zenith Angle comparison"


def _era5_at_observations(era5_ghi: pd.Series, observation_index: pd.DatetimeIndex) -> pd.Series:
    """Interpolate hourly ERA5 to the actual NSRDB observation instants."""
    series = era5_ghi.reindex(era5_ghi.index.union(observation_index)).interpolate(method="time")
    values = series.reindex(observation_index)
    # Never bridge a missing ERA5 hour or extrapolate outside its coverage.
    left = era5_ghi.index.searchsorted(observation_index, side="right") - 1
    right = era5_ghi.index.searchsorted(observation_index, side="left")
    valid = (left >= 0) & (right < len(era5_ghi))
    gaps = np.full(len(observation_index), np.inf)
    gaps[valid] = (era5_ghi.index[right[valid]] - era5_ghi.index[left[valid]]).total_seconds()
    return values.where(valid & (gaps <= 3600))


def calibrate_era5_land_solar(
    era5_path: str | Path,
    nsrdb_path: str | Path,
    *,
    latitude: float,
    longitude: float,
    utc_offset_hours: float,
    nsrdb_time_basis: Literal["auto", "utc", "lst"] = "auto",
    min_overlap_samples: int = 24,
) -> pd.DataFrame:
    """Return calibrated hourly GHI/DNI/DHI in local standard time.

    ERA5-Land surface solar radiation is an hourly energy amount in J m-2, so
    it is converted to W m-2 before calibration. NSRDB overlap is used to learn
    monthly/hourly GHI scale factors and diffuse fractions. The output DATE is
    fixed local standard time (UTC plus utc_offset_hours), without DST.
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
        raise ValueError("NSRDB file contains no valid observations")
    basis, zenith_scores, night_fraction, basis_evidence = _nsrdb_time_basis(
        nsrdb, nsrdb_time_basis, latitude, longitude, utc_offset_hours,
    )
    nsrdb_utc_index = pd.DatetimeIndex(
        nsrdb.index + pd.to_timedelta(-utc_offset_hours if basis == "lst" else 0, unit="h"),
        tz="UTC",
    )
    nsrdb_lst_index = (nsrdb_utc_index + pd.to_timedelta(utc_offset_hours, unit="h")).tz_localize(None)
    nsrdb.index = nsrdb_lst_index
    era_overlap = _era5_at_observations(era5_ghi, nsrdb_lst_index)
    nsr_overlap = nsrdb["GHI"]
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

    # Use the NSRDB zenith at the actual observation instant for closure.
    nsrdb_cosz = (
        np.cos(np.radians(nsrdb["Solar Zenith Angle"].to_numpy(dtype=float)))
        if "Solar Zenith Angle" in nsrdb
        else _cos_zenith(latitude, longitude, nsrdb_utc_index)
    )
    daylight = np.isfinite(nsrdb_cosz) & (nsrdb_cosz > 0.05) & (nsrdb["GHI"].to_numpy() > 10)
    training = nsrdb.loc[daylight].copy()
    training["cosz"] = nsrdb_cosz[daylight]
    training["kd"] = (training["DHI"] / training["GHI"].clip(lower=1)).clip(0, 1)
    closure_error = np.abs(training["GHI"] - training["DNI"] * training["cosz"] - training["DHI"])
    training = training.loc[closure_error.notna() & (closure_error <= training["GHI"] * 0.05)]
    if training.empty:
        raise ValueError("NSRDB has no daylight rows with valid GHI = DNI*cos(zenith) + DHI closure")
    training["month"] = training.index.month
    training["hour"] = training.index.hour
    # Clearness separates cloudy diffuse-dominated hours from clear direct sun.
    training["clearness_bin"] = np.digitize(
        (training["GHI"] / (1361.0 * training["cosz"])).clip(0, 1.5),
        [0.3, 0.6],
    )
    kd_global = float(training["kd"].median())
    kd_by_bin = training.groupby("clearness_bin")["kd"].median()
    kd_month_bin = training.groupby(["month", "clearness_bin"])["kd"].median()
    kd_month_hour_bin = training.groupby(["month", "hour", "clearness_bin"])["kd"].median()

    era5_utc_index = pd.DatetimeIndex(ghi.index + pd.to_timedelta(-utc_offset_hours, unit="h"), tz="UTC")
    cosz = _cos_zenith(latitude, longitude, era5_utc_index)
    calibrated_ghi = ghi.to_numpy(dtype=float).copy()
    calibrated_ghi[cosz <= 0] = 0.0
    output_bins = np.digitize(
        np.divide(calibrated_ghi, 1361.0 * np.maximum(cosz, 0.05)),
        [0.3, 0.6],
    )
    kd = np.array(
        [
            kd_month_hour_bin.get(
                (ts.month, ts.hour, int(bin_id)),
                kd_month_bin.get((ts.month, int(bin_id)), kd_by_bin.get(int(bin_id), kd_global)),
            )
            for ts, bin_id in zip(ghi.index, output_bins)
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
    result.attrs.update({
        "time_basis": "fixed LST",
        "nsrdb_time_basis": basis,
        "nsrdb_time_basis_evidence": basis_evidence,
        "utc_offset_hours": utc_offset_hours,
        "nsrdb_utc_to_lst_hours": utc_offset_hours if basis == "utc" else 0.0,
        "zenith_mae": zenith_scores,
        "ghi_night_fraction": night_fraction,
        "overlap_samples": int(era_overlap.notna().sum()),
        "ghi_fit_samples": int(usable.sum()),
        "closure_daylight_samples": int(daylight.sum()),
        "closure_pass_samples": len(training),
    })
    return result
