"""Shared physical post-processing utilities for forecast outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from samy_wgen.utils.calculate_weather_var import calc_dewpoint_from_RH


def enforce_solar_constraints(
    forecast_df: pd.DataFrame,
    lat: float,
    preserve: str = "GHI",
    night_cos_zenith_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Enforce physical constraints on solar radiation.

    Parameters:
        forecast_df: DataFrame indexed by hourly DatetimeIndex
        lat: Latitude in degrees
        preserve: 'blend' to blend GHI toward closure, 'GHI' to preserve GHI and
            reconcile DHI from GHI - DNI*cos(zenith)
        night_cos_zenith_threshold: hours at or below this cos(zenith) are treated
            as nighttime and forced to zero radiation
    """
    output_df = forecast_df.copy()

    if "GHI" not in output_df.columns:
        return output_df

    has_dni_dhi = all(v in output_df.columns for v in ["DNI", "DHI"])

    lat_rad = np.radians(lat)
    hours = output_df.index.hour.values
    doy = output_df.index.dayofyear.values

    declination = np.radians(23.45) * np.sin(2 * np.pi * (doy - 81) / 365)
    hour_angle_rad = np.radians((hours - 12) * 15)
    cos_zenith = (
        np.sin(lat_rad) * np.sin(declination)
        + np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle_rad)
    )

    night_mask = cos_zenith <= night_cos_zenith_threshold
    output_df.loc[night_mask, "GHI"] = 0.0

    if has_dni_dhi:
        output_df.loc[night_mask, "DNI"] = 0.0
        output_df.loc[night_mask, "DHI"] = 0.0

    output_df.loc[output_df["GHI"] < 0, "GHI"] = 0.0
    if has_dni_dhi:
        output_df.loc[output_df["DNI"] < 0, "DNI"] = 0.0
        output_df.loc[output_df["DHI"] < 0, "DHI"] = 0.0

    day_mask = ~night_mask
    if has_dni_dhi and day_mask.any():
        cos_z = cos_zenith[day_mask]
        if preserve == "blend":
            computed_ghi = output_df.loc[day_mask, "DHI"] + output_df.loc[day_mask, "DNI"] * cos_z
            output_df.loc[day_mask, "GHI"] = (
                0.5 * output_df.loc[day_mask, "GHI"] + 0.5 * computed_ghi
            )
            output_df.loc[output_df["GHI"] < 0, "GHI"] = 0.0
        elif preserve == "GHI":
            ghi = output_df.loc[day_mask, "GHI"].to_numpy(dtype=float)
            dni = output_df.loc[day_mask, "DNI"].to_numpy(dtype=float)
            dhi = ghi - dni * cos_z
            if (dhi < -1e-6).any():
                raise ValueError(
                    "Solar components cannot close while preserving GHI; "
                    "rework the solar signal instead of clipping GHI/DNI"
                )
            output_df.loc[day_mask, "DHI"] = np.maximum(dhi, 0.0)
        else:
            raise ValueError(f"Unsupported solar closure mode: {preserve}")

    return output_df


def apply_physics_qa(
    forecast_df: pd.DataFrame,
    lat: float,
    preserve: str = "GHI",
    night_cos_zenith_threshold: float = 0.0,
) -> pd.DataFrame:
    """Apply shared physical QA to forecast output."""
    output_df = forecast_df.copy()

    if "relative_humidity" in output_df.columns:
        output_df["relative_humidity"] = np.clip(output_df["relative_humidity"], 0, 100)

    if "specific_humidity" in output_df.columns:
        output_df["specific_humidity"] = np.clip(output_df["specific_humidity"], 0, 0.06)

    # The climate delta supplies huss and hurs independently. Treat huss as the
    # moisture signal, then reconcile RH and dewpoint with temperature and
    # pressure. Pressure is hPa in current RS-VAR inputs (accept Pa as well).
    humidity_fields = {"temp", "pressure", "specific_humidity", "relative_humidity"}
    if humidity_fields.issubset(output_df.columns):
        temp = output_df["temp"].to_numpy(dtype=float)
        pressure = output_df["pressure"].to_numpy(dtype=float)
        pressure_hpa = np.where(pressure > 2000, pressure / 100, pressure)
        if not np.isfinite(pressure_hpa).all() or (pressure_hpa <= 0).any():
            raise ValueError("Cannot reconcile humidity with nonpositive/nonfinite pressure")
        q = output_df["specific_humidity"].to_numpy(dtype=float)
        es = 6.112 * np.exp(17.67 * temp / (temp + 243.5))
        if not np.isfinite(es).all() or (es >= pressure_hpa).any():
            raise ValueError("Cannot reconcile humidity with invalid saturation pressure")
        e_min = 0.01 * es  # Match the 1% floor in calc_dewpoint_from_RH.
        q_min = 0.622 * e_min / (pressure_hpa - 0.378 * e_min)
        q_sat = 0.622 * es / (pressure_hpa - 0.378 * es)
        q = np.clip(q, q_min, np.minimum(q_sat, 0.06))
        e = q * pressure_hpa / (0.622 + 0.378 * q)
        rh = np.clip(100 * e / es, 1, 100)
        output_df["specific_humidity"] = q
        output_df["relative_humidity"] = rh

    if "wind_speed" in output_df.columns:
        output_df["wind_speed"] = np.clip(output_df["wind_speed"], 0, None)

    if "temp" in output_df.columns and "relative_humidity" in output_df.columns:
        output_df["dewpoint"] = calc_dewpoint_from_RH(
            output_df["temp"].to_numpy(dtype=float, copy=False),
            output_df["relative_humidity"].to_numpy(dtype=float, copy=False),
        )

    return enforce_solar_constraints(
        output_df,
        lat=lat,
        preserve=preserve,
        night_cos_zenith_threshold=night_cos_zenith_threshold,
    )
