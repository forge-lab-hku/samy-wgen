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
            ghi = output_df.loc[day_mask, "GHI"].values
            dni = output_df.loc[day_mask, "DNI"].values
            dhi = np.maximum(ghi - dni * cos_z, 0.0)
            output_df.loc[day_mask, "DHI"] = dhi
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
