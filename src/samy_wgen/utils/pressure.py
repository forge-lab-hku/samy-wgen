"""Pressure conversion shared by climate inputs and weather-file exporters."""

import numpy as np
import pandas as pd


def station_pressure_pa(pressure_pa, elevation_m: float, pressure_reference: str):
    """Return pressure in Pa at station elevation."""
    if pressure_reference == "station":
        return pressure_pa
    if pressure_reference == "sea_level":
        return pressure_pa * (1 - 2.25577e-5 * elevation_m) ** 5.25588
    raise ValueError("pressure_reference must be 'station' or 'sea_level'")


def projection_pressure_pa(df: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    """Resolve aliases while keeping sea-level and surface pressure distinct."""
    def combined(columns):
        values = pd.Series(np.nan, index=df.index, dtype=float)
        for key in columns:
            if key in df:
                incoming = pd.to_numeric(df[key], errors="coerce")
                values = values.fillna(incoming.where(np.isfinite(incoming)))
        return values

    surface = combined(("ps", "surface_air_pressure"))
    sea_level = combined(("psl", "sea_level_pressure"))
    if surface.notna().any():
        if (sea_level.notna() & surface.isna()).any() and "ps_hPa" not in df:
            raise ValueError("Pressure mixes surface and sea-level values. Supply one consistent reference.")
        return surface, "surface"
    if sea_level.notna().any():
        return sea_level, "sea_level"
    return None, None
