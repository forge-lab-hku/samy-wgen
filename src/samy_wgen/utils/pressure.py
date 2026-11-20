"""Pressure conversion shared by weather-file exporters."""


def station_pressure_pa(pressure_pa, elevation_m: float, pressure_reference: str):
    """Return pressure in Pa at station elevation."""
    if pressure_reference == "station":
        return pressure_pa
    if pressure_reference == "sea_level":
        return pressure_pa * (1 - 2.25577e-5 * elevation_m) ** 5.25588
    raise ValueError("pressure_reference must be 'station' or 'sea_level'")
