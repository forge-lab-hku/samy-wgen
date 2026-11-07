"""Regression checks for the two hourly EPW conversion paths."""

import numpy as np
import pandas as pd
import pytest

from samy_wgen.pipeline.gen_samy import _write_epw_for_year
from samy_wgen.pipeline.run_pipeline import (
    _output_epw, _output_historical_epw, _station_elevation,
)
from samy_wgen.utils.create_epw_ddy import convert_tmy_to_epw, get_timezone_offset


def _weather(index):
    return pd.DataFrame({
        "temp": 25.0,
        "relative_humidity": 60.0,
        "pressure": 101325.0,
        "wind_speed": 2.0,
        "GHI": 100.0,
        "DNI": 60.0,
        "DHI": 40.0,
    }, index=index)


def test_yearly_epw_has_valid_calendar_header_and_missing_codes(tmp_path):
    index = pd.date_range("2028-01-01", "2028-12-31 23:00", freq="h")
    out = tmp_path / "Toronto_2028.epw"
    _write_epw_for_year(_weather(index), 2028, out,
                        location="Toronto-ON-CAN", lat=43.68, lon=-79.63, tz=-5)
    lines = out.read_text().splitlines()
    assert len(lines) == 8768
    assert lines[0].startswith("LOCATION,Toronto,ON,CAN,")
    assert lines[7] == "DATA PERIODS,1,1,Data,Monday, 1/ 1,12/31"
    for line in (lines[8], lines[-1]):
        row = line.split(",")
        assert len(row) == 35
        assert row[0] == "2001"  # leap day omitted from the canonical calendar
        assert row[4] == "60"
        assert len(row[5]) == 40
        assert row[10:13] == ["9999"] * 3
        assert row[16:20] == ["999999", "999999", "999999", "9999"]
        assert row[20] == "999"


def test_yearly_epw_rejects_missing_hour(tmp_path):
    index = pd.date_range("2050-01-01", periods=8760, freq="h")
    with pytest.raises(ValueError, match="missing hourly temperature"):
        _write_epw_for_year(_weather(index.delete(20)), 2050, tmp_path / "bad.epw")


def test_cli_yearly_epw_accepts_city_country_and_preserves_dewpoint(tmp_path):
    index = pd.date_range("2050-01-01", periods=8760, freq="h")
    weather = _weather(index)
    weather["dewpoint"] = 12.0
    paths = _output_epw(weather, [2050], "Miami", "ssp245",
                        {"country": "US", "lat": 25.76, "lon": -80.19,
                         "tz": -5.0, "elevation_m": 2.0}, tmp_path, True)
    lines = paths[0].read_text().splitlines()
    assert lines[0].startswith("LOCATION,Miami,,US,")
    row = lines[8].split(",")
    assert row[7] == "12.0"
    assert row[8] == "60"
    assert float(row[9]) == pytest.approx(
        101325.0 * (1 - 2.25577e-5 * 2.0) ** 5.25588, abs=1)


def test_cli_historical_epw_accepts_city_country(tmp_path):
    index = pd.date_range("2000-01-01", "2000-12-31 23:00", freq="h")
    weather = _weather(index).drop(columns="relative_humidity")
    weather["dewpoint"] = 12.0
    weather["pressure"] = 1013.25  # input hPa
    source = tmp_path / "merged.csv"
    weather.rename_axis("DATE").to_csv(source)
    out = _output_historical_epw(source, "Miami",
                                 {"country": "US", "lat": 25.76, "lon": -80.19,
                                  "tz": -5.0, "elevation_m": 2.0}, tmp_path, True)
    lines = out.read_text().splitlines()
    assert len(lines) == 8768
    assert lines[0].startswith("LOCATION,Miami,,US,")
    assert lines[8].split(",")[0] == "2001"


@pytest.mark.parametrize("pressure_reference", ["station", "sea_level"])
def test_tmy_epw_uses_end_of_hour_and_missing_wind_codes(tmp_path, pressure_reference):
    index = pd.date_range("2001-01-01", periods=8760, freq="h")
    source = pd.DataFrame({
        "DryBulb_C": 25.0,
        "DewPoint_C": 15.0,
        "RelHum_pct": 50.0,
        "Pressure_Pa": 101325.0,
        "WindSpeed_mps": 2.0,
        "GHI_Whm2": 100.0,
        "DNI_Whm2": 60.0,
        "DHI_Whm2": 40.0,
        "specific_humidity": 0.01,
    }, index=index)
    source.loc[index[0], "WindSpeed_mps"] = np.nan
    csv_path = tmp_path / "tmy.csv"
    source.to_csv(csv_path)
    out = tmp_path / "tmy.epw"
    convert_tmy_to_epw(csv_path, out, "Toronto", {
        "Toronto": {"latitude": 43.68, "longitude": -79.63,
                    "country": "CAN", "elevation_m": 173.0,
                    "pressure_reference": pressure_reference}
    })
    lines = out.read_text().splitlines()
    assert len(lines) == 8768
    assert lines[0].split(",")[9] == "173.0"
    row = lines[8].split(",")
    assert len(row) == 35
    assert row[0:5] == ["2001", "1", "1", "1", "60"]
    assert len(row[5]) == 40
    assert row[20:22] == ["999", "999"]
    assert row[16:20] == ["999999", "999999", "999999", "9999"]
    factor = (1 - 2.25577e-5 * 173.0) ** 5.25588
    expected_pressure = 101325.0 * (factor if pressure_reference == "sea_level" else 1)
    assert float(row[9]) == pytest.approx(expected_pressure, abs=1)
    if pressure_reference == "sea_level":
        assert "pressure=station_adjusted_from_sea_level" in lines[6]


def test_cli_elevation_comes_from_selected_isd_station(tmp_path):
    history = tmp_path / "isd-history.csv"
    history.write_text("USAF,WBAN,ELEV(M)\n722020,12839,1.4\n")
    station = tmp_path / "722020-12839_MIAMI_INTERNATIONAL_AIRPORT"
    assert _station_elevation(station, history, None) == 1.4
    assert _station_elevation(station, history, 2.0) == 2.0


def test_epw_timezone_uses_standard_time_in_southern_hemisphere():
    assert get_timezone_offset(-33.87, 151.2) == 10.0  # Sydney, not summer DST
