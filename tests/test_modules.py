"""End-to-end POC tests using the Miami dataset.

All tests are marked @slow — run with:
    pytest -m slow tests/test_modules.py -v
"""
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow


def test_epw_writer_synthetic(tmp_path):
    """EPW writer produces a valid 8760+8-line file from synthetic data."""
    from samy_wgen.pipeline.gen_samy import _write_epw_for_year

    idx = pd.date_range("2050-01-01 00:00", periods=8760, freq="h")
    df = pd.DataFrame(
        {
            "temp": np.full(8760, 28.0),
            "relative_humidity": np.full(8760, 75.0),
            "pressure": np.full(8760, 101000.0),
            "wind_speed": np.full(8760, 3.5),
            "wind_dir": np.full(8760, 180),
            "GHI": np.full(8760, 0.0),
            "DNI": np.full(8760, 0.0),
            "DHI": np.full(8760, 0.0),
        },
        index=idx,
    )
    out = tmp_path / "Miami_2050.epw"
    _write_epw_for_year(
        df,
        year=2050,
        out_path=out,
        location="Miami-FL-USA",
        lat=25.7617,
        lon=-80.1918,
        tz=-5.0,
    )
    assert out.exists(), "EPW file was not created"
    lines = out.read_text().splitlines()
    assert len(lines) == 8760 + 8, f"Expected 8768 lines, got {len(lines)}"
    # Header line 1 must start with LOCATION
    assert lines[0].startswith("LOCATION"), "First line must be LOCATION header"


def test_parse_isd_miami(miami_data, tmp_output):
    """Parse ISD .gz files into a standardized CSV."""
    from samy_wgen.utils.parse_ISD import process_station_directory

    isd_dir = miami_data["isd_dir"]
    if not isd_dir.exists():
        pytest.skip(f"ISD data not found: {isd_dir}")

    out_csv = tmp_output / "miami_isd_parsed.csv"
    process_station_directory(str(isd_dir), str(out_csv))

    assert out_csv.exists(), "Parsed ISD CSV was not created"
    df = pd.read_csv(out_csv, nrows=50)
    assert len(df) > 0, "Parsed CSV is empty"
    # ISD parser should produce a datestamp column
    assert any("date" in c.lower() or "time" in c.lower() for c in df.columns), (
        f"No date/time column found. Columns: {list(df.columns)}"
    )


def test_utc_to_lst_miami(miami_data, tmp_output):
    """Verify UTC→LST shift on a small ISD sample."""
    from samy_wgen.utils.time_utils import utc_to_lst

    isd_csv = tmp_output / "miami_isd_parsed.csv"
    if not isd_csv.exists():
        pytest.skip("Run test_parse_isd_miami first")

    df = pd.read_csv(isd_csv, nrows=500)
    # Find a datetime-like column
    date_col = next(
        (c for c in df.columns if "date" in c.lower() or "time" in c.lower()), None
    )
    if date_col is None:
        pytest.skip("No date column in parsed ISD CSV")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    lst_df = utc_to_lst(df, "Miami")
    # utc_to_lst() returns tz-naive LST index; normalize original UTC index
    # to tz-naive before subtraction.
    utc_idx0 = df.index[0]
    if getattr(df.index, "tz", None) is not None:
        utc_idx0 = utc_idx0.tz_localize(None)

    offset = (lst_df.index[0] - utc_idx0).total_seconds() / 3600
    assert abs(offset - (-5.0)) < 0.1, f"Expected -5h shift, got {offset}h"


def test_rsvar_load_isd(miami_data, tmp_output):
    """RSVAROnlyPipeline can load and preprocess Miami ISD data."""
    isd_csv = tmp_output / "miami_isd_parsed.csv"
    if not isd_csv.exists():
        pytest.skip("Run test_parse_isd_miami first")

    nsrdb_csv = miami_data["nsrdb_csv"]
    if not nsrdb_csv.exists():
        pytest.skip(f"NSRDB data not found: {nsrdb_csv}")

    from samy_wgen.utils.time_utils import utc_to_lst
    from samy_wgen.pipeline.run_rsvar import RSVAROnlyPipeline

    # Build a minimal, valid ISD+solar hourly input expected by RSVAROnlyPipeline
    raw = pd.read_csv(isd_csv, parse_dates=["datestamp"], low_memory=False)
    raw = raw.dropna(subset=["datestamp"]).set_index("datestamp").sort_index()

    rename_map = {
        "air_temperature_observation_air_temperature": "temp",
        "air_temperature_observation_dew_point_temperature": "dewpoint",
        "atmospheric_pressure_observation_sea_level_pressure": "pressure",
        "wind_observation_speed_rate": "wind_speed",
    }
    keep = [c for c in rename_map if c in raw.columns]
    if len(keep) < 3:
        pytest.skip(f"Parsed ISD missing mandatory fields. Found: {list(raw.columns)}")

    met = raw[keep].rename(columns=rename_map)
    for col in met.columns:
        met[col] = pd.to_numeric(met[col], errors="coerce")

    met = met.dropna(how="all")
    met = utc_to_lst(met, "Miami").resample("h").mean()

    nsrdb = pd.read_csv(nsrdb_csv)
    nsrdb["DATE"] = pd.to_datetime(
        nsrdb[["Year", "Month", "Day", "Hour"]].rename(
            columns={"Year": "year", "Month": "month", "Day": "day", "Hour": "hour"}
        )
    )
    solar = (
        nsrdb.set_index("DATE").sort_index()[["GHI", "DNI", "DHI"]]
        .clip(lower=0)
        .resample("h")
        .mean()
    )

    isd_solar = met.join(solar, how="inner").ffill(limit=3).bfill(limit=3)
    isd_solar = isd_solar.dropna(subset=["temp", "dewpoint", "pressure", "GHI"], how="any")
    isd_solar.index.name = "DATE"

    isd_solar_csv = tmp_output / "miami_isd_solar.csv"
    isd_solar.to_csv(isd_solar_csv)

    pipeline = RSVAROnlyPipeline(
        isd_solar_path=str(isd_solar_csv),
        lat=25.7617,
        baseline_start=1991,
        baseline_end=2010,
    )
    pipeline.load_isd_data()

    assert pipeline.isd_data is not None, "isd_data not loaded"
    assert pipeline.isd_baseline is not None, "isd_baseline not extracted"
    assert len(pipeline.isd_baseline) > 0, "isd_baseline is empty"
