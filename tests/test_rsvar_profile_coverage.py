"""Seasonal profiles cover every forecast month/hour without changing observed cells."""

import numpy as np
import pandas as pd
import pytest

from samy_wgen.model.rsvar import ForecastTo2100V5


def test_training_cutoff_follows_baseline_end(tmp_path):
    source = tmp_path / "weather.csv"
    pd.DataFrame({
        "DATE": pd.to_datetime(["2011-01-01", "2012-01-01", "2013-01-01"]),
        "temp": 20.0, "dewpoint": 15.0, "pressure": 1013.0, "wind_speed": 1.0,
    }).to_csv(source, index=False)
    model = ForecastTo2100V5(str(source), str(source), baseline_start=2012, baseline_end=2012)

    model.load_isd_data()

    assert model.isd_training.index.year.tolist() == [2011, 2012]
    assert model.isd_baseline.index.year.tolist() == [2012]


def test_missing_month_hour_uses_within_month_circular_interpolation():
    index = pd.date_range("2000-01-01", "2000-12-31 23:00", freq="h")
    data = pd.DataFrame({"temp": index.hour.astype(float)}, index=index)
    data = data.loc[~((data.index.month == 6) & (data.index.hour == 2))]
    model = ForecastTo2100V5.__new__(ForecastTo2100V5)
    model.isd_baseline = data
    model.seasonal_profiles = {}
    model.seasonal_profile_filled_cells = {}
    model.baseline_start = 1991
    model.baseline_end = 2010

    model.extract_seasonal_profiles(["temp"])

    profile = model.seasonal_profiles["temp"]
    assert len(profile) == 288
    assert np.isclose(profile[(6, 2)], 2)
    assert np.isclose(profile[(6, 1)], 1)
    assert model.seasonal_profile_filled_cells["temp"] == [(6, 2)]


def test_fully_missing_month_fails_closed():
    index = pd.date_range("2000-01-01", "2000-12-31 23:00", freq="h")
    data = pd.DataFrame({"temp": index.hour.astype(float)}, index=index)
    data = data.loc[data.index.month != 6]
    model = ForecastTo2100V5.__new__(ForecastTo2100V5)
    model.isd_baseline = data
    model.seasonal_profiles = {}
    model.seasonal_profile_filled_cells = {}
    model.baseline_start = 1991
    model.baseline_end = 2010

    with pytest.raises(ValueError, match="month 6"):
        model.extract_seasonal_profiles(["temp"])
