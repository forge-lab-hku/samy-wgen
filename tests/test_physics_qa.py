"""Tests for shared forecast physical post-processing."""

import numpy as np
import pandas as pd

from samy_wgen.utils.physics_qa import apply_physics_qa, enforce_solar_constraints


def test_apply_physics_qa_clips_humidity_and_recomputes_dewpoint():
    index = pd.date_range("2030-01-01", periods=4, freq="h")
    df = pd.DataFrame(
        {
            "temp": [30.0, 25.0, 20.0, 15.0],
            "relative_humidity": [120.0, 80.0, -5.0, 40.0],
            "specific_humidity": [0.08, 0.02, -0.01, 0.03],
            "wind_speed": [-2.0, 1.5, -0.1, 3.0],
            "GHI": [100.0, -5.0, 0.0, 50.0],
            "DNI": [200.0, 30.0, 0.0, -10.0],
            "DHI": [10.0, 5.0, 0.0, -2.0],
        },
        index=index,
    )

    corrected = apply_physics_qa(df, lat=25.7617)

    assert corrected["relative_humidity"].between(0.0, 100.0).all()
    assert corrected["specific_humidity"].between(0.0, 0.06).all()
    assert (corrected["wind_speed"] >= 0.0).all()
    assert "dewpoint" in corrected.columns
    assert (corrected["dewpoint"] <= corrected["temp"]).all()
    assert (corrected[["GHI", "DNI", "DHI"]] >= 0.0).all().all()


def test_enforce_solar_constraints_supports_blend_and_preserve_ghi():
    index = pd.date_range("2030-06-21 11:00:00", periods=2, freq="h")
    df = pd.DataFrame(
        {
            "GHI": [800.0, 700.0],
            "DNI": [600.0, 500.0],
            "DHI": [100.0, 120.0],
        },
        index=index,
    )

    blended = enforce_solar_constraints(df, lat=25.7617, preserve="blend")
    preserved = enforce_solar_constraints(df, lat=25.7617, preserve="GHI")

    assert not np.allclose(blended["GHI"].values, df["GHI"].values)
    assert np.allclose(preserved["GHI"].values, df["GHI"].values)


def test_apply_physics_qa_reconciles_q_rh_and_dewpoint_with_pressure():
    index = pd.date_range("2030-01-01", periods=4, freq="h")
    df = pd.DataFrame({
        "temp": [10.0, 25.0, 30.0, 15.0],
        "pressure": [1013.0, 1005.0, 995.0, 1008.0],
        "specific_humidity": [0.005, 0.02, 0.03, 0.0],
        "relative_humidity": [99.0, 10.0, 50.0, 0.0],
    }, index=index)
    corrected = apply_physics_qa(df, lat=25.7617)
    es = 6.112 * np.exp(17.67 * corrected.temp / (corrected.temp + 243.5))
    e = es * corrected.relative_humidity / 100
    q_from_rh = 0.622 * e / (corrected.pressure - 0.378 * e)
    assert np.allclose(corrected.specific_humidity, q_from_rh, atol=1e-10)
    assert (corrected.dewpoint <= corrected.temp).all()
    assert corrected.relative_humidity.between(1, 100).all()



def test_preserve_ghi_rejects_infeasible_direct_component():
    import pytest

    index = pd.date_range("2030-06-21 12:00", periods=1, freq="h")
    df = pd.DataFrame({"GHI": [100.0], "DNI": [900.0], "DHI": [0.0]}, index=index)
    with pytest.raises(ValueError, match="rework the solar signal"):
        enforce_solar_constraints(df, lat=25.7617, preserve="GHI")


def test_preserve_ghi_does_not_silently_cap_to_toa():
    index = pd.date_range("2030-06-21 12:00", periods=1, freq="h")
    df = pd.DataFrame({"GHI": [2500.0], "DNI": [1000.0], "DHI": [1500.0]}, index=index)
    corrected = enforce_solar_constraints(df, lat=25.7617, preserve="GHI")
    assert corrected.GHI.iloc[0] == df.GHI.iloc[0]
    # TOA validity requires longitude-aware validation by the caller.
