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
