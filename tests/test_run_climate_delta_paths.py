from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def test_resolve_city_dir_uses_canonical_ascii_code(tmp_path):
    from samy_wgen.pipeline.run_climate_delta import _resolve_city_dir

    city_dir = tmp_path / "Los_Angeles"
    city_dir.mkdir(parents=True, exist_ok=True)

    resolved = _resolve_city_dir(tmp_path, "Los_Angeles")

    assert resolved == city_dir


def test_normalize_city_args_requires_ascii_codes():
    from samy_wgen.pipeline.run_climate_delta import _get_bundled_cities_config, _load_cities_config, _normalize_city_args

    config_path = _get_bundled_cities_config()
    cities_df = _load_cities_config(config_path)

    resolved = _normalize_city_args(["Montreal", "Toronto"], cities_df, config_path)

    assert resolved == ["Montreal", "Toronto"]


def test_resolve_projection_args_direct_cmip6_mode():
    from argparse import Namespace

    from samy_wgen.pipeline.run_climate_delta import _resolve_projection_args

    args = Namespace(
        projection_type="cmip6",
        projection_root="data/CMIP6",
        ssp="ssp585",
        scenario_label=None,
        gcm=None,
        rcm=None,
    )

    resolved = _resolve_projection_args(args)

    assert resolved["label"] == "CMIP6_ssp585"
    assert str(resolved["root"]) == "data/CMIP6"
    assert resolved["type"] == "cmip6"


def test_resolve_projection_args_cordex_cmip5_mode():
    from argparse import Namespace

    from samy_wgen.pipeline.run_climate_delta import _resolve_projection_args

    args = Namespace(
        projection_type="cordex-cmip5",
        projection_root="data/CORDEX_CMIP5/NAM-22_REMO2015_r1i1p1",
        ssp=None,
        scenario_label=None,
        gcm="MPI-ESM-LR",
        rcm="REMO2015",
    )

    resolved = _resolve_projection_args(args)

    assert resolved["label"] == "CORDEX_CMIP5_rcp85"
    assert str(resolved["root"]) == "data/CORDEX_CMIP5/NAM-22_REMO2015_r1i1p1"
    assert resolved["type"] == "cordex-cmip5"
    assert resolved["ssp"] == "rcp85"


def test_load_cmip6_projection_supports_mri_files(tmp_path):
    from samy_wgen.pipeline.run_climate_delta import load_cmip6_projection

    city_dir = tmp_path / "Beijing"
    city_dir.mkdir()
    pd.DataFrame({"time": ["2014-12-31"], "tas": [280.0]}).to_csv(
        city_dir / "mri_esm2_0_historical.csv", index=False
    )
    pd.DataFrame({"time": ["2015-01-01"], "tas": [281.0]}).to_csv(
        city_dir / "mri_esm2_0_ssp245.csv", index=False
    )

    daily_df, monthly_df, meta = load_cmip6_projection(
        tmp_path, city="Beijing", ssp="ssp245", verbose=False
    )

    assert list(daily_df.index) == list(pd.to_datetime(["2014-12-31", "2015-01-01"]))
    assert monthly_df.empty
    assert meta["model_name"] == "MRI-ESM2-0"


def test_load_cmip6_projection_prioritizes_mri_scenario(tmp_path):
    from samy_wgen.pipeline.run_climate_delta import load_cmip6_projection

    city_dir = tmp_path / "Beijing"
    city_dir.mkdir()
    for model, temperature in (("mpi_esm1_2_lr", 280.0), ("mri_esm2_0", 285.0)):
        pd.DataFrame({"time": ["2014-12-31"], "tas": [temperature]}).to_csv(
            city_dir / f"{model}_historical.csv", index=False
        )
        pd.DataFrame({"time": ["2015-01-01"], "tas": [temperature + 1.0]}).to_csv(
            city_dir / f"{model}_ssp245.csv", index=False
        )

    daily_df, _, meta = load_cmip6_projection(
        tmp_path, city="Beijing", ssp="ssp245", scenario_id="MRI_ssp245", verbose=False
    )

    assert abs(daily_df.iloc[0]["tas_C"] - 11.85) < 1e-9
    assert meta["model_name"] == "MRI-ESM2-0"


def test_load_cmip6_projection_explicit_gcm_selector(tmp_path):
    from samy_wgen.pipeline.run_climate_delta import load_cmip6_projection

    city_dir = tmp_path / "Beijing"
    city_dir.mkdir()
    for model, temperature in (("mpi_esm1_2_lr", 280.0), ("cnrm_esm2_1", 285.0)):
        pd.DataFrame({"time": ["2014-12-31"], "tas": [temperature]}).to_csv(
            city_dir / f"{model}_historical.csv", index=False
        )
        pd.DataFrame({"time": ["2015-01-01"], "tas": [temperature + 1.0]}).to_csv(
            city_dir / f"{model}_ssp245.csv", index=False
        )

    daily_df, _, meta = load_cmip6_projection(
        tmp_path, city="Beijing", ssp="ssp245", gcm="CNRM-ESM2-1", verbose=False
    )

    assert abs(daily_df.iloc[0]["tas_C"] - 11.85) < 1e-9
    assert meta["model_name"] == "CNRM-ESM2-1"


def test_load_cmip6_projection_keeps_monthly_data_with_selected_model(tmp_path):
    from samy_wgen.pipeline.run_climate_delta import load_cmip6_projection

    city_dir = tmp_path / "Beijing"
    city_dir.mkdir()
    for suffix, value in (("historical", 280.0), ("ssp245", 281.0)):
        pd.DataFrame({"time": ["2014-12-31"], "tas": [value]}).to_csv(
            city_dir / f"cnrm_esm2_1_{suffix}.csv", index=False
        )
        pd.DataFrame({"time": ["2014-12-16"], "hurs": [50.0]}).to_csv(
            city_dir / f"cnrm_esm2_1_{suffix}_monthly.csv", index=False
        )
        pd.DataFrame({"time": ["2014-12-16"], "rsds": [200.0]}).to_csv(
            city_dir / f"mri_esm2_0_{suffix}_monthly.csv", index=False
        )

    _, monthly_df, meta = load_cmip6_projection(
        tmp_path, city="Beijing", ssp="ssp245", scenario_id="CNRM_ssp245", verbose=False
    )

    assert meta["model_name"] == "CNRM-ESM2-1"
    assert "hurs" in monthly_df.columns
    assert "rsds" not in monthly_df.columns


def test_mixed_resolution_infers_monthly_wind_from_csv_columns():
    from samy_wgen.pipeline.run_climate_delta import compute_mixed_resolution_baseline

    daily_index = pd.date_range("1991-01-01", "1991-12-31", freq="D")
    monthly_index = pd.date_range("1991-01-01", "1991-12-01", freq="MS")
    daily_df = pd.DataFrame({"tas": 280.0, "ps": 101000.0}, index=daily_index)
    monthly_df = pd.DataFrame({"sfcWind": 2.0}, index=monthly_index)

    baseline = compute_mixed_resolution_baseline(
        daily_df, monthly_df, 1991, 1991, var_resolution_config={}
    )

    assert baseline["sfcWind"]["resolution"] == "monthly"


def test_mixed_delta_raises_when_future_pressure_is_missing():
    from samy_wgen.pipeline.run_climate_delta import (
        apply_mixed_resolution_climate_delta,
        compute_mixed_resolution_baseline,
    )

    hours = pd.date_range("2050-01-01", "2050-12-31 23:00", freq="h")
    rsvar = pd.DataFrame({
        "temp": 20.0,
        "pressure": 1000.0,
        "wind_speed": 3.0,
        "relative_humidity": 60.0,
        "specific_humidity": 0.01,
    }, index=hours)
    historical_days = pd.date_range("1991-01-01", "2010-12-31", freq="D")
    future_days = pd.date_range("2050-01-01", "2050-12-31", freq="D")
    historical = pd.DataFrame({"tas": 293.15, "psl": 100000.0}, index=historical_days)
    complete_future = pd.DataFrame({"tas": 295.15, "psl": 101000.0}, index=future_days)
    missing_pressure = pd.DataFrame({"tas": 295.15}, index=future_days)
    monthly = pd.DataFrame(
        {"rsds": 200.0},
        index=pd.date_range("1991-01-01", "2010-12-01", freq="MS"),
    )

    complete_daily = pd.concat([historical, complete_future])
    complete_baseline = compute_mixed_resolution_baseline(
        complete_daily, monthly, 1991, 2010, var_resolution_config={})
    complete, _ = apply_mixed_resolution_climate_delta(
        rsvar, complete_daily, monthly, complete_baseline,
        gcm="test", rcm=None, lat=25.76, longitude=-80.19, verbose=False)
    assert complete["pressure"].iloc[0] == pytest.approx(1010.0)

    incomplete_daily = pd.concat([historical, missing_pressure])
    incomplete_baseline = compute_mixed_resolution_baseline(
        incomplete_daily, monthly, 1991, 2010, var_resolution_config={})
    with pytest.raises(ValueError, match="pressure.*ps_hPa"):
        apply_mixed_resolution_climate_delta(
            rsvar, incomplete_daily, monthly, incomplete_baseline,
            gcm="test", rcm=None, lat=25.76, longitude=-80.19, verbose=False)
