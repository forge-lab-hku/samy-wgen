from __future__ import annotations

from pathlib import Path

import pandas as pd


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
