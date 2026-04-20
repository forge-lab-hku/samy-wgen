from __future__ import annotations

from pathlib import Path


def test_resolve_city_dir_accepts_spaced_name(tmp_path):
    from samy_wgen.pipeline.run_climate_delta import _resolve_city_dir

    spaced = tmp_path / "Los Angeles"
    spaced.mkdir(parents=True, exist_ok=True)

    resolved = _resolve_city_dir(tmp_path, "Los Angeles")

    assert resolved == spaced


def test_normalize_city_args_accepts_native_and_returns_ascii():
    from samy_wgen.pipeline.run_climate_delta import _get_bundled_cities_config, _load_cities_config, _normalize_city_args

    config_path = _get_bundled_cities_config()
    cities_df = _load_cities_config(config_path)

    resolved = _normalize_city_args(["Montréal", "Toronto"], cities_df, config_path)

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
