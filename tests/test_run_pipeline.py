from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


class TestDownloadDefaults:
    def test_cmip6_download_experiments_follow_ssp(self):
        from samy_wgen.pipeline.run_pipeline import _cmip6_download_experiments

        assert _cmip6_download_experiments("ssp245") == ["historical", "ssp2_4_5"]
        assert _cmip6_download_experiments("ssp585") == ["historical", "ssp5_8_5"]

    def test_cmip6_download_experiments_reject_unknown_ssp(self):
        from samy_wgen.pipeline.run_pipeline import _cmip6_download_experiments

        with pytest.raises(SystemExit, match="unsupported SSP"):
            _cmip6_download_experiments("ssp999")

    def test_find_nsrdb_csv_matches_underscored_city_slug(self, tmp_path):
        from samy_wgen.pipeline.run_pipeline import _find_nsrdb_csv

        nsrdb_dir = tmp_path / "NSRDB"
        nsrdb_dir.mkdir(parents=True, exist_ok=True)
        target = nsrdb_dir / "nsrdb_los_angeles_1998-2024.csv"
        target.write_text("Year,Month,Day,Hour,GHI\n")

        found = _find_nsrdb_csv("Los_Angeles", tmp_path)

        assert found == target

    def test_has_required_cmip6_data_accepts_underscored_city_dir(self, tmp_path):
        from samy_wgen.pipeline.run_pipeline import _has_required_cmip6_data

        city_dir = tmp_path / "CMIP6" / "Los_Angeles"
        city_dir.mkdir(parents=True, exist_ok=True)
        (city_dir / "mpi_esm1_2_lr_historical.csv").write_text("time,tas\n")
        (city_dir / "mpi_esm1_2_lr_ssp245.csv").write_text("time,tas\n")

        assert _has_required_cmip6_data("Los_Angeles", tmp_path / "CMIP6", "ssp245", ["mpi_esm1_2_lr"])


class TestEnsurePipelineInputs:
    def test_downloads_missing_inputs_with_defaults(self, tmp_path):
        from samy_wgen.pipeline.run_pipeline import _ensure_pipeline_inputs

        data_dir = tmp_path / "data"
        captured: dict[str, object] = {}

        def fake_download_isd(station_id: str, city: str, years: list[int], output_dir: str) -> None:
            captured["isd"] = {
                "station_id": station_id,
                "years": (years[0], years[-1]),
                "output_dir": output_dir,
            }
            station_dir = Path(output_dir) / city / f"{station_id}_AUTO"
            station_dir.mkdir(parents=True, exist_ok=True)
            (station_dir / f"{station_id}-2000.gz").write_bytes(b"ok")

        def fake_download_nsrdb(city: str, lat: float, lon: float, years: list[int], output_dir: str) -> None:
            captured["nsrdb"] = {
                "lat": lat,
                "lon": lon,
                "years": (years[0], years[-1]),
                "output_dir": output_dir,
            }
            nsrdb_dir = Path(output_dir)
            nsrdb_dir.mkdir(parents=True, exist_ok=True)
            (nsrdb_dir / "nsrdb_miami_1998-2024.csv").write_text("Year,Month,Day,Hour,GHI\n")

        city_dir = data_dir / "CMIP6" / "Miami"
        city_dir.mkdir(parents=True, exist_ok=True)
        (city_dir / "mpi_esm1_2_lr_historical.csv").write_text("time,tas\n")
        (city_dir / "mpi_esm1_2_lr_ssp245.csv").write_text("time,tas\n")

        args = Namespace(
            city=["Miami"],
            isd_dir=None,
            nsrdb=None,
            cmip6_dir=None,
            skip_download=False,
            download_data=["isd", "nsrdb", "cmip6"],
            download_years=None,
            isd_station=None,
            isd_radius=50.0,
            ssp="ssp245",
            cmip6_models=["mpi_esm1_2_lr"],
            cmip6_experiments=None,
            cmip6_years=None,
        )

        with (
            patch("samy_wgen.pipeline.download_data.select_best_isd_station", return_value="722020-12839") as station_mock,
            patch("samy_wgen.pipeline.download_data._city_meta", return_value={"lat": 25.7617, "lon": -80.1918}),
            patch("samy_wgen.pipeline.download_data.download_isd", side_effect=fake_download_isd),
            patch("samy_wgen.pipeline.download_data.download_nsrdb", side_effect=fake_download_nsrdb),
        ):
            isd_dir, nsrdb_csv, cmip6_dir = _ensure_pipeline_inputs(args, data_dir, "Miami")

        station_mock.assert_called_once_with(
            city="Miami",
            radius_km=50.0,
            cache_dir=str(data_dir / "ISD"),
        )
        assert isd_dir.name == "722020-12839_AUTO"
        assert nsrdb_csv.name == "nsrdb_miami_1998-2024.csv"
        assert cmip6_dir == data_dir / "CMIP6"
        assert captured["isd"]["years"] == (1970, 2024)
        assert captured["nsrdb"]["years"] == (1998, 2024)

    def test_ensure_pipeline_inputs_uses_city_specific_isd_station_mapping(self, tmp_path):
        from samy_wgen.pipeline.run_pipeline import _ensure_pipeline_inputs

        data_dir = tmp_path / "data"
        captured: dict[str, object] = {}

        def fake_download_isd(station_id: str, city: str, years: list[int], output_dir: str) -> None:
            captured["station_id"] = station_id
            station_dir = Path(output_dir) / city / f"{station_id}_AUTO"
            station_dir.mkdir(parents=True, exist_ok=True)
            (station_dir / f"{station_id}-2000.gz").write_bytes(b"ok")

        def fake_download_nsrdb(city: str, lat: float, lon: float, years: list[int], output_dir: str) -> None:
            nsrdb_dir = Path(output_dir)
            nsrdb_dir.mkdir(parents=True, exist_ok=True)
            (nsrdb_dir / "nsrdb_miami_1998-2024.csv").write_text("Year,Month,Day,Hour,GHI\n")

        city_dir = data_dir / "CMIP6" / "Miami"
        city_dir.mkdir(parents=True, exist_ok=True)
        (city_dir / "mpi_esm1_2_lr_historical.csv").write_text("time,tas\n")
        (city_dir / "mpi_esm1_2_lr_ssp245.csv").write_text("time,tas\n")

        args = Namespace(
            city=["Miami", "Los_Angeles"],
            isd_dir=None,
            nsrdb=None,
            cmip6_dir=None,
            skip_download=False,
            download_data=["isd", "nsrdb", "cmip6"],
            download_years=None,
            isd_station=["Miami=722020-12839", "Los_Angeles=722950-23174"],
            isd_radius=50.0,
            ssp="ssp245",
            cmip6_models=["mpi_esm1_2_lr"],
            cmip6_experiments=None,
            cmip6_years=None,
        )

        with (
            patch("samy_wgen.pipeline.download_data.select_best_isd_station") as station_mock,
            patch("samy_wgen.pipeline.download_data._city_meta", return_value={"lat": 25.7617, "lon": -80.1918}),
            patch("samy_wgen.pipeline.download_data.download_isd", side_effect=fake_download_isd),
            patch("samy_wgen.pipeline.download_data.download_nsrdb", side_effect=fake_download_nsrdb),
        ):
            _ensure_pipeline_inputs(args, data_dir, "Miami")

        station_mock.assert_not_called()
        assert captured["station_id"] == "722020-12839"

    def test_batch_cmip6_downloads_missing_cities_once(self, tmp_path):
        from samy_wgen.pipeline.run_pipeline import _ensure_batch_cmip6_inputs

        data_dir = tmp_path / "data"
        existing = data_dir / "CMIP6" / "Miami"
        existing.mkdir(parents=True, exist_ok=True)
        (existing / "mpi_esm1_2_lr_historical.csv").write_text("time,tas\n")
        (existing / "mpi_esm1_2_lr_ssp245.csv").write_text("time,tas\n")

        captured: dict[str, object] = {}

        def fake_download_cmip6_for_cities(cities, models, experiments, year_range, output_dir, variables=None):
            captured["cities"] = cities
            captured["models"] = models
            captured["experiments"] = experiments
            captured["year_range"] = year_range
            captured["output_dir"] = output_dir

        args = Namespace(
            city=["Miami", "Los_Angeles"],
            cmip6_dir=None,
            skip_download=False,
            download_data=["cmip6"],
            ssp="ssp245",
            cmip6_models=["mpi_esm1_2_lr"],
            cmip6_experiments=None,
            cmip6_years=None,
        )

        with patch("samy_wgen.pipeline.download_data.download_cmip6_for_cities", side_effect=fake_download_cmip6_for_cities):
            cmip6_dir = _ensure_batch_cmip6_inputs(args, data_dir, ["Miami", "Los_Angeles"])

        assert cmip6_dir == data_dir / "CMIP6"
        assert captured["cities"] == ["Los_Angeles"]
        assert captured["models"] == ["mpi_esm1_2_lr"]
        assert captured["experiments"] == ["historical", "ssp2_4_5"]


class TestDeterministicArtifacts:
    def test_stage4_uses_mixed_resolution_inputs(self, tmp_path, monkeypatch):
        from samy_wgen.pipeline import run_climate_delta, run_pipeline

        index = pd.DatetimeIndex(["1991-01-01"])
        daily_df = pd.DataFrame({"tas": [280.0], "ps": [101000.0]}, index=index)
        monthly_df = pd.DataFrame({"sfcWind": [2.0]}, index=index)
        captured: dict[str, object] = {}

        monkeypatch.setattr(
            run_climate_delta,
            "load_cmip6_or_cordex6",
            lambda **_kwargs: (daily_df, monthly_df, {"model_name": "CNRM-ESM2-1"}),
        )
        monkeypatch.setattr(run_climate_delta, "load_rsvar_output", lambda _path: (pd.DataFrame(index=index), {}))

        def fake_baseline(daily, monthly, *_args, **_kwargs):
            captured["daily"] = daily
            captured["monthly"] = monthly
            return {"tas_C": {"resolution": "daily"}}

        monkeypatch.setattr(run_climate_delta, "compute_mixed_resolution_baseline", fake_baseline)
        monkeypatch.setattr(
            run_climate_delta,
            "apply_mixed_resolution_climate_delta",
            lambda _rsvar, daily, monthly, **_kwargs: (daily.assign(temp=20.0), {}),
        )

        result = run_pipeline._stage4_climate_delta(
            rsvar_xlsx=tmp_path / "3_rsvar.xlsx",
            cmip6_dir=tmp_path / "CMIP6",
            city="Beijing",
            ssp="ssp245",
            lat=39.9,
            baseline_start=1991,
            baseline_end=2010,
            out_dir=tmp_path,
            overwrite=True,
        )

        assert captured["daily"] is daily_df
        assert captured["monthly"] is monthly_df
        assert "temp" in result.columns

    def test_output_historical_epw_requires_actual_2000(self, tmp_path):
        from samy_wgen.pipeline.run_pipeline import _output_historical_epw

        merged_csv = tmp_path / "2_isd_solar_lst.csv"
        df = pd.DataFrame(
            {"DATE": pd.date_range("1999-01-01", periods=24, freq="h"), "temp": [1.0] * 24, "dewpoint": [0.0] * 24}
        )
        df.to_csv(merged_csv, index=False)

        with pytest.raises(SystemExit, match="does not contain year 2000"):
            _output_historical_epw(
                merged_csv=merged_csv,
                city="Miami",
                meta={"lat": 1.0, "lon": 2.0, "tz": -5.0, "country": "US", "elevation_m": 0.0},
                out_dir=tmp_path,
                overwrite=True,
            )

    def test_output_ddy_files_uses_expected_period_slices(self, tmp_path):
        from samy_wgen.pipeline.run_pipeline import _output_ddy_files

        merged_csv = tmp_path / "2_isd_solar_lst.csv"
        historical = pd.DataFrame(
            {
                "DATE": pd.date_range("1991-01-01", "2010-12-31 23:00:00", freq="h"),
                "temp": 20.0,
                "dewpoint": 15.0,
                "pressure": 101325.0,
                "wind_speed": 3.0,
            }
        )
        historical.to_csv(merged_csv, index=False)

        future_index = pd.date_range("2021-01-01", "2090-12-31 23:00:00", freq="h")
        future_df = pd.DataFrame(
            {
                "temp": 28.0,
                "dewpoint": 22.0,
                "pressure": 101325.0,
                "wind_speed": 4.0,
            },
            index=future_index,
        )

        written = _output_ddy_files(
            merged_csv=merged_csv,
            output_df=future_df,
            city="Miami",
            ssp="ssp245",
            meta={"lat": 25.76, "lon": -80.19, "tz": -5.0, "country": "US", "elevation_m": 2.0},
            baseline_start=1991,
            baseline_end=2010,
            out_dir=tmp_path,
            overwrite=True,
        )

        names = {path.name for path in written}
        assert names == {
            "Miami_baseline_1991_2010.ddy",
            "Miami_ssp245_2030_2021_2040.ddy",
            "Miami_ssp245_2050_2041_2060.ddy",
            "Miami_ssp245_2080_2071_2090.ddy",
        }


class TestEPWHumidityFallback:
    def test_epw_writer_uses_dewpoint_when_rh_missing(self, tmp_path):
        from samy_wgen.pipeline.gen_samy import _write_epw_for_year

        index = pd.date_range("2000-01-01 00:00:00", periods=8760, freq="h")
        df = pd.DataFrame(
            {
                "temp": 30.0,
                "dewpoint": 20.0,
                "pressure": 101325.0,
                "wind_speed": 2.0,
                "GHI": 0.0,
                "DNI": 0.0,
                "DHI": 0.0,
            },
            index=index,
        )

        out_path = tmp_path / "fallback.epw"
        _write_epw_for_year(df, 2000, out_path)

        first_row = out_path.read_text().splitlines()[8].split(",")
        assert float(first_row[7]) == pytest.approx(20.0, abs=0.2)
        assert 50.0 < float(first_row[8]) < 60.0
