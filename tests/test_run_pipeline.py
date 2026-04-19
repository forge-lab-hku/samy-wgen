from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

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

        found = _find_nsrdb_csv("Los Angeles", tmp_path)

        assert found == target

    def test_has_required_cmip6_data_accepts_underscored_city_dir(self, tmp_path):
        from samy_wgen.pipeline.run_pipeline import _has_required_cmip6_data

        city_dir = tmp_path / "CMIP6" / "Los_Angeles"
        city_dir.mkdir(parents=True, exist_ok=True)
        (city_dir / "mpi_esm1_2_lr_historical.csv").write_text("time,tas\n")
        (city_dir / "mpi_esm1_2_lr_ssp245.csv").write_text("time,tas\n")

        assert _has_required_cmip6_data("Los Angeles", tmp_path / "CMIP6", "ssp245", ["mpi_esm1_2_lr"])


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

        def fake_download_cmip6(
            city: str,
            lat: float,
            lon: float,
            models: list[str],
            experiments: list[str],
            year_range: tuple[int, int] | None,
            output_dir: str,
            variables=None,
        ) -> None:
            captured["cmip6"] = {
                "lat": lat,
                "lon": lon,
                "models": models,
                "experiments": experiments,
                "year_range": year_range,
                "output_dir": output_dir,
            }
            city_dir = Path(output_dir) / city
            city_dir.mkdir(parents=True, exist_ok=True)
            (city_dir / "mpi_esm1_2_lr_historical.csv").write_text("time,tas\n")
            (city_dir / "mpi_esm1_2_lr_ssp245.csv").write_text("time,tas\n")

        args = Namespace(
            city="Miami",
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
            patch("samy_wgen.pipeline.download_data.download_cmip6", side_effect=fake_download_cmip6),
        ):
            isd_dir, nsrdb_csv, cmip6_dir = _ensure_pipeline_inputs(args, data_dir)

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
        assert captured["cmip6"]["experiments"] == ["historical", "ssp2_4_5"]
