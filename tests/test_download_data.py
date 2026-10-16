"""Tests for samy_wgen.pipeline.download_data.

Fast unit tests (no network, mocked IO) run with:
    pytest tests/test_download_data.py -v

Slow integration tests (real network) run with:
    pytest -m slow tests/test_download_data.py -v -s
"""

from __future__ import annotations

import io
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_nc_bytes(
    times: pd.DatetimeIndex,
    lat: float = 25.76,
    lon: float = -80.19,
    variables: dict[str, np.ndarray] | None = None,
) -> bytes:
    """Build a minimal in-memory NetCDF dataset and return its raw bytes."""
    import xarray as xr

    if variables is None:
        variables = {"tas": np.random.uniform(295, 305, len(times))}

    data_vars = {
        name: xr.DataArray(
            vals.reshape(len(times), 1, 1),
            dims=["time", "lat", "lon"],
        )
        for name, vals in variables.items()
    }
    ds = xr.Dataset(
        data_vars,
        coords={
            "time": times,
            "lat": [lat],
            "lon": [lon],
        },
    )
    buf = BytesIO()
    ds.to_netcdf(buf)
    return buf.getvalue()


def _make_isd_history_csv(stations: list[dict]) -> str:
    """Return a minimal isd-history.csv string."""
    rows = []
    for s in stations:
        rows.append(
            f"{s['USAF']},{s['WBAN']},{s['name']},US,FL,KMIA,"
            f"{s['lat']},{s['lon']},10,"
            f"{s.get('begin', 19700101)},{s.get('end', 20241231)}"
        )
    header = "USAF,WBAN,STATION NAME,CTRY,ST,ICAO,LAT,LON,ELEV(M),BEGIN,END"
    return header + "\n" + "\n".join(rows)


# ─── _city_meta ───────────────────────────────────────────────────────────────

class TestCityMeta:
    def test_known_city_miami(self):
        from samy_wgen.pipeline.download_data import _city_meta

        meta = _city_meta("Miami")
        assert abs(meta["lat"] - 25.7840) < 0.01
        assert abs(meta["lon"] - (-80.2101)) < 0.01
        assert meta["cordex_domain"] == "NAM-22"

    def test_case_insensitive(self):
        from samy_wgen.pipeline.download_data import _city_meta

        meta = _city_meta("miami")
        assert meta["lat"] > 0

    def test_unknown_city_raises(self):
        from samy_wgen.pipeline.download_data import _city_meta

        with pytest.raises(ValueError, match="not found"):
            _city_meta("Atlantis")


# ─── _nc_bytes_to_point_df ────────────────────────────────────────────────────

class TestNcBytesToPointDf:
    def test_basic_extraction(self):
        from samy_wgen.pipeline.download_data import _nc_bytes_to_point_df

        times = pd.date_range("2000-01-01", periods=10, freq="D")
        values = np.arange(10, dtype=float)
        nc_bytes = _make_nc_bytes(times, variables={"tas": values})

        df = _nc_bytes_to_point_df(nc_bytes, lat=25.76, lon=-80.19)

        assert "time" in df.columns
        assert "tas" in df.columns
        assert len(df) == 10
        np.testing.assert_allclose(df["tas"].values, values)

    def test_nearest_grid_selection(self):
        """Extraction picks the nearest grid cell when the exact point is not in the grid."""
        import xarray as xr

        from samy_wgen.pipeline.download_data import _nc_bytes_to_point_df

        times = pd.date_range("2000-01-01", periods=5, freq="D")
        # Two grid cells at lat 25.0 and 26.0
        tas = np.array([[1.0, 2.0]] * 5).reshape(5, 2, 1)
        ds = xr.Dataset(
            {"tas": xr.DataArray(tas, dims=["time", "lat", "lon"])},
            coords={"time": times, "lat": [25.0, 26.0], "lon": [-80.0]},
        )
        buf = BytesIO()
        ds.to_netcdf(buf)

        # Target lat 25.3 → nearest grid is lat 25.0 → value 1.0
        df = _nc_bytes_to_point_df(buf.getvalue(), lat=25.3, lon=-80.0)
        np.testing.assert_allclose(df["tas"].values, 1.0)

    def test_zip_wrapped_nc(self, tmp_path):
        """zip-wrapped NetCDF (CDS format) is handled transparently."""
        import zipfile

        from samy_wgen.pipeline.download_data import _nc_bytes_to_point_df

        times = pd.date_range("2000-01-01", periods=3, freq="D")
        inner_nc = _make_nc_bytes(times, variables={"pr": np.ones(3)})

        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("data.nc", inner_nc)
        zip_bytes = zip_buf.getvalue()
        assert zip_bytes[:2] == b"PK"  # confirm it's a zip

        df = _nc_bytes_to_point_df(zip_bytes, lat=25.76, lon=-80.19)
        assert "pr" in df.columns
        assert len(df) == 3

    def test_no_time_variable_raises(self):
        """Dataset with no time-indexed variable raises ValueError."""
        import xarray as xr

        from samy_wgen.pipeline.download_data import _nc_bytes_to_point_df

        ds = xr.Dataset({"static": xr.DataArray([1.0])})
        buf = BytesIO()
        ds.to_netcdf(buf)

        with pytest.raises(ValueError, match="lat/lon coordinates not found"):
            _nc_bytes_to_point_df(buf.getvalue(), lat=25.76, lon=-80.19)

    def test_longitude_360_convention(self):
        """Grid using [0, 360] longitude convention is handled correctly."""
        import xarray as xr

        from samy_wgen.pipeline.download_data import _nc_bytes_to_point_df

        times = pd.date_range("2000-01-01", periods=4, freq="D")
        # City lon is -80.19, grid uses [0,360] → wraps to ~279.81
        lon_grid = 279.81
        tas = np.array([10.0, 11.0, 12.0, 13.0]).reshape(4, 1, 1)
        ds = xr.Dataset(
            {"tas": xr.DataArray(tas, dims=["time", "lat", "lon"])},
            coords={"time": times, "lat": [25.76], "lon": [lon_grid]},
        )
        buf = BytesIO()
        ds.to_netcdf(buf)

        df = _nc_bytes_to_point_df(buf.getvalue(), lat=25.76, lon=-80.19)
        assert list(df["tas"]) == pytest.approx([10.0, 11.0, 12.0, 13.0])


class TestExtractCmip6FromFiles:
    def test_merges_local_netcdf_variables_at_nearest_city_grid(self, tmp_path):
        import xarray as xr

        from samy_wgen.pipeline.download_data import extract_cmip6_from_files

        times = pd.date_range("2000-01-01", periods=2, freq="D")
        coords = {"time": times, "lat": [25.0, 26.0], "lon": [-80.5, -80.0]}
        tas = xr.Dataset({"tas": (("time", "lat", "lon"), np.full((2, 2, 2), 300.0))}, coords=coords)
        pr_values = np.zeros((2, 2, 2))
        pr_values[:, 1, 1] = [1.0, 2.0]
        pr = xr.Dataset({"pr": (("time", "lat", "lon"), pr_values)}, coords=coords)
        later_times = pd.date_range("2000-01-03", periods=1, freq="D")
        tas_later = xr.Dataset({"tas": (("time", "lat", "lon"), np.full((1, 2, 2), 301.0))}, coords={**coords, "time": later_times})
        tas_path, pr_path, tas_later_path = tmp_path / "tas.nc", tmp_path / "pr.nc", tmp_path / "tas_later.nc"
        tas.to_netcdf(tas_path)
        pr.to_netcdf(pr_path)
        tas_later.to_netcdf(tas_later_path)

        outputs = extract_cmip6_from_files(
            cities=["Miami"], input_files=[tas_path, pr_path, tas_later_path], output_dir=tmp_path / "out", output_name="test",
        )

        out = pd.read_csv(outputs["Miami"])
        assert outputs["Miami"] == tmp_path / "out" / "Miami" / "test.csv"
        assert out.columns.tolist() == ["time", "tas", "pr"]
        assert out["tas"].tolist() == [300.0, 300.0, 301.0]
        assert out["pr"].tolist()[:2] == [1.0, 2.0]
        assert pd.isna(out["pr"].iloc[2])

    def test_reads_zip_disguised_as_nc(self, tmp_path):
        import zipfile

        from samy_wgen.pipeline.download_data import extract_cmip6_from_files

        times = pd.date_range("2000-01-01", periods=1, freq="D")
        source_nc = tmp_path / "source.nc"
        import xarray as xr
        xr.Dataset(
            {"tas": (("time", "lat", "lon"), np.array([[[300.0]]]))},
            coords={"time": times, "lat": [25.76], "lon": [-80.19]},
        ).to_netcdf(source_nc)
        archive_path = tmp_path / "zip_disguised_as_nc.nc"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.write(source_nc, arcname="inside.nc")

        outputs = extract_cmip6_from_files(
            cities=["Miami"], input_files=[archive_path], output_dir=tmp_path / "out", output_name="zip",
        )

        assert pd.read_csv(outputs["Miami"])["tas"].tolist() == [300.0]


# ─── list_isd_stations (unit, mocked) ────────────────────────────────────────

class TestListIsdStationsUnit:
    def _fake_isd_history(self) -> pd.DataFrame:
        csv = _make_isd_history_csv([
            {"USAF": "722020", "WBAN": "12839", "name": "MIAMI INTL AP",
             "lat": 25.76, "lon": -80.19},
            {"USAF": "722010", "WBAN": "12844", "name": "MIAMI EXEC AP",
             "lat": 25.65, "lon": -80.43},
            {"USAF": "999999", "WBAN": "99999", "name": "FAR AWAY STATION",
             "lat": 40.71, "lon": -74.00},  # ~2100 km away, should be excluded
        ])
        return pd.read_csv(io.StringIO(csv))

    def test_filters_by_radius(self, tmp_path):
        from samy_wgen.pipeline.download_data import list_isd_stations

        with patch("samy_wgen.pipeline.download_data._load_isd_history",
                   return_value=self._fake_isd_history()):
            result = list_isd_stations(25.76, -80.19, radius_km=100,
                                       cache_dir=str(tmp_path))

        assert len(result) == 2
        assert "MIAMI INTL AP" in result["STATION NAME"].values

    def test_sorts_by_distance(self, tmp_path):
        from samy_wgen.pipeline.download_data import list_isd_stations

        with patch("samy_wgen.pipeline.download_data._load_isd_history",
                   return_value=self._fake_isd_history()):
            result = list_isd_stations(25.76, -80.19, radius_km=100,
                                       cache_dir=str(tmp_path))

        # Miami INTL (closer) should come first
        assert result.iloc[0]["STATION NAME"] == "MIAMI INTL AP"

    def test_usaf_wban_column_formatted(self, tmp_path):
        from samy_wgen.pipeline.download_data import list_isd_stations

        with patch("samy_wgen.pipeline.download_data._load_isd_history",
                   return_value=self._fake_isd_history()):
            result = list_isd_stations(25.76, -80.19, radius_km=100,
                                       cache_dir=str(tmp_path))

        assert "722020-12839" in result["USAF-WBAN"].values

    def test_no_stations_in_radius(self, tmp_path, capsys):
        from samy_wgen.pipeline.download_data import list_isd_stations

        with patch("samy_wgen.pipeline.download_data._load_isd_history",
                   return_value=self._fake_isd_history()):
            result = list_isd_stations(0.0, 0.0, radius_km=1,
                                       cache_dir=str(tmp_path))

        assert result.empty
        captured = capsys.readouterr()
        assert "No stations found" in captured.out


# ─── download_isd (unit, mocked HTTPS) ───────────────────────────────────────

class TestDownloadIsdUnit:
    def test_bad_station_id_raises(self, tmp_path):
        from samy_wgen.pipeline.download_data import download_isd

        with pytest.raises(ValueError, match="USAF-WBAN"):
            download_isd("722020", "Miami", [2000], output_dir=str(tmp_path))

    def test_get_isd_station_coverage_returns_begin_end_years(self, tmp_path):
        from samy_wgen.pipeline.download_data import get_isd_station_coverage

        fake_history = pd.DataFrame({
            "USAF": ["722020"], "WBAN": ["12839"],
            "STATION NAME": ["MIAMI INTL AP"], "BEGIN": [19700101], "END": [20241231],
        })

        with patch("samy_wgen.pipeline.download_data._load_isd_history", return_value=fake_history):
            coverage = get_isd_station_coverage("722020-12839", cache_dir=str(tmp_path))

        assert coverage["station_id"] == "722020-12839"
        assert coverage["station_name"] == "MIAMI_INTL_AP"
        assert coverage["begin_year"] == 1970
        assert coverage["end_year"] == 2024
        assert coverage["coverage_years"] == 55

    def test_skips_existing_files(self, tmp_path):
        from samy_wgen.pipeline.download_data import download_isd

        # Pre-create the expected output file so it should be skipped
        station_dir = tmp_path / "Miami" / "722020-12839_MIAMI_INTL_AP"
        station_dir.mkdir(parents=True)
        existing = station_dir / "722020-12839-2000.gz"
        existing.write_bytes(b"fake gz content")

        fake_history = pd.DataFrame({
            "USAF": ["722020"], "WBAN": ["12839"],
            "STATION NAME": ["MIAMI INTL AP"], "BEGIN": [19700101], "END": [20241231],
        })

        get_mock = MagicMock()

        with (
            patch("samy_wgen.pipeline.download_data._load_isd_history",
                  return_value=fake_history),
            patch("requests.get", get_mock),
        ):
            download_isd("722020-12839", "Miami", [2000], output_dir=str(tmp_path))

        get_mock.assert_not_called()

    def test_downloads_missing_years(self, tmp_path):
        from samy_wgen.pipeline.download_data import download_isd

        fake_history = pd.DataFrame({
            "USAF": ["722020"], "WBAN": ["12839"],
            "STATION NAME": ["MIAMI INTL AP"], "BEGIN": [19700101], "END": [20241231],
        })

        downloaded: list[str] = []

        def fake_get(url: str, **kwargs):
            downloaded.append(url.rsplit("/", 1)[-1])
            response = MagicMock()
            response.iter_content.return_value = [b"fake gz"]
            return response

        with (
            patch("samy_wgen.pipeline.download_data._load_isd_history",
                  return_value=fake_history),
            patch("requests.get", side_effect=fake_get),
        ):
            download_isd(
                "722020-12839", "Miami", [2001, 2002], output_dir=str(tmp_path)
            )

        assert "722020-12839-2001.gz" in downloaded
        assert "722020-12839-2002.gz" in downloaded

    def test_filters_requested_years_to_station_metadata_coverage(self, tmp_path):
        from samy_wgen.pipeline.download_data import download_isd

        fake_history = pd.DataFrame({
            "USAF": ["722020"], "WBAN": ["12839"],
            "STATION NAME": ["MIAMI INTL AP"], "BEGIN": [20000101], "END": [20011231],
        })

        downloaded: list[str] = []

        def fake_get(url: str, **kwargs):
            downloaded.append(url.rsplit("/", 1)[-1])
            response = MagicMock()
            response.iter_content.return_value = [b"fake gz"]
            return response

        with (
            patch("samy_wgen.pipeline.download_data._load_isd_history", return_value=fake_history),
            patch("requests.get", side_effect=fake_get),
        ):
            download_isd(
                "722020-12839", "Miami", [1999, 2000, 2001, 2002], output_dir=str(tmp_path)
            )

        assert downloaded == ["722020-12839-2000.gz", "722020-12839-2001.gz"]

    def test_continues_after_http_failure(self, tmp_path, caplog):
        from samy_wgen.pipeline.download_data import download_isd

        fake_history = pd.DataFrame({
            "USAF": ["722020"], "WBAN": ["12839"],
            "STATION NAME": ["MIAMI INTL AP"], "BEGIN": [20000101], "END": [20011231],
        })
        failed = MagicMock()
        failed.raise_for_status.side_effect = RuntimeError("HTTP 503")
        succeeded = MagicMock()
        succeeded.iter_content.return_value = [b"fake gz"]

        with (
            patch("samy_wgen.pipeline.download_data._load_isd_history", return_value=fake_history),
            patch("requests.get", side_effect=[failed, succeeded]),
        ):
            download_isd("722020-12839", "Miami", [2000, 2001], output_dir=str(tmp_path))

        station_dir = tmp_path / "Miami" / "722020-12839_MIAMI_INTL_AP"
        assert not (station_dir / "722020-12839-2000.gz").exists()
        assert (station_dir / "722020-12839-2001.gz").read_bytes() == b"fake gz"
        assert "2000 failed: HTTP 503" in caplog.text


# ─── download_nsrdb (unit, mocked HTTP) ──────────────────────────────────────

class TestDownloadNsrdbUnit:
    def _mock_nsrdb_csv(self, year: int = 2000) -> str:
        header = "Source,Location ID,City,State,Country,Latitude,Longitude,Time Zone,Elevation\n"
        header += "NSRDB,123,Miami,FL,US,25.76,-80.19,-5,4\n"
        data = "Year,Month,Day,Hour,Minute,GHI,DNI,DHI,air_temperature,dew_point,relative_humidity,wind_speed,wind_direction,surface_pressure\n"
        for h in range(8760):
            data += f"{year},1,1,{h % 24},0,0,0,0,25.0,20.0,75.0,3.5,180,1013\n"
        return header + data

    def test_missing_env_vars_raises(self, tmp_path, monkeypatch):
        from samy_wgen.pipeline.download_data import download_nsrdb

        with patch(
            "samy_wgen.pipeline.download_data._load_nsrdb_credentials",
            side_effect=EnvironmentError("NSRDB_API_KEY missing"),
        ):
            with pytest.raises(EnvironmentError, match="NSRDB_API_KEY"):
                download_nsrdb("Miami", 25.76, -80.19, [2000], output_dir=str(tmp_path))

    def test_saves_merged_csv(self, tmp_path, monkeypatch):
        from samy_wgen.pipeline.download_data import download_nsrdb

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = self._mock_nsrdb_csv(2000)

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with (
            patch("samy_wgen.pipeline.download_data._load_nsrdb_credentials", return_value=("test_key", "test@example.com")),
            patch(
                "samy_wgen.pipeline.download_data._resolve_nsrdb_sources_with_dynamic_years",
                return_value=[("https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download", 1998, 2024)],
            ),
            patch("requests.Session", return_value=mock_session),
        ):
            download_nsrdb(
                "Miami", 25.76, -80.19, [2000], output_dir=str(tmp_path)
            )

        output_files = list(tmp_path.glob("nsrdb_miami_*.csv"))
        assert len(output_files) == 1
        df = pd.read_csv(output_files[0])
        assert "GHI" in df.columns

    def test_filters_years_outside_coverage(self, tmp_path, monkeypatch):
        """Years outside query-reported coverage are silently skipped."""
        from samy_wgen.pipeline.download_data import download_nsrdb

        selection = {
            "query_success": True,
            "query_error": None,
            "fallback_used": False,
            "fallback_reason": "",
            "raw_discovered_sources": [],
            "candidate_groups": [
                {
                    "group_key": "goes_aggregated",
                    "label": "goes_aggregated",
                    "source_keys": ["goes_aggregated"],
                    "years": list(range(1998, 2025)),
                    "year_ranges": "1998-2024",
                    "coverage_years": 27,
                }
            ],
            "selected_group": {
                "group_key": "goes_aggregated",
                "label": "goes_aggregated",
                "source_keys": ["goes_aggregated"],
                "years": list(range(1998, 2025)),
                "year_ranges": "1998-2024",
                "coverage_years": 27,
            },
            "selected_sources": [
                ("https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download", 1998, 2024)
            ],
        }

        with (
            patch("samy_wgen.pipeline.download_data._load_nsrdb_credentials", return_value=("test_key", "test@example.com")),
            patch("samy_wgen.pipeline.download_data._resolve_nsrdb_source_selection", return_value=selection),
            patch("requests.Session") as mock_session,
        ):
            download_nsrdb(
                "Miami", 25.76, -80.19, [1980, 2030],  # both out of range
                output_dir=str(tmp_path),
            )
        mock_session.return_value.get.assert_not_called()

    def test_dynamic_year_query_updates_open_ended_goes_range(self):
        from samy_wgen.pipeline.download_data import _resolve_nsrdb_sources_with_dynamic_years

        payload = {
            "outputs": [
                {
                    "availableYears": [1998, 1999, 2023, 2024, "tmy"],
                    "links": [
                        {
                            "year": 2024,
                            "interval": 30,
                            "link": "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv?names=2024",
                        }
                    ],
                }
            ]
        }
        session = MagicMock()
        session.get.return_value.raise_for_status = MagicMock()
        session.get.return_value.json.return_value = payload

        sources = _resolve_nsrdb_sources_with_dynamic_years(
            city="Miami",
            lat=25.76,
            lon=-80.19,
            api_key="test_key",
            session=session,
        )

        assert sources == [
            ("https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download", 1998, 2024)
        ]

    def test_dynamic_year_query_falls_back_to_static_closed_range_when_query_misses_source(self):
        from samy_wgen.pipeline.download_data import _resolve_nsrdb_sources_with_dynamic_years

        session = MagicMock()
        session.get.return_value.raise_for_status = MagicMock()
        session.get.return_value.json.return_value = {"outputs": []}

        with patch(
            "samy_wgen.pipeline.download_data._resolve_nsrdb_source",
            return_value=[
                ("https://developer.nlr.gov/api/nsrdb/v2/solar/himawari7-download", 2011, 2015),
                ("https://developer.nlr.gov/api/nsrdb/v2/solar/himawari-download", 2016, 2020),
            ],
        ):
            sources = _resolve_nsrdb_sources_with_dynamic_years(
                city="AnyCity",
                lat=35.68,
                lon=139.76,
                api_key="test_key",
                session=session,
            )

        assert sources == [
            ("https://developer.nlr.gov/api/nsrdb/v2/solar/himawari7-download", 2011, 2015),
            ("https://developer.nlr.gov/api/nsrdb/v2/solar/himawari-download", 2016, 2020),
        ]

    def test_selects_largest_coverage_group_from_query(self):
        from samy_wgen.pipeline.download_data import _resolve_nsrdb_source_selection

        payload = {
            "outputs": [
                {
                    "availableYears": list(range(2011, 2016)),
                    "links": [
                        {"link": "https://developer.nlr.gov/api/nsrdb/v2/solar/himawari7-download.csv?names=2015"}
                    ],
                },
                {
                    "availableYears": list(range(2016, 2021)),
                    "links": [
                        {"link": "https://developer.nlr.gov/api/nsrdb/v2/solar/himawari-download.csv?names=2020"}
                    ],
                },
                {
                    "availableYears": list(range(2000, 2015)),
                    "links": [
                        {"link": "https://developer.nlr.gov/api/nsrdb/v2/solar/suny-india-download.csv?names=2014"}
                    ],
                },
            ]
        }
        session = MagicMock()
        session.get.return_value.raise_for_status = MagicMock()
        session.get.return_value.json.return_value = payload

        selection = _resolve_nsrdb_source_selection(
            city="Chennai",
            lat=13.08,
            lon=80.27,
            api_key="test_key",
            session=session,
        )

        assert selection["selected_group"]["group_key"] == "suny_india"
        assert selection["selected_group"]["coverage_years"] == 15
        assert selection["fallback_used"] is False

    def test_resolves_himawari_pair_as_single_group(self):
        from samy_wgen.pipeline.download_data import _resolve_nsrdb_source_selection

        payload = {
            "outputs": [
                {
                    "availableYears": list(range(2011, 2016)),
                    "links": [
                        {"link": "https://developer.nlr.gov/api/nsrdb/v2/solar/himawari7-download.csv?names=2015"}
                    ],
                },
                {
                    "availableYears": list(range(2016, 2021)),
                    "links": [
                        {"link": "https://developer.nlr.gov/api/nsrdb/v2/solar/himawari-download.csv?names=2020"}
                    ],
                },
            ]
        }
        session = MagicMock()
        session.get.return_value.raise_for_status = MagicMock()
        session.get.return_value.json.return_value = payload

        selection = _resolve_nsrdb_source_selection(
            city="Beijing",
            lat=39.90,
            lon=116.39,
            api_key="test_key",
            session=session,
        )

        assert selection["selected_group"]["group_key"] == "himawari_pair"
        assert selection["selected_group"]["coverage_years"] == 10
        assert selection["selected_sources"] == [
            ("https://developer.nlr.gov/api/nsrdb/v2/solar/himawari7-download", 2011, 2015),
            ("https://developer.nlr.gov/api/nsrdb/v2/solar/himawari-download", 2016, 2020),
        ]

    def test_query_failure_falls_back_to_static_india_source(self):
        from samy_wgen.pipeline.download_data import _resolve_nsrdb_source_selection

        session = MagicMock()
        session.get.side_effect = Exception("network down")

        selection = _resolve_nsrdb_source_selection(
            city="Ahmedabad",
            lat=23.02,
            lon=72.57,
            api_key="test_key",
            session=session,
        )

        assert selection["fallback_used"] is True
        assert selection["selected_group"]["group_key"] == "suny_india"

    def test_download_prints_selected_source_summary(self, tmp_path, capsys):
        from samy_wgen.pipeline.download_data import download_nsrdb

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = self._mock_nsrdb_csv(2000)

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        selection = {
            "query_success": True,
            "query_error": None,
            "fallback_used": False,
            "fallback_reason": "",
            "raw_discovered_sources": [
                {
                    "url": "https://developer.nlr.gov/api/nsrdb/v2/solar/suny-india-download",
                    "source_key": "suny_india",
                    "supported": True,
                    "years": list(range(2000, 2015)),
                    "year_ranges": "2000-2014",
                }
            ],
            "candidate_groups": [
                {
                    "group_key": "suny_india",
                    "label": "suny_india",
                    "source_keys": ["suny_india"],
                    "years": list(range(2000, 2015)),
                    "year_ranges": "2000-2014",
                    "coverage_years": 15,
                }
            ],
            "selected_group": {
                "group_key": "suny_india",
                "label": "suny_india",
                "source_keys": ["suny_india"],
                "years": list(range(2000, 2015)),
                "year_ranges": "2000-2014",
                "coverage_years": 15,
            },
            "selected_sources": [
                ("https://developer.nlr.gov/api/nsrdb/v2/solar/suny-india-download", 2000, 2014)
            ],
        }

        with (
            patch("samy_wgen.pipeline.download_data._load_nsrdb_credentials", return_value=("test_key", "test@example.com")),
            patch("samy_wgen.pipeline.download_data._resolve_nsrdb_source_selection", return_value=selection),
            patch("requests.Session", return_value=mock_session),
        ):
            download_nsrdb("Ahmedabad", 23.02, 72.57, [2000], output_dir=str(tmp_path))

        captured = capsys.readouterr()
        assert "Selected default: suny_india (2000-2014; 15 years)" in captured.out

    def test_retries_transient_failure_then_succeeds(self, tmp_path):
        from samy_wgen.pipeline.download_data import download_nsrdb

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = self._mock_nsrdb_csv(2000)

        session = MagicMock()
        session.get.side_effect = [Exception("tls eof"), mock_resp]

        with (
            patch("samy_wgen.pipeline.download_data._load_nsrdb_credentials", return_value=("test_key", "test@example.com")),
            patch(
                "samy_wgen.pipeline.download_data._resolve_nsrdb_sources_with_dynamic_years",
                return_value=[("https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download", 1998, 2024)],
            ),
            patch("requests.Session", return_value=session),
            patch("time.sleep"),
        ):
            download_nsrdb(
                "Miami", 25.76, -80.19, [2000], output_dir=str(tmp_path), max_retries=2
            )

        output_files = list(tmp_path.glob("nsrdb_miami_*.csv"))
        assert len(output_files) == 1
        assert session.get.call_count == 2

    def test_retries_exhausted_skips_year(self, tmp_path):
        from samy_wgen.pipeline.download_data import download_nsrdb

        session = MagicMock()
        session.get.side_effect = Exception("tls eof")

        selection = {
            "query_success": True,
            "query_error": None,
            "fallback_used": False,
            "fallback_reason": "",
            "raw_discovered_sources": [],
            "candidate_groups": [
                {
                    "group_key": "goes_aggregated",
                    "label": "goes_aggregated",
                    "source_keys": ["goes_aggregated"],
                    "years": list(range(1998, 2025)),
                    "year_ranges": "1998-2024",
                    "coverage_years": 27,
                }
            ],
            "selected_group": {
                "group_key": "goes_aggregated",
                "label": "goes_aggregated",
                "source_keys": ["goes_aggregated"],
                "years": list(range(1998, 2025)),
                "year_ranges": "1998-2024",
                "coverage_years": 27,
            },
            "selected_sources": [
                ("https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download", 1998, 2024)
            ],
        }

        with (
            patch("samy_wgen.pipeline.download_data._load_nsrdb_credentials", return_value=("test_key", "test@example.com")),
            patch("samy_wgen.pipeline.download_data._resolve_nsrdb_source_selection", return_value=selection),
            patch("requests.Session", return_value=session),
            patch("time.sleep"),
        ):
            download_nsrdb(
                "Miami", 25.76, -80.19, [2000], output_dir=str(tmp_path), max_retries=3
            )

        output_files = list(tmp_path.glob("nsrdb_miami_*.csv"))
        assert output_files == []
        assert session.get.call_count == 3


# ─── CLI argument parsing ─────────────────────────────────────────────────────

class TestCliParsing:
    def _parser(self):
        from samy_wgen.pipeline.download_data import _build_parser
        return _build_parser()

    def test_list_stations_requires_city(self):
        parser = self._parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["list-stations"])

    def test_list_stations_defaults(self):
        parser = self._parser()
        args = parser.parse_args(["list-stations", "--city", "Miami"])
        assert args.city == "Miami"
        assert args.radius == 150.0
        assert args.isd_dir == "data/ISD"

    def test_list_nsrdb_requires_city(self):
        parser = self._parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["list-nsrdb"])

    def test_list_nsrdb_accepts_multiple_cities(self):
        parser = self._parser()
        args = parser.parse_args(["list-nsrdb", "--city", "Miami", "Chennai"])
        assert args.command == "list-nsrdb"
        assert args.city == ["Miami", "Chennai"]

    def test_download_default_sources(self):
        parser = self._parser()
        args = parser.parse_args(["download", "--city", "Miami"])
        assert set(args.data) == {"isd", "nsrdb", "cmip6"}
        assert args.isd_station is None
        assert args.isd_radius == 50.0
        assert args.city == ["Miami"]

    def test_download_accepts_multiple_cities(self):
        parser = self._parser()
        args = parser.parse_args(["download", "--city", "Miami", "Los_Angeles"])
        assert args.city == ["Miami", "Los_Angeles"]

    def test_download_accepts_isd_station_mapping(self):
        parser = self._parser()
        args = parser.parse_args([
            "download",
            "--city", "Miami", "Los_Angeles",
            "--isd-station", "Miami=722020-12839", "Los_Angeles=722950-23174",
        ])
        assert args.isd_station == ["Miami=722020-12839", "Los_Angeles=722950-23174"]

    def test_download_accepts_single_city_legacy_isd_station(self):
        parser = self._parser()
        args = parser.parse_args([
            "download",
            "--city", "Miami",
            "--isd-station", "722020-12839",
        ])
        assert args.isd_station == ["722020-12839"]

    def test_download_custom_experiments(self):
        parser = self._parser()
        args = parser.parse_args([
            "download", "--city", "Miami",
            "--data", "cmip6",
            "--experiments", "historical", "ssp5_8_5",
        ])
        assert "ssp5_8_5" in args.experiments

    def test_download_cmip6_years(self):
        parser = self._parser()
        args = parser.parse_args([
            "download", "--city", "Miami", "--data", "cmip6",
            "--cmip6-years", "1990", "2010",
        ])
        assert args.cmip6_years == [1990, 2010]

    def test_download_cmip6_years_default_none(self):
        parser = self._parser()
        args = parser.parse_args(["download", "--city", "Miami", "--data", "cmip6"])
        assert args.cmip6_years is None
        assert args.cmip6_var is None

    def test_download_cmip6_vars(self):
        parser = self._parser()
        args = parser.parse_args([
            "download", "--city", "Miami", "--data", "cmip6",
            "--cmip6-var", "near_surface_air_temperature", "precipitation",
        ])
        assert args.cmip6_var == ["near_surface_air_temperature", "precipitation"]

    def test_download_cordex_options(self):
        parser = self._parser()
        args = parser.parse_args([
            "download", "--city", "Los_Angeles", "--data", "cordex",
            "--cordex-rcm", "ncar_regcm4",
            "--cordex-var", "2m_air_temperature", "surface_pressure",
            "--cordex-historical-years", "2000", "2000",
            "--cordex-rcp85-years", "2006", "2006",
            "--cordex-months", "01",
        ])
        assert args.data == ["cordex"]
        assert args.cordex_var == ["2m_air_temperature", "surface_pressure"]
        assert args.cordex_historical_years == [2000, 2000]
        assert args.cordex_rcp85_years == [2006, 2006]
        assert args.cordex_months == ["01"]

    def test_extract_cmip6_parses_multiple_inputs_and_cities(self):
        parser = self._parser()
        args = parser.parse_args([
            "extract-cmip6", "--city", "Miami", "Los_Angeles",
            "--input", "tas.nc", "pr.nc", "--cmip6-dir", "raw/CMIP6", "--name", "mpi_ssp245",
        ])
        assert args.command == "extract-cmip6"
        assert args.city == ["Miami", "Los_Angeles"]
        assert args.input == ["tas.nc", "pr.nc"]
        assert args.cmip6_dir == "raw/CMIP6"
        assert args.name == "mpi_ssp245"


class TestIsdStationSelection:
    def _parser(self):
        from samy_wgen.pipeline.download_data import _build_parser
        return _build_parser()

    def test_select_best_isd_station_prefers_best_coverage(self):
        from samy_wgen.pipeline.download_data import select_best_isd_station

        history = pd.read_csv(
            io.StringIO(
                _make_isd_history_csv(
                    [
                        {
                            "USAF": "111111",
                            "WBAN": "11111",
                            "name": "NEAR_SHORT",
                            "lat": 25.77,
                            "lon": -80.19,
                            "begin": 19900101,
                            "end": 20101231,
                        },
                        {
                            "USAF": "222222",
                            "WBAN": "22222",
                            "name": "FAR_LONG",
                            "lat": 26.05,
                            "lon": -80.19,
                            "begin": 19700101,
                            "end": 20241231,
                        },
                    ]
                )
            )
        )

        with (
            patch("samy_wgen.pipeline.download_data._resolve_coords", return_value=(25.7617, -80.1918)),
            patch("samy_wgen.pipeline.download_data._load_isd_history", return_value=history),
        ):
            station_id = select_best_isd_station("Miami", radius_km=50.0, cache_dir="data/ISD")

        assert station_id == "222222-22222"

    def test_check_station_subcommand(self):
        parser = self._parser()
        args = parser.parse_args(["check-station", "--isd-station", "722020-12839"])
        assert args.command == "check-station"
        assert args.isd_station == "722020-12839"
        assert args.years is None

    def test_list_models_subcommand(self):
        parser = self._parser()
        args = parser.parse_args(["list-cmip6"])
        assert args.command == "list-cmip6"
        assert args.resolution is None
        assert args.var is None
        assert args.detail is False

    def test_list_models_detail_flag(self):
        parser = self._parser()
        args = parser.parse_args(["list-cmip6", "--detail"])
        assert args.detail is True

    def test_list_models_filters(self):
        parser = self._parser()
        args = parser.parse_args(
            [
                "list-cmip6",
                "--experiment", "ssp2_4_5",
                "--var", "near_surface_air_temperature",
                "--resolution", "monthly",
            ]
        )
        assert args.experiment == "ssp2_4_5"
        assert args.var == "near_surface_air_temperature"
        assert args.resolution == "monthly"

    def test_list_cordex_subcommand(self):
        parser = self._parser()
        args = parser.parse_args(["list-cordex"])
        assert args.command == "list-cordex"
        assert args.domain is None
        assert args.detail is False

    def test_list_cordex_filters(self):
        parser = self._parser()
        args = parser.parse_args(
            [
                "list-cordex",
                "--detail",
                "--domain", "NAM-22",
                "--experiment", "historical",
                "--gcm", "mpi_m_mpi_esm_lr",
                "--rcm", "ncar_regcm4",
                "--ensemble", "r1i1p1",
                "--resolution", "0_22_degree_x_0_22_degree",
                "--var", "2m_air_temperature",
            ]
        )
        assert args.detail is True
        assert args.domain == "NAM-22"
        assert args.experiment == "historical"
        assert args.gcm == "mpi_m_mpi_esm_lr"
        assert args.rcm == "ncar_regcm4"
        assert args.ensemble == "r1i1p1"
        assert args.resolution == "0_22_degree_x_0_22_degree"
        assert args.var == "2m_air_temperature"


# ─── CDS constraint helpers ───────────────────────────────────────────────────

class TestCdsConstraintHelpers:
    def test_constraint_cache_key_stable(self):
        from samy_wgen.pipeline.download_data import _constraint_cache_key

        req1 = {"experiment": ["ssp2_4_5"], "temporal_resolution": ["daily"]}
        req2 = {"temporal_resolution": ["daily"], "experiment": ["ssp2_4_5"]}
        assert _constraint_cache_key("projections-cmip6", req1) == _constraint_cache_key("projections-cmip6", req2)

    def test_requested_values_available(self):
        from samy_wgen.pipeline.download_data import _requested_values_available

        constrained = {
            "model": ["mpi_esm1_2_lr"],
            "experiment": ["historical", "ssp2_4_5"],
        }
        request = {"model": ["mpi_esm1_2_lr"], "experiment": ["ssp2_4_5"]}
        assert _requested_values_available(constrained, request, ["model", "experiment"]) is True

    def test_requested_values_missing(self):
        from samy_wgen.pipeline.download_data import _requested_values_available

        constrained = {"model": ["mpi_esm1_2_lr"], "experiment": []}
        request = {"model": ["mpi_esm1_2_lr"], "experiment": ["ssp2_4_5"]}
        assert _requested_values_available(constrained, request, ["model", "experiment"]) is False

    def test_get_cmip6_variable_groups(self):
        from samy_wgen.pipeline.download_data import _get_cmip6_variable_groups

        def fake_constraints(collection_id, request):
            assert collection_id == "projections-cmip6"
            if request["temporal_resolution"] == ["daily"]:
                return {
                    "model": ["mpi_esm1_2_lr"],
                    "experiment": ["ssp2_4_5"],
                    "temporal_resolution": ["daily"],
                    "variable": [
                        "near_surface_air_temperature",
                        "precipitation",
                        "not_supported_here",
                    ],
                }
            return {
                "model": ["mpi_esm1_2_lr"],
                "experiment": ["ssp2_4_5"],
                "temporal_resolution": ["monthly"],
                "variable": [
                    "near_surface_air_temperature",
                    "surface_downwelling_shortwave_radiation",
                ],
            }

        with patch("samy_wgen.pipeline.download_data._apply_collection_constraints", side_effect=fake_constraints):
            groups = _get_cmip6_variable_groups("mpi_esm1_2_lr", "ssp2_4_5")

        assert groups == {
            "daily": ["near_surface_air_temperature", "precipitation"],
            "monthly": ["surface_downwelling_shortwave_radiation"],
        }

    def test_list_cmip6_models_invalid_filter_returns_empty(self):
        from samy_wgen.pipeline.download_data import _list_cmip6_models

        fake = {
            "model": [],
            "experiment": [],
            "variable": [],
            "temporal_resolution": [],
        }
        with patch("samy_wgen.pipeline.download_data._apply_collection_constraints", return_value=fake):
            out = _list_cmip6_models(experiment="ssp2_4_5", temporal_resolution="daily")
        assert out["model"] == []

    def test_list_cmip6_model_rows(self):
        from samy_wgen.pipeline.download_data import _list_cmip6_model_rows

        def fake_list_models(experiment=None, variable=None, temporal_resolution=None):
            assert experiment == "ssp2_4_5"
            assert variable is None
            assert temporal_resolution is None
            return {
                "model": ["mpi_esm1_2_lr"],
                "experiment": ["ssp2_4_5"],
                "variable": ["near_surface_air_temperature"],
                "temporal_resolution": ["daily", "monthly"],
            }

        def fake_constraints(collection_id, request):
            assert collection_id == "projections-cmip6"
            if request == {"model": ["mpi_esm1_2_lr"], "experiment": ["ssp2_4_5"], "temporal_resolution": ["daily"]}:
                return {"variable": ["near_surface_air_temperature", "precipitation"]}
            if request == {"model": ["mpi_esm1_2_lr"], "experiment": ["ssp2_4_5"], "temporal_resolution": ["monthly"]}:
                return {"variable": ["surface_downwelling_shortwave_radiation"]}
            return {
                "experiment": ["ssp2_4_5"],
                "temporal_resolution": ["daily", "monthly"],
            }

        with (
            patch("samy_wgen.pipeline.download_data._list_cmip6_models", side_effect=fake_list_models),
            patch("samy_wgen.pipeline.download_data._apply_collection_constraints", side_effect=fake_constraints),
        ):
            rows = _list_cmip6_model_rows(experiment="ssp2_4_5", show_progress=False)

        assert rows == [
            {
                "model": "mpi_esm1_2_lr",
                "experiment": "ssp2_4_5",
                "resolution": "daily",
                "variable": "tas,pr",
            },
            {
                "model": "mpi_esm1_2_lr",
                "experiment": "ssp2_4_5",
                "resolution": "monthly",
                "variable": "rsds",
            },
        ]

    def test_get_cmip6_variable_groups_prefers_highest_resolution(self):
        from samy_wgen.pipeline.download_data import _get_cmip6_variable_groups

        def fake_constraints(collection_id, request):
            assert collection_id == "projections-cmip6"
            if request["temporal_resolution"] == ["daily"]:
                return {
                    "model": ["mpi_esm1_2_lr"],
                    "experiment": ["ssp2_4_5"],
                    "temporal_resolution": ["daily"],
                    "variable": ["near_surface_air_temperature"],
                }
            if request["temporal_resolution"] == ["monthly"]:
                return {
                    "model": ["mpi_esm1_2_lr"],
                    "experiment": ["ssp2_4_5"],
                    "temporal_resolution": ["monthly"],
                    "variable": ["near_surface_air_temperature", "near_surface_wind_speed"],
                }
            return {
                "model": ["mpi_esm1_2_lr"],
                "experiment": ["ssp2_4_5"],
                "temporal_resolution": ["fixed"],
                "variable": ["near_surface_air_temperature"],
            }

        with patch("samy_wgen.pipeline.download_data._apply_collection_constraints", side_effect=fake_constraints):
            groups = _get_cmip6_variable_groups("mpi_esm1_2_lr", "ssp2_4_5")

        assert groups == {"daily": ["near_surface_air_temperature"], "monthly": ["near_surface_wind_speed"]}

    def test_filter_cmip6_variable_groups(self):
        from samy_wgen.pipeline.download_data import _filter_cmip6_variable_groups

        groups = {
            "daily": ["near_surface_air_temperature", "precipitation"],
            "monthly": ["surface_downwelling_shortwave_radiation"],
        }
        filtered = _filter_cmip6_variable_groups(groups, ["precipitation"])
        assert filtered == {"daily": ["precipitation"]}

    def test_list_cordex_options_invalid_filter_returns_empty(self):
        from samy_wgen.pipeline.download_data import _list_cordex_options

        fake = {
            "domain": [],
            "experiment": [],
            "gcm_model": [],
            "rcm_model": [],
            "ensemble_member": [],
            "horizontal_resolution": [],
            "variable": [],
        }
        with patch("samy_wgen.pipeline.download_data._apply_collection_constraints", return_value=fake):
            out = _list_cordex_options(domain="NAM-22", gcm="missing_model")
        assert out["gcm_model"] == []

    def test_list_cordex_rows(self):
        from samy_wgen.pipeline.download_data import _list_cordex_rows

        def fake_constraints(collection_id, request):
            assert collection_id == "projections-cordex-domains-single-levels"
            if request == {
                "gcm_model": ["mpi_m_mpi_esm_lr"],
                "rcm_model": ["ncar_regcm4"],
            }:
                return {
                    "domain": ["north_america"],
                    "experiment": ["historical"],
                    "gcm_model": ["mpi_m_mpi_esm_lr"],
                    "rcm_model": ["ncar_regcm4"],
                    "ensemble_member": ["r1i1p1", "r2i1p1"],
                    "horizontal_resolution": ["0_22_degree_x_0_22_degree"],
                }
            if request == {
                "domain": ["north_america"],
                "experiment": ["historical"],
                "gcm_model": ["mpi_m_mpi_esm_lr"],
                "rcm_model": ["ncar_regcm4"],
                "ensemble_member": ["r1i1p1"],
                "horizontal_resolution": ["0_22_degree_x_0_22_degree"],
            }:
                return {
                    "variable": [
                        "2m_air_temperature",
                        "mean_precipitation_flux",
                        "not_supported",
                    ]
                }
            raise AssertionError(f"Unexpected request: {request}")

        with patch(
            "samy_wgen.pipeline.download_data._list_cordex_options",
            return_value={
                "domain": ["NAM-22"],
                "experiment": ["historical"],
                "gcm_model": ["mpi_m_mpi_esm_lr"],
                "rcm_model": ["ncar_regcm4"],
                "ensemble_member": ["r1i1p1"],
                "horizontal_resolution": ["0_22_degree_x_0_22_degree"],
                "variable": ["2m_air_temperature", "mean_precipitation_flux"],
            },
        ), patch(
            "samy_wgen.pipeline.download_data._apply_collection_constraints",
            side_effect=fake_constraints,
        ):
            rows = _list_cordex_rows(show_progress=False)

        assert rows == [
            {
                "domain": "NAM-22",
                "experiment": "historical",
                "gcm_model": "mpi_m_mpi_esm_lr",
                "rcm_model": "ncar_regcm4",
                "ensemble_member": "r1i1p1",
                "resolution": "0_22_degree_x_0_22_degree",
                "variable": "2m_air_temperature,mean_precipitation_flux",
            }
        ]

    def test_list_cordex_rows_expand_ensemble(self):
        from samy_wgen.pipeline.download_data import _list_cordex_rows

        def fake_constraints(collection_id, request):
            assert collection_id == "projections-cordex-domains-single-levels"
            if request == {
                "gcm_model": ["mpi_m_mpi_esm_lr"],
                "rcm_model": ["ncar_regcm4"],
                "ensemble_member": ["r1i1p1"],
            }:
                return {
                    "domain": ["north_america"],
                    "experiment": ["historical"],
                    "gcm_model": ["mpi_m_mpi_esm_lr"],
                    "rcm_model": ["ncar_regcm4"],
                    "ensemble_member": ["r1i1p1"],
                    "horizontal_resolution": ["0_22_degree_x_0_22_degree"],
                }
            if request == {
                "domain": ["north_america"],
                "experiment": ["historical"],
                "gcm_model": ["mpi_m_mpi_esm_lr"],
                "rcm_model": ["ncar_regcm4"],
                "ensemble_member": ["r1i1p1"],
                "horizontal_resolution": ["0_22_degree_x_0_22_degree"],
            }:
                return {"variable": ["2m_air_temperature"]}
            raise AssertionError(f"Unexpected request: {request}")

        with patch(
            "samy_wgen.pipeline.download_data._list_cordex_options",
            return_value={
                "domain": ["NAM-22"],
                "experiment": ["historical"],
                "gcm_model": ["mpi_m_mpi_esm_lr"],
                "rcm_model": ["ncar_regcm4"],
                "ensemble_member": ["r1i1p1"],
                "horizontal_resolution": ["0_22_degree_x_0_22_degree"],
                "variable": ["2m_air_temperature"],
            },
        ), patch(
            "samy_wgen.pipeline.download_data._apply_collection_constraints",
            side_effect=fake_constraints,
        ):
            rows = _list_cordex_rows(show_progress=False, expand_ensemble=True)

        assert rows[0]["ensemble_member"] == "r1i1p1"

    def test_resolve_cordex_download_plan_prefers_domain_default_resolution(self):
        from samy_wgen.pipeline.download_data import _resolve_cordex_download_plan

        def fake_constraints(collection_id, request):
            assert collection_id == "projections-cordex-domains-single-levels"
            if request == {
                "domain": ["north_america"],
                "experiment": ["historical"],
                "gcm_model": ["mpi_m_mpi_esm_lr"],
                "rcm_model": ["ncar_regcm4"],
                "ensemble_member": ["r1i1p1"],
            }:
                return {
                    "domain": ["north_america"],
                    "experiment": ["historical"],
                    "gcm_model": ["mpi_m_mpi_esm_lr"],
                    "rcm_model": ["ncar_regcm4"],
                    "ensemble_member": ["r1i1p1"],
                    "horizontal_resolution": [
                        "0_11_degree_x_0_11_degree",
                        "0_22_degree_x_0_22_degree",
                    ],
                }
            if request == {
                "domain": ["north_america"],
                "experiment": ["historical"],
                "gcm_model": ["mpi_m_mpi_esm_lr"],
                "rcm_model": ["ncar_regcm4"],
                "ensemble_member": ["r1i1p1"],
                "horizontal_resolution": ["0_22_degree_x_0_22_degree"],
            }:
                return {
                    "variable": ["2m_air_temperature", "surface_pressure"]
                }
            raise AssertionError(f"Unexpected request: {request}")

        with patch("samy_wgen.pipeline.download_data._apply_collection_constraints", side_effect=fake_constraints):
            plan = _resolve_cordex_download_plan(
                cordex_domain="NAM-22",
                gcm="mpi_m_mpi_esm_lr",
                rcm="ncar_regcm4",
                ensemble="r1i1p1",
                experiments=["historical"],
                variables=["2m_air_temperature", "surface_pressure"],
            )

        assert plan == [
            {
                "domain": "north_america",
                "experiment": "historical",
                "gcm_model": "mpi_m_mpi_esm_lr",
                "rcm_model": "ncar_regcm4",
                "ensemble_member": "r1i1p1",
                "horizontal_resolution": "0_22_degree_x_0_22_degree",
                "variable": ["2m_air_temperature", "surface_pressure"],
            }
        ]

    def test_resolve_cordex_download_plan_auto_selects_first_ensemble(self):
        from samy_wgen.pipeline.download_data import _resolve_cordex_download_plan

        def fake_constraints(collection_id, request):
            assert collection_id == "projections-cordex-domains-single-levels"
            if request == {
                "domain": ["north_america"],
                "experiment": ["historical"],
                "gcm_model": ["mpi_m_mpi_esm_lr"],
                "rcm_model": ["ncar_regcm4"],
            }:
                return {
                    "domain": ["north_america"],
                    "experiment": ["historical"],
                    "gcm_model": ["mpi_m_mpi_esm_lr"],
                    "rcm_model": ["ncar_regcm4"],
                    "ensemble_member": ["r1i1p1", "r2i1p1"],
                    "horizontal_resolution": ["0_22_degree_x_0_22_degree"],
                }
            if request == {
                "domain": ["north_america"],
                "experiment": ["historical"],
                "gcm_model": ["mpi_m_mpi_esm_lr"],
                "rcm_model": ["ncar_regcm4"],
                "ensemble_member": ["r1i1p1"],
                "horizontal_resolution": ["0_22_degree_x_0_22_degree"],
            }:
                return {"variable": ["2m_air_temperature"]}
            raise AssertionError(f"Unexpected request: {request}")

        with patch("samy_wgen.pipeline.download_data._apply_collection_constraints", side_effect=fake_constraints):
            plan = _resolve_cordex_download_plan(
                cordex_domain="NAM-22",
                gcm="mpi_m_mpi_esm_lr",
                rcm="ncar_regcm4",
                ensemble=None,
                experiments=["historical"],
                variables=["2m_air_temperature"],
            )

        assert plan[0]["ensemble_member"] == "r1i1p1"


# ─── CMIP6 year_range filtering ───────────────────────────────────────────────

class TestCmip6YearRange:
    def test_year_range_filters_historical(self):
        from samy_wgen.pipeline.download_data import CMIP6_EXP_YEARS
        year_range = (1990, 2000)
        years = [y for y in CMIP6_EXP_YEARS["historical"] if year_range[0] <= y <= year_range[1]]
        assert years == list(range(1990, 2001))

    def test_year_range_empty_returns_nothing(self):
        from samy_wgen.pipeline.download_data import CMIP6_EXP_YEARS
        year_range = (1800, 1850)
        years = [y for y in CMIP6_EXP_YEARS["historical"] if year_range[0] <= y <= year_range[1]]
        assert years == []

    def test_download_cmip6_skips_unknown_model(self, tmp_path):
        from samy_wgen.pipeline.download_data import download_cmip6

        with (
            patch("cdsapi.Client"),
            patch("samy_wgen.pipeline.download_data._get_cmip6_variable_groups", return_value={}),
        ):
            download_cmip6(
                city="Miami", lat=25.76, lon=-80.19,
                models=["nonexistent_model_xyz"],
                output_dir=str(tmp_path),
            )
        assert list(tmp_path.iterdir()) == [tmp_path / "Miami"]
        assert not any((tmp_path / "Miami").iterdir()), "Unknown model should not produce output files"

    def test_download_cmip6_year_range_passed_to_request(self, tmp_path):
        from samy_wgen.pipeline.download_data import download_cmip6
        captured_requests: list[dict] = []

        def fake_retrieve(dataset, req, path):
            captured_requests.append(req.copy())
            Path(path).write_bytes(b"")

        times = pd.date_range("1995-01-01", periods=3, freq="D")
        mock_client = MagicMock()
        mock_client.retrieve.side_effect = fake_retrieve

        _call = [0]
        def fake_nc_df(nc_bytes, lat, lon):
            col = f"col_{_call[0]}"
            _call[0] += 1
            return pd.DataFrame({"time": times, col: [1.0, 2.0, 3.0]})

        with (
            patch("cdsapi.Client", return_value=mock_client),
            patch(
                "samy_wgen.pipeline.download_data._get_cmip6_variable_groups",
                return_value={"daily": ["near_surface_air_temperature"]},
            ),
            patch("samy_wgen.pipeline.download_data._nc_bytes_to_point_df",
                  side_effect=fake_nc_df),
        ):
            download_cmip6(
                city="Miami", lat=25.76, lon=-80.19,
                models=["mpi_esm1_2_lr"],
                experiments=["historical"],
                year_range=(1990, 2000),
                output_dir=str(tmp_path),
            )

        assert captured_requests, "Expected at least one CDS request"
        years_in_req = [int(y) for y in captured_requests[0]["year"]]
        assert min(years_in_req) >= 1990
        assert max(years_in_req) <= 2000

    def test_download_cmip6_respects_variable_filter(self, tmp_path):
        from samy_wgen.pipeline.download_data import download_cmip6
        captured_requests: list[dict] = []

        def fake_retrieve(dataset, req, path):
            captured_requests.append(req.copy())
            Path(path).write_bytes(b"")

        mock_client = MagicMock()
        mock_client.retrieve.side_effect = fake_retrieve

        with (
            patch("cdsapi.Client", return_value=mock_client),
            patch(
                "samy_wgen.pipeline.download_data._get_cmip6_variable_groups",
                return_value={
                    "daily": ["near_surface_air_temperature", "precipitation"],
                    "monthly": ["surface_downwelling_shortwave_radiation"],
                },
            ),
            patch(
                "samy_wgen.pipeline.download_data._nc_bytes_to_point_df",
                return_value=pd.DataFrame({"time": pd.date_range("1995-01-01", periods=1), "tas": [1.0]}),
            ),
        ):
            download_cmip6(
                city="Miami",
                lat=25.76,
                lon=-80.19,
                models=["mpi_esm1_2_lr"],
                experiments=["historical"],
                variables=["precipitation"],
                year_range=(1995, 1995),
                output_dir=str(tmp_path),
            )

        assert len(captured_requests) == 1
        assert captured_requests[0]["variable"] == "precipitation"

    def test_download_cmip6_for_cities_batches_request(self, tmp_path):
        from samy_wgen.pipeline.download_data import download_cmip6_for_cities

        captured_requests: list[dict] = []

        def fake_retrieve(dataset, req, path):
            captured_requests.append(req.copy())
            Path(path).write_bytes(b"fake_nc")

        mock_client = MagicMock()
        mock_client.retrieve.side_effect = fake_retrieve

        def fake_extract(_nc_bytes, city_points):
            times = pd.date_range("1995-01-01", periods=2, freq="D")
            return {
                city: pd.DataFrame({"time": times, "near_surface_air_temperature": [1.0, 2.0]})
                for city in city_points
            }

        with (
            patch("cdsapi.Client", return_value=mock_client),
            patch(
                "samy_wgen.pipeline.download_data._get_cmip6_variable_groups",
                return_value={"daily": ["near_surface_air_temperature"]},
            ),
            patch(
                "samy_wgen.pipeline.download_data._resolve_city_points",
                return_value={
                    "Miami": (25.76, -80.19),
                    "Los_Angeles": (34.05, -118.24),
                },
            ),
            patch("samy_wgen.pipeline.download_data._nc_bytes_to_point_dfs", side_effect=fake_extract),
        ):
            download_cmip6_for_cities(
                cities=["Miami", "Los_Angeles"],
                models=["mpi_esm1_2_lr"],
                experiments=["historical"],
                year_range=(1995, 1995),
                output_dir=str(tmp_path),
            )

        assert len(captured_requests) == 1
        assert captured_requests[0]["area"] == [35.55, -119.74, 24.26, -78.69]
        assert (tmp_path / "Miami" / "mpi_esm1_2_lr_historical.csv").exists()
        assert (tmp_path / "Los_Angeles" / "mpi_esm1_2_lr_historical.csv").exists()

    def test_download_cordex_uses_constrained_plan(self, tmp_path):
        from samy_wgen.pipeline.download_data import download_cordex_cmip5

        captured_requests: list[dict] = []

        def fake_retrieve(dataset, req, path):
            captured_requests.append(req.copy())
            Path(path).write_bytes(b"fake nc bytes")

        mock_client = MagicMock()
        mock_client.retrieve.side_effect = fake_retrieve

        with (
            patch("cdsapi.Client", return_value=mock_client),
            patch(
                "samy_wgen.pipeline.download_data._resolve_cordex_download_plan",
                return_value=[
                    {
                        "domain": "north_america",
                        "experiment": "historical",
                        "gcm_model": "mpi_m_mpi_esm_lr",
                        "rcm_model": "ncar_regcm4",
                        "ensemble_member": "r1i1p1",
                        "horizontal_resolution": "0_22_degree_x_0_22_degree",
                        "variable": ["2m_air_temperature", "surface_pressure"],
                    }
                ],
            ),
            patch(
                "samy_wgen.pipeline.download_data.CORDEX_EXP_YEARS",
                {"historical": [2000]},
            ),
            patch(
                "samy_wgen.pipeline.download_data._nc_bytes_to_point_df",
                return_value=pd.DataFrame({"time": pd.date_range("2000-01-01", periods=1), "tas": [280.0], "ps": [101000.0]}),
            ),
            patch("samy_wgen.pipeline.download_data.time.sleep"),
        ):
            download_cordex_cmip5(
                city="Miami",
                lat=25.76,
                lon=-80.19,
                cordex_domain="NAM-22",
                gcm="mpi_m_mpi_esm_lr",
                rcm="ncar_regcm4",
                ensemble="r1i1p1",
                experiments=["historical"],
                output_dir=str(tmp_path),
            )

        assert len(captured_requests) == 2
        assert captured_requests[0]["domain"] == "north_america"
        assert captured_requests[0]["horizontal_resolution"] == "0_22_degree_x_0_22_degree"
        assert {request["variable"] for request in captured_requests} == {
            "2m_air_temperature", "surface_pressure"
        }
        assert captured_requests[0]["year"] == ["2000"]
        assert captured_requests[0]["month"] == [f"{month:02d}" for month in range(1, 13)]
        assert "start_year" not in captured_requests[0]
        assert captured_requests[0]["area"] == [26.26, -80.69, 25.26, -79.69]
        assert (tmp_path / "NAM-22_mpi_m_mpi_esm_lr_ncar_regcm4_r1i1p1" / "Miami" / "historical.csv").exists()

    def test_cordex_zip_outputs_feed_delta_reader(self, tmp_path):
        from samy_wgen.pipeline.download_data import download_cordex_cmip5
        from samy_wgen.pipeline.run_climate_delta import (
            compute_cordex_baseline,
            compute_daily_delta,
            load_cordex_cmip5_projection,
        )

        plans = [
            {
                "domain": "north_america",
                "experiment": experiment,
                "gcm_model": "mpi_m_mpi_esm_lr",
                "rcm_model": "ncar_regcm4",
                "ensemble_member": "r1i1p1",
                "horizontal_resolution": "0_22_degree_x_0_22_degree",
                "variable": ["2m_air_temperature", "surface_pressure"],
            }
            for experiment in ("historical", "rcp_8_5")
        ]

        def fake_retrieve(_dataset, request, path):
            year = int(request["year"][0])
            times = pd.date_range(f"{year}-01-01", periods=3, freq="D")
            temperature = 280.0 if year == 2000 else 282.0
            with zipfile.ZipFile(path, "w") as archive:
                if request["variable"] == "2m_air_temperature":
                    archive.writestr("tas.nc", _make_nc_bytes(times, lat=34.1141, lon=-118.4068,
                                                              variables={"tas": np.full(3, temperature)}))
                else:
                    archive.writestr("ps.nc", _make_nc_bytes(times, lat=34.1141, lon=-118.4068,
                                                             variables={"ps": np.full(3, 101000.0)}))

        client = MagicMock()
        client.retrieve.side_effect = fake_retrieve
        with (
            patch("cdsapi.Client", return_value=client),
            patch("samy_wgen.pipeline.download_data._resolve_cordex_download_plan", return_value=plans),
            patch("samy_wgen.pipeline.download_data.CORDEX_EXP_YEARS", {"historical": [2000], "rcp_8_5": [2006]}),
            patch("samy_wgen.pipeline.download_data.time.sleep"),
        ):
            outputs = download_cordex_cmip5("Los_Angeles", output_dir=str(tmp_path))

        variant_root = outputs["historical"].parent.parent
        assert outputs["rcp_8_5"] == variant_root / "Los_Angeles" / "rcp85.csv"
        projection, _, _ = load_cordex_cmip5_projection(variant_root, "Los_Angeles", verbose=False)
        baseline = compute_cordex_baseline(projection, 2000, 2000)
        delta = compute_daily_delta(projection, baseline, pd.Timestamp("2006-01-01"))
        assert delta["tas_C"] == pytest.approx(2.0)
        assert delta["ps_hPa"] == pytest.approx(0.0)

    def test_cordex_existing_csv_must_cover_requested_month(self, tmp_path):
        from samy_wgen.pipeline.download_data import _cordex_csv_covers_request

        path = tmp_path / "historical.csv"
        frame = pd.DataFrame({
            "time": pd.date_range("2000-01-01", periods=31, freq="D"),
            "tas": np.full(31, 280.0),
            "ps": np.full(31, 101000.0),
        })
        frame.to_csv(path, index=False)
        variables = ["2m_air_temperature", "surface_pressure"]
        assert _cordex_csv_covers_request(path, [2000], ["01"], variables)
        assert not _cordex_csv_covers_request(path, [2000], ["01", "02"], variables)
        assert not _cordex_csv_covers_request(path, [2000, 2001], ["01"], variables)
        frame.iloc[:3].to_csv(path, index=False)
        assert not _cordex_csv_covers_request(path, [2000], ["01"], variables)

    def test_download_cordex_skips_when_plan_empty(self, tmp_path):
        from samy_wgen.pipeline.download_data import download_cordex_cmip5

        mock_client = MagicMock()

        with (
            patch("cdsapi.Client", return_value=mock_client),
            patch("samy_wgen.pipeline.download_data._resolve_cordex_download_plan", return_value=[]),
        ):
            with pytest.raises(ValueError, match="No CORDEX download combinations"):
                download_cordex_cmip5(
                    city="Miami",
                    lat=25.76,
                    lon=-80.19,
                    cordex_domain="NAM-22",
                    output_dir=str(tmp_path),
                )

        mock_client.retrieve.assert_not_called()


# ─── Integration tests (slow, real network) ───────────────────────────────────

@pytest.mark.slow
class TestListIsdStationsIntegration:
    def test_miami_finds_airport(self, tmp_path):
        """List stations for Miami — expects Miami Intl Airport in results."""
        from samy_wgen.pipeline.download_data import list_isd_stations

        result = list_isd_stations(25.7617, -80.1918, radius_km=50,
                                   cache_dir=str(tmp_path))

        assert not result.empty, "No stations returned for Miami"
        names = result["STATION NAME"].str.upper()
        assert any("MIAMI" in n for n in names), (
            f"Expected a Miami station, got: {result['STATION NAME'].tolist()}"
        )

    def test_isd_history_cached(self, tmp_path):
        """Second call uses the cached isd-history.csv (no repeated download)."""
        from samy_wgen.pipeline.download_data import _load_isd_history

        cache = tmp_path / "isd-history.csv"
        df1 = _load_isd_history(cache)
        mtime1 = cache.stat().st_mtime

        df2 = _load_isd_history(cache)
        mtime2 = cache.stat().st_mtime

        assert mtime1 == mtime2, "Cache file was re-downloaded on second call"
        assert len(df1) == len(df2)
