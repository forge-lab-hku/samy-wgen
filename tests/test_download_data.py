"""Tests for samy_wgen.pipeline.download_data.

Fast unit tests (no network, mocked IO) run with:
    pytest tests/test_download_data.py -v

Slow integration tests (real network) run with:
    pytest -m slow tests/test_download_data.py -v -s
"""

from __future__ import annotations

import io
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
        assert abs(meta["lat"] - 25.7617) < 0.01
        assert abs(meta["lon"] - (-80.1918)) < 0.01
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


# ─── download_isd (unit, mocked FTP) ─────────────────────────────────────────

class TestDownloadIsdUnit:
    def test_bad_station_id_raises(self, tmp_path):
        from samy_wgen.pipeline.download_data import download_isd

        with pytest.raises(ValueError, match="USAF-WBAN"):
            download_isd("722020", "Miami", [2000], output_dir=str(tmp_path))

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

        ftp_mock = MagicMock()

        with (
            patch("samy_wgen.pipeline.download_data._load_isd_history",
                  return_value=fake_history),
            patch("samy_wgen.pipeline.download_data.FTP", return_value=ftp_mock),
        ):
            download_isd("722020-12839", "Miami", [2000], output_dir=str(tmp_path))

        # retrbinary should NOT have been called (file already exists)
        ftp_mock.retrbinary.assert_not_called()

    def test_downloads_missing_years(self, tmp_path):
        from samy_wgen.pipeline.download_data import download_isd

        fake_history = pd.DataFrame({
            "USAF": ["722020"], "WBAN": ["12839"],
            "STATION NAME": ["MIAMI INTL AP"], "BEGIN": [19700101], "END": [20241231],
        })

        downloaded: list[str] = []

        def fake_retrbinary(cmd: str, callback):
            filename = cmd.split()[-1]
            downloaded.append(filename)
            # Write dummy content so tmp.rename() works
            callback(b"fake gz")

        ftp_mock = MagicMock()
        ftp_mock.retrbinary.side_effect = fake_retrbinary

        with (
            patch("samy_wgen.pipeline.download_data._load_isd_history",
                  return_value=fake_history),
            patch("samy_wgen.pipeline.download_data.FTP", return_value=ftp_mock),
        ):
            download_isd(
                "722020-12839", "Miami", [2001, 2002], output_dir=str(tmp_path)
            )

        assert "722020-12839-2001.gz" in downloaded
        assert "722020-12839-2002.gz" in downloaded


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

        monkeypatch.delenv("NSRDB_API_KEY", raising=False)
        monkeypatch.delenv("NSRDB_EMAIL",   raising=False)

        with pytest.raises(EnvironmentError, match="NSRDB_API_KEY"):
            download_nsrdb("Miami", 25.76, -80.19, [2000], output_dir=str(tmp_path))

    def test_saves_merged_csv(self, tmp_path, monkeypatch):
        from samy_wgen.pipeline.download_data import download_nsrdb

        monkeypatch.setenv("NSRDB_API_KEY", "test_key")
        monkeypatch.setenv("NSRDB_EMAIL",   "test@example.com")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = self._mock_nsrdb_csv(2000)

        with patch("requests.get", return_value=mock_resp):
            download_nsrdb(
                "Miami", 25.76, -80.19, [2000], output_dir=str(tmp_path)
            )

        output_files = list(tmp_path.glob("nsrdb_miami_*.csv"))
        assert len(output_files) == 1
        df = pd.read_csv(output_files[0])
        assert "GHI" in df.columns

    def test_filters_years_outside_coverage(self, tmp_path, monkeypatch):
        """Years outside 1998–2024 are silently skipped."""
        from samy_wgen.pipeline.download_data import download_nsrdb

        monkeypatch.setenv("NSRDB_API_KEY", "test_key")
        monkeypatch.setenv("NSRDB_EMAIL",   "test@example.com")

        with patch("requests.get") as mock_get:
            download_nsrdb(
                "Miami", 25.76, -80.19, [1980, 2030],  # both out of range
                output_dir=str(tmp_path),
            )
            mock_get.assert_not_called()


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

    def test_download_default_sources(self):
        parser = self._parser()
        args = parser.parse_args(["download", "--city", "Miami",
                                   "--isd-station", "722020-12839"])
        assert set(args.data) == {"isd", "nsrdb", "cmip6"}

    def test_download_cordex_only(self):
        parser = self._parser()
        args = parser.parse_args(["download", "--city", "Miami", "--data", "cordex"])
        assert args.data == ["cordex"]

    def test_download_custom_experiments(self):
        parser = self._parser()
        args = parser.parse_args([
            "download", "--city", "Miami",
            "--data", "cmip6",
            "--experiments", "historical", "ssp5_8_5",
        ])
        assert "ssp5_8_5" in args.experiments


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
