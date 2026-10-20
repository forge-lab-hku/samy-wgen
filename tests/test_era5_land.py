from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd


def _write_era5_nc(path: Path, times: pd.DatetimeIndex, ssrd: np.ndarray) -> None:
    import xarray as xr

    dataset = xr.Dataset(
        {"ssrd": ("valid_time", ssrd)},
        coords={"valid_time": times},
    )
    dataset.to_netcdf(path)


def _write_nsrdb_csv(path: Path, timestamps: pd.DatetimeIndex, ghi: np.ndarray,
                     dni: np.ndarray, dhi: np.ndarray) -> None:
    half_hour = timestamps + pd.Timedelta(minutes=30)
    frame = pd.DataFrame(
        {
            "Year": half_hour.year,
            "Month": half_hour.month,
            "Day": half_hour.day,
            "Hour": half_hour.hour,
            "Minute": half_hour.minute,
            "GHI": ghi,
            "DNI": dni,
            "DHI": dhi,
        }
    )
    frame.to_csv(path, index=False)


def test_download_era5_land_uses_cds_point_request_and_skips_existing(tmp_path):
    from samy_wgen.pipeline.download_data import download_era5_land

    calls = []

    class FakeClient:
        def retrieve(self, dataset, request, target):
            calls.append((dataset, request, target))
            start_date, end_date = request["date"][0].split("/")
            times = pd.date_range(start_date, pd.Timestamp(end_date) + pd.Timedelta(days=1),
                                  freq="h", inclusive="left")
            _write_era5_nc(Path(target), times, np.ones(len(times)))

    with patch.dict("sys.modules", {"cdsapi": SimpleNamespace(Client=lambda: FakeClient())}):
        output = download_era5_land(
            "Miami",
            start_date="2001-01-01",
            end_date="2001-01-02",
            output_dir=str(tmp_path),
            variables=["surface_solar_radiation_downwards"],
        )
        cache_files = list((tmp_path / ".cache" / "ERA5_Land" / "Miami").glob("*.nc"))
        assert len(cache_files) == 1
        output[0].write_bytes(b"damaged output")
        second_output = download_era5_land(
            "Miami",
            start_date="2001-01-01",
            end_date="2001-01-02",
            output_dir=str(tmp_path),
            variables=["surface_solar_radiation_downwards"],
        )

    assert output == second_output
    assert output[0].is_file()
    assert output[0].read_bytes() == cache_files[0].read_bytes()
    assert len(calls) == 1
    dataset, request, _ = calls[0]
    assert dataset == "reanalysis-era5-land-timeseries"
    assert request["variable"] == ["surface_solar_radiation_downwards"]
    assert request["date"] == ["2001-01-01/2001-01-02"]
    assert request["location"]["latitude"] == 25.784
    assert request["location"]["longitude"] == -80.2101


def test_calibrate_era5_land_solar_uses_nsrdb_overlap_and_closes_components(tmp_path):
    from samy_wgen.utils.solar_calibration import _cos_zenith, calibrate_era5_land_solar

    lst = pd.date_range("2001-03-01", periods=24 * 8, freq="h")
    offset = -5.0
    utc = lst + pd.to_timedelta(-offset, unit="h")
    cosz = _cos_zenith(25.78, -80.21, pd.DatetimeIndex(utc, tz="UTC"))
    ghi = np.where(cosz > 0, 50.0 + 300.0 * cosz, 0.0)
    dhi = np.where(cosz > 0.05, 0.3 * ghi, ghi)
    dni = np.where(cosz > 0.05, (ghi - dhi) / cosz, 0.0)

    era5_path = tmp_path / "era5.nc"
    nsrdb_path = tmp_path / "nsrdb.csv"
    # ERA5 is UTC and has half the NSRDB irradiance, with hourly energy in J/m2.
    _write_era5_nc(era5_path, utc, ghi * 0.5 * 3600.0)
    _write_nsrdb_csv(nsrdb_path, lst, ghi, dni, dhi)

    calibrated = calibrate_era5_land_solar(
        era5_path,
        nsrdb_path,
        latitude=25.78,
        longitude=-80.21,
        utc_offset_hours=offset,
        nsrdb_time_basis="lst",
        min_overlap_samples=24,
    )

    assert np.median(calibrated.loc[calibrated.GHI > 10, "GHI"]) > 0
    assert calibrated.index.equals(lst.rename("DATE"))
    assert (calibrated[["GHI", "DNI", "DHI"]] >= 0).all().all()
    daytime = calibrated["GHI"] > 10
    np.testing.assert_allclose(
        calibrated.loc[daytime, "DHI"]
        + calibrated.loc[daytime, "DNI"] * np.maximum(cosz, 0)[daytime.to_numpy()],
        calibrated.loc[daytime, "GHI"],
        atol=1e-8,
    )


def test_calibrate_era5_land_solar_rejects_insufficient_overlap(tmp_path):
    from samy_wgen.utils.solar_calibration import calibrate_era5_land_solar

    timestamps = pd.date_range("2001-01-01", periods=4, freq="h")
    era5_path = tmp_path / "era5.nc"
    nsrdb_path = tmp_path / "nsrdb.csv"
    _write_era5_nc(era5_path, timestamps, np.full(4, 3600.0))
    _write_nsrdb_csv(nsrdb_path, timestamps, np.full(4, 100.0), np.full(4, 50.0), np.full(4, 50.0))

    import pytest

    with pytest.raises(ValueError, match="Insufficient daytime ERA5/NSRDB overlap"):
        calibrate_era5_land_solar(
            era5_path,
            nsrdb_path,
            latitude=25.78,
            longitude=-80.21,
            utc_offset_hours=-5,
            nsrdb_time_basis="lst",
            min_overlap_samples=10,
        )


def test_download_parser_accepts_era5_land_and_range():
    from samy_wgen.pipeline.download_data import _build_parser

    args = _build_parser().parse_args(
        [
            "download",
            "--city",
            "Miami",
            "--data",
            "era5-land",
            "--era5-land-years",
            "1980",
            "2020",
            "--era5-land-var",
            "surface_solar_radiation_downwards",
        ]
    )
    assert args.data == ["era5-land"]
    assert args.era5_land_years == [1980, 2020]
    assert args.era5_land_var == ["surface_solar_radiation_downwards"]

    calibration = _build_parser().parse_args(
        [
            "calibrate-solar",
            "--city",
            "Miami",
            "--era5",
            "era5.nc",
            "--nsrdb",
            "nsrdb.csv",
            "--utc-offset",
            "-5",
        ]
    )
    assert calibration.command == "calibrate-solar"
    assert calibration.era5 == Path("era5.nc")
    assert calibration.nsrdb == Path("nsrdb.csv")
    assert calibration.utc_offset == -5.0
