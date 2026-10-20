"""Offline tests of solar time bases, physical closure, and Stage 2 input."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from samy_wgen.pipeline.run_pipeline import _stage2_merge_nsrdb
from samy_wgen.utils.solar_calibration import _cos_zenith, calibrate_era5_land_solar


LAT, LON, OFFSET = 43.6532, -79.3832, -5.0


def _make_inputs(tmp_path, end="2001-01-09"):
    utc = pd.date_range("2000-12-24", end, freq="h", tz="UTC")
    lst = utc.tz_localize(None) + pd.Timedelta(hours=OFFSET)
    cos_hour = _cos_zenith(LAT, LON, utc)
    cloudy_hour = lst.day % 2 == 0
    era_ghi = np.maximum(cos_hour, 0) * np.where(cloudy_hour, 80.0, 400.0)
    era_path = tmp_path / "era.nc"
    xr.Dataset({"ssrd": ("valid_time", era_ghi * 3600)},
               coords={"valid_time": utc.tz_localize(None)}).to_netcdf(era_path)

    obs_utc = utc[:-1] + pd.Timedelta(minutes=30)
    obs_lst = obs_utc.tz_localize(None) + pd.Timedelta(hours=OFFSET)
    cos_obs = _cos_zenith(LAT, LON, obs_utc)
    ghi = np.where(cos_obs > 0, era_ghi[:-1] + era_ghi[1:], 0.0)
    cloudy_obs = obs_lst.day % 2 == 0
    kd = np.where(cloudy_obs, 0.8, 0.2)
    dhi = ghi * kd
    dni = np.divide(ghi - dhi, cos_obs, out=np.zeros_like(ghi), where=cos_obs > 0.05)
    # Low-sun rows need exact closure as well.
    dhi = ghi - dni * np.maximum(cos_obs, 0)
    frame = pd.DataFrame({
        "GHI": ghi, "DNI": dni, "DHI": dhi,
        "Solar Zenith Angle": np.degrees(np.arccos(np.clip(cos_obs, -1, 1))),
    })
    paths = {}
    for basis, times in (("utc", obs_utc), ("lst", obs_lst)):
        out = frame.copy()
        out["Year"], out["Month"], out["Day"] = times.year, times.month, times.day
        out["Hour"], out["Minute"] = times.hour, times.minute
        paths[basis] = tmp_path / f"nsrdb_{basis}.csv"
        out.to_csv(paths[basis], index=False)
    return era_path, paths, utc, era_ghi


def test_utc_and_lst_observations_align_across_new_year_and_fit_sky_states(tmp_path):
    era, paths, utc, era_ghi = _make_inputs(tmp_path)
    results = {}
    for basis in ("utc", "lst"):
        results[basis] = calibrate_era5_land_solar(
            era, paths[basis], latitude=LAT, longitude=LON,
            utc_offset_hours=OFFSET,
        )
        assert results[basis].attrs["nsrdb_time_basis"] == basis
        assert results[basis].attrs["closure_pass_samples"] == results[basis].attrs["closure_daylight_samples"]
        assert results[basis].attrs["overlap_samples"] == len(utc) - 1
    pd.testing.assert_frame_equal(results["utc"], results["lst"])

    result = results["utc"]
    assert result.index[0] == pd.Timestamp("2000-12-23 19:00")
    assert result.index[-1] == pd.Timestamp("2001-01-08 19:00")
    np.testing.assert_allclose(result.GHI, era_ghi * 2, atol=1e-7)
    for stamp, expected_kd in (("2001-01-01 12:00", 0.2), ("2001-01-02 12:00", 0.8)):
        row = result.loc[stamp]
        assert row.GHI > 0
        assert row.DHI / row.GHI == pytest.approx(expected_kd, abs=0.02)
        cosz = _cos_zenith(LAT, LON, pd.DatetimeIndex([pd.Timestamp(stamp) - pd.Timedelta(hours=OFFSET)], tz="UTC"))[0]
        assert row.DNI * cosz + row.DHI == pytest.approx(row.GHI, abs=1e-8)
    assert (result[["GHI", "DNI", "DHI"]] >= 0).all().all()
    assert (result.loc["2001-01-01 00:00", ["GHI", "DNI", "DHI"]] == 0).all()


def test_missing_or_ambiguous_zenith_requires_explicit_basis(tmp_path):
    era, paths, _, _ = _make_inputs(tmp_path)
    no_zenith = pd.read_csv(paths["utc"]).drop(columns=["Solar Zenith Angle"])
    no_zenith_path = tmp_path / "no_zenith.csv"
    no_zenith.to_csv(no_zenith_path, index=False)
    with pytest.raises(ValueError, match="specify nsrdb_time_basis"):
        calibrate_era5_land_solar(era, no_zenith_path, latitude=LAT, longitude=LON,
                                  utc_offset_hours=OFFSET)
    explicit = calibrate_era5_land_solar(
        era, no_zenith_path, latitude=LAT, longitude=LON,
        utc_offset_hours=OFFSET, nsrdb_time_basis="utc",
    )
    assert explicit.attrs["closure_pass_samples"] > 100

    ambiguous = pd.read_csv(paths["utc"])
    ambiguous["Solar Zenith Angle"] = 90.0
    ambiguous_path = tmp_path / "ambiguous.csv"
    ambiguous.to_csv(ambiguous_path, index=False)
    with pytest.raises(ValueError, match="specify nsrdb_time_basis"):
        calibrate_era5_land_solar(era, ambiguous_path, latitude=LAT, longitude=LON,
                                  utc_offset_hours=OFFSET)


def test_long_records_without_zenith_use_daylight_not_filenames(tmp_path):
    era, paths, _, _ = _make_inputs(tmp_path, end="2001-12-31")
    results = {}
    for basis, name in (("utc", "alpha.csv"), ("lst", "beta.csv")):
        raw = pd.read_csv(paths[basis]).drop(columns=["Solar Zenith Angle"])
        path = tmp_path / name
        raw.to_csv(path, index=False)
        results[basis] = calibrate_era5_land_solar(
            era, path, latitude=LAT, longitude=LON, utc_offset_hours=OFFSET,
        )
        assert results[basis].attrs["nsrdb_time_basis"] == basis
        assert results[basis].attrs["nsrdb_time_basis_evidence"].startswith("GHI daylight")
        fractions = results[basis].attrs["ghi_night_fraction"]
        assert fractions[basis] < 0.03
        assert fractions["lst" if basis == "utc" else "utc"] > fractions[basis] + 0.03
    pd.testing.assert_frame_equal(results["utc"], results["lst"])

    ambiguous = pd.read_csv(paths["utc"]).drop(columns=["Solar Zenith Angle"])
    ambiguous["GHI"] = 200.0
    path = tmp_path / "ambiguous_without_zenith.csv"
    ambiguous.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Ambiguous NSRDB time basis"):
        calibrate_era5_land_solar(era, path, latitude=LAT, longitude=LON,
                                  utc_offset_hours=OFFSET)


def test_stage2_consumes_calibrated_date_csv(tmp_path):
    era, paths, _, _ = _make_inputs(tmp_path)
    solar = calibrate_era5_land_solar(
        era, paths["utc"], latitude=LAT, longitude=LON, utc_offset_hours=OFFSET,
    )
    solar_path = tmp_path / "calibrated.csv"
    solar.to_csv(solar_path)
    isd = pd.DataFrame({"temp": 10.0, "dewpoint": 5.0}, index=solar.index)
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    output = _stage2_merge_nsrdb(isd, solar_path, out_dir, overwrite=True)
    merged = pd.read_csv(output, parse_dates=["DATE"]).set_index("DATE")
    assert len(merged) == len(solar)
    assert {"temp", "dewpoint", "GHI", "DNI", "DHI"} <= set(merged.columns)
    assert merged.index.equals(solar.index.astype(merged.index.dtype))
    np.testing.assert_allclose(merged.GHI, solar.GHI)

    # The existing Year/Month/Day/Hour NSRDB contract still works.
    legacy_dir = tmp_path / "legacy_output"
    legacy_dir.mkdir()
    legacy_output = _stage2_merge_nsrdb(isd, paths["lst"], legacy_dir, overwrite=True)
    legacy = pd.read_csv(legacy_output)
    assert {"DATE", "temp", "GHI", "DNI", "DHI"} <= set(legacy.columns)
    assert len(legacy) > 100
