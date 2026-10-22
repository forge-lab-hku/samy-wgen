"""Regression coverage for the default bounded solar workflow."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from samy_wgen.pipeline.run_climate_delta import (
    apply_climate_delta,
    apply_mixed_resolution_climate_delta,
    compute_cordex_baseline,
)
from samy_wgen.utils.solar_latent import (
    LATENT_COLUMNS,
    apply_monthly_solar_signal,
    calibrate_baseline,
    decode_solar,
    encode_solar,
    monthly_rsds_ratios,
    observed_monthly_targets,
    solar_geometry,
    solar_qa,
)


LAT, LON = 33.4484, -112.074


def _latent(index, kt=0.55, beam=0.6):
    return pd.DataFrame({LATENT_COLUMNS[0]: logit(kt),
                         LATENT_COLUMNS[1]: logit(beam)}, index=index)


def test_encode_decode_screens_invalid_hour_and_preserves_closure():
    index = pd.date_range("2025-06-01", periods=48, freq="h")
    solar = decode_solar(_latent(index), LAT, LON)
    encoded, screen = encode_solar(solar, LAT, LON)
    assert screen["training_solar_rejected_hours"] == 0
    restored = decode_solar(encoded, LAT, LON)
    np.testing.assert_allclose(restored.GHI, solar.GHI, atol=1e-8)
    assert solar_qa(restored, LAT, LON)["closure_gt1_hours"] == 0

    bad = solar.copy()
    noon = int(np.argmax(bad.GHI.to_numpy()))
    bad.iloc[noon, bad.columns.get_loc("GHI")] *= 2
    encoded_bad, bad_screen = encode_solar(bad, LAT, LON)
    assert bad_screen["training_solar_rejected_hours"] == 1
    assert encoded_bad.iloc[noon].isna().all()


def test_baseline_and_monthly_signal_match_feasible_targets_without_clipping():
    index = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h")
    base = decode_solar(_latent(index), LAT, LON)
    targets, screen = observed_monthly_targets(base, LAT, LON, 2025, 2025)
    assert screen["training_solar_rejected_hours"] == 0
    corrected, latent, calibration = calibrate_baseline(
        base, targets, LAT, LON, generated_fit_years=(2025, 2025))
    assert len(calibration) == 12
    assert (calibration.calibrated_GHI_mean_w_m2 - calibration.observed_GHI_mean_w_m2).abs().max() < 1e-7
    assert (calibration.calibrated_DNI_mean_w_m2 - calibration.observed_DNI_mean_w_m2).abs().max() < 1e-7
    ratios = pd.Series(1.2, index=index.to_period("M").unique())
    future, energy = apply_monthly_solar_signal(latent, LAT, LON, ratios)
    assert len(energy) == 12
    assert (energy.achieved_ghi_sum_wh_m2 - energy.target_ghi_sum_wh_m2).abs().max() < 1e-4
    assert all(solar_qa(future, LAT, LON)[key] == 0 for key in (
        "night_ghi_gt1_hours", "ghi_above_toa_gt1_hours", "closure_gt1_hours"))
    impossible = ratios.copy()
    impossible.iloc[0] = 10
    with pytest.raises(ValueError, match="outside"):
        apply_monthly_solar_signal(latent, LAT, LON, impossible)


def test_trailing_monthly_signal_uses_climate_mean_not_single_day_ratio():
    index = pd.date_range("1991-01-01", "2025-12-01", freq="MS")
    rsds = np.full(len(index), 100.0)
    rsds[(index.year == 2025) & (index.month == 1)] = 200.0
    projection = pd.DataFrame({"rsds": rsds}, index=index)
    target_index = pd.date_range("2025-01-01", periods=24, freq="h")
    ratios = monthly_rsds_ratios(projection, target_index)
    assert ratios.iloc[0] == pytest.approx(1.05)


def test_mixed_delta_rejects_old_direct_solar_and_reconstructs_new_solar():
    hours = pd.date_range("2025-01-01", "2025-01-31 23:00", freq="h")
    daily = pd.date_range("1991-01-01", "2025-12-31", freq="D")
    climate = pd.DataFrame({"rsds": 100.0, "tas_C": 20.0,
                            "ps_hPa": 1010.0}, index=daily)
    baseline = compute_cordex_baseline(climate, 1991, 2010, mode="daily")
    latent = _latent(hours)
    rsvar = pd.DataFrame({"temp": 20.0}, index=hours).join(latent)
    solar = decode_solar(latent, LAT, LON)
    rsvar = rsvar.join(solar)
    with pytest.raises(ValueError, match="regenerate RS-VAR"):
        apply_mixed_resolution_climate_delta(
            rsvar.drop(columns=list(LATENT_COLUMNS)), climate, pd.DataFrame(), baseline,
            gcm="test", rcm=None, lat=LAT, longitude=LON, verbose=False)
    final, _ = apply_mixed_resolution_climate_delta(
        rsvar, climate, pd.DataFrame(), baseline,
        gcm="test", rcm=None, lat=LAT, longitude=LON, verbose=False)
    assert final.index.equals(hours)
    assert "solar_kt_logit" not in final
    assert "solar_monthly_energy" in final.attrs
    assert solar_qa(final, LAT, LON)["closure_gt1_hours"] == 0
    daily_final, _ = apply_climate_delta(
        rsvar, climate, baseline, gcm="test", rcm=None,
        lat=LAT, longitude=LON, verbose=False)
    assert daily_final.index.equals(hours)
    assert len(daily_final.attrs["solar_monthly_energy"]) == 1
    assert solar_qa(daily_final, LAT, LON)["closure_gt1_hours"] == 0


def test_solar_geometry_uses_actual_longitude_and_fixed_lst():
    index = pd.date_range("2025-03-20", periods=24, freq="h")
    west, _, _ = solar_geometry(index, LAT, -112.074)
    east, _, _ = solar_geometry(index, LAT, -100.0)
    assert np.argmax(west) != np.argmax(east)
