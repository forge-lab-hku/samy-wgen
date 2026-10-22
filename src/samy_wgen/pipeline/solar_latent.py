"""Default RS-VAR solar fit and monthly projection-signal workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from samy_wgen.pipeline.run_rsvar import RSVAROnlyPipeline
from samy_wgen.utils.physics_qa import apply_physics_qa
from samy_wgen.utils.solar_latent import (
    LATENT_COLUMNS,
    apply_monthly_solar_signal,
    calibrate_baseline,
    decode_solar,
    encode_solar,
    monthly_rsds_ratios,
    observed_monthly_targets,
    solar_qa,
)


NON_SOLAR = ("temp", "pressure", "wind_speed", "relative_humidity", "specific_humidity")
_QA_FAILURES = ("nonfinite_hours", "negative_hours", "night_ghi_gt1_hours",
                "ghi_above_toa_gt1_hours", "dni_above_toa_gt1_hours",
                "closure_gt1_hours")


def fit_solar_latent_rsvar(
    isd_path: str | Path, latitude: float, longitude: float,
    baseline_start: int = 1991, baseline_end: int = 2010,
    forecast_start: str = "2025-01-01 00:00",
    forecast_end: str = "2100-12-31 23:00",
    generated_calibration_years: tuple[int, int] | None = None,
    seed: int = 42, verbose: bool = False,
) -> tuple[pd.DataFrame, RSVAROnlyPipeline, pd.DataFrame, dict]:
    """Fit bounded solar jointly with weather and return calibrated base.

    The calibrated base retains the two latent columns so a later climate
    delta can reconstruct GHI/DNI/DHI without fitting RS-VAR again. Historical
    solar hours outside the accepted envelope are excluded from the joint fit.
    """
    model = RSVAROnlyPipeline(str(isd_path), lat=latitude,
                              baseline_start=baseline_start,
                              baseline_end=baseline_end, seed=seed, verbose=verbose)
    model.load_isd_data()
    missing = set(NON_SOLAR) - set(model.isd_data)
    if missing:
        raise ValueError(f"Missing historical weather fields: {sorted(missing)}")
    encoded, screen = encode_solar(model.isd_data, latitude, longitude)
    model.isd_baseline = model.isd_baseline.join(encoded).dropna(subset=list(LATENT_COLUMNS))
    model.isd_training = model.isd_training.join(encoded).dropna(subset=list(LATENT_COLUMNS))
    if model.isd_baseline.empty or model.isd_training.empty:
        raise ValueError("No accepted solar/weather training hours")
    targets = list(NON_SOLAR + LATENT_COLUMNS)
    model.extract_seasonal_profiles(targets)
    model.engineer_features()
    model.standardize_features(targets)
    model.train_regime_switching(k_regimes=3)
    model.train_regime_conditional_vars(targets)
    forecast = model.forecast_rsvar_only(targets, start_date=forecast_start,
                                        end_date=forecast_end, chunk_years=10)
    expected = pd.date_range(forecast_start, forecast_end, freq="h")
    if forecast is None or not forecast.index.equals(expected):
        raise ValueError("RS-VAR did not return the complete hourly forecast")
    if generated_calibration_years is None:
        generated_calibration_years = (expected[0].year, min(expected[0].year + 19, expected[-1].year))
    raw_solar = decode_solar(forecast, latitude, longitude)
    monthly_targets, target_screen = observed_monthly_targets(
        model.isd_data, latitude, longitude, baseline_start, baseline_end)
    calibrated_solar, calibrated_latent, calibration = calibrate_baseline(
        raw_solar, monthly_targets, latitude, longitude, generated_calibration_years)
    non_solar = apply_physics_qa(forecast[list(NON_SOLAR)], lat=latitude)
    base = non_solar.join(calibrated_solar).join(calibrated_latent)
    qa = solar_qa(base, latitude, longitude)
    if any(qa[key] for key in _QA_FAILURES):
        raise ValueError(f"Calibrated RS-VAR solar failed physical QA: {qa}")
    info = {"training_solar_screen": screen, "monthly_target_screen": target_screen,
            "base_solar_qa": qa, "generated_calibration_years": generated_calibration_years,
            "training_hours": len(model.isd_training)}
    return base, model, calibration, info


def apply_solar_latent_signal(
    rsvar_df: pd.DataFrame, non_solar_future: pd.DataFrame,
    projection_rsds: pd.DataFrame, latitude: float, longitude: float,
    baseline_start: int = 1991, baseline_end: int = 2010,
    trailing_years: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply bounded monthly solar change to a non-solar climate-delta result."""
    if not rsvar_df.index.equals(non_solar_future.index):
        raise ValueError("RS-VAR and climate-delta hours do not align")
    ratios = monthly_rsds_ratios(projection_rsds, rsvar_df.index,
                                 baseline_start, baseline_end, trailing_years)
    solar, monthly_energy = apply_monthly_solar_signal(
        rsvar_df[list(LATENT_COLUMNS)], latitude, longitude, ratios)
    final = non_solar_future.join(solar)
    qa = solar_qa(final, latitude, longitude)
    if any(qa[key] for key in _QA_FAILURES):
        raise ValueError(f"Signaled solar failed physical QA: {qa}")
    final.attrs["solar_monthly_energy"] = monthly_energy
    final.attrs["solar_physical_qa"] = qa
    return final, monthly_energy
