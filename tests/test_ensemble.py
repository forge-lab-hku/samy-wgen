"""Regression tests for ensemble anomaly generation."""

from pathlib import Path

import numpy as np
import pytest

from samy_wgen.model.anomaly_generator import AnomalyGenerator
from samy_wgen.pipeline.run_ensemble import EnsemblePipeline
from samy_wgen.pipeline.run_rsvar import RSVAROnlyPipeline


pytestmark = pytest.mark.slow


MIAMI_MERGED_CSV = (
    Path(__file__).parent / "output" / "miami_poc_cmip6_ssp245" / "stage3_isd_solar_lst.csv"
)
TARGET_VARS = [
    "temp",
    "pressure",
    "wind_speed",
    "GHI",
    "DNI",
    "DHI",
    "relative_humidity",
    "specific_humidity",
]


def _require_miami_merged_csv() -> Path:
    if not MIAMI_MERGED_CSV.exists():
        pytest.skip(f"Cached Miami merged CSV not found: {MIAMI_MERGED_CSV}")
    return MIAMI_MERGED_CSV


def _train_ensemble_pipeline() -> EnsemblePipeline:
    pipeline = EnsemblePipeline(
        isd_solar_path=str(_require_miami_merged_csv()),
        lat=25.7617,
        baseline_start=1991,
        baseline_end=2010,
        seed=0,
    )
    pipeline.train(TARGET_VARS)
    return pipeline


def _train_rsvar_pipeline() -> RSVAROnlyPipeline:
    pipeline = RSVAROnlyPipeline(
        isd_solar_path=str(_require_miami_merged_csv()),
        lat=25.7617,
        baseline_start=1991,
        baseline_end=2010,
        seed=0,
    )
    pipeline.load_isd_data()
    pipeline.extract_seasonal_profiles(TARGET_VARS)
    pipeline.engineer_features()
    pipeline.standardize_features(TARGET_VARS)
    pipeline.train_regime_switching(k_regimes=3)
    pipeline.train_regime_conditional_vars(TARGET_VARS)
    return pipeline


def test_soft_deterministic_matches_rsvar_baseline_path():
    ensemble_pipeline = _train_ensemble_pipeline()
    generator = AnomalyGenerator(
        fitted_model=ensemble_pipeline,
        regime_mode="soft",
        residual_mode="deterministic",
    )
    ensemble_member = generator.generate(
        n_members=1,
        start_date="2030-01-01",
        end_date="2030-01-03 23:00:00",
        seed=0,
    )[0]

    rsvar_pipeline = _train_rsvar_pipeline()
    rsvar_output = rsvar_pipeline.forecast_rsvar_only(
        TARGET_VARS,
        start_date="2030-01-01",
        end_date="2030-01-03 23:00:00",
        chunk_years=10,
    )

    assert list(ensemble_member.columns) == TARGET_VARS
    assert ensemble_member.index.equals(rsvar_output.index)

    temp_diff = np.abs(ensemble_member["temp"].values - rsvar_output["temp"].values)
    assert temp_diff.max() < 0.5, f"soft+deter temp drift too large: {temp_diff.max():.3f} C"

    for var in TARGET_VARS:
        assert np.isfinite(ensemble_member[var]).all(), f"{var} contains non-finite values"


def test_ensemble_block_member_stays_finite_and_bounded():
    pipeline = _train_ensemble_pipeline()
    generator = AnomalyGenerator(
        fitted_model=pipeline,
        regime_mode="ensemble",
        residual_mode="block",
        regime_strength=1.0,
        residual_scale=1.0,
    )
    member = generator.generate(
        n_members=1,
        start_date="2030-01-01",
        end_date="2030-12-31 23:00:00",
        seed=0,
    )[0]

    for var in TARGET_VARS:
        assert np.isfinite(member[var]).all(), f"{var} contains non-finite values"

    assert member["temp"].abs().max() < 100.0
    assert member["temp"].between(-50.0, 70.0).all()
    assert member["relative_humidity"].abs().max() < 200.0
    assert member["specific_humidity"].abs().max() < 0.1
