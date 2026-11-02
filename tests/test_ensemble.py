"""Regression tests for ensemble anomaly generation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
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

    pd.testing.assert_frame_equal(ensemble_member, rsvar_output)

    for var in TARGET_VARS:
        assert np.isfinite(ensemble_member[var]).all(), f"{var} contains non-finite values"


def test_regime_paths_use_true_labels_for_noncontiguous_regimes():
    idx = pd.date_range("2000-01-01 00:00:00", periods=4, freq="h")
    regime_model = SimpleNamespace(
        coefs=np.zeros((1, 1, 1), dtype=float),
        intercept=np.zeros(1, dtype=float),
        model=SimpleNamespace(endog_names=["temp_deviation"]),
    )
    fitted = SimpleNamespace(
        regime_models={0: regime_model, 2: regime_model},
        regime_residuals={
            0: pd.DataFrame([[0.0], [0.0]], index=idx[:2], columns=["temp_deviation"]),
            2: pd.DataFrame([[0.0], [0.0]], index=idx[2:], columns=["temp_deviation"]),
        },
        ms_result=SimpleNamespace(
            regime_transition=np.array(
                [
                    [0.70, 0.20, 0.10],
                    [0.15, 0.70, 0.15],
                    [0.10, 0.25, 0.65],
                ],
                dtype=float,
            )
        ),
        regime_probs=np.array([[0.80], [0.05], [0.15]], dtype=float),
        isd_baseline=pd.DataFrame({"temp": [0.0] * 24}, index=pd.date_range("1999-12-31 00:00:00", periods=24, freq="h")),
        seasonal_profiles={"temp": {(12, h): 0.0 for h in range(24)}},
    )

    hard_gen = AnomalyGenerator(
        fitted_model=fitted,
        regime_mode="hard",
    )
    hard_path, hard_probs = hard_gen._get_regime_path(n_days=8, rng=np.random.default_rng(0))
    assert set(np.unique(hard_path)).issubset({0, 2})
    hard_weights = hard_gen._day_weights(0, hard_path, hard_probs)
    assert hard_weights.shape == (2,)

    ens_gen = AnomalyGenerator(
        fitted_model=fitted,
        regime_mode="ensemble",
    )
    ens_path, ens_probs = ens_gen._get_regime_path(n_days=8, rng=np.random.default_rng(1))
    assert set(np.unique(ens_path)).issubset({0, 2})
    ens_weights = ens_gen._day_weights(0, ens_path, ens_probs)
    assert ens_weights.shape == (2,)


def test_generator_reconciles_dense_markov_space_to_available_regimes():
    idx = pd.date_range("2000-01-01 00:00:00", periods=4, freq="h")
    regime_model = SimpleNamespace(
        coefs=np.zeros((1, 1, 1), dtype=float),
        intercept=np.zeros(1, dtype=float),
        model=SimpleNamespace(endog_names=["temp_deviation"]),
    )
    fitted = SimpleNamespace(
        regime_models={1: regime_model, 3: regime_model},
        regime_residuals={},
        ms_result=SimpleNamespace(
            regime_transition=np.array(
                [
                    [0.55, 0.15, 0.20, 0.10],
                    [0.10, 0.70, 0.10, 0.10],
                    [0.20, 0.20, 0.40, 0.20],
                    [0.05, 0.15, 0.20, 0.60],
                ],
                dtype=float,
            )
        ),
        regime_probs=np.array([[0.05], [0.70], [0.05], [0.20]], dtype=float),
        isd_baseline=pd.DataFrame({"temp": [0.0] * 24}, index=pd.date_range("1999-12-31 00:00:00", periods=24, freq="h")),
        seasonal_profiles={"temp": {(12, h): 0.0 for h in range(24)}},
    )

    gen = AnomalyGenerator(
        fitted_model=fitted,
        regime_mode="ensemble",
    )
    assert gen._regimes == [1, 3]
    assert gen._k == 2
    assert gen._transition.shape == (2, 2)
    assert gen._last_posterior.shape == (2,)

    path, probs = gen._get_regime_path(n_days=12, rng=np.random.default_rng(0))
    assert set(np.unique(path)).issubset({1, 3})
    assert probs.shape == (12, 2)
