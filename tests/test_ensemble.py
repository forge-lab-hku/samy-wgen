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

    pd.testing.assert_frame_equal(ensemble_member, rsvar_output)

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


def _make_minimal_generator(*, residual_mode: str = "block", block_size: int = 3) -> AnomalyGenerator:
    idx = pd.date_range("2000-01-01 00:00:00", periods=4, freq="h")
    resid0 = pd.DataFrame(
        [[1.0], [2.0], [3.0], [4.0]],
        index=idx,
        columns=["temp_deviation"],
    )
    gap_idx = pd.to_datetime(
        ["2000-01-01 00:00:00", "2000-01-01 01:00:00", "2000-01-01 03:00:00", "2000-01-01 04:00:00"]
    )
    resid1 = pd.DataFrame(
        [[10.0], [11.0], [13.0], [14.0]],
        index=gap_idx,
        columns=["temp_deviation"],
    )

    regime_model = SimpleNamespace(
        coefs=np.zeros((1, 1, 1), dtype=float),
        intercept=np.zeros(1, dtype=float),
        model=SimpleNamespace(endog_names=["temp_deviation"]),
    )
    fitted = SimpleNamespace(
        regime_models={0: regime_model, 1: regime_model},
        regime_residuals={0: resid0, 1: resid1},
        ms_result=SimpleNamespace(regime_transition=np.eye(2)),
        regime_probs=np.array([[1.0], [0.0]], dtype=float),
        isd_baseline=pd.DataFrame({"temp": [0.0] * 24}, index=pd.date_range("1999-12-31 00:00:00", periods=24, freq="h")),
        seasonal_profiles={"temp": {(12, h): 0.0 for h in range(24)}},
    )
    return AnomalyGenerator(
        fitted_model=fitted,
        regime_mode="soft",
        residual_mode=residual_mode,
        block_size=block_size,
        residual_scale=1.0,
    )


def test_block_sampling_uses_only_truly_contiguous_windows():
    generator = _make_minimal_generator(residual_mode="block", block_size=3)

    blocks0 = generator._residual_blocks[0]
    assert blocks0.shape == (2, 3, 1)
    assert np.allclose(blocks0[0, :, 0], [1.0, 2.0, 3.0])
    assert np.allclose(blocks0[1, :, 0], [2.0, 3.0, 4.0])

    blocks1 = generator._residual_blocks[1]
    assert blocks1.shape == (0, 3, 1)


def test_soft_residual_sampling_respects_weight_mixture():
    generator = _make_minimal_generator(residual_mode="iid", block_size=3)
    rng = np.random.default_rng(0)

    eps = generator._draw_residuals(
        n_hours=12,
        rng=rng,
        weights=np.array([1.0, 0.0], dtype=float),
    )
    assert np.isin(eps[:, 0], [1.0, 2.0, 3.0, 4.0]).all()

    eps_other = generator._draw_residuals(
        n_hours=12,
        rng=np.random.default_rng(1),
        weights=np.array([0.0, 1.0], dtype=float),
    )
    assert np.isin(eps_other[:, 0], [10.0, 11.0, 13.0, 14.0]).all()
