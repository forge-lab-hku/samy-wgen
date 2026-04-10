"""Validation-side temperature distribution comparison for ensemble pathways."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from samy_wgen.model.anomaly_generator import AnomalyGenerator
from samy_wgen.pipeline.run_ensemble import EnsemblePipeline
from samy_wgen.pipeline.run_rsvar import RSVAROnlyPipeline


pytestmark = pytest.mark.slow


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

VALIDATION_START = "2011-01-01"
VALIDATION_END = "2011-03-31 23:00:00"
N_STOCHASTIC_MEMBERS = 8

MIAMI_MERGED_CSV = (
    Path(__file__).parent / "output" / "miami_poc_cmip6_ssp245" / "stage3_isd_solar_lst.csv"
)
OUTPUT_CSV = (
    Path(__file__).parent / "output" / "miami_temp_distribution_validation_2011Q1.csv"
)


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


def _summarize_df(df: pd.DataFrame) -> dict[str, float]:
    series = df["temp"]
    daily_max = series.resample("D").max()
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "p05": float(series.quantile(0.05)),
        "p50": float(series.quantile(0.50)),
        "p95": float(series.quantile(0.95)),
        "daily_max_p50": float(daily_max.quantile(0.50)),
        "daily_max_p95": float(daily_max.quantile(0.95)),
    }


def _summarize_members(members: list[pd.DataFrame]) -> dict[str, float]:
    temp_values = pd.Series(np.concatenate([member["temp"].to_numpy() for member in members]))
    daily_max_values = pd.Series(
        np.concatenate([member["temp"].resample("D").max().to_numpy() for member in members])
    )
    return {
        "mean": float(temp_values.mean()),
        "std": float(temp_values.std()),
        "p05": float(temp_values.quantile(0.05)),
        "p50": float(temp_values.quantile(0.50)),
        "p95": float(temp_values.quantile(0.95)),
        "daily_max_p50": float(daily_max_values.quantile(0.50)),
        "daily_max_p95": float(daily_max_values.quantile(0.95)),
    }


def test_validation_temp_distributions_print_and_save_csv():
    merged_csv = _require_miami_merged_csv()

    ensemble_pipeline = _train_ensemble_pipeline()
    rsvar_pipeline = _train_rsvar_pipeline()
    legacy = rsvar_pipeline.forecast_rsvar_only(
        TARGET_VARS,
        start_date=VALIDATION_START,
        end_date=VALIDATION_END,
        chunk_years=10,
    )

    generator_configs = {
        "soft + deterministic": dict(
            regime_mode="soft",
            residual_mode="deterministic",
            regime_strength=1.0,
        ),
        "ensemble + deterministic": dict(
            regime_mode="ensemble",
            residual_mode="deterministic",
            regime_strength=1.0,
        ),
        "ensemble + block": dict(
            regime_mode="ensemble",
            residual_mode="block",
            regime_strength=1.0,
        ),
    }

    outputs: dict[str, list[pd.DataFrame]] = {}
    for name, kwargs in generator_configs.items():
        generator = AnomalyGenerator(
            fitted_model=ensemble_pipeline,
            residual_scale=1.0,
            **kwargs,
        )
        n_members = 1 if name == "soft + deterministic" else N_STOCHASTIC_MEMBERS
        outputs[name] = generator.generate(
            n_members=n_members,
            start_date=VALIDATION_START,
            end_date=VALIDATION_END,
            seed=0,
        )

    observed = pd.read_csv(merged_csv, parse_dates=["DATE"]).set_index("DATE").loc[
        VALIDATION_START:VALIDATION_END, ["temp"]
    ]

    summary_rows = {
        "observed temp": _summarize_df(observed),
        "legacy raw": _summarize_df(legacy),
        "soft + deterministic": _summarize_df(outputs["soft + deterministic"][0]),
        f"ensemble + deterministic ({N_STOCHASTIC_MEMBERS}m)": _summarize_members(
            outputs["ensemble + deterministic"]
        ),
        f"ensemble + block ({N_STOCHASTIC_MEMBERS}m)": _summarize_members(
            outputs["ensemble + block"]
        ),
    }
    summary = pd.DataFrame(summary_rows).T
    summary["delta_p95_vs_legacy"] = summary["p95"] - summary.loc["legacy raw", "p95"]
    summary["delta_daily_max_p95_vs_legacy"] = (
        summary["daily_max_p95"] - summary.loc["legacy raw", "daily_max_p95"]
    )

    soft_det = outputs["soft + deterministic"][0]
    max_abs_diff = float((soft_det - legacy).abs().to_numpy().max())

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.round(6).to_csv(OUTPUT_CSV, index=True)

    print(f"Validation window: {VALIDATION_START} to {VALIDATION_END}")
    print(f"soft + deterministic max abs diff vs legacy raw = {max_abs_diff}")
    print(summary.round(3).to_string())
    print(f"Saved CSV: {OUTPUT_CSV}")

    assert max_abs_diff == 0.0
    assert OUTPUT_CSV.exists()
