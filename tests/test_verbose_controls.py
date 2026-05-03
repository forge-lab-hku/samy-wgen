"""Unit tests for verbose output suppression in pipeline helpers."""

import numpy as np
import pandas as pd

from samy_wgen.pipeline.run_climate_delta import apply_climate_delta
from samy_wgen.pipeline.run_ensemble import EnsemblePipeline


def test_ensemble_pipeline_train_suppresses_stdout_when_verbose_false(monkeypatch, capsys):
    pipeline = EnsemblePipeline(
        isd_solar_path="dummy.csv",
        lat=25.0,
        baseline_start=1991,
        baseline_end=2010,
        seed=0,
        verbose=False,
    )

    def _printer(_self, *args, **kwargs):
        print("noisy training step")
        return _self

    monkeypatch.setattr(EnsemblePipeline, "load_isd_data", _printer)
    monkeypatch.setattr(EnsemblePipeline, "extract_seasonal_profiles", _printer)
    monkeypatch.setattr(EnsemblePipeline, "engineer_features", _printer)
    monkeypatch.setattr(EnsemblePipeline, "standardize_features", _printer)
    monkeypatch.setattr(EnsemblePipeline, "train_regime_switching", _printer)
    monkeypatch.setattr(EnsemblePipeline, "train_regime_conditional_vars", _printer)

    pipeline.train(["temp"])

    captured = capsys.readouterr()
    assert captured.out == ""


def test_apply_climate_delta_suppresses_stdout_when_verbose_false(monkeypatch, capsys):
    rsvar_df = pd.DataFrame(
        {"temp": [20.0, 21.0]},
        index=pd.to_datetime(["2030-01-01 00:00:00", "2030-01-01 01:00:00"]),
    )
    cordex = pd.DataFrame(
        {"tas_C": [0.0]},
        index=pd.to_datetime(["2030-01-01"]),
    )
    baseline = {"tas_C": {"overall": 0.0}}

    monkeypatch.setattr(
        "samy_wgen.pipeline.run_climate_delta.compute_daily_delta",
        lambda *args, **kwargs: {"tas_C": 1.5},
    )
    monkeypatch.setattr(
        "samy_wgen.pipeline.run_climate_delta.apply_physics_qa",
        lambda df, lat, preserve="GHI": df,
    )

    output_df, delta_stats = apply_climate_delta(
        rsvar_df=rsvar_df,
        cordex=cordex,
        baseline=baseline,
        gcm="test-model",
        rcm=None,
        lat=25.0,
        verbose=False,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    np.testing.assert_allclose(output_df["temp"].to_numpy(), np.array([21.5, 22.5]))
    assert delta_stats["temp"] == [1.5]
