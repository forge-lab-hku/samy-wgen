from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import samy_wgen.model.rsvar as rsvar


class _FakeVARResult:
    def __init__(self, n_obs: int):
        self.aic = 1.23
        self.resid = pd.DataFrame(
            np.zeros((n_obs, 2), dtype=float),
            columns=["temp_deviation", "pressure_deviation"],
        )


class _RecordingVAR:
    fit_calls: list[object] = []
    seen_frames: list[pd.DataFrame] = []

    def __init__(self, frame: pd.DataFrame):
        self.__class__.seen_frames.append(frame.copy())

    def fit(self, *args, **kwargs):
        self.__class__.fit_calls.append((args, kwargs))
        return _FakeVARResult(len(self.__class__.seen_frames[-1]))


class _FallbackVAR(_RecordingVAR):
    def fit(self, *args, **kwargs):
        self.__class__.fit_calls.append((args, kwargs))
        if kwargs == {"maxlags": 1, "ic": "aic"}:
            raise RuntimeError("aic fit failed")
        return _FakeVARResult(len(self.__class__.seen_frames[-1]))


def _build_pipeline() -> rsvar.ForecastTo2100V5:
    pipeline = rsvar.ForecastTo2100V5.__new__(rsvar.ForecastTo2100V5)
    idx = pd.date_range("2000-01-01 00:00:00", periods=120, freq="h")
    pipeline.isd_training = pd.DataFrame(
        {
            "temp": np.linspace(20.0, 30.0, len(idx)),
            "pressure": np.linspace(1000.0, 1010.0, len(idx)),
        },
        index=idx,
    )
    pipeline.seasonal_profiles = {
        "temp": {(1, h): 0.0 for h in range(24)},
        "pressure": {(1, h): 0.0 for h in range(24)},
    }
    pipeline.regime_models = {}
    pipeline.regime_residuals = {}
    return pipeline


def test_train_regime_conditional_vars_falls_back_to_hard_assignment_and_drops_nonfinite(monkeypatch):
    monkeypatch.setattr(rsvar, "VAR", _RecordingVAR)
    _RecordingVAR.fit_calls = []
    _RecordingVAR.seen_frames = []

    pipeline = _build_pipeline()
    regime_probs = np.zeros((2, len(pipeline.isd_training)), dtype=float)
    regime_probs[0, :2] = 0.9
    regime_probs[1, :2] = 0.1
    regime_probs[0, 2:107] = 0.4
    regime_probs[1, 2:107] = 0.3
    regime_probs[1, 107:] = 0.9
    regime_probs[0, 107:] = 0.1
    pipeline.regime_probs = regime_probs

    pipeline.isd_training.iloc[5, pipeline.isd_training.columns.get_loc("temp")] = np.inf
    pipeline.isd_training.iloc[6, pipeline.isd_training.columns.get_loc("pressure")] = np.nan

    pipeline.train_regime_conditional_vars(["temp", "pressure"], k_regimes=2)

    assert 0 in pipeline.regime_models
    assert len(_RecordingVAR.seen_frames) == 1
    seen = _RecordingVAR.seen_frames[0]
    assert len(seen) == 105
    assert np.isfinite(seen.to_numpy()).all()
    assert _RecordingVAR.fit_calls[0] == ((), {"maxlags": 1, "ic": "aic"})


def test_train_regime_conditional_vars_retries_with_fixed_lag_one(monkeypatch):
    monkeypatch.setattr(rsvar, "VAR", _FallbackVAR)
    _FallbackVAR.fit_calls = []
    _FallbackVAR.seen_frames = []

    pipeline = _build_pipeline()
    pipeline.regime_probs = np.vstack(
        [
            np.ones(len(pipeline.isd_training), dtype=float),
            np.zeros(len(pipeline.isd_training), dtype=float),
        ]
    )

    pipeline.train_regime_conditional_vars(["temp", "pressure"], k_regimes=1)

    assert 0 in pipeline.regime_models
    assert _FallbackVAR.fit_calls[0] == ((), {"maxlags": 1, "ic": "aic"})
    assert _FallbackVAR.fit_calls[1] == ((1,), {})
