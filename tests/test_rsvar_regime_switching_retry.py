from __future__ import annotations

import numpy as np
import pandas as pd

import samy_wgen.model.rsvar as rsvar


class _BadResult:
    def __init__(self):
        self.aic = np.nan
        self.llf = np.nan
        self.smoothed_marginal_probabilities = pd.DataFrame(
            np.full((4, 3), np.nan),
            columns=[0, 1, 2],
        )
        self.regime_transition = np.eye(3)


class _GoodResult:
    def __init__(self):
        self.aic = 123.0
        self.llf = -50.0
        self.smoothed_marginal_probabilities = pd.DataFrame(
            [
                [0.8, 0.1, 0.1],
                [0.7, 0.2, 0.1],
                [0.1, 0.7, 0.2],
                [0.1, 0.2, 0.7],
            ],
            columns=[0, 1, 2],
        )
        self.regime_transition = np.eye(3)


class _FakeMarkovAutoregression:
    fit_calls: list[dict] = []

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, **kwargs):
        self.__class__.fit_calls.append(kwargs)
        if len(self.__class__.fit_calls) == 1:
            return _BadResult()
        return _GoodResult()


def test_train_regime_switching_retries_multistart_on_nonfinite_fit(monkeypatch):
    monkeypatch.setattr(rsvar, "MarkovAutoregression", _FakeMarkovAutoregression)
    _FakeMarkovAutoregression.fit_calls = []

    pipeline = rsvar.ForecastTo2100V5.__new__(rsvar.ForecastTo2100V5)
    idx = pd.date_range("2000-01-01 00:00:00", periods=24 * 4, freq="h")
    pipeline.isd_training = pd.DataFrame(
        {
            "temp": np.linspace(20.0, 23.0, len(idx)),
            "temp_std": np.linspace(-1.0, 1.0, len(idx)),
        },
        index=idx,
    )
    pipeline.regime_probs = None
    pipeline.ms_result = None

    result = pipeline.train_regime_switching(k_regimes=3)

    assert result.aic == 123.0
    assert len(_FakeMarkovAutoregression.fit_calls) == 2
    assert _FakeMarkovAutoregression.fit_calls[0]["method"] == "nm"
    assert "search_reps" not in _FakeMarkovAutoregression.fit_calls[0]
    assert _FakeMarkovAutoregression.fit_calls[1]["method"] == "nm"
    assert _FakeMarkovAutoregression.fit_calls[1]["search_reps"] == 10
    assert pipeline.regime_probs.shape == (3, len(idx))
