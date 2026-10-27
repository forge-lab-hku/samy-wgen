"""Generation-chunk centering must leave stochastic propagation and draws intact."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from samy_wgen.model.anomaly_generator import AnomalyGenerator


def _generator(chunk_years: int) -> AnomalyGenerator:
    names = ["temp_deviation", "pressure_deviation"]
    models = {
        regime: SimpleNamespace(
            coefs=np.array([np.eye(2) * 0.4]),
            intercept=np.array([regime + 0.5, 2 * regime - 0.3]),
            model=SimpleNamespace(endog_names=names),
        )
        for regime in (0, 1)
    }
    index = pd.date_range("2000-01-01", periods=48, freq="h")
    residuals = {
        regime: pd.DataFrame(
            np.column_stack((np.arange(48) / 100, np.arange(48)[::-1] / 200))
            + regime,
            index=index,
            columns=names,
        )
        for regime in (0, 1)
    }
    baseline_index = pd.date_range("2000-12-31", periods=24, freq="h")
    model = SimpleNamespace(
        regime_models=models,
        regime_residuals=residuals,
        ms_result=SimpleNamespace(regime_transition=np.array([[0.8, 0.2], [0.3, 0.7]])),
        regime_probs=np.array([[0.6], [0.4]]),
        isd_baseline=pd.DataFrame({"temp": 0.0, "pressure": 0.0}, index=baseline_index),
        seasonal_profiles={
            var: {(month, hour): (month + hour / 24) * scale
                  for month in range(1, 13) for hour in range(24)}
            for var, scale in (("temp", 1.0), ("pressure", 2.0))
        },
    )
    return AnomalyGenerator(model, regime_mode="ensemble", residual_mode="block",
                            chunk_years=chunk_years)


def test_centering_uses_nominal_hour_chunks_and_preserves_interannual_means():
    generator = _generator(chunk_years=2)
    anomaly = np.column_stack((
        np.repeat([1.0, 5.0], 8760).tolist() + [20.0] * 37,
        np.repeat([-3.0, 7.0], 8760).tolist() + [30.0] * 37,
    ))
    anomaly[100, 0] += 10.0
    anomaly[9000, 1] -= 8.0
    original = anomaly.copy()

    centered = generator._center_generation_chunks(anomaly)

    assert centered is anomaly
    np.testing.assert_allclose(centered[:17520].mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(centered[17520:].mean(axis=0), 0, atol=1e-12)
    assert centered[:8760, 0].mean() != pytest.approx(centered[8760:17520, 0].mean())
    np.testing.assert_allclose(centered[8760, 0] - centered[0, 0],
                               original[8760, 0] - original[0, 0])


def test_fixed_seed_draws_and_carried_var_are_independent_of_chunk_size():
    index = pd.date_range("2025-01-01", periods=8760 + 48, freq="h")
    recorded = []
    outputs = []
    seasonal = _generator(1)._compute_seasonal(index)
    for chunk_years in (1, 2):
        generator = _generator(chunk_years)
        paths = []
        residuals = []
        original_path = generator._get_regime_path
        original_residuals = generator._draw_residuals
        original_propagate = generator._propagate
        raw = []

        def record_path(n_days, rng):
            path, probs = original_path(n_days, rng)
            paths.append((path.copy(), probs.copy()))
            return path, probs

        def record_residuals(*args, **kwargs):
            result = original_residuals(*args, **kwargs)
            residuals.append(result.copy())
            return result

        def record_propagate(*args, **kwargs):
            result = original_propagate(*args, **kwargs)
            raw.append(result.copy())
            return result

        generator._get_regime_path = record_path
        generator._draw_residuals = record_residuals
        generator._propagate = record_propagate
        members = generator.generate(2, str(index[0]), str(index[-1]), seed=17)
        recorded.append((paths, residuals, raw))
        outputs.append(members)

    # Fixed-seed draw snapshot from the pre-centering sampling sequence.
    np.testing.assert_array_equal(
        recorded[0][0][0][0][:12], [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]
    )
    np.testing.assert_allclose(
        recorded[0][1][0][:3], [[0.09, 0.19], [0.10, 0.185], [0.11, 0.18]]
    )

    for member in range(2):
        np.testing.assert_array_equal(recorded[0][0][member][0], recorded[1][0][member][0])
        np.testing.assert_array_equal(recorded[0][0][member][1], recorded[1][0][member][1])
        np.testing.assert_array_equal(recorded[0][2][member], recorded[1][2][member])
        for day in range(len(index) // 24):
            draw_index = member * (len(index) // 24) + day
            np.testing.assert_array_equal(recorded[0][1][draw_index], recorded[1][1][draw_index])

        for output, chunk_hours in ((outputs[0][member], 8760),
                                    (outputs[1][member], 17520)):
            anomaly = output.to_numpy() - seasonal
            raw = recorded[0][2][member]
            for start in range(0, len(index), chunk_hours):
                stop = min(start + chunk_hours, len(index))
                np.testing.assert_allclose(anomaly[start:stop].mean(axis=0), 0, atol=1e-12)
                np.testing.assert_allclose(
                    anomaly[start:stop],
                    raw[start:stop] - raw[start:stop].mean(axis=0),
                    atol=1e-12,
                )

        # The hour after a chunk boundary uses the preceding raw VAR state.
        day = 8760 // 24
        weights = _generator(1)._day_weights(
            day, recorded[0][0][member][0], recorded[0][0][member][1]
        )
        previous = recorded[0][2][member][8759]
        expected = sum(
            weights[regime] * (np.array([regime + 0.5, 2 * regime - 0.3])
                               + 0.4 * previous)
            for regime in (0, 1)
        ) + recorded[0][1][member * (len(index) // 24) + day][0]
        np.testing.assert_allclose(recorded[0][2][member][8760], expected)

    assert not outputs[0][0].equals(outputs[1][0])


def test_chunk_years_must_be_positive():
    with pytest.raises(ValueError, match="chunk_years must be positive"):
        _generator(0)
