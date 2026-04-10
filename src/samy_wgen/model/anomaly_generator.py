"""
AnomalyGenerator: post-inference stochastic anomaly generation for RS-VAR ensemble.

Two independently controllable layers:
  Layer 1 (regime_mode):    how the regime sequence is drawn per member
  Layer 2 (residual_mode):  how the VAR innovation epsilon is sampled

Layer 1 — regime_mode:
  'soft'      deterministic probability-weighted blend (existing behavior)
  'hard'      deterministic argmax path (no member diversity from regimes)
  'ensemble'  per-member Markov chain sampling (stochastic regime paths)

  regime_strength in [0, 1]  (applies to 'ensemble' only):
    blends one-hot sampled label toward soft probability vector
    0 = collapses to soft; 1 = pure sampled Markov path

Layer 2 — residual_mode:
  'deterministic'  no noise (reproduces existing forecast_without_climate_delta)
  'iid'            i.i.d. residual row per hourly step
  'block'          24h contiguous residual block per day (preserves within-day autocorrelation)
  'parametric'     N(0, Sigma_regime) draw

  residual_scale in [0, inf):
    scalar multiplier on the bootstrapped epsilon before adding to VAR propagation
    0 = no noise; 1 = full empirical amplitude
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from samy_wgen.model.rsvar import ForecastTo2100V5

_REGIME_MODES = ('soft', 'hard', 'ensemble')
_RESIDUAL_MODES = ('deterministic', 'iid', 'block', 'parametric')


class AnomalyGenerator:
    """
    Generates M ensemble members of synthetic weather anomalies from a fitted
    ForecastTo2100V5 model, without touching the model's internals.

    Usage:
        gen = AnomalyGenerator(fitted_model, regime_mode='ensemble',
                               residual_mode='block', regime_strength=1.0,
                               residual_scale=1.0)
        members = gen.generate(n_members=50, start_date='2025-01-01',
                               end_date='2025-12-31', seed=0)
        # members: list of 50 DataFrames, each (8760 x n_vars)
        # values = seasonal_profile + anomaly  (no climate delta applied)
    """

    def __init__(
        self,
        fitted_model: "ForecastTo2100V5",
        regime_mode: str = 'ensemble',
        residual_mode: str = 'block',
        block_size: int = 24,
        regime_strength: float = 1.0,
        residual_scale: float = 1.0,
        centering_period: int = 365,
    ):
        if regime_mode not in _REGIME_MODES:
            raise ValueError(f"regime_mode must be one of {_REGIME_MODES}")
        if residual_mode not in _RESIDUAL_MODES:
            raise ValueError(f"residual_mode must be one of {_RESIDUAL_MODES}")

        self.model = fitted_model
        self.regime_mode = regime_mode
        self.residual_mode = residual_mode
        self.block_size = block_size
        self.regime_strength = float(regime_strength)
        self.residual_scale = float(residual_scale)
        self.centering_period = int(centering_period)  # days between drift corrections

        # Validate required model attributes
        if not fitted_model.regime_models:
            raise RuntimeError("Model has no fitted regime_models. Run train_regime_conditional_vars() first.")
        if residual_mode != 'deterministic' and not fitted_model.regime_residuals:
            raise RuntimeError(
                "Model has no regime_residuals. Ensure rsvar.py stores result.resid.values "
                "in train_regime_conditional_vars()."
            )

        self._regimes = sorted(fitted_model.regime_models.keys())
        self._k = len(self._regimes)
        self._transition = self._get_transition_matrix()
        self._last_posterior = self._get_last_posterior()

        # Pre-extract VAR coefficients and intercepts per regime
        self._coefs = {}    # regime -> (n_vars, n_vars) coefficient matrix (lag-1)
        self._intercepts = {}  # regime -> (n_vars,) intercept
        self._n_vars = None
        self._deviation_vars = None
        self._build_var_params()

        # Pre-slice residuals into blocks
        self._residual_blocks = {}    # regime -> list of (block_size, n_vars) arrays
        self._residual_pool = {}      # regime -> (n_obs, n_vars) for iid sampling
        self._residual_cov = {}       # regime -> (n_vars, n_vars) for parametric
        if residual_mode != 'deterministic':
            self._build_residual_structures()

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _get_transition_matrix(self) -> np.ndarray:
        """Extract (k, k) Markov transition matrix from fitted model."""
        tm = self.model.ms_result.regime_transition  # shape (k, k) or (k, k, lag)
        if tm.ndim == 3:
            tm = tm[:, :, 0]
        return np.array(tm, dtype=float)

    def _get_last_posterior(self) -> np.ndarray:
        """Get last training posterior regime probabilities as (k,) array."""
        probs = np.array(self.model.regime_probs[:, -1], dtype=float)
        probs = np.maximum(probs, 1e-12)
        return probs / probs.sum()

    def _build_var_params(self):
        """Extract coefficient matrices and intercepts from fitted VAR results."""
        for regime_i, result in self.model.regime_models.items():
            # statsmodels VAR result: coefs shape (n_lags, n_vars, n_vars)
            # intercept shape (n_vars,). Keep statsmodels orientation so
            # A_t = intercept + coefs[0] @ A_{t-1} matches VARResults.forecast().
            self._coefs[regime_i] = result.coefs[0]  # (n_vars, n_vars)
            self._intercepts[regime_i] = result.intercept  # (n_vars,)
            if self._n_vars is None:
                self._n_vars = result.coefs.shape[2]
                self._deviation_vars = result.model.endog_names

    def _build_residual_structures(self):
        """Build residual pools and contiguous block candidates from stored residuals."""
        for regime_i, resid in self.model.regime_residuals.items():
            if isinstance(resid, pd.DataFrame):
                resid_df = resid.copy()
            else:
                resid_df = pd.DataFrame(np.asarray(resid, dtype=float))

            resid_values = resid_df.to_numpy(dtype=float, copy=False)
            self._residual_pool[regime_i] = resid_values
            self._residual_blocks[regime_i] = self._extract_contiguous_blocks(resid_df)

            if resid_values.shape[0] > 1:
                self._residual_cov[regime_i] = np.cov(resid_values.T)
            else:
                self._residual_cov[regime_i] = np.eye(resid_values.shape[1])

    def _extract_contiguous_blocks(self, resid_df: pd.DataFrame) -> np.ndarray:
        """Return only truly contiguous hourly residual windows of length block_size."""
        values = resid_df.to_numpy(dtype=float, copy=False)
        if len(values) == 0:
            return np.empty((0, self.block_size, self._n_vars), dtype=float)

        if not isinstance(resid_df.index, pd.DatetimeIndex):
            n_blocks = len(values) // self.block_size
            if n_blocks == 0:
                return np.empty((0, self.block_size, self._n_vars), dtype=float)
            trimmed = values[: n_blocks * self.block_size]
            return trimmed.reshape(n_blocks, self.block_size, self._n_vars)

        if len(values) < self.block_size:
            return np.empty((0, self.block_size, self._n_vars), dtype=float)

        blocks = []
        timestamps = resid_df.index
        expected_step = pd.Timedelta(hours=1)
        for start in range(0, len(values) - self.block_size + 1):
            window = timestamps[start:start + self.block_size]
            deltas = np.diff(window.to_numpy())
            if len(deltas) == self.block_size - 1 and np.all(deltas == expected_step.to_numpy()):
                blocks.append(values[start:start + self.block_size])

        if not blocks:
            return np.empty((0, self.block_size, self._n_vars), dtype=float)
        return np.stack(blocks, axis=0)

    # ------------------------------------------------------------------
    # Layer 1: regime path generation
    # ------------------------------------------------------------------

    def _propagate_markov(self, n_days: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """
        Propagate Markov chain for n_days steps.
        Returns:
            sampled_path: (n_days,) int array of regime labels
            soft_probs:   (n_days, k) float probability vectors
        """
        sampled_path = np.empty(n_days, dtype=int)
        soft_probs = np.empty((n_days, self._k), dtype=float)

        current_probs = self._last_posterior.copy()
        # Sample initial regime
        current_regime = int(rng.choice(self._k, p=current_probs))

        for d in range(n_days):
            # Propagate probability vector
            current_probs = self._transition.T @ current_probs
            current_probs = np.maximum(current_probs, 1e-12)
            current_probs /= current_probs.sum()
            soft_probs[d] = current_probs

            # Sample or assign regime label
            if self.regime_mode == 'ensemble':
                current_regime = int(rng.choice(self._k, p=current_probs))
            else:
                # 'hard': argmax
                current_regime = int(np.argmax(current_probs))

            sampled_path[d] = current_regime

        return sampled_path, soft_probs

    def _get_regime_path(
        self, n_days: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (sampled_path, soft_probs) based on regime_mode.
          sampled_path: (n_days,) int array
          soft_probs:   (n_days, k) float array
        For 'soft': sampled_path = argmax of soft_probs (not used in propagation).
        For 'hard': sampled_path = argmax; soft_probs also returned.
        For 'ensemble': sampled_path drawn from Categorical per step.
        """
        if self.regime_mode == 'soft':
            # Deterministic probability propagation; sampled path is argmax (unused)
            soft_probs = np.empty((n_days, self._k), dtype=float)
            current_probs = self._last_posterior.copy()
            for d in range(n_days):
                current_probs = self._transition.T @ current_probs
                current_probs = np.maximum(current_probs, 1e-12)
                current_probs /= current_probs.sum()
                soft_probs[d] = current_probs
            sampled_path = np.argmax(soft_probs, axis=1).astype(int)
            return sampled_path, soft_probs
        else:
            return self._propagate_markov(n_days, rng)

    def _day_weights(
        self, d: int, sampled_path: np.ndarray, soft_probs: np.ndarray
    ) -> np.ndarray:
        """
        Compute per-regime blending weights for day d.
        Returns (k,) weight array that sums to 1.
        """
        if self.regime_mode == 'soft':
            return soft_probs[d]

        # 'hard' or 'ensemble'
        one_hot = np.zeros(self._k, dtype=float)
        # Map sampled regime label to its index in self._regimes
        regime_label = sampled_path[d]
        idx = self._regimes.index(regime_label)
        one_hot[idx] = 1.0

        if self.regime_mode == 'hard':
            return one_hot

        # 'ensemble': blend with soft via regime_strength
        blended = self.regime_strength * one_hot + (1.0 - self.regime_strength) * soft_probs[d]
        blended = np.maximum(blended, 0.0)
        total = blended.sum()
        if total > 0:
            blended /= total
        return blended

    # ------------------------------------------------------------------
    # Layer 2: residual sampling
    # ------------------------------------------------------------------

    def _normalize_weights(self, weights: np.ndarray | None, regime: int | None = None) -> np.ndarray:
        """Convert optional regime weights or a hard regime label to a probability vector."""
        if weights is not None:
            probs = np.asarray(weights, dtype=float).copy()
        elif regime is not None:
            probs = np.zeros(self._k, dtype=float)
            probs[self._regimes.index(regime)] = 1.0
        else:
            probs = np.full(self._k, 1.0 / self._k, dtype=float)

        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total <= 0:
            return np.full(self._k, 1.0 / self._k, dtype=float)
        return probs / total

    def _draw_residuals(
        self,
        n_hours: int,
        rng: np.random.Generator,
        weights: np.ndarray | None = None,
        regime: int | None = None,
    ) -> np.ndarray:
        """
        Returns (n_hours, n_vars) array of innovations scaled by residual_scale.
        For 'block': samples ceil(n_hours/block_size) blocks, trims to n_hours.
        For 'iid':   samples n_hours rows independently.
        For 'parametric': draws from N(0, Sigma).
        For 'deterministic': returns zeros.
        """
        if self.residual_mode == 'deterministic' or self.residual_scale == 0.0:
            return np.zeros((n_hours, self._n_vars), dtype=float)

        probs = self._normalize_weights(weights, regime=regime)

        if self.residual_mode == 'block':
            n_blocks_needed = int(np.ceil(n_hours / self.block_size))
            chosen_blocks = []
            for _ in range(n_blocks_needed):
                regime_idx = int(rng.choice(self._k, p=probs))
                regime_label = self._regimes[regime_idx]
                blocks = self._residual_blocks.get(regime_label)
                if blocks is not None and len(blocks) > 0:
                    chosen_blocks.append(blocks[rng.integers(0, len(blocks))])
                    continue

                pool = self._residual_pool.get(regime_label)
                if pool is None or len(pool) == 0:
                    chosen_blocks.append(np.zeros((self.block_size, self._n_vars), dtype=float))
                    continue
                idx = rng.integers(0, len(pool), size=self.block_size)
                chosen_blocks.append(pool[idx])

            eps = np.vstack(chosen_blocks)[:n_hours]

        elif self.residual_mode == 'iid':
            regime_draws = rng.choice(self._k, size=n_hours, p=probs)
            eps = np.zeros((n_hours, self._n_vars), dtype=float)
            for regime_idx, regime_label in enumerate(self._regimes):
                mask = regime_draws == regime_idx
                if not np.any(mask):
                    continue
                pool = self._residual_pool.get(regime_label)
                if pool is None or len(pool) == 0:
                    continue
                idx = rng.integers(0, len(pool), size=int(mask.sum()))
                eps[mask] = pool[idx]

        elif self.residual_mode == 'parametric':
            regime_draws = rng.choice(self._k, size=n_hours, p=probs)
            eps = np.zeros((n_hours, self._n_vars), dtype=float)
            for regime_idx, regime_label in enumerate(self._regimes):
                mask = regime_draws == regime_idx
                if not np.any(mask):
                    continue
                cov = self._residual_cov.get(regime_label, np.eye(self._n_vars))
                eps[mask] = rng.multivariate_normal(
                    np.zeros(self._n_vars), cov, size=int(mask.sum())
                )

        else:
            return np.zeros((n_hours, self._n_vars), dtype=float)

        return eps * self.residual_scale

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def _get_initial_y(self) -> np.ndarray:
        """Extract initial VAR state from the end of baseline data."""
        baseline = self.model.isd_baseline
        target_vars = [v.replace('_deviation', '') for v in self._deviation_vars]
        recent = baseline[target_vars].iloc[-24:].copy()
        recent_month = recent.index.month
        recent_hour = recent.index.hour

        init_deviations = np.zeros((len(recent), self._n_vars), dtype=float)
        for j, dev_var in enumerate(self._deviation_vars):
            var = dev_var.replace('_deviation', '')
            if var in self.model.seasonal_profiles:
                profile = self.model.seasonal_profiles[var]
                seasonal = np.array([
                    float(profile.get((m, h), 0.0))
                    for m, h in zip(recent_month, recent_hour)
                ])
                init_deviations[:, j] = recent[var].values - seasonal
            else:
                init_deviations[:, j] = 0.0

        return init_deviations  # (24, n_vars); VAR uses the last lag_order rows

    def _propagate(
        self,
        sampled_path: np.ndarray,
        soft_probs: np.ndarray,
        initial_y: np.ndarray,
        rng: np.random.Generator,
        forecast_index: pd.DatetimeIndex,
    ) -> np.ndarray:
        """
        Core VAR propagation with regime-weighted coefficients and bootstrapped residuals.

        Drift correction: every `centering_period` days the accumulated anomaly for that
        segment is mean-subtracted, and the carry-forward state A_prev is adjusted by the
        same offset.  This mirrors the chunk-level centering in the original deterministic
        pipeline (rsvar.py:972) and prevents long-term drift from accumulating when the
        blended VAR intercept sum_r(w_r * mu_r) is non-zero.

        Args:
            sampled_path:   (n_days,) regime label per day
            soft_probs:     (n_days, k) Markov probability vectors
            initial_y:      (lag_order, n_vars) initial state
            rng:            random generator for this member
            forecast_index: hourly DatetimeIndex for the forecast period

        Returns:
            anomaly: (n_hours, n_vars) array (deviation from seasonal profile)
        """
        n_hours = len(forecast_index)
        n_days = len(sampled_path)
        assert n_days * 24 >= n_hours, "sampled_path must cover the full forecast period"

        anomaly = np.empty((n_hours, self._n_vars), dtype=float)
        A_prev = initial_y[-1].copy()  # (n_vars,) carry-forward state

        segment_start_hour = 0  # hour index where the current centering segment began

        for d in range(n_days):
            # Blending weights for this day
            weights = self._day_weights(d, sampled_path, soft_probs)  # (k,)

            eps = self._draw_residuals(
                self.block_size,
                rng,
                weights=weights,
                regime=sampled_path[d],
            )  # (24, n_vars)

            for h in range(self.block_size):
                t = d * self.block_size + h
                if t >= n_hours:
                    break

                # Weighted VAR step: A_t = sum_r(w_r * (mu_r + Phi_r @ A_{t-1})) + eps
                A_t = np.zeros(self._n_vars, dtype=float)
                for i, regime_i in enumerate(self._regimes):
                    if weights[i] < 1e-10:
                        continue
                    mu = self._intercepts[regime_i]
                    phi = self._coefs[regime_i]
                    A_t += weights[i] * (mu + phi @ A_prev)
                A_t += eps[h]

                anomaly[t] = A_t
                A_prev = A_t

            # Periodic drift correction: at the end of each centering_period-day segment,
            # subtract the segment mean from the anomaly and from A_prev so the carry-
            # forward state does not import the drift into the next segment.
            day_in_segment = (d + 1) % self.centering_period
            segment_end_day = d + 1
            is_segment_boundary = (day_in_segment == 0)
            is_last_day = (segment_end_day * 24 >= n_hours)

            if is_segment_boundary or is_last_day:
                segment_end_hour = min(segment_end_day * 24, n_hours)
                seg = anomaly[segment_start_hour:segment_end_hour]
                if len(seg) > 0:
                    seg_mean = seg.mean(axis=0)
                    anomaly[segment_start_hour:segment_end_hour] -= seg_mean
                    A_prev -= seg_mean  # prevent bias from carrying into next segment
                segment_start_hour = segment_end_hour

        return anomaly

    def _generate_soft_deterministic_compatible(
        self,
        forecast_index: pd.DatetimeIndex,
        chunk_years: int = 5,
    ) -> pd.DataFrame:
        """
        Reproduce the legacy soft deterministic path inside AnomalyGenerator.

        This mirrors ForecastTo2100V5.forecast_without_climate_delta() but stays
        inside the generator rather than delegating to the legacy pipeline entrypoint.
        """
        target_vars = [c.replace('_deviation', '') for c in self._deviation_vars]
        n_steps = len(forecast_index)
        chunk_size = chunk_years * 8760
        n_chunks = int(np.ceil(n_steps / chunk_size))

        start_date = str(forecast_index[0])
        end_date = str(forecast_index[-1])
        daily_regime_probs = self.model._forecast_regime_probabilities(start_date, end_date)
        regime_prob_cols = sorted(
            [col for col in daily_regime_probs.columns if col.endswith('_prob')],
            key=lambda name: int(name.split('_')[1]) if name.split('_')[1].isdigit() else name
        )

        recent_data = self.model.isd_baseline[target_vars].iloc[-1000:].copy()
        recent_data['month'] = recent_data.index.month
        recent_data['hour'] = recent_data.index.hour

        deviation_vars = []
        for var in target_vars:
            if var not in self.model.seasonal_profiles:
                continue
            seasonal_vals = recent_data.apply(
                lambda row: self.model.seasonal_profiles[var][(row['month'], row['hour'])],
                axis=1,
            )
            recent_data[f'{var}_deviation'] = recent_data[var] - seasonal_vals
            deviation_vars.append(f'{var}_deviation')

        hourly_regime_probs = daily_regime_probs[regime_prob_cols].reindex(forecast_index, method='ffill')
        hourly_regime_probs = hourly_regime_probs.div(hourly_regime_probs.sum(1), axis=0)

        all_forecasts = []
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, n_steps)
            chunk_len = chunk_end - chunk_start
            chunk_index = forecast_index[chunk_start:chunk_end]

            chunk_df = pd.DataFrame(index=chunk_index, columns=target_vars, dtype=float)
            chunk_df['month'] = chunk_index.month
            chunk_df['hour'] = chunk_index.hour
            chunk_df['year'] = chunk_index.year

            for var in target_vars:
                if var not in self.model.seasonal_profiles:
                    continue
                seasonal_values = chunk_df.apply(
                    lambda row: self.model.seasonal_profiles[var][(row['month'], row['hour'])],
                    axis=1,
                )
                chunk_df[var] = seasonal_values

            chunk_probs = hourly_regime_probs.loc[chunk_index]
            blended_deviations = pd.DataFrame(
                np.zeros((chunk_len, len(deviation_vars))),
                index=chunk_index,
                columns=deviation_vars,
            )
            regime_weights = pd.DataFrame(
                0.0, index=chunk_index, columns=regime_prob_cols, dtype=float
            )

            for regime_i, var_model in sorted(self.model.regime_models.items()):
                regime_prob_col = f'regime_{regime_i}_prob'
                if regime_prob_col not in chunk_probs:
                    continue

                last_deviation = recent_data[deviation_vars].iloc[-var_model.k_ar:].values
                regime_deviation_forecast = var_model.forecast(y=last_deviation, steps=chunk_len)
                regime_probs_vals = chunk_probs[regime_prob_col].values.reshape(-1, 1)

                blended_deviations += regime_deviation_forecast * regime_probs_vals
                regime_weights.loc[chunk_index, regime_prob_col] = chunk_probs[regime_prob_col].values

            total_weights = regime_weights.sum(axis=1).clip(lower=1e-10)
            blended_deviations = blended_deviations.div(total_weights, axis=0)

            for dev_var in blended_deviations.columns:
                dev_mean = blended_deviations[dev_var].mean()
                blended_deviations[dev_var] = blended_deviations[dev_var] - dev_mean

            for var in target_vars:
                dev_var = f'{var}_deviation'
                if dev_var in blended_deviations.columns:
                    chunk_df[var] = chunk_df[var] + blended_deviations[dev_var]

            chunk_df = chunk_df[target_vars]

            recent_deviation_update = chunk_df.copy()
            recent_deviation_update['month'] = chunk_index.month
            recent_deviation_update['hour'] = chunk_index.hour
            for var in target_vars:
                if var not in self.model.seasonal_profiles:
                    continue
                profile = self.model.seasonal_profiles[var]
                seasonal_vals = pd.Series(
                    [float(profile.get((m, h), np.nan)) for m, h in zip(
                        recent_deviation_update['month'],
                        recent_deviation_update['hour'],
                    )],
                    index=recent_deviation_update.index,
                )
                recent_deviation_update[f'{var}_deviation'] = chunk_df[var] - seasonal_vals

            recent_data = pd.concat([recent_data, recent_deviation_update]).iloc[-1000:]
            all_forecasts.append(chunk_df)

        forecast_df = pd.concat(all_forecasts)
        forecast_df = self.model._enforce_solar_constraints(forecast_df)
        return forecast_df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        n_members: int,
        start_date: str = '2025-01-01',
        end_date: str = '2025-12-31',
        seed: int | None = None,
    ) -> list[pd.DataFrame]:
        """
        Generate M ensemble members.

        Each member is jointly sampled from both layers:
          - Layer 1: one independent regime path (or deterministic path for 'soft'/'hard')
          - Layer 2: one set of residual draws for that member

        Args:
            n_members:  number of ensemble members M
            start_date: forecast start (inclusive)
            end_date:   forecast end (inclusive)
            seed:       base random seed; member m uses seed + m for reproducibility

        Returns:
            list of n_members DataFrames, each indexed by hourly DatetimeIndex,
            with columns = target variable names.
            Values = seasonal_profile + anomaly  (no climate delta).
        """
        base_seed = seed if seed is not None else 0
        forecast_index = pd.date_range(start=start_date, end=end_date, freq='h')
        n_hours = len(forecast_index)
        n_days = int(np.ceil(n_hours / 24))

        # Pre-compute seasonal profile for the forecast period
        seasonal = self._compute_seasonal(forecast_index)
        initial_y = self._get_initial_y()

        members = []
        if self.regime_mode == 'soft' and self.residual_mode == 'deterministic':
            compatible = self._generate_soft_deterministic_compatible(forecast_index)
            return [compatible.copy() for _ in range(n_members)]

        for m in range(n_members):
            rng = np.random.default_rng(base_seed + m)

            sampled_path, soft_probs = self._get_regime_path(n_days, rng)
            anomaly = self._propagate(sampled_path, soft_probs, initial_y, rng, forecast_index)

            output = seasonal + anomaly  # (n_hours, n_vars)
            df = pd.DataFrame(output, index=forecast_index, columns=self._deviation_vars)
            # Rename deviation columns to plain variable names
            df.columns = [c.replace('_deviation', '') for c in df.columns]
            members.append(df)

        return members

    def _compute_seasonal(self, forecast_index: pd.DatetimeIndex) -> np.ndarray:
        """
        Compute baseline seasonal profile for each hour in forecast_index.
        Returns (n_hours, n_vars) array.
        """
        n_hours = len(forecast_index)
        seasonal = np.zeros((n_hours, self._n_vars), dtype=float)
        profiles = self.model.seasonal_profiles

        for j, dev_var in enumerate(self._deviation_vars):
            var = dev_var.replace('_deviation', '')
            if var not in profiles:
                continue
            profile = profiles[var]
            for t, ts in enumerate(forecast_index):
                seasonal[t, j] = float(profile.get((ts.month, ts.hour), 0.0))

        return seasonal
