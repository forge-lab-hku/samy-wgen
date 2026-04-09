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
        """Pre-slice residuals into blocks and compute covariance matrices."""
        for regime_i, resid in self.model.regime_residuals.items():
            resid = np.array(resid, dtype=float)
            self._residual_pool[regime_i] = resid

            # Trim to full blocks
            n_blocks = len(resid) // self.block_size
            if n_blocks > 0:
                trimmed = resid[: n_blocks * self.block_size]
                blocks = trimmed.reshape(n_blocks, self.block_size, -1)
                self._residual_blocks[regime_i] = blocks
            else:
                # Not enough residuals for even one block — fall back to single block
                self._residual_blocks[regime_i] = resid[np.newaxis, ...]

            if resid.shape[0] > 1:
                self._residual_cov[regime_i] = np.cov(resid.T)
            else:
                self._residual_cov[regime_i] = np.eye(resid.shape[1])

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

    def _draw_residuals(
        self, regime: int, n_hours: int, rng: np.random.Generator
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

        if self.residual_mode == 'block':
            blocks = self._residual_blocks.get(regime)
            if blocks is None or len(blocks) == 0:
                return np.zeros((n_hours, self._n_vars), dtype=float)
            n_blocks_needed = int(np.ceil(n_hours / self.block_size))
            chosen = [blocks[rng.integers(0, len(blocks))] for _ in range(n_blocks_needed)]
            eps = np.vstack(chosen)[:n_hours]

        elif self.residual_mode == 'iid':
            pool = self._residual_pool.get(regime)
            if pool is None or len(pool) == 0:
                return np.zeros((n_hours, self._n_vars), dtype=float)
            idx = rng.integers(0, len(pool), size=n_hours)
            eps = pool[idx]

        elif self.residual_mode == 'parametric':
            cov = self._residual_cov.get(regime, np.eye(self._n_vars))
            eps = rng.multivariate_normal(np.zeros(self._n_vars), cov, size=n_hours)

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

            # Draw residuals for this 24h block (from sampled regime)
            regime_label = sampled_path[d]
            eps = self._draw_residuals(regime_label, self.block_size, rng)  # (24, n_vars)

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
