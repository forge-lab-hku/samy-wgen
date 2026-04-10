#!/usr/bin/env python3
"""
Ensemble Anomaly Generation Pipeline

Generates M stochastic ensemble members of synthetic weather anomalies using the
RS-VAR model with two independently controllable randomization layers:

  Layer 1 (regime_mode):    how the regime sequence is drawn per member
    'soft'      deterministic probability-weighted blend (existing behavior)
    'hard'      deterministic argmax path (no member diversity)
    'ensemble'  per-member Markov chain sampling

  Layer 2 (residual_mode):  how the VAR innovation is sampled
    'deterministic'  no noise (reproduces existing forecast_without_climate_delta)
    'iid'            i.i.d. residual row per hourly step
    'block'          24h contiguous block per day (preserves within-day autocorrelation)
    'parametric'     N(0, Sigma_regime) draw

  Strength controls:
    --regime-strength [0,1]     blending between soft and hard regime assignment (ensemble mode only)
    --residual-scale  [0,inf)   scalar multiplier on bootstrapped residuals

OUTPUT:
    output_dir/member_000.parquet, member_001.parquet, ...
    output_dir/ensemble_summary.xlsx  (median, p05, p95 per hour)

USAGE:
    samy-ensemble --isd data/ISD/Miami_with_solar.csv --city Miami \\
                  --members 50 --out-dir output/ensemble/ \\
                  --regime-mode soft --residual-mode deterministic \\
                  --start 2025-01-01 --end 2025-12-31

    # Reproduce existing soft-mode behavior (regression check)
    samy-ensemble --isd data/ISD/Miami_with_solar.csv --city Miami \\
                  --members 1 --regime-mode soft --residual-mode deterministic

    # Ablation: regime diversity only
    samy-ensemble ... --regime-mode ensemble --residual-mode deterministic

    # Ablation: residual diversity only
    samy-ensemble ... --regime-mode soft --residual-mode block
"""

import argparse
import os
import importlib.resources
from pathlib import Path

import numpy as np
import pandas as pd

from samy_wgen.model.rsvar import ForecastTo2100V5
from samy_wgen.model.anomaly_generator import AnomalyGenerator


# ==========================================================================
# CONFIGURATION
# ==========================================================================
BASELINE_START = 1991
BASELINE_END = 2010

TARGET_VARS = [
    'temp', 'pressure', 'wind_speed', 'GHI', 'DNI', 'DHI',
    'relative_humidity', 'specific_humidity',
]

ABS_LIMITS = {
    'temp': 100.0,
    'pressure': 2.0e5,
    'wind_speed': 200.0,
    'GHI': 2.0e3,
    'DNI': 2.0e3,
    'DHI': 2.0e3,
    'relative_humidity': 200.0,
    'specific_humidity': 0.1,
}


# ==========================================================================
# TRAINING PIPELINE
# ==========================================================================

class EnsemblePipeline(ForecastTo2100V5):
    """ForecastTo2100V5 subclass that skips CORDEX loading (not needed for anomaly generation)."""

    def __init__(self, isd_solar_path, lat=34.0, baseline_start=1991, baseline_end=2010, seed=42):
        super().__init__(
            isd_solar_path=isd_solar_path,
            cordex_rcp85_path=os.devnull,
            lat=lat,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            seed=seed,
        )

    def train(self, target_vars=None):
        """Run all training steps required for AnomalyGenerator."""
        if target_vars is None:
            target_vars = TARGET_VARS

        self.load_isd_data()
        self.extract_seasonal_profiles(target_vars)
        self.engineer_features()
        self.standardize_features(target_vars)
        self.train_regime_switching(k_regimes=3)
        self.train_regime_conditional_vars(target_vars, k_regimes=3)
        return self


def _validate_member(member_df: pd.DataFrame, member_idx: int) -> None:
    """Fail fast on non-finite or obviously exploded member values."""
    values = member_df.to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"member_{member_idx:03d} contains non-finite values")

    for col, limit in ABS_LIMITS.items():
        if col not in member_df.columns:
            continue
        col_max = float(np.nanmax(np.abs(member_df[col].to_numpy(dtype=float, copy=False))))
        if col_max > limit:
            raise ValueError(
                f"member_{member_idx:03d} column '{col}' exceeded sanity bound "
                f"{limit:g} with abs max {col_max:.3f}"
            )


# ==========================================================================
# ENSEMBLE GENERATION
# ==========================================================================

def run_and_save_ensemble(
    isd_path: str | Path,
    lat: float,
    output_dir: str | Path,
    n_members: int = 50,
    start_date: str = '2025-01-01',
    end_date: str = '2025-12-31',
    regime_mode: str = 'soft',
    residual_mode: str = 'deterministic',
    regime_strength: float = 1.0,
    residual_scale: float = 1.0,
    block_size: int = 24,
    centering_period: int = 365,
    seed: int = 0,
    target_vars: list[str] | None = None,
) -> Path:
    """
    Train RS-VAR and generate M ensemble members.

    Args:
        isd_path:        Path to merged ISD + solar CSV (hourly, LST)
        lat:             Station latitude
        output_dir:      Directory for output parquet files + summary Excel
        n_members:       Number of ensemble members to generate
        start_date:      Forecast start (inclusive)
        end_date:        Forecast end (inclusive)
        regime_mode:     'soft' | 'hard' | 'ensemble'
        residual_mode:   'deterministic' | 'iid' | 'block' | 'parametric'
        regime_strength: [0,1] blending factor for 'ensemble' mode
        residual_scale:  [0,inf) multiplier on bootstrapped residuals
        block_size:      Hours per residual block (default 24)
        seed:            Base random seed; member m uses seed + m
        target_vars:     Variables to generate (default: all 8 weather vars)

    Returns:
        Path to output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if target_vars is None:
        target_vars = TARGET_VARS

    # --- Train ---
    print("\n" + "="*80)
    print("ENSEMBLE PIPELINE: TRAINING RS-VAR")
    print("="*80)
    pipeline = EnsemblePipeline(
        isd_solar_path=str(isd_path),
        lat=lat,
        baseline_start=BASELINE_START,
        baseline_end=BASELINE_END,
        seed=seed,
    )
    pipeline.train(target_vars)

    # --- Generate ensemble ---
    print("\n" + "="*80)
    print(f"ENSEMBLE GENERATION: {n_members} members")
    print(f"  regime_mode={regime_mode!r}  regime_strength={regime_strength}")
    print(f"  residual_mode={residual_mode!r}  residual_scale={residual_scale}")
    print(f"  period: {start_date} → {end_date}")
    print("="*80)

    gen = AnomalyGenerator(
        fitted_model=pipeline,
        regime_mode=regime_mode,
        residual_mode=residual_mode,
        block_size=block_size,
        regime_strength=regime_strength,
        residual_scale=residual_scale,
        centering_period=centering_period,
    )

    members = gen.generate(
        n_members=n_members,
        start_date=start_date,
        end_date=end_date,
        seed=seed,
    )

    # --- Save individual members ---
    print(f"\nSaving {n_members} members to {output_dir}/")
    for m, df in enumerate(members):
        _validate_member(df, m)
        out_path = output_dir / f"member_{m:03d}.parquet"
        df.to_parquet(out_path)

    print(f"✓ Saved {n_members} member parquet files")

    # --- Save ensemble summary ---
    summary_path = output_dir / "ensemble_summary.xlsx"
    _save_summary(members, summary_path, regime_mode, residual_mode,
                  regime_strength, residual_scale, n_members, start_date, end_date, seed)
    print(f"✓ Summary saved: {summary_path}")

    return output_dir


def _save_summary(
    members: list[pd.DataFrame],
    path: Path,
    regime_mode: str,
    residual_mode: str,
    regime_strength: float,
    residual_scale: float,
    n_members: int,
    start_date: str,
    end_date: str,
    seed: int,
):
    """Save ensemble summary statistics to Excel."""
    all_data = np.stack([df.values for df in members], axis=0)  # (M, n_hours, n_vars)
    index = members[0].index
    columns = members[0].columns

    median_df = pd.DataFrame(np.median(all_data, axis=0), index=index, columns=columns)
    p05_df = pd.DataFrame(np.percentile(all_data, 5, axis=0), index=index, columns=columns)
    p95_df = pd.DataFrame(np.percentile(all_data, 95, axis=0), index=index, columns=columns)

    meta = pd.DataFrame({
        'Parameter': [
            'n_members', 'start_date', 'end_date', 'seed',
            'regime_mode', 'regime_strength', 'residual_mode', 'residual_scale',
        ],
        'Value': [
            n_members, start_date, end_date, seed,
            regime_mode, regime_strength, residual_mode, residual_scale,
        ],
    })

    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        meta.to_excel(writer, sheet_name='Metadata', index=False)
        median_df.to_excel(writer, sheet_name='Median')
        p05_df.to_excel(writer, sheet_name='P05')
        p95_df.to_excel(writer, sheet_name='P95')


# ==========================================================================
# CLI
# ==========================================================================

def _get_lat_for_city(city: str) -> float:
    """Look up latitude from bundled cities_config.csv."""
    ref = importlib.resources.files("samy_wgen.config").joinpath("cities_config.csv")
    cfg = pd.read_csv(str(ref))
    row = cfg[cfg['city'] == city]
    if row.empty:
        raise SystemExit(f"ERROR: City '{city}' not found in cities_config.csv")
    return float(row.iloc[0]['latitude'])


def main():
    parser = argparse.ArgumentParser(
        description="Generate RS-VAR ensemble members for HDC analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--isd', required=True, help="Path to merged ISD+solar CSV (LST, hourly)")
    parser.add_argument('--city', required=True, help="City name (used to look up latitude)")
    parser.add_argument('--lat', type=float, default=None,
                        help="Override latitude (optional; auto-looked up from city if not given)")
    parser.add_argument('--out-dir', default='output/ensemble',
                        help="Output directory for member parquet files + summary Excel")
    parser.add_argument('--members', type=int, default=50,
                        help="Number of ensemble members M")
    parser.add_argument('--start', default='2025-01-01', help="Forecast start date")
    parser.add_argument('--end', default='2025-12-31', help="Forecast end date")
    parser.add_argument('--regime-mode', default='soft',
                        choices=['soft', 'hard', 'ensemble'],
                        help="Layer 1: regime path generation mode")
    parser.add_argument('--regime-strength', type=float, default=1.0,
                        help="Layer 1 intensity [0,1]: 0=pure soft, 1=pure Markov sample (ensemble mode only)")
    parser.add_argument('--residual-mode', default='deterministic',
                        choices=['deterministic', 'iid', 'block', 'parametric'],
                        help="Layer 2: residual sampling mode")
    parser.add_argument('--residual-scale', type=float, default=1.0,
                        help="Layer 2 intensity [0,inf): multiplier on bootstrapped residuals")
    parser.add_argument('--block-size', type=int, default=24,
                        help="Residual block size in hours (for block mode)")
    parser.add_argument('--centering-period', type=int, default=365,
                        help="Days between drift corrections (annual centering = 365)")
    parser.add_argument('--seed', type=int, default=0, help="Base random seed")

    args = parser.parse_args()

    lat = args.lat if args.lat is not None else _get_lat_for_city(args.city)

    run_and_save_ensemble(
        isd_path=args.isd,
        lat=lat,
        output_dir=args.out_dir,
        n_members=args.members,
        start_date=args.start,
        end_date=args.end,
        regime_mode=args.regime_mode,
        residual_mode=args.residual_mode,
        regime_strength=args.regime_strength,
        residual_scale=args.residual_scale,
        block_size=args.block_size,
        centering_period=args.centering_period,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
