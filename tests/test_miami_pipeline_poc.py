"""
End-to-end POC test: Miami CMIP6 MPI-ESM1-2-LR SSP245 pipeline.

Covers all pipeline stages and saves every intermediate result to
  tests/output/miami_poc_cmip6_ssp245/
for manual inspection after the run.

Pipeline stages:
  1. Parse ISD .gz files  → raw UTC CSV
  2. Remap ISD columns + UTC→LST  → standardised meteorology DataFrame
  3. Merge with NSRDB solar data  → combined ISD+solar CSV (DATE index, LST)
  4. RS-VAR training + forecast (2025-2100)  → RSVAR_Output Excel + CSV
  5. Load CMIP6 MPI-ESM1-2-LR SSP245 data
  6. Compute CORDEX baseline climatology (1991-2010)
  7. Apply climate delta to RS-VAR output  → delta-corrected hourly CSV
  8. Write EPW files for years 2030, 2050, 2080

Run with:
  pytest -m slow tests/test_miami_pipeline_poc.py -v -s
  pytest -m slow tests/test_miami_pipeline_poc.py -v -s --overwrite   # force rerun all stages
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIAMI_LAT    = 25.7617
MIAMI_LON    = -80.1918
MIAMI_TZ     = -5.0        # EST — fixed offset, no DST
MIAMI_ELEV_M = 2.0
MIAMI_GCM    = "MPI-ESM1-2-LR"
MIAMI_SSP    = "ssp245"
BASELINE_START = 1991
BASELINE_END   = 2010

REPO_ROOT   = Path(__file__).parent.parent
DATA_DIR    = REPO_ROOT / "data"
OUTPUT_DIR  = Path(__file__).parent / "output" / "miami_poc_cmip6_ssp245"

ISD_STATION_DIR = DATA_DIR / "ISD" / "Miami" / "722020-12839_MIAMI_INTERNATIONAL_AIRPORT"
NSRDB_CSV       = DATA_DIR / "NSRDB" / "nsrdb_miami_1998-2024.csv"
CMIP6_DIR       = DATA_DIR / "CMIP6"

from samy_wgen.utils.parse_ISD import ISD_RENAME, remap_isd_columns, resample_isd_hourly


# ---------------------------------------------------------------------------
# Stage helpers  (each accepts overwrite=False to skip if output already exists)
# ---------------------------------------------------------------------------

def _stage1_parse_isd() -> pd.DataFrame:
    """Parse all ISD .gz files in the Miami station directory to a raw UTC DataFrame."""
    import tempfile
    from samy_wgen.utils.parse_ISD import process_station_directory

    print(f"  [stage 1] Parsing ISD .gz files from:\n    {ISD_STATION_DIR}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        result_path = process_station_directory(str(ISD_STATION_DIR), tmp_dir)
        assert result_path is not None, "process_station_directory returned None"
        raw = pd.read_csv(result_path, parse_dates=["datestamp"], low_memory=False)
    print(f"  [stage 1] Parsed {len(raw):,} records")
    return raw


def _stage2_remap_to_lst(isd_raw: pd.DataFrame, out_dir: Path, overwrite: bool = False) -> pd.DataFrame:
    """Remap ISD field names, UTC → LST, resample to hourly.

    Output columns: temp (°C), dewpoint (°C), pressure (hPa), wind_speed (m/s), wind_dir (°)
    Index: tz-naive LST DatetimeIndex, hourly
    """
    from samy_wgen.utils.time_utils import utc_to_lst

    csv_path = out_dir / "1_isd_lst.csv"
    if csv_path.exists() and not overwrite:
        print(f"  [stage 2] Cache hit: {csv_path.name}")
        return pd.read_csv(csv_path, parse_dates=["DATE"]).set_index("DATE")

    raw = isd_raw.dropna(subset=["datestamp"]).set_index("datestamp").sort_index()

    df = remap_isd_columns(raw)

    key_cols = [c for c in ["temp", "dewpoint", "pressure", "wind_speed"] if c in df.columns]
    df = df.dropna(subset=key_cols, how="all")

    # UTC → LST (Miami = EST = UTC-5, fixed, no DST)
    df = utc_to_lst(df, "Miami")

    # Resample to hourly: mean for continuous vars, mode for wind_dir
    df = resample_isd_hourly(df)

    yr_min, yr_max = df.index.year.min(), df.index.year.max()
    print(f"  [stage 2] ISD LST: {len(df):,} hourly rows, {yr_min}–{yr_max}")
    for col in key_cols:
        print(f"    {col:20s}: mean={df[col].mean():.2f}, NaN%={df[col].isna().mean()*100:.1f}%")

    df.index.name = "DATE"
    df.to_csv(csv_path)
    print(f"  [stage 2] Saved: {csv_path.name}")
    return df


def _stage3_merge_with_nsrdb(isd_lst: pd.DataFrame, out_dir: Path,
                               overwrite: bool = False) -> Path:
    """Merge ISD LST DataFrame with NSRDB solar data (GHI, DNI, DHI).

    NSRDB timestamps (Minute=30) are floored to the hour start.
    Returns path to the merged CSV (DATE index, LST).
    """
    csv_path = out_dir / "2_isd_solar_lst.csv"
    if csv_path.exists() and not overwrite:
        print(f"  [stage 3] Cache hit: {csv_path.name}")
        return csv_path

    print(f"  [stage 3] Loading NSRDB: {NSRDB_CSV.name}")
    nsrdb = pd.read_csv(NSRDB_CSV)
    nsrdb["DATE"] = pd.to_datetime(
        nsrdb[["Year", "Month", "Day", "Hour"]].rename(
            columns={"Year": "year", "Month": "month", "Day": "day", "Hour": "hour"}
        )
    )
    nsrdb = nsrdb.set_index("DATE").sort_index()

    solar_cols = ["GHI", "DNI", "DHI"]
    missing_solar = [c for c in solar_cols if c not in nsrdb.columns]
    assert not missing_solar, f"NSRDB missing columns: {missing_solar}"

    solar = nsrdb[solar_cols].clip(lower=0).resample("h").mean()

    merged = isd_lst.join(solar, how="inner")
    merged.index.name = "DATE"
    merged = merged.ffill(limit=3).bfill(limit=3)
    merged = merged.dropna(subset=["temp", "GHI"])

    yr_min, yr_max = merged.index.year.min(), merged.index.year.max()
    print(f"  [stage 3] Merged: {len(merged):,} hourly rows, {yr_min}–{yr_max}")
    for col in merged.columns:
        print(f"    {col:20s}: mean={merged[col].mean():.3f}, NaN%={merged[col].isna().mean()*100:.1f}%")

    merged.to_csv(csv_path)
    print(f"  [stage 3] Saved: {csv_path.name}")
    return csv_path


def _stage4_run_rsvar(merged_csv: Path, out_dir: Path, overwrite: bool = False) -> Path:
    """Run RS-VAR training + forecast (2025-2100). Returns path to Excel output."""
    from samy_wgen.pipeline.run_rsvar import run_and_save_rsvar

    excel_path = out_dir / "3_rsvar_miami.xlsx"
    if excel_path.exists() and not overwrite:
        print(f"  [stage 4] Cache hit: {excel_path.name}")
        return excel_path

    print(f"  [stage 4] Training RS-VAR on: {merged_csv.name}")
    run_and_save_rsvar(
        isd_path=merged_csv,
        lat=MIAMI_LAT,
        output_path=excel_path,
        city="Miami",
    )
    assert excel_path.exists(), f"RS-VAR Excel not created at {excel_path}"
    print(f"  [stage 4] Saved: {excel_path.name}")
    return excel_path


def _stage5_load_cmip6(out_dir: Path, overwrite: bool = False):
    """Load CMIP6 MPI-ESM1-2-LR SSP245. Returns (daily_df, monthly_df, meta)."""
    from samy_wgen.pipeline.run_climate_delta import load_cmip6_or_cordex6

    csv_path = out_dir / "4_cmip6_mpi_ssp245_daily.csv"

    print(f"  [stage 5] Loading CMIP6 MPI-SSP245 from: {CMIP6_DIR.name}/Miami/")
    daily_df, monthly_df, meta = load_cmip6_or_cordex6(
        base_path=CMIP6_DIR,
        city="Miami",
        ssp=MIAMI_SSP,
    )

    if not csv_path.exists() or overwrite:
        daily_df.to_csv(csv_path)
        print(f"  [stage 5] Saved daily: {csv_path.name}")
        if not monthly_df.empty:
            monthly_csv = out_dir / "4_cmip6_mpi_ssp245_monthly.csv"
            monthly_df.to_csv(monthly_csv)
            print(f"  [stage 5] Saved monthly: {monthly_csv.name}")

    print(f"  [stage 5] {len(daily_df):,} daily rows, "
          f"{daily_df.index.min().date()} → {daily_df.index.max().date()}")
    print(f"  [stage 5] Columns: {list(daily_df.columns)}")
    print(f"  [stage 5] Model: {meta.get('model_name', 'unknown')}")
    return daily_df, monthly_df, meta


def _stage6_compute_baseline(daily_df: pd.DataFrame, out_dir: Path,
                               overwrite: bool = False) -> dict:
    """Compute CORDEX DOY baseline climatology (1991-2010). Returns baseline dict."""
    from samy_wgen.pipeline.run_climate_delta import compute_cordex_baseline

    csv_path = out_dir / "5_cordex_baseline_doy.csv"

    print(f"  [stage 6] Computing CORDEX baseline ({BASELINE_START}–{BASELINE_END})")
    baseline = compute_cordex_baseline(daily_df, BASELINE_START, BASELINE_END, mode="daily")

    if not csv_path.exists() or overwrite:
        rows = []
        for var, info in baseline.items():
            if "doy" in info:
                for doy, val in info["doy"].items():
                    rows.append({"variable": var, "doy": doy, "baseline_mean": val,
                                 "overall_mean": info["overall"], "transform": info["transform"]})
        if rows:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"  [stage 6] Saved: {csv_path.name}")

    for var, info in baseline.items():
        print(f"    {var:15s}: overall={info['overall']:.3f}, transform={info['transform']}")
    return baseline


def _stage7_apply_delta(rsvar_excel: Path, daily_df: pd.DataFrame,
                         baseline: dict, out_dir: Path,
                         overwrite: bool = False) -> pd.DataFrame:
    """Apply MPI-SSP245 climate delta to RS-VAR output. Returns hourly output DataFrame."""
    from samy_wgen.pipeline.run_climate_delta import load_rsvar_output, apply_climate_delta

    csv_path = out_dir / "6_miami_mpi_ssp245_hourly.csv"
    if csv_path.exists() and not overwrite:
        print(f"  [stage 7] Cache hit: {csv_path.name}")
        return pd.read_csv(csv_path, parse_dates=["datetime"]).set_index("datetime")

    print(f"  [stage 7] Loading RS-VAR output: {rsvar_excel.name}")
    rsvar_df, _ = load_rsvar_output(rsvar_excel)
    print(f"  [stage 7] RS-VAR: {len(rsvar_df):,} rows, cols={list(rsvar_df.columns)}")

    print(f"  [stage 7] Applying delta: {MIAMI_GCM} {MIAMI_SSP}")
    output_df, delta_stats = apply_climate_delta(
        rsvar_df=rsvar_df,
        cordex=daily_df,
        baseline=baseline,
        gcm=MIAMI_GCM,
        rcm=None,
        lat=MIAMI_LAT,
        baseline_start=BASELINE_START,
        baseline_end=BASELINE_END,
        isd_path=None,
        debug=False,
        delta_mode="daily",
    )

    # Save delta statistics summary
    delta_rows = []
    for var, vals in delta_stats.items():
        if vals:
            arr = np.array(vals)
            delta_rows.append({
                "variable": var, "n_days": len(arr),
                "mean": float(np.mean(arr)), "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p5": float(np.percentile(arr, 5)),
                "p95": float(np.percentile(arr, 95)),
            })
    if delta_rows:
        pd.DataFrame(delta_rows).to_csv(
            out_dir / "6_delta_stats.csv", index=False
        )

    output_df.index.name = "datetime"
    output_df.to_csv(csv_path)
    print(f"  [stage 7] Saved: {csv_path.name} ({len(output_df):,} rows)")
    return output_df


def _stage8_write_epw_samples(output_df: pd.DataFrame, out_dir: Path,
                               sample_years: tuple = (2030, 2050, 2080),
                               overwrite: bool = False) -> list:
    """Write EPW files for a sample set of years. Returns list of written paths."""
    from samy_wgen.pipeline.gen_samy import _write_epw_for_year

    epw_dir = out_dir / "EPW"
    epw_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for year in sample_years:
        epw_path = epw_dir / f"Miami_MPI_SSP245_{year}.epw"

        if epw_path.exists() and not overwrite:
            print(f"  [stage 8] Cache hit: {epw_path.name}")
            written.append(epw_path)
            continue

        if (output_df.index.year == year).sum() == 0:
            print(f"  [stage 8] WARNING: no data for {year}, skipping")
            continue

        _write_epw_for_year(
            df=output_df, year=year, out_path=epw_path,
            location="Miami-FL-USA",
            lat=MIAMI_LAT, lon=MIAMI_LON,
            tz=MIAMI_TZ, elevation_m=MIAMI_ELEV_M,
        )
        lines = epw_path.read_text().splitlines()
        print(f"  [stage 8] {epw_path.name}: {len(lines) - 8} data rows")
        written.append(epw_path)

    return written


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_temp_warming(output_df: pd.DataFrame) -> None:
    for label, years in [("2026–2040", range(2026, 2041)), ("2081–2100", range(2081, 2101))]:
        mask = output_df.index.year.isin(years)
        if mask.any():
            print(f"  temp mean {label}: {output_df.loc[mask, 'temp'].mean():.2f}°C")


def _check_physical_plausibility(output_df: pd.DataFrame) -> None:
    temp = output_df["temp"]
    assert temp.min() > -30, f"Unrealistically cold: {temp.min():.1f}°C"
    assert temp.max() < 60,  f"Unrealistically hot: {temp.max():.1f}°C"
    if "relative_humidity" in output_df.columns:
        rh = output_df["relative_humidity"]
        assert (rh >= 0).all(),   "Negative RH found"
        assert (rh <= 100).all(), f"RH > 100% found: max={rh.max():.1f}%"
    if "GHI" in output_df.columns:
        assert (output_df["GHI"] >= 0).all(), "Negative GHI found"
    if "wind_speed" in output_df.columns:
        assert (output_df["wind_speed"] >= 0).all(), "Negative wind speed found"


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_miami_cmip6_ssp245_full_pipeline(overwrite):
    """Full pipeline POC: Miami MPI-ESM1-2-LR SSP245.

    Each stage writes its output to tests/output/miami_poc_cmip6_ssp245/ and
    is skipped on re-runs unless --overwrite is passed.

    Expected runtime: 5–15 min (dominated by RS-VAR training + forecast).
    Run with --overwrite to force all stages to recompute from scratch.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ow = overwrite  # shorthand
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Overwrite mode: {ow}")

    # stage 1+2 — parse ISD → remap to LST
    assert ISD_STATION_DIR.exists(), f"ISD dir not found: {ISD_STATION_DIR}"
    assert NSRDB_CSV.exists(), f"NSRDB not found: {NSRDB_CSV}"
    isd_lst_cache = OUTPUT_DIR / "1_isd_lst.csv"
    if isd_lst_cache.exists() and not ow:
        print("  [stage 2] Cache hit: 1_isd_lst.csv")
        isd_lst = pd.read_csv(isd_lst_cache, parse_dates=["DATE"]).set_index("DATE")
    else:
        isd_raw = _stage1_parse_isd()
        isd_lst = _stage2_remap_to_lst(isd_raw, OUTPUT_DIR, overwrite=ow)
    assert not isd_lst.empty
    assert "temp" in isd_lst.columns
    assert isd_lst.index.tzinfo is None, "Index must be tz-naive (LST)"

    # stage 3 — merge with NSRDB
    merged_csv = _stage3_merge_with_nsrdb(isd_lst, OUTPUT_DIR, overwrite=ow)
    sample = pd.read_csv(merged_csv, nrows=5)
    for col in ["temp", "GHI", "DNI", "DHI"]:
        assert col in sample.columns, f"Merged CSV missing '{col}'"

    # stage 4 — RS-VAR
    rsvar_excel = _stage4_run_rsvar(merged_csv, OUTPUT_DIR, overwrite=ow)
    assert rsvar_excel.exists()

    # stage 5 — load CMIP6
    assert CMIP6_DIR.exists(), f"CMIP6 dir not found: {CMIP6_DIR}"
    daily_df, monthly_df, meta = _stage5_load_cmip6(OUTPUT_DIR, overwrite=ow)
    assert not daily_df.empty
    assert "tas_C" in daily_df.columns, f"CMIP6 missing 'tas_C'. Got: {list(daily_df.columns)}"
    assert meta["model_name"] == MIAMI_GCM, f"Expected {MIAMI_GCM}, got {meta['model_name']}"

    # stage 6 — compute baseline
    baseline = _stage6_compute_baseline(daily_df, OUTPUT_DIR, overwrite=ow)
    assert "tas_C" in baseline, "Baseline missing 'tas_C'"
    assert "ps_hPa" in baseline, "Baseline missing 'ps_hPa'"

    # stage 7 — apply delta
    output_df = _stage7_apply_delta(rsvar_excel, daily_df, baseline, OUTPUT_DIR, overwrite=ow)
    assert not output_df.empty
    assert output_df.index.year.max() >= 2095, "Forecast doesn't reach 2095"
    _check_physical_plausibility(output_df)
    print("\n  === Temperature warming signal ===")
    _check_temp_warming(output_df)

    # stage 8 — write EPW samples
    epw_paths = _stage8_write_epw_samples(output_df, OUTPUT_DIR,
                                           sample_years=(2030, 2050, 2080), overwrite=ow)
    assert len(epw_paths) == 3, f"Expected 3 EPW files, got {len(epw_paths)}"
    for p in epw_paths:
        n_lines = len(p.read_text().splitlines())
        assert n_lines >= 8 + 8760, f"{p.name}: expected ≥8768 lines, got {n_lines}"

    print(f"\n  All stages completed. Outputs in: {OUTPUT_DIR}")
