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
  pytest -m slow tests/test_miami_pipeline_poc.py -v
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

# ISD mandatory-field name → model internal name
ISD_RENAME = {
    "air_temperature_observation_air_temperature":     "temp",
    "air_temperature_observation_dew_point_temperature": "dewpoint",
    "atmospheric_pressure_observation_sea_level_pressure": "pressure",
    "wind_observation_speed_rate":                      "wind_speed",
}


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def _stage1_parse_isd(out_dir: Path) -> Path:
    """Parse all ISD .gz files in the Miami station directory to a raw UTC CSV.

    Returns path to the saved CSV.
    """
    from samy_wgen.utils.parse_ISD import process_station_directory

    csv_path = out_dir / "stage1_isd_raw_utc.csv"
    if csv_path.exists():
        print(f"  [stage 1] Cache hit: {csv_path}")
        return csv_path

    print(f"  [stage 1] Parsing ISD .gz files from:\n    {ISD_STATION_DIR}")
    result_path = process_station_directory(str(ISD_STATION_DIR), str(out_dir))
    assert result_path is not None, "process_station_directory returned None"

    # Rename output file to stage1 name for clarity
    import shutil
    shutil.move(result_path, str(csv_path))
    print(f"  [stage 1] Saved: {csv_path}")
    return csv_path


def _stage2_remap_to_lst(isd_csv: Path, out_dir: Path) -> pd.DataFrame:
    """Remap ISD field names, convert UTC → LST, return standardised DataFrame.

    Output columns: temp (°C), dewpoint (°C), pressure (hPa), wind_speed (m/s)
    Index: naive LST DatetimeIndex (hourly, may be irregular)

    Also saves the result to out_dir/stage2_isd_lst.csv for inspection.
    """
    from samy_wgen.utils.time_utils import utc_to_lst

    csv_path = out_dir / "stage2_isd_lst.csv"
    if csv_path.exists():
        print(f"  [stage 2] Cache hit: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["DATE"]).set_index("DATE")
        return df

    print(f"  [stage 2] Loading ISD raw CSV: {isd_csv}")
    raw = pd.read_csv(isd_csv, parse_dates=["datestamp"], low_memory=False)
    raw = raw.dropna(subset=["datestamp"])
    raw = raw.set_index("datestamp").sort_index()

    # Keep only the four mandatory weather fields (others are quality codes / extras)
    keep = [c for c in ISD_RENAME if c in raw.columns]
    missing_fields = [c for c in ISD_RENAME if c not in raw.columns]
    if missing_fields:
        print(f"  [stage 2] WARNING: ISD fields not found (will be NaN): {missing_fields}")

    df = raw[keep].rename(columns=ISD_RENAME)

    # Ensure numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where all key weather fields are NaN
    key_cols = [c for c in ["temp", "dewpoint", "pressure", "wind_speed"] if c in df.columns]
    df = df.dropna(subset=key_cols, how="all")

    # UTC → LST (Miami = EST = UTC-5, fixed, no DST)
    df = utc_to_lst(df, "Miami")

    # Resample to hourly means to standardise timestamps and remove within-hour duplicates
    df = df.resample("h").mean()

    # Print coverage summary
    n_years = df.index.year.nunique()
    yr_min, yr_max = df.index.year.min(), df.index.year.max()
    print(f"  [stage 2] ISD LST: {len(df):,} hourly rows, {yr_min}-{yr_max} ({n_years} years)")
    for col in key_cols:
        if col in df.columns:
            print(f"    {col:20s}: mean={df[col].mean():.2f}, NaN%={df[col].isna().mean()*100:.1f}%")

    df.index.name = "DATE"
    df.to_csv(csv_path)
    print(f"  [stage 2] Saved: {csv_path}")
    return df


def _stage3_merge_with_nsrdb(isd_lst: pd.DataFrame, out_dir: Path) -> Path:
    """Merge ISD LST DataFrame with NSRDB solar data.

    NSRDB provides GHI, DNI, DHI (W/m²) already in LST.
    NSRDB timestamps have Minute=30; these are floored to hour start.

    The merged CSV has index named DATE and columns:
      temp, dewpoint, pressure, wind_speed, GHI, DNI, DHI

    Returns path to the merged CSV.
    """
    csv_path = out_dir / "stage3_isd_solar_lst.csv"
    if csv_path.exists():
        print(f"  [stage 3] Cache hit: {csv_path}")
        return csv_path

    print(f"  [stage 3] Loading NSRDB: {NSRDB_CSV}")
    nsrdb = pd.read_csv(NSRDB_CSV)

    # Build LST DatetimeIndex from separate Year/Month/Day/Hour columns.
    # NSRDB uses hour 0-23 with Minute=30; we floor to the start of the hour.
    nsrdb["DATE"] = pd.to_datetime(
        nsrdb[["Year", "Month", "Day", "Hour"]].rename(
            columns={"Year": "year", "Month": "month", "Day": "day", "Hour": "hour"}
        )
    )
    nsrdb = nsrdb.set_index("DATE").sort_index()

    # Extract solar irradiance columns
    solar_cols = ["GHI", "DNI", "DHI"]
    missing_solar = [c for c in solar_cols if c not in nsrdb.columns]
    assert not missing_solar, f"NSRDB missing columns: {missing_solar}"
    solar = nsrdb[solar_cols].copy()

    # Clip to non-negative (NSRDB sometimes has −9999 fill values)
    solar = solar.clip(lower=0)

    # Resample to hourly mean (in case of duplicates)
    solar = solar.resample("h").mean()

    # Merge: ISD meteorology (left) + NSRDB solar (right) on common hourly index
    merged = isd_lst.join(solar, how="inner")
    merged.index.name = "DATE"

    # Forward-fill short gaps (≤ 3 h) in all columns
    merged = merged.ffill(limit=3).bfill(limit=3)

    # Drop rows still missing temp or solar
    merged = merged.dropna(subset=["temp", "GHI"], how="any")

    yr_min = merged.index.year.min()
    yr_max = merged.index.year.max()
    print(f"  [stage 3] Merged: {len(merged):,} hourly rows, {yr_min}-{yr_max}")
    for col in merged.columns:
        print(f"    {col:20s}: mean={merged[col].mean():.3f}, NaN%={merged[col].isna().mean()*100:.1f}%")

    merged.to_csv(csv_path)
    print(f"  [stage 3] Saved: {csv_path}")
    return csv_path


def _stage4_run_rsvar(merged_csv: Path, out_dir: Path) -> Path:
    """Run RS-VAR training + forecast (2025-2100) on the merged CSV.

    Returns path to the saved Excel file (RSVAR_Output sheet).
    """
    from samy_wgen.pipeline.run_rsvar import run_and_save_rsvar

    excel_path = out_dir / "stage4_rsvar_miami.xlsx"
    if excel_path.exists():
        print(f"  [stage 4] Cache hit: {excel_path}")
        return excel_path

    print(f"  [stage 4] Training RS-VAR on: {merged_csv}")
    run_and_save_rsvar(
        isd_path=merged_csv,
        lat=MIAMI_LAT,
        output_path=excel_path,
        city="Miami",
    )
    # run_and_save_rsvar calls pipeline.forecast_rsvar_only internally and saves to Excel.
    assert excel_path.exists(), f"RS-VAR Excel not created at {excel_path}"
    print(f"  [stage 4] Saved: {excel_path}")
    return excel_path


def _stage5_load_cmip6(out_dir: Path):
    """Load CMIP6 MPI-ESM1-2-LR SSP245 data and save daily CSV for inspection.

    Returns (daily_df, monthly_df, meta).
    """
    from samy_wgen.pipeline.run_climate_delta import load_cmip6_or_cordex6

    csv_path = out_dir / "stage5_cmip6_mpi_ssp245_daily.csv"

    print(f"  [stage 5] Loading CMIP6 MPI-SSP245 from: {CMIP6_DIR}")
    daily_df, monthly_df, meta = load_cmip6_or_cordex6(
        base_path=CMIP6_DIR,
        city="Miami",
        ssp=MIAMI_SSP,
    )

    if not csv_path.exists():
        daily_df.to_csv(csv_path)
        print(f"  [stage 5] Saved daily: {csv_path}")
        if not monthly_df.empty:
            monthly_csv = out_dir / "stage5_cmip6_mpi_ssp245_monthly.csv"
            monthly_df.to_csv(monthly_csv)
            print(f"  [stage 5] Saved monthly: {monthly_csv}")

    print(f"  [stage 5] CMIP6 daily: {len(daily_df):,} rows, "
          f"{daily_df.index.min().date()} to {daily_df.index.max().date()}")
    print(f"  [stage 5] Columns: {list(daily_df.columns)}")
    print(f"  [stage 5] Model detected: {meta.get('model_name', 'unknown')}")

    return daily_df, monthly_df, meta


def _stage6_compute_baseline(daily_df: pd.DataFrame, out_dir: Path) -> dict:
    """Compute CORDEX baseline climatology (1991-2010 DOY means).

    Saves baseline summary CSV for inspection.
    Returns baseline dict.
    """
    from samy_wgen.pipeline.run_climate_delta import compute_cordex_baseline

    print(f"  [stage 6] Computing CORDEX baseline ({BASELINE_START}-{BASELINE_END})")
    baseline = compute_cordex_baseline(daily_df, BASELINE_START, BASELINE_END, mode="daily")

    # Save summary: one row per (variable, DOY) with the climatology value
    rows = []
    for var, info in baseline.items():
        if "doy" in info:
            for doy, val in info["doy"].items():
                rows.append({"variable": var, "doy": doy, "baseline_mean": val,
                             "overall_mean": info["overall"], "transform": info["transform"]})
    if rows:
        summary_df = pd.DataFrame(rows)
        csv_path = out_dir / "stage6_cordex_baseline_doy.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"  [stage 6] Saved baseline DOY climatology: {csv_path}")

    print(f"  [stage 6] Baseline variables: {list(baseline.keys())}")
    for var, info in baseline.items():
        print(f"    {var:15s}: overall={info['overall']:.3f}, transform={info['transform']}")

    return baseline


def _stage7_apply_delta(rsvar_excel: Path, daily_df: pd.DataFrame,
                         baseline: dict, out_dir: Path) -> pd.DataFrame:
    """Apply MPI-SSP245 climate delta to RS-VAR output.

    Returns output DataFrame indexed by datetime.
    """
    from samy_wgen.pipeline.run_climate_delta import (
        load_rsvar_output, apply_climate_delta,
    )

    csv_path = out_dir / "stage7_miami_mpi_ssp245_hourly.csv"
    if csv_path.exists():
        print(f"  [stage 7] Cache hit: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["datetime"]).set_index("datetime")
        return df

    print(f"  [stage 7] Loading RS-VAR output from: {rsvar_excel}")
    rsvar_df, metadata = load_rsvar_output(rsvar_excel)
    print(f"  [stage 7] RS-VAR rows: {len(rsvar_df):,}, cols: {list(rsvar_df.columns)}")

    print(f"  [stage 7] Applying climate delta: {MIAMI_GCM} {MIAMI_SSP}")
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

    # Save delta statistics
    delta_rows = []
    for var, vals in delta_stats.items():
        if vals:
            arr = np.array(vals)
            delta_rows.append({
                "variable": var,
                "n_days": len(arr),
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p5": float(np.percentile(arr, 5)),
                "p95": float(np.percentile(arr, 95)),
            })
    if delta_rows:
        pd.DataFrame(delta_rows).to_csv(
            out_dir / "stage7_delta_stats.csv", index=False
        )

    # Save hourly output
    output_df.index.name = "datetime"
    output_df.to_csv(csv_path)
    print(f"  [stage 7] Saved hourly output ({len(output_df):,} rows): {csv_path}")

    return output_df


def _stage8_write_epw_samples(output_df: pd.DataFrame, out_dir: Path,
                               sample_years=(2030, 2050, 2080)) -> list:
    """Write EPW files for a sample set of years.

    Returns list of written EPW paths.
    """
    from samy_wgen.pipeline.gen_samy import _write_epw_for_year

    epw_dir = out_dir / "EPW"
    epw_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for year in sample_years:
        year_data = output_df.loc[str(year)] if str(year) in output_df.index.astype(str) else None
        # Check using year filter
        mask = output_df.index.year == year
        if mask.sum() == 0:
            print(f"  [stage 8] WARNING: no data for year {year}, skipping EPW")
            continue

        epw_path = epw_dir / f"Miami_MPI_SSP245_{year}.epw"
        _write_epw_for_year(
            df=output_df,
            year=year,
            out_path=epw_path,
            location="Miami-FL-USA",
            lat=MIAMI_LAT,
            lon=MIAMI_LON,
            tz=MIAMI_TZ,
            elevation_m=MIAMI_ELEV_M,
        )
        assert epw_path.exists(), f"EPW not written: {epw_path}"
        lines = epw_path.read_text().splitlines()
        data_rows = len(lines) - 8  # 8 header lines
        print(f"  [stage 8] {epw_path.name}: {data_rows} data rows")
        written.append(epw_path)

    return written


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_temp_warming(output_df: pd.DataFrame) -> None:
    """Print temperature trend checks: 2030s vs 2080s warming should be visible."""
    for window_label, years in [("2026-2040", range(2026, 2041)),
                                 ("2081-2100", range(2081, 2101))]:
        mask = output_df.index.year.isin(years)
        if mask.sum() == 0:
            continue
        mean_t = output_df.loc[mask, "temp"].mean()
        print(f"  temp mean {window_label}: {mean_t:.2f}°C")


def _check_physical_plausibility(output_df: pd.DataFrame) -> None:
    """Assert basic physical plausibility of the output."""
    temp = output_df["temp"]
    assert temp.min() > -30, f"Unrealistically cold: min={temp.min():.1f}°C"
    assert temp.max() < 60, f"Unrealistically hot: max={temp.max():.1f}°C"

    if "relative_humidity" in output_df.columns:
        rh = output_df["relative_humidity"]
        assert (rh >= 0).all(), "Negative RH"
        assert (rh <= 100).all(), f"RH exceeds 100%: max={rh.max():.1f}%"

    if "GHI" in output_df.columns:
        ghi = output_df["GHI"]
        assert (ghi >= 0).all(), "Negative GHI"

    if "wind_speed" in output_df.columns:
        ws = output_df["wind_speed"]
        assert (ws >= 0).all(), "Negative wind speed"


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_miami_cmip6_ssp245_full_pipeline():
    """Full pipeline POC: Miami MPI-ESM1-2-LR SSP245.

    This test is self-contained: each stage reads its inputs from files saved
    by the previous stage, so individual stages can be re-run after a cache hit.
    Expected runtime: 5-15 minutes (dominated by RS-VAR training + forecast).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # ---------- stage 1: parse ISD ----------
    assert ISD_STATION_DIR.exists(), f"ISD station dir not found: {ISD_STATION_DIR}"
    isd_csv = _stage1_parse_isd(OUTPUT_DIR)
    assert isd_csv.exists()

    # ---------- stage 2: remap ISD + UTC→LST ----------
    isd_lst = _stage2_remap_to_lst(isd_csv, OUTPUT_DIR)
    assert not isd_lst.empty, "ISD LST DataFrame is empty"
    assert "temp" in isd_lst.columns, f"'temp' not in ISD columns: {list(isd_lst.columns)}"
    assert isd_lst.index.tzinfo is None, "Index should be tz-naive (LST)"

    # ---------- stage 3: merge with NSRDB solar ----------
    assert NSRDB_CSV.exists(), f"NSRDB CSV not found: {NSRDB_CSV}"
    merged_csv = _stage3_merge_with_nsrdb(isd_lst, OUTPUT_DIR)
    assert merged_csv.exists()

    # Quick sanity: merged CSV has expected columns
    sample = pd.read_csv(merged_csv, nrows=5)
    for col in ["temp", "GHI", "DNI", "DHI"]:
        assert col in sample.columns, f"Merged CSV missing column '{col}'"

    # ---------- stage 4: RS-VAR training + forecast ----------
    rsvar_excel = _stage4_run_rsvar(merged_csv, OUTPUT_DIR)
    assert rsvar_excel.exists()

    # ---------- stage 5: load CMIP6 MPI-SSP245 ----------
    assert CMIP6_DIR.exists(), f"CMIP6 dir not found: {CMIP6_DIR}"
    daily_df, monthly_df, meta = _stage5_load_cmip6(OUTPUT_DIR)
    assert not daily_df.empty, "CMIP6 daily DataFrame is empty"
    assert "tas_C" in daily_df.columns, (
        f"CMIP6 missing 'tas_C' after unit conversion. Columns: {list(daily_df.columns)}"
    )
    assert meta["model_name"] == MIAMI_GCM, (
        f"Expected {MIAMI_GCM}, got {meta['model_name']}"
    )

    # ---------- stage 6: compute CORDEX baseline ----------
    baseline = _stage6_compute_baseline(daily_df, OUTPUT_DIR)
    assert "tas_C" in baseline, "Baseline missing 'tas_C'"
    assert "ps_hPa" in baseline, "Baseline missing 'ps_hPa'"

    # ---------- stage 7: apply climate delta ----------
    output_df = _stage7_apply_delta(rsvar_excel, daily_df, baseline, OUTPUT_DIR)
    assert not output_df.empty, "Climate-delta output is empty"
    assert output_df.index.year.max() >= 2095, "Forecast doesn't reach 2095"

    # Physical plausibility check
    _check_physical_plausibility(output_df)

    # Print warming signal for visual inspection
    print("\n  === Temperature warming signal ===")
    _check_temp_warming(output_df)

    # ---------- stage 8: write EPW files ----------
    epw_paths = _stage8_write_epw_samples(output_df, OUTPUT_DIR, sample_years=(2030, 2050, 2080))
    assert len(epw_paths) == 3, f"Expected 3 EPW files, got {len(epw_paths)}"
    for p in epw_paths:
        lines = p.read_text().splitlines()
        assert len(lines) >= 8 + 8760, (
            f"{p.name}: expected ≥8768 lines, got {len(lines)}"
        )

    print(f"\n  All stages completed. Outputs in: {OUTPUT_DIR}")
