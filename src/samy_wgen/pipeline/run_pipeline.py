#!/usr/bin/env python3
"""
samy-run: One-click SAMY weather generation pipeline.

Chains:
  ISD parse → LST remap → NSRDB merge → RS-VAR → climate delta → EPW
                                                 ↳ (ensemble) stochastic members

Modes:
  deterministic  RS-VAR forecast + CMIP6 delta + EPW files  [default]
  ensemble       Stochastic ensemble members (parquet + summary Excel)

Outputs land in --out-dir/<city>/ with numbered intermediate files:
  1_isd_lst.csv            LST-converted ISD (cached)
  2_isd_solar_lst.csv      + NSRDB solar merge (cached)
  3_rsvar.xlsx             RS-VAR training output (cached, deterministic only)
  4_delta_corrected.csv    climate-delta-corrected hourly series (cached, deterministic only)
  EPW/<city>_<ssp>_<year>.epw  one file per requested year (deterministic only)
  member_NNN.parquet       ensemble members (ensemble only)
  ensemble_summary.xlsx    p05/median/p95 (ensemble only)

Intermediate files are skipped on re-runs unless --overwrite is passed.

USAGE:
  samy-run --city Miami
  samy-run --city Miami Los_Angeles
  samy-run --city Miami --mode ensemble --members 50
  samy-run --city Miami --start 2025-01-01 --end 2050-12-31 --ssp ssp585
  samy-run --city Miami --overwrite
"""

import argparse
import math
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd
from samy_wgen.utils.city_utils import resolve_city, resolve_city_codes


# ==========================================================================
# DEFAULTS
# ==========================================================================

BASELINE_START = 1991
BASELINE_END   = 2010
FORECAST_START = "2025-01-01"
FORECAST_END   = "2100-12-31 23:00"
DEFAULT_EPW_YEARS = "2030,2050,2080"
DEFAULT_SSP    = "ssp245"
DEFAULT_DOWNLOAD_DATA = ["isd", "nsrdb", "cmip6"]
DEFAULT_CMIP6_MODEL = "mpi_esm1_2_lr"
DEFAULT_ISD_RADIUS_KM = 50.0
DEFAULT_ISD_YEAR_RANGE = (1970, 2024)
DEFAULT_NSRDB_YEAR_RANGE = (1998, 2024)

SSP_TO_CMIP6_EXPERIMENT = {
    "ssp126": "ssp1_2_6",
    "ssp245": "ssp2_4_5",
    "ssp370": "ssp3_7_0",
    "ssp585": "ssp5_8_5",
}

FUTURE_DDY_PERIODS = {
    2030: (2021, 2040),
    2050: (2041, 2060),
    2080: (2071, 2090),
}


def _print_title(title: str) -> None:
    print(f"\n{'-' * 12} {title} {'-' * 12}")


# ==========================================================================
# CITY METADATA
# ==========================================================================

def _load_city_meta(city: str) -> dict:
    """Return lat, lon, tz_offset for an ASCII city code."""
    from samy_wgen.utils.time_utils import get_utc_offset

    record = resolve_city(city)
    lat = record.latitude
    lon = record.longitude
    tz = get_utc_offset(city=record.city_ascii, latitude=lat, longitude=lon)
    return {
        "lat": lat,
        "lon": lon,
        "tz":  tz,
        "country": record.country_iso2,
    }


def _station_elevation(station_dir: Path, history_path: Path, override: float | None) -> float:
    """Resolve selected ISD station elevation, with an explicit CLI override."""
    if override is not None:
        return override
    station_id = station_dir.name.split("_", 1)[0]
    if not history_path.exists():
        raise SystemExit(f"ERROR: {history_path} is unavailable; pass --elevation")
    try:
        usaf, wban = station_id.split("-", 1)
        history = pd.read_csv(history_path, dtype={"USAF": str, "WBAN": str})
        match = history.loc[(history["USAF"] == usaf) & (history["WBAN"] == wban)]
        elevation = float(match.iloc[0]["ELEV(M)"])
    except (ValueError, IndexError, KeyError) as exc:
        raise SystemExit(f"ERROR: no elevation for ISD station {station_id}; pass --elevation") from exc
    if not math.isfinite(elevation) or not -1000 <= elevation < 9999.9:
        raise SystemExit(f"ERROR: invalid elevation for ISD station {station_id}; pass --elevation")
    return elevation


def _find_isd_dir(city: str, base: Path) -> Path:
    """Return the first station sub-directory under base/ISD/<city>/."""
    city_dir = base / "ISD" / city
    if not city_dir.exists():
        raise SystemExit(f"ERROR: ISD directory not found: {city_dir}")
    stations = [d for d in city_dir.iterdir() if d.is_dir()]
    if not stations:
        raise SystemExit(f"ERROR: No station sub-directories found in {city_dir}")
    if len(stations) > 1:
        print(f"  [info] Multiple ISD stations found, using: {stations[0].name}")
    return stations[0]


def _find_nsrdb_csv(city: str, base: Path) -> Path:
    """Auto-detect NSRDB CSV under base/NSRDB/ matching nsrdb_<city_lower>*.csv."""
    nsrdb_dir = base / "NSRDB"
    if not nsrdb_dir.exists():
        raise SystemExit(f"ERROR: NSRDB directory not found: {nsrdb_dir}")
    city_slug = city.lower().replace(" ", "_")
    pattern = f"nsrdb_{city_slug}*.csv"
    matches = list(nsrdb_dir.glob(pattern))
    if not matches:
        raise SystemExit(
            f"ERROR: No NSRDB CSV found for '{city}' in {nsrdb_dir}.\n"
            f"  Expected pattern: {pattern}"
        )
    if len(matches) > 1:
        print(f"  [info] Multiple NSRDB files found, using: {matches[0].name}")
    return matches[0]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-click SAMY weather generation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--city", required=True, nargs="+",
                        help="ASCII city code(s) matching cities_config.csv, e.g. 'Miami' 'Los_Angeles'")

    parser.add_argument("--data-dir", default="data",
                        help="Root data directory (ISD/, NSRDB/, CMIP6/ live here)")
    parser.add_argument("--isd-dir", default=None,
                        help="ISD station directory (overrides auto-detect under data-dir/ISD/<city>/)")
    parser.add_argument("--nsrdb", default=None,
                        help="NSRDB solar CSV path (overrides auto-detect under data-dir/NSRDB/)")
    parser.add_argument("--cmip6-dir", default=None,
                        help="CMIP6 root directory (overrides data-dir/CMIP6)")
    parser.add_argument("--gcm", default=None,
                        help="Optional CMIP6 GCM selector (e.g. MPI-ESM1-2-LR, CNRM-ESM2-1, MRI-ESM2-0). Omit to retain automatic selection.")
    parser.add_argument("--rsvar-input", default=None,
                        help="Optional existing 3_rsvar.xlsx/CSV to reuse instead of running Stage 3; single-city mode only.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: output/<city>)")
    parser.add_argument("--elevation", type=float, default=None,
                        help="Station elevation in metres; defaults to selected station in ISD history")

    parser.add_argument("--baseline-start", type=int, default=BASELINE_START,
                        help="Baseline period start year")
    parser.add_argument("--baseline-end", type=int, default=BASELINE_END,
                        help="Baseline period end year")

    parser.add_argument("--start", default=FORECAST_START, help="Forecast start date")
    parser.add_argument("--end",   default=FORECAST_END,   help="Forecast end date")

    parser.add_argument("--ssp", default=DEFAULT_SSP,
                        help="CMIP6 SSP scenario (e.g. ssp245, ssp585)")

    parser.add_argument("--mode", default="deterministic",
                        choices=["deterministic", "ensemble"],
                        help="Output mode")
    parser.add_argument("--epw-years", default=DEFAULT_EPW_YEARS,
                        help="Comma-separated years for EPW output (deterministic mode)")

    parser.add_argument("--members", type=int, default=50,
                        help="Number of ensemble members (ensemble mode)")
    parser.add_argument("--regime-mode", default="ensemble",
                        choices=["soft", "hard", "ensemble"],
                        help="Regime sampling mode (ensemble mode)")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")

    parser.add_argument("--download-data", nargs="+",
                        choices=["isd", "nsrdb", "cmip6"],
                        default=DEFAULT_DOWNLOAD_DATA,
                        help="Auto-download these datasets when required inputs are missing")
    parser.add_argument("--skip-download", action="store_true",
                        help="Disable automatic download of missing input data")
    parser.add_argument("--isd-station", nargs="+", default=None,
                        help="Override automatic ISD station selection. Single-city mode accepts USAF-WBAN; multi-city mode accepts City=USAF-WBAN mappings")
    parser.add_argument("--isd-radius", type=float, default=DEFAULT_ISD_RADIUS_KM,
                        help="Radius in km for automatic ISD station selection")
    parser.add_argument("--download-years", nargs=2, type=int, metavar=("START", "END"), default=None,
                        help="Year range for automatic ISD/NSRDB downloads")
    parser.add_argument("--cmip6-models", nargs="+", default=[DEFAULT_CMIP6_MODEL],
                        help="CMIP6 models used for automatic downloads")
    parser.add_argument("--cmip6-experiments", nargs="+", default=None,
                        help="CMIP6 CDS experiments for automatic downloads (default: historical + selected SSP)")
    parser.add_argument("--cmip6-years", nargs=2, type=int, metavar=("START", "END"), default=None,
                        help="Year range for automatic CMIP6 downloads")

    parser.add_argument("--overwrite", action="store_true",
                        help="Ignore cached intermediate files and rerun all stages")
    return parser


def _cmip6_download_experiments(ssp: str, requested: Iterable[str] | None = None) -> list[str]:
    if requested:
        return list(requested)
    cds_experiment = SSP_TO_CMIP6_EXPERIMENT.get(ssp)
    if cds_experiment is None:
        raise SystemExit(
            f"ERROR: unsupported SSP '{ssp}' for auto-download. "
            f"Expected one of: {sorted(SSP_TO_CMIP6_EXPERIMENT)}"
        )
    return ["historical", cds_experiment]


def _has_required_cmip6_data(city: str, cmip6_dir: Path, ssp: str, models: Iterable[str]) -> bool:
    city_dir = cmip6_dir / city
    if not city_dir.exists():
        return False
    required = []
    for model in models:
        required.append(city_dir / f"{model}_historical.csv")
        required.append(city_dir / f"{model}_{ssp}.csv")
    return all(path.exists() for path in required)


def _normalize_city_args(requested: list[str]) -> list[str]:
    return resolve_city_codes(requested)


def _parse_isd_station_args(values: list[str] | None, cities: list[str]) -> dict[str, str] | None:
    from samy_wgen.pipeline.download_data import _parse_isd_station_args as _parse

    return _parse(values, cities)


def _ensure_pipeline_inputs(args, data_dir: Path, city: str, cmip6_dir: Path | None = None) -> tuple[Path, Path, Path]:
    from samy_wgen.pipeline.download_data import (
        _city_meta,
        download_isd,
        download_nsrdb,
        select_best_isd_station,
    )

    if args.isd_dir:
        isd_station_dir = Path(args.isd_dir)
    else:
        try:
            isd_station_dir = _find_isd_dir(city, data_dir)
        except SystemExit:
            if args.skip_download or "isd" not in set(args.download_data):
                raise
            _print_title("Download ISD")
            yr_start, yr_end = args.download_years if args.download_years else DEFAULT_ISD_YEAR_RANGE
            station_map = _parse_isd_station_args(args.isd_station, args.city)
            station_id = (station_map or {}).get(city) or select_best_isd_station(
                city=city,
                radius_km=args.isd_radius,
                cache_dir=str(data_dir / "ISD"),
            )
            download_isd(
                station_id=station_id,
                city=city,
                years=list(range(yr_start, yr_end + 1)),
                output_dir=str(data_dir / "ISD"),
            )
            isd_station_dir = _find_isd_dir(city, data_dir)

    if args.nsrdb:
        nsrdb_csv = Path(args.nsrdb)
    else:
        try:
            nsrdb_csv = _find_nsrdb_csv(city, data_dir)
        except SystemExit:
            if args.skip_download or "nsrdb" not in set(args.download_data):
                raise
            _print_title("Download NSRDB")
            meta = _city_meta(city)
            yr_start, yr_end = args.download_years if args.download_years else DEFAULT_NSRDB_YEAR_RANGE
            download_nsrdb(
                city=city,
                lat=meta["lat"],
                lon=meta["lon"],
                years=list(range(yr_start, yr_end + 1)),
                output_dir=str(data_dir / "NSRDB"),
            )
            nsrdb_csv = _find_nsrdb_csv(city, data_dir)

    cmip6_dir = cmip6_dir or (Path(args.cmip6_dir) if args.cmip6_dir else data_dir / "CMIP6")
    if not _has_required_cmip6_data(city, cmip6_dir, args.ssp, args.cmip6_models):
        missing_city_dir = cmip6_dir / city
        raise SystemExit(
            f"ERROR: Required CMIP6 files not found under {missing_city_dir}. "
            "Provide --cmip6-dir, pre-download the files, or allow auto-download."
        )

    return isd_station_dir, nsrdb_csv, cmip6_dir


def _ensure_batch_cmip6_inputs(args, data_dir: Path, cities: list[str]) -> Path:
    from samy_wgen.pipeline.download_data import download_cmip6_for_cities

    cmip6_dir = Path(args.cmip6_dir) if args.cmip6_dir else data_dir / "CMIP6"
    missing_cities = [
        city for city in cities
        if not _has_required_cmip6_data(city, cmip6_dir, args.ssp, args.cmip6_models)
    ]
    if not missing_cities:
        return cmip6_dir

    if args.skip_download or "cmip6" not in set(args.download_data):
        missing_city_dir = cmip6_dir / missing_cities[0]
        raise SystemExit(
            f"ERROR: Required CMIP6 files not found under {missing_city_dir}. "
            "Provide --cmip6-dir, pre-download the files, or allow auto-download."
        )

    _print_title("Download CMIP6")
    print(f"[batch] Downloading shared CMIP6 inputs for: {', '.join(missing_cities)}")
    download_cmip6_for_cities(
        cities=missing_cities,
        models=args.cmip6_models,
        experiments=_cmip6_download_experiments(args.ssp, args.cmip6_experiments),
        year_range=tuple(args.cmip6_years) if args.cmip6_years else None,
        output_dir=str(cmip6_dir),
    )
    return cmip6_dir


# ==========================================================================
# PIPELINE STAGES  (0-indexed: stage 0 = ISD parse, no output file)
# ==========================================================================

def _stage0_parse_isd(isd_station_dir: Path) -> pd.DataFrame:
    """Parse ISD .gz files → raw UTC DataFrame (not written to disk)."""
    from samy_wgen.utils.parse_ISD import process_station_directory

    _print_title("Stage 0 Parse ISD")
    print(f"[stage 0] Parsing ISD .gz from: {isd_station_dir.name}")
    with tempfile.TemporaryDirectory() as tmp:
        result_path = process_station_directory(str(isd_station_dir), tmp)
        assert result_path is not None, "process_station_directory returned None"
        raw = pd.read_csv(result_path, parse_dates=["datestamp"], low_memory=False)
    print(f"  parsed {len(raw):,} records")
    return raw


def _stage1_remap_to_lst(isd_raw: pd.DataFrame, city: str,
                          lat: float, lon: float,
                          out_dir: Path, overwrite: bool) -> pd.DataFrame:
    """Remap ISD columns + UTC→LST + resample to hourly → 1_isd_lst.csv."""
    from samy_wgen.utils.parse_ISD import remap_isd_columns, resample_isd_hourly
    from samy_wgen.utils.time_utils import utc_to_lst

    csv_path = out_dir / "1_isd_lst.csv"
    if csv_path.exists() and not overwrite:
        _print_title("Stage 1 Remap To LST")
        print(f"[stage 1] Cache hit: {csv_path.name}")
        return pd.read_csv(csv_path, parse_dates=["DATE"]).set_index("DATE")

    _print_title("Stage 1 Remap To LST")
    print(f"[stage 1] Remapping ISD columns + UTC→LST")
    raw = isd_raw.dropna(subset=["datestamp"]).set_index("datestamp").sort_index()
    df = remap_isd_columns(raw)

    key_cols = [c for c in ["temp", "dewpoint", "pressure", "wind_speed"] if c in df.columns]
    df = df.dropna(subset=key_cols, how="all")
    df = utc_to_lst(df, city=city, latitude=lat, longitude=lon)
    df = resample_isd_hourly(df)

    yr_min, yr_max = df.index.year.min(), df.index.year.max()
    print(f"  {len(df):,} hourly rows, {yr_min}–{yr_max}")
    df.index.name = "DATE"
    df.to_csv(csv_path)
    print(f"  saved: {csv_path.name}")
    return df


def _stage2_merge_nsrdb(isd_lst: pd.DataFrame, nsrdb_csv: Path,
                         out_dir: Path, overwrite: bool) -> Path:
    """Merge ISD LST + NSRDB solar (GHI/DNI/DHI) → 2_isd_solar_lst.csv."""
    csv_path = out_dir / "2_isd_solar_lst.csv"
    if csv_path.exists() and not overwrite:
        _print_title("Stage 2 Merge NSRDB")
        print(f"[stage 2] Cache hit: {csv_path.name}")
        return csv_path

    _print_title("Stage 2 Merge NSRDB")
    print(f"[stage 2] Merging NSRDB solar: {nsrdb_csv.name}")
    nsrdb = pd.read_csv(nsrdb_csv)
    if "DATE" in nsrdb:
        nsrdb["DATE"] = pd.to_datetime(nsrdb["DATE"], errors="raise")
    else:
        nsrdb["DATE"] = pd.to_datetime(
            nsrdb[["Year", "Month", "Day", "Hour"]].rename(
                columns={"Year": "year", "Month": "month", "Day": "day", "Hour": "hour"}
            )
        )
    nsrdb = nsrdb.set_index("DATE").sort_index()

    solar_cols = [c for c in ["GHI", "DNI", "DHI"] if c in nsrdb.columns]
    if not solar_cols:
        raise ValueError(f"NSRDB CSV missing solar columns (GHI/DNI/DHI). Found: {nsrdb.columns.tolist()}")

    solar = nsrdb[solar_cols].clip(lower=0).resample("h").mean()
    merged = isd_lst.join(solar, how="inner").ffill(limit=3).bfill(limit=3)
    merged = merged.dropna(subset=["temp"] + solar_cols[:1])

    yr_min, yr_max = merged.index.year.min(), merged.index.year.max()
    print(f"  {len(merged):,} hourly rows, {yr_min}–{yr_max}")
    merged.index.name = "DATE"
    merged.to_csv(csv_path)
    print(f"  saved: {csv_path.name}")
    return csv_path


def _stage3_rsvar(merged_csv: Path, lat: float, longitude: float, city: str,
                   baseline_start: int, baseline_end: int,
                   forecast_start: str, forecast_end: str,
                   out_dir: Path, overwrite: bool) -> Path:
    """RS-VAR training + forecast → 3_rsvar.xlsx (deterministic mode)."""
    from samy_wgen.pipeline.run_rsvar import run_and_save_rsvar

    xlsx_path = out_dir / "3_rsvar.xlsx"
    calibration_path = xlsx_path.with_name(xlsx_path.stem + "_solar_calibration.csv")
    if xlsx_path.exists() and calibration_path.exists() and not overwrite:
        _print_title("Stage 3 Run RS-VAR")
        print(f"[stage 3] Cache hit: {xlsx_path.name}")
        return xlsx_path

    _print_title("Stage 3 Run RS-VAR")
    print(f"[stage 3] Training RS-VAR ({city}, lat={lat:.2f}°N)")
    # Temporarily patch the module-level constants so run_and_save_rsvar uses our periods
    import samy_wgen.pipeline.run_rsvar as _rsvar_mod
    _orig_start = _rsvar_mod.BASELINE_START
    _orig_end   = _rsvar_mod.BASELINE_END
    _orig_fs    = _rsvar_mod.FORECAST_START
    _orig_fe    = _rsvar_mod.FORECAST_END
    try:
        _rsvar_mod.BASELINE_START = baseline_start
        _rsvar_mod.BASELINE_END   = baseline_end
        _rsvar_mod.FORECAST_START = forecast_start
        _rsvar_mod.FORECAST_END   = forecast_end
        run_and_save_rsvar(isd_path=merged_csv, lat=lat, output_path=xlsx_path,
                           city=city, verbose=False, longitude=longitude)
    finally:
        _rsvar_mod.BASELINE_START = _orig_start
        _rsvar_mod.BASELINE_END   = _orig_end
        _rsvar_mod.FORECAST_START = _orig_fs
        _rsvar_mod.FORECAST_END   = _orig_fe

    print(f"  saved: {xlsx_path.name}")
    return xlsx_path


def _stage4_climate_delta(rsvar_xlsx: Path, cmip6_dir: Path,
                           city: str, ssp: str, lat: float, longitude: float,
                           baseline_start: int, baseline_end: int,
                           out_dir: Path, overwrite: bool,
                           gcm: str | None = None) -> pd.DataFrame:
    """Apply per-variable daily/monthly CMIP6 deltas → 4_delta_corrected.csv."""
    from samy_wgen.pipeline.run_climate_delta import (
        load_rsvar_output,
        load_cmip6_or_cordex6,
        compute_mixed_resolution_baseline,
        apply_mixed_resolution_climate_delta,
    )

    csv_path = out_dir / "4_delta_corrected.csv"
    if csv_path.exists() and (out_dir / "4_solar_monthly_energy.csv").exists() and not overwrite:
        _print_title("Stage 4 Apply Climate Delta")
        print(f"[stage 4] Cache hit: {csv_path.name}")
        return pd.read_csv(csv_path, parse_dates=["datetime"]).set_index("datetime")

    _print_title("Stage 4 Apply Climate Delta")
    print(f"[stage 4] Applying CMIP6 climate delta ({ssp})")
    daily_df, monthly_df, meta = load_cmip6_or_cordex6(
        base_path=cmip6_dir, city=city, ssp=ssp,
        scenario_id=f"{gcm}_{ssp}" if gcm else None,
        gcm=gcm, verbose=False
    )
    gcm = meta.get("model_name", "unknown")
    print(f"  GCM: {gcm}, {len(daily_df):,} daily rows, {len(monthly_df):,} monthly rows")

    baseline = compute_mixed_resolution_baseline(
        daily_df, monthly_df, baseline_start, baseline_end, var_resolution_config={}
    )
    rsvar_df, _ = load_rsvar_output(rsvar_xlsx)

    output_df, _stats = apply_mixed_resolution_climate_delta(
        rsvar_df,
        daily_df,
        monthly_df,
        baseline=baseline,
        gcm=gcm,
        rcm=None,
        lat=lat,
        longitude=longitude,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        verbose=False,
    )
    output_df.index.name = "datetime"
    output_df.to_csv(csv_path)
    monthly_energy = output_df.attrs.get("solar_monthly_energy")
    if monthly_energy is not None:
        monthly_energy.to_csv(out_dir / "4_solar_monthly_energy.csv", index=False)
    print(f"  saved: {csv_path.name} ({len(output_df):,} rows)")
    return output_df


def _load_hourly_csv(csv_path: Path, datetime_col: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, parse_dates=[datetime_col]).set_index(datetime_col)


def _output_historical_epw(merged_csv: Path, city: str, meta: dict, out_dir: Path, overwrite: bool) -> Path:
    """Write the actual historical 2000 EPW from the merged ISD+NSRDB sequence."""
    from samy_wgen.pipeline.gen_samy import _write_epw_for_year

    historical_df = _load_hourly_csv(merged_csv, "DATE")
    epw_dir = out_dir / "EPW"
    epw_dir.mkdir(parents=True, exist_ok=True)
    epw_path = epw_dir / f"{city}_historical_2000.epw"

    if epw_path.exists() and not overwrite:
        print(f"  [EPW] Cache hit: {epw_path.name}")
        return epw_path

    if (historical_df.index.year == 2000).sum() == 0:
        raise SystemExit(
            f"ERROR: historical merged sequence for {city} does not contain year 2000; cannot write {epw_path.name}"
        )

    _write_epw_for_year(
        df=historical_df,
        year=2000,
        out_path=epw_path,
        location=f"{city}-{meta.get('country', 'XX')}",
        lat=meta["lat"],
        lon=meta["lon"],
        tz=meta["tz"],
        elevation_m=meta.get("elevation_m", 0.0),
        pressure_reference="sea_level",
    )
    print(f"  [EPW] {epw_path.name}")
    return epw_path


def _output_epw(output_df: pd.DataFrame, epw_years: list[int],
                city: str, ssp: str, meta: dict,
                out_dir: Path, overwrite: bool) -> list[Path]:
    """Write one EPW per requested year → EPW/<city>_<ssp>_<year>.epw."""
    from samy_wgen.pipeline.gen_samy import _write_epw_for_year

    _print_title("Write EPW")
    epw_dir = out_dir / "EPW"
    epw_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for year in epw_years:
        epw_path = epw_dir / f"{city}_{ssp}_{year}.epw"
        if epw_path.exists() and not overwrite:
            print(f"  [EPW] Cache hit: {epw_path.name}")
            written.append(epw_path)
            continue
        if (output_df.index.year == year).sum() == 0:
            print(f"  [EPW] WARNING: no data for {year}, skipping")
            continue
        _write_epw_for_year(
            df=output_df,
            year=year,
            out_path=epw_path,
            location=f"{city}-{meta.get('country', 'XX')}",
            lat=meta["lat"],
            lon=meta["lon"],
            tz=meta["tz"],
            elevation_m=meta.get("elevation_m", 0.0),
            pressure_reference="sea_level",
        )
        print(f"  [EPW] {epw_path.name}")
        written.append(epw_path)
    return written


def _output_ddy_files(merged_csv: Path, output_df: pd.DataFrame,
                      city: str, ssp: str, meta: dict,
                      baseline_start: int, baseline_end: int,
                      out_dir: Path, overwrite: bool) -> list[Path]:
    """Write baseline and future DDYs from their multi-year period slices."""
    from samy_wgen.utils.create_epw_ddy import write_ddy_from_dataframe

    historical_df = _load_hourly_csv(merged_csv, "DATE")
    ddy_dir = out_dir / "DDY"
    ddy_dir.mkdir(parents=True, exist_ok=True)

    period_specs = [("baseline", baseline_start, baseline_end, historical_df, f"{city} baseline {baseline_start}-{baseline_end}")]
    for rep_year, (start_year, end_year) in FUTURE_DDY_PERIODS.items():
        label = f"{city} {ssp} {rep_year}"
        period_specs.append((str(rep_year), start_year, end_year, output_df, label))

    written: list[Path] = []
    for key, start_year, end_year, df, label in period_specs:
        period_df = df[(df.index.year >= start_year) & (df.index.year <= end_year)].copy()
        if period_df.empty:
            print(f"  [DDY] WARNING: no data for {key} period {start_year}-{end_year}, skipping")
            continue

        if key == "baseline":
            ddy_path = ddy_dir / f"{city}_baseline_{start_year}_{end_year}.ddy"
        else:
            ddy_path = ddy_dir / f"{city}_{ssp}_{key}_{start_year}_{end_year}.ddy"

        if ddy_path.exists() and not overwrite:
            print(f"  [DDY] Cache hit: {ddy_path.name}")
            written.append(ddy_path)
            continue

        write_ddy_from_dataframe(
            period_df,
            ddy_path,
            city=city,
            country=meta.get("country", "XX"),
            lat=meta["lat"],
            lon=meta["lon"],
            tz=meta["tz"],
            elevation_m=meta.get("elevation_m", 0.0),
            label=label,
        )
        print(f"  [DDY] {ddy_path.name}")
        written.append(ddy_path)

    return written


def _stage3_ensemble(merged_csv: Path, lat: float,
                      n_members: int, regime_mode: str,
                      forecast_start: str, forecast_end: str,
                      baseline_start: int, baseline_end: int,
                      seed: int, out_dir: Path) -> Path:
    """Generate stochastic ensemble members (ensemble mode)."""
    from samy_wgen.pipeline.run_ensemble import run_and_save_ensemble

    _print_title("Stage 3 Run Ensemble")
    print(f"[stage 3] Generating {n_members} ensemble members")
    print(f"  regime_mode={regime_mode!r}")

    run_and_save_ensemble(
        isd_path=merged_csv,
        lat=lat,
        output_dir=out_dir / "ensemble",
        n_members=n_members,
        start_date=forecast_start,
        end_date=forecast_end,
        regime_mode=regime_mode,
        seed=seed,
        verbose=False,
    )
    return out_dir / "ensemble"


# ==========================================================================
# MAIN ENTRY
# ==========================================================================

def _city_out_dir(args, city: str) -> Path:
    if args.out_dir is None:
        return Path("output") / city
    base = Path(args.out_dir)
    if len(args.city) == 1:
        return base
    return base / city


def _run_city_pipeline(args, city: str, data_dir: Path, cmip6_dir: Path) -> None:
    out_dir = _city_out_dir(args, city)
    out_dir.mkdir(parents=True, exist_ok=True)

    isd_station_dir, nsrdb_csv, cmip6_dir = _ensure_pipeline_inputs(args, data_dir, city, cmip6_dir)

    meta = _load_city_meta(city)
    meta["elevation_m"] = _station_elevation(
        isd_station_dir, data_dir / "ISD" / "isd-history.csv", args.elevation)

    epw_years = [int(y.strip()) for y in args.epw_years.split(",")]
    ow = args.overwrite

    print("=" * 70)
    print(f"SAMY PIPELINE  city={city}  mode={args.mode}")
    print(f"  baseline: {args.baseline_start}–{args.baseline_end}")
    print(f"  forecast: {args.start} → {args.end}")
    print(f"  output:   {out_dir}")
    print("=" * 70)

    isd_lst_cache = out_dir / "1_isd_lst.csv"
    if isd_lst_cache.exists() and not ow:
        _print_title("Stage 0 Parse ISD")
        print("[stage 1] Cache hit: 1_isd_lst.csv  (skipping ISD parse)")
        isd_lst = pd.read_csv(isd_lst_cache, parse_dates=["DATE"]).set_index("DATE")
    else:
        isd_raw = _stage0_parse_isd(isd_station_dir)
        isd_lst = _stage1_remap_to_lst(
            isd_raw,
            city,
            meta["lat"],
            meta["lon"],
            out_dir,
            overwrite=ow,
        )

    merged_csv = _stage2_merge_nsrdb(isd_lst, nsrdb_csv, out_dir, overwrite=ow)

    if args.mode == "deterministic":
        if args.rsvar_input:
            if len(args.city) != 1:
                raise SystemExit("ERROR: --rsvar-input is supported only in single-city mode.")
            xlsx = Path(args.rsvar_input)
            if not xlsx.exists():
                raise SystemExit(f"ERROR: --rsvar-input not found: {xlsx}")
            print(f"[stage 3] Reusing explicit RS-VAR input: {xlsx}")
        else:
            xlsx = _stage3_rsvar(
                merged_csv=merged_csv,
                lat=meta["lat"],
                longitude=meta["lon"],
                city=city,
                baseline_start=args.baseline_start,
                baseline_end=args.baseline_end,
                forecast_start=args.start,
                forecast_end=args.end,
                out_dir=out_dir,
                overwrite=ow,
            )
        output_df = _stage4_climate_delta(
            rsvar_xlsx=xlsx,
            cmip6_dir=cmip6_dir,
            city=city,
            ssp=args.ssp,
            lat=meta["lat"],
            longitude=meta["lon"],
            baseline_start=args.baseline_start,
            baseline_end=args.baseline_end,
            out_dir=out_dir,
            overwrite=ow,
            gcm=args.gcm,
        )
        print(f"\n[output] Writing historical EPW for 2000")
        historical_epw = _output_historical_epw(
            merged_csv=merged_csv,
            city=city,
            meta=meta,
            out_dir=out_dir,
            overwrite=ow,
        )
        print(f"\n[output] Writing future EPW files for years: {epw_years}")
        future_epws = _output_epw(
            output_df=output_df,
            epw_years=epw_years,
            city=city,
            ssp=args.ssp,
            meta=meta,
            out_dir=out_dir,
            overwrite=ow,
        )
        print(f"\n[output] Writing baseline/future DDY files")
        ddys = _output_ddy_files(
            merged_csv=merged_csv,
            output_df=output_df,
            city=city,
            ssp=args.ssp,
            meta=meta,
            baseline_start=args.baseline_start,
            baseline_end=args.baseline_end,
            out_dir=out_dir,
            overwrite=ow,
        )
        print(f"\n{'='*70}")
        print(
            f"DONE — {1 + len(future_epws)} EPW file(s) in {out_dir / 'EPW'} "
            f"and {len(ddys)} DDY file(s) in {out_dir / 'DDY'}"
        )
    else:
        ens_dir = _stage3_ensemble(
            merged_csv=merged_csv,
            lat=meta["lat"],
            n_members=args.members,
            regime_mode=args.regime_mode,
            forecast_start=args.start,
            forecast_end=args.end,
            baseline_start=args.baseline_start,
            baseline_end=args.baseline_end,
            seed=args.seed,
            out_dir=out_dir,
        )
        print(f"\n{'='*70}")
        print(f"DONE — {args.members} ensemble members in {ens_dir}")


def main():
    parser = _build_parser()
    args = parser.parse_args()
    args.city = _normalize_city_args(args.city)
    data_dir = Path(args.data_dir)

    cmip6_dir = _ensure_batch_cmip6_inputs(args, data_dir, args.city)
    for city in args.city:
        _run_city_pipeline(args, city, data_dir, cmip6_dir)


if __name__ == "__main__":
    main()
