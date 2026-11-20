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

Intermediate files are reused when their inputs and parameters match.
Pass --overwrite to rebuild them.

USAGE:
  samy-run --city Miami
  samy-run --city Miami Los_Angeles
  samy-run --city Miami --mode ensemble --members 50
  samy-run --city Miami --start 2025-01-01 --end 2050-12-31 --ssp ssp585
  samy-run --city Miami --overwrite
"""

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd
from samy_wgen.utils.city_utils import resolve_city, resolve_city_codes
from samy_wgen.utils.file_lock import file_lock

from samy_wgen.pipeline.input_checks import (
    CMIP6_SCENARIO_START_YEAR,
    CMIP6_SOLAR_TRAILING_YEARS,
    cmip6_coverage_issues,
    format_cmip6_issues,
    report_training_coverage,
)


# ==========================================================================
# DEFAULTS
# ==========================================================================

BASELINE_START = 1991
BASELINE_END   = 2010
FORECAST_START = "2025-01-01"
FORECAST_END   = "2100-12-31 23:00"
DEFAULT_EPW_YEARS = "2030,2050,2080"
DEFAULT_SSP    = "ssp245"
DEFAULT_DOWNLOAD_DATA = ["isd", "nsrdb", "era5-land", "cmip6"]
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


def _file_signature(path: Path) -> list:
    resolved = path.resolve()
    try:
        stat = resolved.stat()
    except FileNotFoundError:
        return [str(resolved), None]
    return [str(resolved), stat.st_size, stat.st_mtime_ns]


def _cached_outputs(path: Path, inputs: dict) -> list[Path] | None:
    try:
        manifest = json.loads(path.read_text())
        if manifest["inputs"] != inputs:
            return None
        signatures = manifest["outputs"]
        outputs = [Path(signature[0]) for signature in signatures]
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None
    if all(output.is_file() for output in outputs) and signatures == [
        _file_signature(output) for output in outputs
    ]:
        return outputs
    return None


def _cache_matches(path: Path, inputs: dict, *outputs: Path) -> bool:
    return _cached_outputs(path, inputs) == [output.resolve() for output in outputs]


def _save_cache(path: Path, inputs: dict, *outputs: Path) -> None:
    manifest = {"inputs": inputs, "outputs": [_file_signature(output) for output in outputs]}
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(manifest, stream)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


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


def _find_isd_dir(city: str, base: Path, station_id: str | None = None) -> Path:
    """Find a station directory with usable ISD archives."""
    from samy_wgen.pipeline.download_data import _usable_isd_archive

    city_dir = base / "ISD" / city
    if not city_dir.exists():
        raise SystemExit(f"ERROR: ISD directory not found: {city_dir}")
    stations = [
        path for path in sorted(city_dir.iterdir())
        if path.is_dir() and (station_id is None or path.name.split("_", 1)[0] == station_id)
        and any(_usable_isd_archive(archive) for archive in path.glob("*.gz"))
    ]
    if not stations:
        raise SystemExit(f"ERROR: No usable ISD archives for {station_id or city} under {city_dir}")
    if len(stations) > 1:
        print(f"  [info] Multiple ISD stations found, using: {stations[0].name}")
    return stations[0]


def _find_nsrdb_csv(city: str, base: Path, required_years: Iterable[int] = ()) -> Path:
    """Select a local solar CSV by usable coverage."""
    nsrdb_dir = base / "NSRDB"
    if not nsrdb_dir.exists():
        raise SystemExit(f"ERROR: NSRDB directory not found: {nsrdb_dir}")
    city_slug = city.lower().replace(" ", "_")
    pattern = f"nsrdb_{city_slug}*.csv"
    required = set(required_years)
    candidates = []
    for path in sorted(nsrdb_dir.glob(pattern)):
        try:
            solar = _load_solar_hourly(path).dropna()
        except (OSError, KeyError, ValueError):
            continue
        # NSRDB downloads omit Feb 29 by request, even in leap years.
        hours = solar.index[~((solar.index.month == 2) & (solar.index.day == 29))]
        counts = hours.to_series().groupby(hours.year).size()
        complete = set(counts.index[counts == 8760])
        if complete and required.issubset(complete):
            candidates.append((len(complete), path))
    if not candidates:
        raise SystemExit(
            f"ERROR: No NSRDB CSV with usable hourly solar coverage for '{city}' in {nsrdb_dir}.\n"
            f"  Expected pattern: {pattern}, required years: {sorted(required)}"
        )
    return max(candidates)[1]


def _load_solar_hourly(path: Path) -> pd.DataFrame:
    columns = ["GHI", "DNI", "DHI"]
    source = pd.read_csv(path, usecols=lambda col: col in {*columns, "DATE", "Year", "Month", "Day", "Hour"})
    if "DATE" in source:
        times = pd.to_datetime(source["DATE"])
    else:
        times = pd.to_datetime(source[["Year", "Month", "Day", "Hour"]].rename(columns=str.lower))
    solar = source[columns].apply(pd.to_numeric, errors="coerce")
    solar.index = pd.DatetimeIndex(times, name="DATE")
    return solar.where(solar.abs().lt(float("inf"))).clip(lower=0).resample("h").mean()


def _solar_months(path: Path) -> pd.PeriodIndex:
    frame = pd.read_csv(
        path, usecols=lambda col: col in {"DATE", "Year", "Month", "GHI", "DNI", "DHI"},
    )
    solar = ["GHI", "DNI", "DHI"]
    if not set(solar).issubset(frame):
        return pd.PeriodIndex([], freq="M")
    frame = frame.dropna(subset=solar)
    if "DATE" in frame:
        dates = pd.to_datetime(frame["DATE"], errors="coerce")
    elif {"Year", "Month"}.issubset(frame):
        dates = pd.to_datetime(
            {"year": frame["Year"], "month": frame["Month"], "day": 1}, errors="coerce",
        )
    else:
        return pd.PeriodIndex([], freq="M")
    return pd.PeriodIndex(dates.dropna(), freq="M").unique().sort_values()


def _cached_calibrated_solar(args, city: str, data_dir: Path) -> Path | None:
    from samy_wgen.utils.time_utils import get_utc_offset

    calibrated = data_dir / "ERA5_Land" / "calibrated" / f"{city.lower()}_solar_calibrated.csv"
    if not calibrated.is_file():
        return None
    try:
        offset = json.loads(calibrated.with_suffix(".metadata.json").read_text())["utc_offset_hours"]
        if offset != get_utc_offset(city=city):
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    years = _baseline_years(args)
    expected = pd.period_range(f"{years[0]}-01", f"{years[-1]}-12", freq="M")
    return calibrated if expected.isin(_solar_months(calibrated)).all() else None


def _resolve_solar_for_baseline(args, city: str, data_dir: Path, out_dir: Path,
                                nsrdb_csv: Path, meta: dict) -> Path:
    """Extend the local NSRDB record when solar coverage misses baseline months."""
    from samy_wgen.pipeline.download_data import _era5_land_file_covers_request, download_era5_land
    from samy_wgen.utils.solar_calibration import calibrate_era5_land_solar

    expected = pd.period_range(f"{args.baseline_start}-01", f"{args.baseline_end}-12", freq="M")
    observed = _solar_months(nsrdb_csv)
    if expected.isin(observed).all():
        return nsrdb_csv

    calibrated = _cached_calibrated_solar(args, city, data_dir)
    if calibrated is not None:
        print(f"  [solar] Using calibrated input: {calibrated}")
        return calibrated
    if observed.empty:
        raise SystemExit(f"ERROR: No complete solar observations in {nsrdb_csv}; supply --nsrdb.")

    # The fixed-LST baseline needs one extra UTC day at each edge.
    start = min(expected[0].start_time - pd.Timedelta(days=1), observed[0].start_time)
    end = max(expected[-1].end_time.normalize() + pd.Timedelta(days=1),
              observed[-1].end_time.normalize())
    era_dir = data_dir / "ERA5_Land"
    variable = "surface_solar_radiation_downwards"
    era_path = None
    for candidate in sorted((era_dir / city).glob(f"era5_land_{city.lower()}_{variable}_*.nc")):
        try:
            first, last = map(pd.Timestamp, candidate.stem.rsplit("_", 1)[-1].split("-"))
        except ValueError:
            continue
        if first <= start and last >= end and _era5_land_file_covers_request(candidate, variable, first, last):
            era_path = candidate
            break
    if era_path is None:
        if args.skip_download or "era5-land" not in args.download_data:
            raise SystemExit(
                f"ERROR: NSRDB misses months in the {args.baseline_start}–{args.baseline_end} baseline for {city}. "
                "Supply a calibrated LST solar CSV via --nsrdb or allow ERA5-Land download."
            )
        _print_title("Download ERA5-Land Solar")
        era_path, = download_era5_land(
            city, lat=meta["lat"], lon=meta["lon"], start_date=start.date().isoformat(),
            end_date=end.date().isoformat(), output_dir=str(era_dir), variables=[variable],
        )

    output = out_dir / "2_solar_extended_lst.csv"
    cache_path = out_dir / ".2_solar_extended_lst.json"
    cache_inputs = {
        "era5": _file_signature(era_path), "nsrdb": _file_signature(nsrdb_csv),
        "lat": meta["lat"], "longitude": meta["lon"], "tz": meta["tz"],
    }
    if not args.overwrite and _cache_matches(cache_path, cache_inputs, output):
        return output

    _print_title("Extend NSRDB With ERA5-Land")
    cache_path.unlink(missing_ok=True)
    solar = calibrate_era5_land_solar(
        era_path, nsrdb_csv, latitude=meta["lat"], longitude=meta["lon"],
        utc_offset_hours=meta["tz"], nsrdb_time_basis="lst", preserve_observations=True,
    )
    solar.to_csv(output)
    if not expected.isin(_solar_months(output)).all():
        raise SystemExit(f"ERROR: Calibrated solar still misses baseline months for {city}: {output}")
    _save_cache(cache_path, cache_inputs, output)
    return output


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
                        help="Explicit solar CSV in fixed local standard time")
    parser.add_argument("--cmip6-dir", default=None,
                        help="CMIP6 root directory (overrides data-dir/CMIP6)")
    parser.add_argument("--gcm", default=None,
                        help="CMIP6 GCM used for coverage, download, and generation. Overrides --cmip6-models.")
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
                        choices=["isd", "nsrdb", "era5-land", "cmip6"],
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
                        help="Ordered CMIP6 model list. The deterministic run uses the first when --gcm is omitted.")
    parser.add_argument("--cmip6-experiments", nargs="+", default=None,
                        help="CMIP6 CDS experiments for automatic downloads (default: historical + selected SSP)")
    parser.add_argument("--cmip6-years", nargs=2, type=int, metavar=("START", "END"), default=None,
                        help="Year range for automatic CMIP6 downloads (extended as needed to cover the baseline and solar trailing window)")

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


def _forecast_years(args) -> list[int]:
    start = pd.Timestamp(getattr(args, "start", FORECAST_START)).year
    end = pd.Timestamp(getattr(args, "end", FORECAST_END)).year
    return list(range(start, end + 1))


def _baseline_years(args) -> list[int]:
    start = int(getattr(args, "baseline_start", BASELINE_START))
    end = int(getattr(args, "baseline_end", BASELINE_END))
    return list(range(start, end + 1))


def _cmip6_coverage_issues(city: str, cmip6_dir: Path, args) -> list[str]:
    return cmip6_coverage_issues(
        city,
        cmip6_dir,
        args.ssp,
        _pipeline_cmip6_models(args),
        forecast_years=_forecast_years(args),
        baseline_years=_baseline_years(args),
        trailing_years=CMIP6_SOLAR_TRAILING_YEARS,
    )


def _pipeline_cmip6_models(args) -> list[str]:
    """Use the model selected for the deterministic calculation throughout."""
    if getattr(args, "mode", "deterministic") == "ensemble":
        return args.cmip6_models
    from samy_wgen.pipeline.run_climate_delta import cmip6_model_stem

    return [cmip6_model_stem(getattr(args, "gcm", None) or args.cmip6_models[0])]


def _has_required_cmip6_data(
    city: str,
    cmip6_dir: Path,
    ssp: str,
    models: Iterable[str],
    *,
    forecast_years: Iterable[int] | None = None,
    baseline_years: Iterable[int] | None = None,
    trailing_years: int = 20,
) -> bool:
    """Return whether existing CMIP6 files cover the variables and years needed.

    File presence alone is not sufficient: the bounded solar calculation needs
    a complete monthly rsds trailing window in addition to the daily baseline
    and forecast fields.
    """
    if forecast_years is None:
        forecast_years = range(
            int(FORECAST_START[:4]), int(FORECAST_END[:4]) + 1)
    if baseline_years is None:
        baseline_years = range(BASELINE_START, BASELINE_END + 1)
    return not cmip6_coverage_issues(
        city,
        cmip6_dir,
        ssp,
        models,
        forecast_years=forecast_years,
        baseline_years=baseline_years,
        trailing_years=trailing_years,
    )


def _normalize_city_args(requested: list[str]) -> list[str]:
    return resolve_city_codes(requested)


def _parse_isd_station_args(values: list[str] | None, cities: list[str]) -> dict[str, str] | None:
    from samy_wgen.pipeline.download_data import _parse_isd_station_args as _parse

    return _parse(values, cities)


def _cmip6_download_range(args) -> tuple[int, int]:
    baseline = _baseline_years(args)
    forecast = _forecast_years(args)
    start = min(min(baseline), min(forecast) - CMIP6_SOLAR_TRAILING_YEARS + 1,
                CMIP6_SCENARIO_START_YEAR - 1)
    end = max(max(baseline), max(forecast), CMIP6_SCENARIO_START_YEAR)
    requested = getattr(args, "cmip6_years", None)
    if requested:
        start = min(start, int(requested[0]))
        end = max(end, int(requested[1]))
    return start, end


def _ensure_pipeline_inputs(args, data_dir: Path, city: str, cmip6_dir: Path | None = None) -> tuple[Path, Path, Path]:
    from samy_wgen.pipeline.download_data import (
        _city_meta,
        _resolve_nsrdb_source,
        download_isd,
        download_nsrdb,
        select_best_isd_station,
    )

    station_map = _parse_isd_station_args(args.isd_station, args.city)
    station_id = (station_map or {}).get(city)
    if args.isd_dir:
        isd_station_dir = Path(args.isd_dir)
    else:
        try:
            isd_station_dir = _find_isd_dir(city, data_dir, station_id)
        except SystemExit:
            if args.skip_download or "isd" not in set(args.download_data):
                raise
            _print_title("Download ISD")
            yr_start, yr_end = args.download_years if args.download_years else DEFAULT_ISD_YEAR_RANGE
            station_id = station_id or select_best_isd_station(
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
            isd_station_dir = _find_isd_dir(city, data_dir, station_id)

    if args.nsrdb:
        nsrdb_csv = Path(args.nsrdb)
    elif (calibrated := _cached_calibrated_solar(args, city, data_dir)) is not None:
        nsrdb_csv = calibrated
    else:
        yr_start, yr_end = args.download_years or DEFAULT_NSRDB_YEAR_RANGE
        requested_years = list(range(yr_start, yr_end + 1))
        baseline = set(_baseline_years(args))
        try:
            static_sources = _resolve_nsrdb_source(city)
        except ValueError:
            static_sources = []
        required = {
            year for _, first, last in static_sources
            for year in requested_years if year in baseline and first <= year <= last
        }
        try:
            nsrdb_csv = _find_nsrdb_csv(city, data_dir, required)
        except SystemExit:
            if args.skip_download or "nsrdb" not in set(args.download_data):
                raise
            _print_title("Download NSRDB")
            meta = _city_meta(city)
            result = download_nsrdb(
                city=city,
                lat=meta["lat"],
                lon=meta["lon"],
                years=requested_years,
                output_dir=str(data_dir / "NSRDB"),
            )
            if not result["success"]:
                raise SystemExit(f"ERROR: NSRDB download did not produce complete data for {city}: {result}")
            nsrdb_csv = Path(result["success"][0]["path"])

    cmip6_dir = cmip6_dir or (Path(args.cmip6_dir) if args.cmip6_dir else data_dir / "CMIP6")
    coverage_issues = _cmip6_coverage_issues(city, cmip6_dir, args)
    if coverage_issues:
        raise SystemExit(
            f"ERROR: CMIP6 inputs under {cmip6_dir / city} are incomplete:\n"
            f"{format_cmip6_issues({city: coverage_issues})}\n"
            "Provide --cmip6-dir, pre-download the files, or allow auto-download."
        )

    return isd_station_dir, nsrdb_csv, cmip6_dir


def _ensure_batch_cmip6_inputs(args, data_dir: Path, cities: list[str]) -> Path:
    from samy_wgen.pipeline.download_data import download_cmip6_for_cities

    cmip6_dir = Path(args.cmip6_dir) if args.cmip6_dir else data_dir / "CMIP6"
    issues_by_city = {}
    for city in cities:
        issues = _cmip6_coverage_issues(city, cmip6_dir, args)
        if issues:
            issues_by_city[city] = issues
    if not issues_by_city:
        return cmip6_dir

    details = format_cmip6_issues(issues_by_city)
    if args.skip_download or "cmip6" not in set(args.download_data):
        raise SystemExit(
            "ERROR: CMIP6 inputs are incomplete:\n"
            f"{details}\n"
            "Provide --cmip6-dir, pre-download the files, or allow auto-download."
        )

    _print_title("Download CMIP6")
    missing_cities = list(issues_by_city)
    print(f"[batch] CMIP6 inputs are incomplete for: {', '.join(missing_cities)}")
    print(details)
    download_start, download_end = _cmip6_download_range(args)
    print(f"[batch] Requesting CMIP6 years {download_start}-{download_end} "
          "(includes baseline and solar trailing window)")
    download_cmip6_for_cities(
        cities=missing_cities,
        models=_pipeline_cmip6_models(args),
        experiments=_cmip6_download_experiments(args.ssp, args.cmip6_experiments),
        year_range=(download_start, download_end),
        output_dir=str(cmip6_dir),
    )

    remaining = {}
    for city in missing_cities:
        issues = _cmip6_coverage_issues(city, cmip6_dir, args)
        if issues:
            remaining[city] = issues
    if remaining:
        raise SystemExit(
            "ERROR: CMIP6 download did not fill the required coverage:\n"
            f"{format_cmip6_issues(remaining)}\n"
            "Check the download log and retry, or provide complete files via --cmip6-dir."
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
        if result_path is None:
            raise SystemExit(f"ERROR: could not read weather records from {isd_station_dir}")
        raw = pd.read_csv(result_path, parse_dates=["datestamp"], low_memory=False)
    print(f"  parsed {len(raw):,} records")
    return raw


def _stage1_remap_to_lst(isd_raw: pd.DataFrame, city: str,
                          lat: float, lon: float,
                          out_dir: Path) -> pd.DataFrame:
    """Remap ISD columns + UTC→LST + resample to hourly → 1_isd_lst.csv."""
    from samy_wgen.utils.parse_ISD import remap_isd_columns, resample_isd_hourly
    from samy_wgen.utils.time_utils import utc_to_lst

    csv_path = out_dir / "1_isd_lst.csv"
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
                         out_dir: Path, overwrite: bool, *, isd_csv: Path | None = None) -> Path:
    """Merge ISD LST + NSRDB solar (GHI/DNI/DHI) → 2_isd_solar_lst.csv."""
    csv_path = out_dir / "2_isd_solar_lst.csv"
    cache_path = out_dir / ".2_isd_solar_lst.json"
    cache_inputs = (
        {"solar_policy": 2, "isd": _file_signature(isd_csv), "solar": _file_signature(nsrdb_csv)}
        if isd_csv is not None else None
    )
    if not overwrite and cache_inputs is not None and _cache_matches(cache_path, cache_inputs, csv_path):
        _print_title("Stage 2 Merge NSRDB")
        print(f"[stage 2] Cache hit: {csv_path.name}")
        return csv_path

    _print_title("Stage 2 Merge NSRDB")
    cache_path.unlink(missing_ok=True)
    print(f"[stage 2] Merging NSRDB solar: {nsrdb_csv.name}")
    solar = _load_solar_hourly(nsrdb_csv)
    merged = isd_lst.join(solar, how="inner").ffill(limit=3).bfill(limit=3)
    merged = merged.dropna(subset=["temp", "GHI", "DNI", "DHI"])

    report_training_coverage(merged, key="temp", context="ISD+NSRDB training series")

    yr_min, yr_max = merged.index.year.min(), merged.index.year.max()
    print(f"  {len(merged):,} hourly rows, {yr_min}–{yr_max}")
    merged.index.name = "DATE"
    merged.to_csv(csv_path)
    if cache_inputs is not None:
        _save_cache(cache_path, cache_inputs, csv_path)
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
    cache_path = out_dir / ".3_rsvar.json"
    cache_inputs = {
        "coverage_policy": 2,
        "weather": _file_signature(merged_csv),
        "city": city, "lat": lat, "longitude": longitude,
        "baseline": [baseline_start, baseline_end],
        "forecast": [pd.Timestamp(forecast_start).isoformat(), pd.Timestamp(forecast_end).isoformat()],
    }
    outputs = (xlsx_path, xlsx_path.with_suffix(".csv"), calibration_path)
    if not overwrite and _cache_matches(cache_path, cache_inputs, *outputs):
        _print_title("Stage 3 Run RS-VAR")
        print(f"[stage 3] Cache hit: {xlsx_path.name}")
        return xlsx_path

    _print_title("Stage 3 Run RS-VAR")
    print(f"[stage 3] Training RS-VAR ({city}, lat={lat:.2f}°N)")
    cache_path.unlink(missing_ok=True)
    run_and_save_rsvar(
        isd_path=merged_csv, lat=lat, output_path=xlsx_path, city=city,
        verbose=False, longitude=longitude, baseline_start=baseline_start,
        baseline_end=baseline_end, forecast_start=forecast_start, forecast_end=forecast_end,
    )
    _save_cache(cache_path, cache_inputs, *outputs)

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
        cmip6_model_stem,
    )

    csv_path = out_dir / "4_delta_corrected.csv"
    energy_path = out_dir / "4_solar_monthly_energy.csv"
    cache_path = out_dir / ".4_delta.json"
    climate_dir = cmip6_dir / city.replace(" ", "_")
    model_prefix = cmip6_model_stem(gcm) if gcm else "*"
    climate_paths = sorted({
        path for label in ("historical", ssp)
        for suffix in ("", "_monthly")
        for path in climate_dir.glob(f"{model_prefix}_{label}{suffix}.csv")
    })
    rsvar_paths = [rsvar_xlsx]
    if rsvar_xlsx.suffix == ".xlsx":
        rsvar_paths.append(rsvar_xlsx.with_suffix(".csv"))
    cache_inputs = {
        "coverage_policy": 2,
        "rsvar": [_file_signature(path) for path in rsvar_paths],
        "climate_dir": str(climate_dir.resolve()),
        "climate_files": [_file_signature(path) for path in climate_paths],
        "city": city, "lat": lat, "longitude": longitude, "ssp": ssp, "gcm": gcm,
        "baseline": [baseline_start, baseline_end],
    }
    if not overwrite and _cache_matches(cache_path, cache_inputs, csv_path, energy_path):
        _print_title("Stage 4 Apply Climate Delta")
        print(f"[stage 4] Cache hit: {csv_path.name}")
        return pd.read_csv(csv_path, parse_dates=["datetime"]).set_index("datetime")

    _print_title("Stage 4 Apply Climate Delta")
    print(f"[stage 4] Applying CMIP6 climate delta ({ssp})")
    cache_path.unlink(missing_ok=True)
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
        monthly_energy.to_csv(energy_path, index=False)
        _save_cache(cache_path, cache_inputs, csv_path, energy_path)
    else:
        energy_path.unlink(missing_ok=True)
    print(f"  saved: {csv_path.name} ({len(output_df):,} rows)")
    return output_df


def _load_hourly_csv(csv_path: Path, datetime_col: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, parse_dates=[datetime_col]).set_index(datetime_col)


def _output_historical_epw(
    merged_csv: Path, city: str, meta: dict, out_dir: Path, overwrite: bool,
    baseline_start: int = BASELINE_START, baseline_end: int = BASELINE_END,
) -> Path | None:
    """Prefer 2000, then try the latest exportable year in the fit baseline."""
    from samy_wgen.pipeline.gen_samy import _write_epw_for_year

    historical_df = _load_hourly_csv(merged_csv, "DATE")
    epw_dir = out_dir / "EPW"
    epw_dir.mkdir(parents=True, exist_ok=True)
    years = historical_df.index.year.unique()
    candidates = sorted(
        (year for year in years if year == 2000 or baseline_start <= year <= baseline_end),
        key=lambda year: (year == 2000, year), reverse=True,
    )
    for year in candidates:
        epw_path = epw_dir / f"{city}_historical_{year}.epw"
        if epw_path.exists() and not overwrite:
            print(f"  [EPW] Cache hit: {epw_path.name}")
            return epw_path
        try:
            _write_epw_for_year(
                df=historical_df,
                year=year,
                out_path=epw_path,
                location=f"{city}-{meta.get('country', 'XX')}",
                lat=meta["lat"],
                lon=meta["lon"],
                tz=meta["tz"],
                elevation_m=meta.get("elevation_m", 0.0),
                pressure_reference="sea_level",
            )
        except ValueError as exc:
            print(f"  [EPW] WARNING: historical {year} skipped ({exc})")
            continue
        print(f"  [EPW] {epw_path.name}")
        return epw_path
    print("  [EPW] WARNING: no complete historical year available for export")
    return None


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
            pressure_reference="sea_level",
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
    isd_manifest = out_dir / ".1_isd_lst.json"
    isd_inputs = {
        "station": str(isd_station_dir.resolve()),
        "files": [_file_signature(path) for path in sorted(isd_station_dir.glob("*.gz"))],
        "city": city, "lat": meta["lat"], "longitude": meta["lon"], "tz": meta["tz"],
    }
    if not ow and _cache_matches(isd_manifest, isd_inputs, isd_lst_cache):
        _print_title("Stage 1 Remap To LST")
        print("[stage 1] Cache hit: 1_isd_lst.csv  (skipping ISD parse)")
        isd_lst = pd.read_csv(isd_lst_cache, parse_dates=["DATE"]).set_index("DATE")
    else:
        isd_manifest.unlink(missing_ok=True)
        isd_raw = _stage0_parse_isd(isd_station_dir)
        isd_lst = _stage1_remap_to_lst(
            isd_raw,
            city,
            meta["lat"],
            meta["lon"],
            out_dir,
        )
        _save_cache(isd_manifest, isd_inputs, isd_lst_cache)

    if not args.nsrdb:
        nsrdb_csv = _resolve_solar_for_baseline(args, city, data_dir, out_dir, nsrdb_csv, meta)
    merged_csv = _stage2_merge_nsrdb(isd_lst, nsrdb_csv, out_dir, overwrite=ow, isd_csv=isd_lst_cache)

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
            gcm=_pipeline_cmip6_models(args)[0],
        )
        export_manifest = out_dir / ".exports.json"
        export_inputs = {
            "ddy_pressure_reference": "sea_level",
            "historical": _file_signature(merged_csv),
            "future": _file_signature(out_dir / "4_delta_corrected.csv"),
            "city": city, "ssp": args.ssp, "meta": meta, "epw_years": epw_years,
            "baseline": [args.baseline_start, args.baseline_end],
        }
        exports = None if ow else _cached_outputs(export_manifest, export_inputs)
        if exports is None:
            export_manifest.unlink(missing_ok=True)
            print("\n[output] Writing historical EPW")
            historical_epw = _output_historical_epw(
                merged_csv=merged_csv,
                city=city,
                meta=meta,
                out_dir=out_dir,
                overwrite=True,
                baseline_start=args.baseline_start,
                baseline_end=args.baseline_end,
            )
            print(f"\n[output] Writing future EPW files for years: {epw_years}")
            future_epws = _output_epw(
                output_df=output_df,
                epw_years=epw_years,
                city=city,
                ssp=args.ssp,
                meta=meta,
                out_dir=out_dir,
                overwrite=True,
            )
            print("\n[output] Writing baseline/future DDY files")
            ddys = _output_ddy_files(
                merged_csv=merged_csv,
                output_df=output_df,
                city=city,
                ssp=args.ssp,
                meta=meta,
                baseline_start=args.baseline_start,
                baseline_end=args.baseline_end,
                out_dir=out_dir,
                overwrite=True,
            )
            exports = ([historical_epw] if historical_epw else []) + future_epws + ddys
            _save_cache(export_manifest, export_inputs, *exports)
        else:
            print(f"\n[output] Cache hit: {', '.join(path.name for path in exports) or 'no exportable files'}")
        print(f"\n{'='*70}")
        print(
            f"DONE — {sum(path.suffix == '.epw' for path in exports)} EPW file(s) in {out_dir / 'EPW'} "
            f"and {sum(path.suffix == '.ddy' for path in exports)} DDY file(s) in {out_dir / 'DDY'}"
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
    if args.baseline_start > args.baseline_end:
        parser.error("--baseline-start must be on or before --baseline-end")
    args.city = _normalize_city_args(args.city)
    data_dir = Path(args.data_dir)

    cmip6_dir = _ensure_batch_cmip6_inputs(args, data_dir, args.city)
    for city in args.city:
        out_dir = _city_out_dir(args, city)
        out_dir.mkdir(parents=True, exist_ok=True)
        with file_lock(out_dir / ".samy-run.lock"):
            _run_city_pipeline(args, city, data_dir, cmip6_dir)


if __name__ == "__main__":
    main()
