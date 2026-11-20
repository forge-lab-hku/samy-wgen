"""Shared input-integrity checks for every pipeline entry point.

CMIP6 daily/monthly coverage and the ISD+NSRDB training-series hourly
coverage are validated here so that ``samy-run``, ``samy-delta``,
``samy-rsvar`` and ``samy-ensemble`` apply the same checks before consuming
data.  ``samy-run`` composes the RS-VAR and climate-delta stages, so it picks
up both checks through the same functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from functools import lru_cache

import numpy as np
import pandas as pd

from samy_wgen.utils.pressure import projection_pressure_pa

# Full monthly rsds trailing window required by the bounded solar calculation.
CMIP6_SOLAR_TRAILING_YEARS = 20
CMIP6_SCENARIO_START_YEAR = 2015

# Hourly coverage thresholds for the ISD+NSRDB training series.  A year is
# "thin" below WARN (reported) and cannot train below FAIL (aborts).  Edge
# years (first/last of the series) are exempt because a station record
# legitimately starts or stops mid-year.
TRAINING_COVERAGE_WARN = 0.70
TRAINING_COVERAGE_FAIL = 0.30


@lru_cache(maxsize=512)
def _calendar_days(year: int, calendar: str) -> pd.DatetimeIndex:
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    if calendar == "360_day":
        return dates[np.rint(np.linspace(0, len(dates) - 1, 360)).astype(int)]
    if calendar == "noleap":
        return dates[~((dates.month == 2) & (dates.day == 29))]
    return dates


def cmip6_daily_calendar(times: pd.DatetimeIndex) -> str:
    """Infer the source calendar from complete annual patterns in legacy CSVs."""
    dates = times.normalize().unique().sort_values()
    if not len(dates):
        return "gregorian"
    groups = {year: dates[dates.year == year] for year in dates.year.unique()}
    for calendar in ("360_day", "noleap"):
        if (all(not days.difference(_calendar_days(year, calendar)).size
                for year, days in groups.items())
                and any(days.equals(_calendar_days(year, calendar))
                        and len(days) < len(_calendar_days(year, "gregorian"))
                        for year, days in groups.items())):
            return calendar
    return "gregorian"


def cmip6_missing_days(times: pd.DatetimeIndex, valid_times: pd.DatetimeIndex,
                       years: Iterable[int]) -> pd.DatetimeIndex:
    calendar = cmip6_daily_calendar(times)
    actual = valid_times.normalize()
    parts = [_calendar_days(year, calendar).difference(actual) for year in sorted(set(years))]
    return pd.DatetimeIndex([]).append(parts)


def cmip6_calendar_gaps(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return only date labels absent by design from the inferred calendar."""
    calendar = cmip6_daily_calendar(times)
    if calendar == "gregorian" or times.empty:
        return pd.DatetimeIndex([])
    parts = [_calendar_days(year, "gregorian").difference(_calendar_days(year, calendar))
             for year in range(times.min().year, times.max().year + 1)]
    return pd.DatetimeIndex([]).append(parts)


def _format_int_ranges(values: list[int]) -> str:
    """Render sorted integers as compact ranges (``1991-2010, 2025``)."""
    values = sorted({int(value) for value in values})
    if not values:
        return "none"
    parts: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        parts.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    parts.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(parts)


def _format_missing_months(pairs: set[tuple[int, int]]) -> str:
    """Summarize missing (year, month) pairs by year."""
    by_year: dict[int, set[int]] = {}
    for year, month in pairs:
        by_year.setdefault(year, set()).add(month)
    full_years = sorted(year for year, months in by_year.items() if len(months) == 12)
    partial_years = {year: sorted(months) for year, months in by_year.items()
                     if len(months) < 12}
    parts: list[str] = []
    if full_years:
        parts.append(f"{_format_int_ranges(full_years)} (all months)")
    for year in sorted(partial_years):
        parts.append(f"{year} (month(s) {_format_int_ranges(partial_years[year])})")
    return ", ".join(parts)


def _read_cmip6_csv(path: Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(path, parse_dates=["time"])
    except (OSError, pd.errors.ParserError, ValueError):
        return None
    if "time" not in frame.columns or frame.empty:
        return None
    return frame


def _daily_coverage_issues(
    model: str,
    label: str,
    path: Path,
    frame: pd.DataFrame | None,
    required_years: list[int],
) -> list[str]:
    """Return coverage problems for one historical/scenario daily file.

    Require finite temperature and pressure for every source-calendar day.
    """
    if frame is None:
        return [f"{model}: missing or unreadable {label} file {path.name}"]
    issues: list[str] = []
    temperature_column = next((column for column in ("tas", "tas_C")
                               if column in frame.columns), None)
    try:
        pressure, reference = projection_pressure_pa(frame)
    except ValueError as error:
        return [f"{model}: {label} file {path.name}: {error}"]
    if "ps_hPa" in frame:
        pressure, reference = frame["ps_hPa"], "normalized"
    missing_columns = []
    if temperature_column is None:
        missing_columns.append("tas")
    if pressure is None:
        missing_columns.append("psl (or ps/surface_air_pressure)")
    if missing_columns:
        issues.append(
            f"{model}: {label} file {path.name} is missing required variable(s): "
            f"{', '.join(missing_columns)}")
        return issues

    times = pd.DatetimeIndex(pd.to_datetime(frame["time"], errors="coerce"))
    if times.isna().any():
        issues.append(f"{model}: {label} file {path.name} has invalid timestamps")
        return issues
    if times.normalize().has_duplicates:
        return [f"{model}: {label} file {path.name} has duplicate daily dates"]
    pressure_label = {"surface": "ps", "sea_level": "psl", "normalized": "ps_hPa"}[reference]
    for column, values in ((temperature_column, frame[temperature_column]), (pressure_label, pressure)):
        valid = np.isfinite(pd.to_numeric(values, errors="coerce"))
        missing_days = cmip6_missing_days(times, times[valid], required_years)
        if len(missing_days):
            examples = ", ".join(missing_days[:5].strftime("%Y-%m-%d"))
            missing_pairs = set(zip(missing_days.year, missing_days.month))
            issues.append(
                f"{model}: {label} file {path.name} does not cover {column} for "
                f"{len(missing_days)} day(s) in {_format_missing_months(missing_pairs)} "
                f"(first missing: {examples})")
    return issues


def _solar_coverage_issues(
    model: str,
    label: str,
    frame: pd.DataFrame,
    forecast_years: list[int],
    baseline_years: list[int],
    trailing_years: int,
) -> list[str]:
    """Return monthly rsds coverage problems for the trailing solar window."""
    if "rsds" not in frame.columns:
        return [f"{model}: {label} input is missing the required rsds column"]
    times = pd.to_datetime(frame["time"], errors="coerce")
    if times.isna().any():
        return [f"{model}: {label} input has invalid timestamps"]
    if times.duplicated().any():
        return [f"{model}: {label} input has duplicate timestamps"]

    valid = np.isfinite(pd.to_numeric(frame["rsds"], errors="coerce"))
    available = set(zip(times[valid].dt.year, times[valid].dt.month))
    issues: list[str] = []

    baseline_set = set(baseline_years)
    baseline_months = {(year, month) for year in baseline_set for month in range(1, 13)}
    missing_baseline_months = baseline_months - available
    if missing_baseline_months:
        issues.append(
            f"{model}: {label} has no baseline rsds for month(s) "
            f"{_format_missing_months(missing_baseline_months)}")

    missing_pairs: set[tuple[int, int]] = set()
    for year in forecast_years:
        for month in range(1, 13):
            for window_year in range(year - trailing_years + 1, year + 1):
                if (window_year, month) not in available:
                    missing_pairs.add((window_year, month))
    if missing_pairs:
        available_years = sorted({year for year, _ in available})
        issues.append(
            f"{model}: {label} does not cover the {trailing_years}-year solar "
            f"trailing window; missing {_format_missing_months(missing_pairs)}; "
            f"available years: {_format_int_ranges(available_years)}")
    return issues


def cmip6_coverage_issues(
    city: str,
    cmip6_dir: Path,
    ssp: str,
    models: Iterable[str],
    *,
    forecast_years: Iterable[int],
    baseline_years: Iterable[int],
    trailing_years: int = CMIP6_SOLAR_TRAILING_YEARS,
) -> list[str]:
    """Return human-readable reasons the CMIP6 inputs cannot drive the pipeline.

    Validates required variables, daily baseline/forecast months, and the full
    monthly rsds trailing window used by the bounded solar calculation.  An
    empty list means the inputs are usable.
    """
    city_dir = Path(cmip6_dir) / city
    forecast_years = sorted({int(year) for year in forecast_years})
    baseline_years = sorted({int(year) for year in baseline_years})
    if not forecast_years or not baseline_years:
        return [f"{city}: forecast/baseline year ranges are empty"]

    required_years = sorted(set(baseline_years) | set(forecast_years))
    historical_years = [year for year in required_years if year < CMIP6_SCENARIO_START_YEAR]
    scenario_years = [year for year in required_years if year >= CMIP6_SCENARIO_START_YEAR]
    issues: list[str] = []
    for model in models:
        historical_daily_path = city_dir / f"{model}_historical.csv"
        scenario_daily_path = city_dir / f"{model}_{ssp}.csv"
        historical_monthly_path = city_dir / f"{model}_historical_monthly.csv"
        scenario_monthly_path = city_dir / f"{model}_{ssp}_monthly.csv"

        historical_daily = _read_cmip6_csv(historical_daily_path)
        scenario_daily = _read_cmip6_csv(scenario_daily_path)
        issues.extend(_daily_coverage_issues(
            model, "historical daily", historical_daily_path,
            historical_daily, historical_years))
        issues.extend(_daily_coverage_issues(
            model, "scenario daily", scenario_daily_path,
            scenario_daily, scenario_years))

        daily_frames = [frame for frame in (historical_daily, scenario_daily)
                        if frame is not None]
        if any("rsds" in frame.columns for frame in daily_frames):
            solar_frame = pd.concat(daily_frames, ignore_index=True)
            solar_label = "daily rsds"
        else:
            monthly_frames = []
            for path, label in (
                (historical_monthly_path, "historical monthly"),
                (scenario_monthly_path, "scenario monthly"),
            ):
                frame = _read_cmip6_csv(path)
                if frame is None:
                    issues.append(
                        f"{model}: missing or unreadable {label} file {path.name} "
                        "(required for rsds)")
                else:
                    monthly_frames.append(frame)
            if not monthly_frames:
                continue
            solar_frame = pd.concat(monthly_frames, ignore_index=True)
            solar_label = "monthly rsds"

        issues.extend(_solar_coverage_issues(
            model, solar_label, solar_frame,
            forecast_years, baseline_years, trailing_years))
    return issues


def format_cmip6_issues(issues_by_city: dict[str, list[str]]) -> str:
    """Render a city → issues mapping as an indented, human-readable block."""
    lines: list[str] = []
    for city, issues in issues_by_city.items():
        lines.append(f"  {city}:")
        lines.extend(f"    - {issue}" for issue in issues)
    return "\n".join(lines)


def report_training_coverage(
    frame: pd.DataFrame,
    *,
    key: str,
    context: str,
    baseline_years: Iterable[int] | None = None,
    required_columns: Iterable[str] = (),
) -> None:
    """Report per-year hourly coverage of the ISD+NSRDB training series.

    The merged series is resampled to one row per hour, so coverage is
    ``present hours / (days_in_year * 24)``.  Interior years below
    ``TRAINING_COVERAGE_WARN`` are reported; interior years below
    ``TRAINING_COVERAGE_FAIL`` abort, since such thin years cannot train a
    stable weather generator (for example a file that only contains January).
    Edge years are exempt: a station record legitimately starts or stops
    mid-year.
    """
    columns = [key, *required_columns]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise SystemExit(
            f"ERROR: {context}: missing coverage columns: {', '.join(sorted(missing))}.")
    present = np.isfinite(frame[columns].apply(pd.to_numeric, errors="coerce")).all(axis=1)
    hours = present.groupby(frame.index.floor("h")).any()
    if not hours.any():
        raise SystemExit(f"ERROR: {context}: empty series, no coverage to report.")
    first, last = hours.index[0].year, hours.index[-1].year
    counts = hours.groupby(hours.index.year).sum()
    per_year = {
        year: counts.get(year, 0) / (pd.Timestamp(year, 12, 31).dayofyear * 24)
        for year in range(first, last + 1)
    }
    overall = sum(per_year.values()) / len(per_year)
    print(f"  [coverage] {context}: overall {overall:.0%} across "
          f"{min(per_year)}-{max(per_year)} ({len(per_year)} year(s))")

    if baseline_years is not None:
        requested = sorted(set(baseline_years))
        if not any(per_year.get(year, 0) > 0 for year in requested):
            raise SystemExit(f"ERROR: {context}: no observations in baseline {_format_int_ranges(requested)}.")
        thin_baseline = [year for year in requested
                         if per_year.get(year, 0) < TRAINING_COVERAGE_WARN]
        if thin_baseline:
            print(f"  [warning] {context}: baseline years below {TRAINING_COVERAGE_WARN:.0%} "
                  f"coverage: {_format_int_ranges(thin_baseline)}")
        available = hours.index[hours.to_numpy()].to_period("M").unique()
        requested_months = {(year, month) for year in requested for month in range(1, 13)}
        missing_months = requested_months - set(zip(available.year, available.month))
        if missing_months:
            print(f"  [warning] {context}: no usable observations for baseline months "
                  f"{_format_missing_months(missing_months)}")

    edge_years = {min(per_year), max(per_year)}
    interior = {y: r for y, r in per_year.items() if y not in edge_years}
    critical = {y: r for y, r in interior.items() if r < TRAINING_COVERAGE_FAIL}
    thin = {y: r for y, r in interior.items() if r < TRAINING_COVERAGE_WARN}

    if critical:
        lines = "\n".join(
            f"    - {y}: {r:.0%}" for y, r in sorted(critical.items()))
        raise SystemExit(
            f"ERROR: {context}: {len(critical)} year(s) below "
            f"{TRAINING_COVERAGE_FAIL:.0%} hourly coverage:\n{lines}\n"
            "Check the ISD/NSRDB inputs or extend the download year range.")
    if thin:
        lines = "\n".join(
            f"    - {y}: {r:.0%}" for y, r in sorted(thin.items()))
        print(f"  [warning] {context}: {len(thin)} year(s) below "
              f"{TRAINING_COVERAGE_WARN:.0%} hourly coverage:\n{lines}")


def report_training_coverage_from_csv(
    csv_path: str | Path,
    *,
    context: str,
    datetime_col: str = "DATE",
    key: str = "temp",
    baseline_years: Iterable[int] | None = None,
    required_columns: Iterable[str] = (),
) -> None:
    """Load a merged ISD+NSRDB CSV and run :func:`report_training_coverage`."""
    frame = pd.read_csv(csv_path, parse_dates=[datetime_col]).set_index(datetime_col)
    report_training_coverage(
        frame, key=key, context=context, baseline_years=baseline_years,
        required_columns=required_columns,
    )
