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

import pandas as pd

# Full monthly rsds trailing window required by the bounded solar calculation.
CMIP6_SOLAR_TRAILING_YEARS = 20


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

    Daily temperature/pressure must have at least one valid value in every
    requested month (not every day), which matches how the DOY climatology
    consumes them and avoids false failures from a few missing days.
    """
    if frame is None:
        return [f"{model}: missing or unreadable {label} file {path.name}"]
    issues: list[str] = []
    temperature_column = next((column for column in ("tas", "tas_C")
                               if column in frame.columns), None)
    pressure_column = next((column for column in ("psl", "ps", "surface_air_pressure")
                            if column in frame.columns), None)
    missing_columns = []
    if temperature_column is None:
        missing_columns.append("tas")
    if pressure_column is None:
        missing_columns.append("psl (or ps/surface_air_pressure)")
    if missing_columns:
        issues.append(
            f"{model}: {label} file {path.name} is missing required variable(s): "
            f"{', '.join(missing_columns)}")
        return issues

    times = pd.to_datetime(frame["time"], errors="coerce")
    if times.isna().any():
        issues.append(f"{model}: {label} file {path.name} has invalid timestamps")
        return issues
    requested = set(required_years)
    expected_months = {(year, month) for year in requested for month in range(1, 13)}
    for column in (temperature_column, pressure_column):
        valid = frame[column].notna()
        available = set(zip(times[valid].dt.year, times[valid].dt.month))
        missing_pairs = expected_months - available
        if missing_pairs:
            issues.append(
                f"{model}: {label} file {path.name} does not cover {column} for "
                f"month(s) {_format_missing_months(missing_pairs)} "
                f"(required {_format_int_ranges(sorted(requested))}, all months)")
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

    valid = frame["rsds"].notna()
    available = set(zip(times[valid].dt.year, times[valid].dt.month))
    issues: list[str] = []

    baseline_set = set(baseline_years)
    baseline_months = {month for year, month in available if year in baseline_set}
    missing_baseline_months = sorted(set(range(1, 13)) - baseline_months)
    if missing_baseline_months:
        issues.append(
            f"{model}: {label} has no baseline rsds for month(s) "
            f"{_format_int_ranges(missing_baseline_months)} in "
            f"{baseline_years[0]}-{baseline_years[-1]}")

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
            historical_daily, baseline_years))
        issues.extend(_daily_coverage_issues(
            model, "scenario daily", scenario_daily_path,
            scenario_daily, forecast_years))

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
