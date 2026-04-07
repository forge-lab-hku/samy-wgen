"""
Timezone utilities for the samy_wgen pipeline.

Strategy: All internal processing uses LST (Local Standard Time, no DST).
UTC→LST conversion happens exactly once at the data ingestion boundary,
after ISD parsing and before merging with NSRDB/ERA5 and feeding the model.

Typical usage:
    from samy_wgen.utils.time_utils import utc_to_lst, assert_is_lst, get_utc_offset

    isd_df = parse_isd(...)          # UTC output from parse_ISD.py
    isd_lst = utc_to_lst(isd_df, "Miami")   # convert once at boundary
    assert_is_lst(nsrdb_df, "NSRDB")         # verify other sources
    assert_is_lst(era5_df, "ERA5")
    merged = merge(isd_lst, nsrdb_df, era5_df)  # all LST from here on
"""

from __future__ import annotations

import importlib.resources
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _load_city_timezones() -> dict[str, float]:
    """Return {city: utc_offset_hours} from bundled config.yaml."""
    ref = importlib.resources.files("samy_wgen.config").joinpath("config.yaml")
    with ref.open("r") as f:
        cfg = yaml.safe_load(f)
    raw = cfg.get("city_timezones", {})
    # config stores IANA names; we need fixed offsets.
    # Extend this mapping as new cities are added.
    iana_to_offset: dict[str, float] = {
        "America/Los_Angeles": -8.0,
        "America/New_York": -5.0,
        "America/Toronto": -5.0,
        "America/Montreal": -5.0,
        "America/Vancouver": -8.0,
        "America/Chicago": -6.0,
        "America/Denver": -7.0,
        "America/Phoenix": -7.0,   # Arizona — no DST, fixed MST
    }
    result: dict[str, float] = {}
    for city, iana in raw.items():
        if iana in iana_to_offset:
            result[city] = iana_to_offset[iana]
        else:
            raise ValueError(
                f"No fixed UTC offset known for timezone '{iana}' (city: {city}). "
                "Add it to iana_to_offset in time_utils.py."
            )
    return result


def get_utc_offset(city: str) -> float:
    """Return fixed LST offset from UTC (hours) for a city.

    Examples:
        Miami  → -5.0  (EST, America/New_York standard offset)
        LA     → -8.0  (PST, America/Los_Angeles standard offset)
        Toronto → -5.0

    Args:
        city: City name matching keys in config.yaml city_timezones section
              (e.g. "Miami", "Los_Angeles", "Toronto", "Montreal").

    Returns:
        UTC offset in hours (negative for Americas).

    Raises:
        KeyError: If city is not found in config.
    """
    offsets = _load_city_timezones()
    if city not in offsets:
        available = ", ".join(sorted(offsets))
        raise KeyError(
            f"City '{city}' not found in config city_timezones. "
            f"Available: {available}"
        )
    return offsets[city]


def utc_to_lst(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Shift a UTC-indexed DataFrame to Local Standard Time (no DST).

    This is the single UTC→LST conversion point in the pipeline.
    Call this on raw ISD output before merging with other data sources.

    Args:
        df: DataFrame with a timezone-naive or UTC DatetimeIndex.
        city: City name (must match config.yaml city_timezones).

    Returns:
        New DataFrame with index shifted to LST (timezone-naive).
    """
    offset_hours = get_utc_offset(city)
    offset = pd.Timedelta(hours=offset_hours)
    out = df.copy()
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index + offset
    logger.debug("utc_to_lst: shifted %s by %.1fh for %s", len(df), offset_hours, city)
    return out


def assert_is_lst(df: pd.DataFrame, name: str = "") -> None:
    """Raise ValueError if the DataFrame index looks like it contains UTC or DST jumps.

    Checks:
    1. Index must be timezone-naive (tz-aware → likely UTC, not yet converted).
    2. No duplicate timestamps (UTC→LST shift should be monotone).
    3. No hour jumps of ±1h that suggest DST transitions.

    Args:
        df: DataFrame to validate.
        name: Descriptive name for error messages (e.g. "NSRDB", "ERA5").

    Raises:
        ValueError: If any check fails.
    """
    label = f"[{name}] " if name else ""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{label}Index is not a DatetimeIndex.")

    if df.index.tz is not None:
        raise ValueError(
            f"{label}Index is timezone-aware ({df.index.tz}). "
            "Expected naive LST. Call utc_to_lst() first for ISD data, "
            "or strip tz info if source is already in local time."
        )

    if df.index.duplicated().any():
        n = df.index.duplicated().sum()
        raise ValueError(
            f"{label}{n} duplicate timestamps found. "
            "This may indicate a timezone conversion error."
        )

    # Check for ±1h jumps indicative of DST (only for sub-daily frequency)
    if len(df) > 1:
        diffs = df.index.to_series().diff().dropna()
        expected = diffs.mode().iloc[0]
        if expected <= pd.Timedelta(hours=2):
            # Only check for hourly-or-finer data
            dst_jumps = ((diffs == pd.Timedelta(hours=2)) | (diffs == pd.Timedelta(0))).sum()
            if dst_jumps > 0:
                raise ValueError(
                    f"{label}{dst_jumps} potential DST transition(s) detected "
                    "(consecutive timestamps differ by 0h or 2h). "
                    "Ensure input uses fixed LST offset, not wall-clock local time."
                )
