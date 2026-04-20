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

import logging
import os
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from samy_wgen.utils.city_utils import resolve_city

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _resolve_city_coordinates(city: str) -> tuple[float, float, str]:
    """Return latitude, longitude, and canonical city label for a requested city."""
    record = resolve_city(city)
    return record.latitude, record.longitude, record.city_ascii


@lru_cache(maxsize=1)
def _get_timezone_finder():
    # timezonefinder imports numba-decorated modules at import time; under some
    # conda setups it needs an explicit writable cache directory.
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba-cache"))
    from timezonefinder import TimezoneFinder

    return TimezoneFinder()


def get_timezone_name(
    city: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> str:
    """Return the IANA timezone name for a city or coordinate pair."""
    if latitude is None or longitude is None:
        if city is None:
            raise ValueError("Either city or both latitude/longitude must be provided.")
        latitude, longitude, _ = _resolve_city_coordinates(city)

    tz_name = _get_timezone_finder().timezone_at(lat=latitude, lng=longitude)
    if tz_name is None:
        raise ValueError(
            f"Could not determine timezone for lat={latitude}, lon={longitude}."
        )
    return tz_name


def get_utc_offset(
    city: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> float:
    """Return fixed LST offset from UTC (hours) for a city or coordinate pair.

    The returned offset is the standard-time offset, evaluated on January 15 to
    avoid daylight-saving shifts.
    """
    tz_name = get_timezone_name(city=city, latitude=latitude, longitude=longitude)
    standard_dt = datetime(2020, 1, 15, 12, 0, tzinfo=ZoneInfo(tz_name))
    offset = standard_dt.utcoffset()
    if offset is None:
        raise ValueError(f"Could not determine UTC offset for timezone '{tz_name}'.")
    return offset.total_seconds() / 3600.0


def utc_to_lst(
    df: pd.DataFrame,
    city: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> pd.DataFrame:
    """Shift a UTC-indexed DataFrame to Local Standard Time (no DST).

    This is the single UTC→LST conversion point in the pipeline.
    Call this on raw ISD output before merging with other data sources.

    Args:
        df: DataFrame with a timezone-naive or UTC DatetimeIndex.
        city: City name, used to look up coordinates from cities_config.csv.
        latitude: Optional explicit latitude.
        longitude: Optional explicit longitude.

    Returns:
        New DataFrame with index shifted to LST (timezone-naive).
    """
    offset_hours = get_utc_offset(city=city, latitude=latitude, longitude=longitude)
    offset = pd.Timedelta(hours=offset_hours)
    out = df.copy()
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index + offset
    label = city or f"lat={latitude}, lon={longitude}"
    logger.debug("utc_to_lst: shifted %s by %.1fh for %s", len(df), offset_hours, label)
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

    if len(df) > 1:
        diffs = df.index.to_series().diff().dropna()
        expected = diffs.mode().iloc[0]
        if expected <= pd.Timedelta(hours=2):
            dst_jumps = ((diffs == pd.Timedelta(hours=2)) | (diffs == pd.Timedelta(0))).sum()
            if dst_jumps > 0:
                raise ValueError(
                    f"{label}{dst_jumps} potential DST transition(s) detected "
                    "(consecutive timestamps differ by 0h or 2h). "
                    "Ensure input uses fixed LST offset, not wall-clock local time."
                )
