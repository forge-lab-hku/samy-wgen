#!/usr/bin/env python3
"""
Solar calculation utilities for nighttime detection and solar angle calculations.

This module provides shared functions for calculating solar angles and
identifying nighttime hours, ensuring consistency across different scripts.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union


def calculate_solar_zenith_angle(timestamps: pd.DatetimeIndex, lat_deg: float) -> np.ndarray:
    """
    Calculate solar zenith angle cosine for given timestamps and latitude.

    This calculation assumes timestamps are in LOCAL TIME (not UTC).
    The formula assumes that 12:00 local time is solar noon.

    Args:
        timestamps: DatetimeIndex in local time
        lat_deg: Latitude in degrees (positive for North, negative for South)

    Returns:
        cos_zenith: Cosine of solar zenith angle (ndarray)
                   - Positive values indicate sun above horizon
                   - Negative values indicate sun below horizon
                   - cos(zenith) ≈ 1.0 means sun directly overhead
                   - cos(zenith) ≈ 0.0 means sun at horizon

    Notes:
        - Uses simplified declination formula (Spencer, 1971)
        - Assumes 12:00 local time = solar noon (ignores equation of time)
        - Suitable for approximate solar position calculations

    References:
        Spencer, J.W. (1971). "Fourier series representation of the position of the sun."
        Search, 2(5), 172.
    """
    lat_rad = np.radians(lat_deg)

    # Extract hour (0-23) and day of year (1-365)
    hours = timestamps.hour.values
    doy = timestamps.dayofyear.values

    # Solar declination angle (simplified formula)
    # Assumes perihelion is around day 81 (March 21, vernal equinox)
    declination = np.radians(23.45) * np.sin(2 * np.pi * (doy - 81) / 365)

    # Hour angle: measures time from solar noon
    # 15 degrees per hour (360° / 24 hours)
    # Assumes 12:00 local time = solar noon
    hour_angle_deg = (hours - 12) * 15
    hour_angle_rad = np.radians(hour_angle_deg)

    # Solar zenith angle (using spherical trigonometry)
    # cos(zenith) = sin(lat) × sin(decl) + cos(lat) × cos(decl) × cos(hour_angle)
    cos_zenith = (np.sin(lat_rad) * np.sin(declination) +
                  np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle_rad))

    return cos_zenith


def calculate_night_mask(timestamps: pd.DatetimeIndex,
                         lat_deg: float,
                         threshold: float = 0.065) -> np.ndarray:
    """
    Calculate nighttime mask based on solar zenith angle.

    Args:
        timestamps: DatetimeIndex in local time
        lat_deg: Latitude in degrees
        threshold: Cosine of zenith angle threshold for night detection
                  Default 0.065 corresponds to zenith angle > 86.3°
                  (sun ~3.7° below horizon, including civil twilight)

    Returns:
        night_mask: Boolean array, True for nighttime hours

    Notes:
        Default threshold (0.065) is consistent with solar constraints
        in rsvar.py

        Threshold interpretation:
        - cos(zenith) = 0.065 → zenith = 86.3° → sun 3.7° below horizon
        - This captures nighttime including civil twilight
        - Solar radiation should be zero for cos(zenith) ≤ 0.065
    """
    cos_zenith = calculate_solar_zenith_angle(timestamps, lat_deg)
    night_mask = cos_zenith <= threshold
    return night_mask


def get_nighttime_statistics(df: pd.DataFrame,
                            lat_deg: float,
                            ghi_column: str = 'GHI',
                            threshold: float = 0.065) -> dict:
    """
    Calculate nighttime GHI statistics based on solar angle.

    Args:
        df: DataFrame with DatetimeIndex (local time) and GHI column
        lat_deg: Latitude in degrees
        ghi_column: Name of GHI column (default: 'GHI')
        threshold: Night detection threshold (default: 0.065)

    Returns:
        Dictionary containing:
        - 'night_mask': Boolean array for nighttime hours
        - 'night_hours': Total number of nighttime hours
        - 'night_percentage': Percentage of hours that are nighttime
        - 'mean_ghi': Mean GHI during nighttime (should be ~0.0)
        - 'max_ghi': Maximum GHI during nighttime
        - 'non_zero_count': Count of nighttime hours with GHI > 0

    Example:
        >>> stats = get_nighttime_statistics(df, lat=34.05)
        >>> print(f"Nighttime mean GHI: {stats['mean_ghi']:.4f} W/m²")
        >>> print(f"Nighttime hours: {stats['night_hours']:,} ({stats['night_percentage']:.1f}%)")
    """
    if ghi_column not in df.columns:
        raise ValueError(f"Column '{ghi_column}' not found in DataFrame")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    # Calculate night mask
    night_mask = calculate_night_mask(df.index, lat_deg, threshold)

    # Extract nighttime GHI values
    night_ghi = df.loc[night_mask, ghi_column]

    # Calculate statistics
    stats = {
        'night_mask': night_mask,
        'night_hours': night_mask.sum(),
        'night_percentage': night_mask.sum() / len(df) * 100,
        'mean_ghi': night_ghi.mean(),
        'max_ghi': night_ghi.max(),
        'min_ghi': night_ghi.min(),
        'non_zero_count': (night_ghi > 0).sum(),
        'threshold_used': threshold,
    }

    return stats


def calculate_sunrise_sunset_hours(date: pd.Timestamp, lat_deg: float) -> Tuple[float, float]:
    """
    Calculate approximate sunrise and sunset times (in local hours).

    Args:
        date: Date for calculation
        lat_deg: Latitude in degrees

    Returns:
        (sunrise_hour, sunset_hour): Tuple of floats representing hours (0-24)
                                     Returns (None, None) for polar night
                                     Returns (0, 24) for polar day

    Notes:
        This is a simplified calculation that:
        - Ignores equation of time
        - Ignores atmospheric refraction
        - Suitable for approximate calculations only

    Example:
        >>> sunrise, sunset = calculate_sunrise_sunset_hours(pd.Timestamp('2023-06-21'), 34.05)
        >>> print(f"Sunrise: {sunrise:.2f}h ({int(sunrise)}:{int((sunrise%1)*60):02d})")
        >>> print(f"Sunset: {sunset:.2f}h ({int(sunset)}:{int((sunset%1)*60):02d})")
    """
    lat_rad = np.radians(lat_deg)
    doy = date.timetuple().tm_yday

    # Solar declination
    declination = np.radians(23.45) * np.sin(2 * np.pi * (doy - 81) / 365)

    # Calculate hour angle at sunrise/sunset
    # At sunrise/sunset: cos(zenith) = 0
    # cos(hour_angle) = -tan(lat) × tan(decl)
    cos_hour_angle = -np.tan(lat_rad) * np.tan(declination)

    if cos_hour_angle > 1:
        # Polar night (sun never rises)
        return None, None
    elif cos_hour_angle < -1:
        # Polar day (sun never sets)
        return 0.0, 24.0
    else:
        # Normal case
        hour_angle_deg = np.degrees(np.arccos(cos_hour_angle))
        sunrise_hour = 12 - hour_angle_deg / 15
        sunset_hour = 12 + hour_angle_deg / 15
        return sunrise_hour, sunset_hour


def validate_local_time(df: pd.DataFrame, lat_deg: float, ghi_column: str = 'GHI') -> dict:
    """
    Validate that timestamps are in local time by checking GHI peak hour.

    A properly timezone-converted dataset should have peak GHI around 12:00 local time.

    Args:
        df: DataFrame with DatetimeIndex and GHI column
        lat_deg: Latitude in degrees
        ghi_column: Name of GHI column (default: 'GHI')

    Returns:
        Dictionary containing:
        - 'peak_hour': Hour with maximum average GHI
        - 'is_likely_local_time': Boolean, True if peak is between 11:00-13:00
        - 'hourly_mean_ghi': Series of mean GHI by hour

    Example:
        >>> validation = validate_local_time(df, lat=34.05)
        >>> if validation['is_likely_local_time']:
        >>>     print(f"✓ Data appears to be in local time (peak at {validation['peak_hour']}:00)")
        >>> else:
        >>>     print(f"⚠ Warning: GHI peak at {validation['peak_hour']}:00, may not be local time")
    """
    if ghi_column not in df.columns:
        raise ValueError(f"Column '{ghi_column}' not found in DataFrame")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    # Calculate hourly mean GHI
    hourly_mean_ghi = df.groupby(df.index.hour)[ghi_column].mean()
    peak_hour = hourly_mean_ghi.idxmax()

    # Check if peak is around solar noon (11:00-13:00)
    is_likely_local_time = 11 <= peak_hour <= 13

    return {
        'peak_hour': peak_hour,
        'is_likely_local_time': is_likely_local_time,
        'hourly_mean_ghi': hourly_mean_ghi,
        'peak_ghi': hourly_mean_ghi.max(),
    }

