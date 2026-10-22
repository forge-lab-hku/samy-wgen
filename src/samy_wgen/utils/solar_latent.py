"""Bounded solar representation for fixed-local-standard-time weather series.

The two RS-VAR coordinates are logit clearness index and logit horizontal
beam share. Reconstructed components obey GHI = DHI + DNI*cos(zenith).
The geometry follows the calibrated ERA5/NSRDB input convention; its
extraterrestrial envelope is a modeling bound, not a universal observed cap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from samy_wgen.utils.solar_calibration import _cos_zenith
from samy_wgen.utils.time_utils import get_utc_offset


LATENT_COLUMNS = ("solar_kt_logit", "solar_beam_logit")
SOLAR_COLUMNS = ("GHI", "DNI", "DHI")


def solar_geometry(
    index: pd.DatetimeIndex, latitude: float, longitude: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cosine zenith, extraterrestrial normal, and horizontal W/m².

    The naive hourly index is fixed local standard time, as in the calibrated
    solar input. No daylight-saving adjustment is applied.
    """
    index = pd.DatetimeIndex(index)
    if index.tz is not None or index.hasnans or not index.is_monotonic_increasing:
        raise ValueError("Solar index must be ordered, timezone-naive fixed LST")
    offset = get_utc_offset(latitude=latitude, longitude=longitude)
    utc = pd.DatetimeIndex(index - pd.to_timedelta(offset, unit="h"), tz="UTC")
    cosz = _cos_zenith(latitude, longitude, utc)
    normal = 1361.0 * (1 + 0.033 * np.cos(2 * np.pi * utc.dayofyear.to_numpy() / 365))
    horizontal = normal * np.maximum(cosz, 0)
    return cosz, normal, horizontal


def encode_solar(
    frame: pd.DataFrame, latitude: float, longitude: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Screen historical solar and encode two bounded coordinates.

    Hours failing component closure, nonnegativity, or the chosen envelope
    become NaN and must be dropped from the joint RS-VAR fit. Endpoint clipping
    is only for the finite logit representation of otherwise valid 0/1 values.
    """
    if not set(SOLAR_COLUMNS).issubset(frame.columns):
        raise ValueError(f"Historical input requires {SOLAR_COLUMNS}")
    cosz, normal, horizontal = solar_geometry(frame.index, latitude, longitude)
    ghi, dni, dhi = (frame[name].to_numpy(dtype=float) for name in SOLAR_COLUMNS)
    daylight = cosz > 0
    closure = np.abs(ghi - dhi - dni * np.maximum(cosz, 0))
    valid = np.isfinite(ghi + dni + dhi) & (ghi >= 0) & (dni >= 0) & (dhi >= 0)
    valid &= closure <= 1
    valid &= np.where(daylight, ghi <= horizontal + 1, ghi <= 1)
    valid &= dni <= normal + 1
    beam = np.divide(dni * np.maximum(cosz, 0), ghi,
                     out=np.zeros_like(ghi), where=ghi > 1)
    valid &= (beam >= -0.01) & (beam <= 1.01)
    kt = np.divide(ghi, horizontal, out=np.zeros_like(ghi), where=daylight)
    eps = 1e-4
    encoded = pd.DataFrame({
        LATENT_COLUMNS[0]: np.where(daylight, logit(np.clip(kt, eps, 1 - eps)), 0.0),
        LATENT_COLUMNS[1]: np.where(daylight, logit(np.clip(beam, eps, 1 - eps)), 0.0),
    }, index=frame.index)
    encoded.loc[~valid, list(LATENT_COLUMNS)] = np.nan
    qa = {
        "loaded_hours": len(frame),
        "training_solar_valid_hours": int(valid.sum()),
        "training_solar_rejected_hours": int((~valid).sum()),
        "input_closure_gt1w_hours": int((closure > 1).sum()),
        "input_ghi_above_toa_gt1w_hours": int((ghi - horizontal > 1).sum()),
    }
    return encoded, qa


def decode_solar(
    latent: pd.DataFrame, latitude: float, longitude: float,
) -> pd.DataFrame:
    """Reconstruct nonnegative, bounded, exactly closing solar components."""
    if not set(LATENT_COLUMNS).issubset(latent.columns):
        raise ValueError(f"Latent input requires {LATENT_COLUMNS}")
    values = latent[list(LATENT_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Latent solar coordinates must be finite")
    cosz, normal, horizontal = solar_geometry(latent.index, latitude, longitude)
    kt = expit(values[:, 0])
    beam = expit(values[:, 1])
    ghi = horizontal * kt
    dni = np.where(cosz > 0, normal * kt * beam, 0.0)
    dhi = ghi - dni * np.maximum(cosz, 0)
    return pd.DataFrame({"GHI": ghi, "DNI": dni, "DHI": dhi}, index=latent.index)


def observed_monthly_targets(
    observed: pd.DataFrame, latitude: float, longitude: float,
    baseline_start: int = 1991, baseline_end: int = 2010,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Get 12 screened GHI/DNI monthly means with equal hour-of-day weight."""
    subset = observed.loc[observed.index.year >= baseline_start]
    subset = subset.loc[subset.index.year <= baseline_end]
    if subset.empty:
        raise ValueError("No historical solar records in baseline years")
    encoded, screen = encode_solar(subset, latitude, longitude)
    subset = subset.loc[encoded.notna().all(axis=1)]
    records = []
    for month in range(1, 13):
        values = subset.loc[subset.index.month == month]
        if values.empty:
            raise ValueError(f"No accepted historical solar in month {month}")
        record = {"month": month, "valid_observed_hours": len(values)}
        for name in ("GHI", "DNI"):
            by_hour = values[name].groupby(values.index.hour).mean()
            hours = np.arange(24)
            x = by_hour.index.to_numpy(dtype=float)
            y = by_hour.to_numpy(dtype=float)
            profile = np.interp(hours, np.r_[x - 24, x, x + 24], np.tile(y, 3))
            record[f"observed_{name}_mean_w_m2"] = float(profile.mean())
        records.append(record)
    return pd.DataFrame(records).set_index("month"), screen


def _solve_shift(logits: np.ndarray, weights: np.ndarray, target: float) -> float:
    """Solve a common logit shift for a physically feasible weighted sum."""
    capacity = float(weights.sum())
    if not np.isfinite(target) or target < 0 or target > capacity + 1e-6:
        raise ValueError(f"Solar target {target} outside 0..{capacity}")
    lo, hi = -50.0, 50.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if float(np.dot(weights, expit(logits + mid))) < target:
            lo = mid
        else:
            hi = mid
    shift = (lo + hi) / 2
    achieved = float(np.dot(weights, expit(logits + shift)))
    if abs(achieved - target) > max(1e-4, target * 1e-8):
        raise ValueError("Could not match feasible solar target")
    return shift


def calibrate_baseline(
    base: pd.DataFrame, targets: pd.DataFrame, latitude: float, longitude: float,
    generated_fit_years: tuple[int, int] = (2025, 2044),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit 12 GHI and 12 DNI shifts to screened historical monthly means.

    ``base`` is generated unsignaled solar. The same monthly shifts are applied
    to every forecast year. The generated fit years are not a held-out test.
    """
    if not set(SOLAR_COLUMNS).issubset(base.columns):
        raise ValueError(f"Base solar requires {SOLAR_COLUMNS}")
    cosz, normal, horizontal = solar_geometry(base.index, latitude, longitude)
    ghi = base.GHI.to_numpy(dtype=float)
    dni = base.DNI.to_numpy(dtype=float)
    kt = np.divide(ghi, horizontal, out=np.full_like(ghi, 0.5), where=horizontal > 0)
    beam = np.divide(dni * np.maximum(cosz, 0), ghi,
                     out=np.full_like(ghi, 0.5), where=ghi > 0)
    eps = 1e-12
    kt_logit = logit(np.clip(kt, eps, 1 - eps))
    beam_logit = logit(np.clip(beam, eps, 1 - eps))
    months = base.index.month.to_numpy()
    fit = ((base.index.year >= generated_fit_years[0])
           & (base.index.year <= generated_fit_years[1]))
    corrected_kt = kt_logit.copy()
    corrected_beam = beam_logit.copy()
    records = []
    for month in range(1, 13):
        fit_month = fit & (months == month)
        if not fit_month.any():
            raise ValueError(f"No generated calibration hours in month {month}")
        count = int(fit_month.sum())
        target_ghi = float(targets.loc[month, "observed_GHI_mean_w_m2"]) * count
        ghi_shift = _solve_shift(kt_logit[fit_month], horizontal[fit_month], target_ghi)
        corrected_kt[months == month] += ghi_shift
        fitted_kt = expit(corrected_kt[fit_month])
        dni_weights = normal[fit_month] * fitted_kt * (cosz[fit_month] > 0)
        target_dni = float(targets.loc[month, "observed_DNI_mean_w_m2"]) * count
        beam_shift = _solve_shift(beam_logit[fit_month], dni_weights, target_dni)
        corrected_beam[months == month] += beam_shift
        records.append({
            "month": month, "generated_calibration_hours": count,
            "observed_GHI_mean_w_m2": float(targets.loc[month, "observed_GHI_mean_w_m2"]),
            "observed_DNI_mean_w_m2": float(targets.loc[month, "observed_DNI_mean_w_m2"]),
            "ghi_logit_shift": ghi_shift, "beam_logit_shift": beam_shift,
            "calibrated_GHI_mean_w_m2": float(np.mean(horizontal[fit_month] * fitted_kt)),
            "calibrated_DNI_mean_w_m2": float(np.mean(dni_weights * expit(corrected_beam[fit_month]))),
            "valid_observed_hours": int(targets.loc[month, "valid_observed_hours"]),
        })
    corrected = pd.DataFrame({LATENT_COLUMNS[0]: corrected_kt,
                              LATENT_COLUMNS[1]: corrected_beam}, index=base.index)
    return decode_solar(corrected, latitude, longitude), corrected, pd.DataFrame(records)


def monthly_rsds_ratios(
    daily: pd.DataFrame, forecast_index: pd.DatetimeIndex,
    baseline_start: int = 1991, baseline_end: int = 2010,
    trailing_years: int = 20,
) -> pd.Series:
    """Return trailing monthly CORDEX rsds means / baseline monthly means."""
    if "rsds" not in daily or daily.index.has_duplicates:
        raise ValueError("CORDEX daily input requires unique rsds timestamps")
    rsds = daily["rsds"].astype(float)
    if not np.isfinite(rsds.to_numpy()).all():
        raise ValueError("CORDEX daily rsds must be finite")
    monthly = rsds.groupby([rsds.index.year, rsds.index.month]).mean()
    reference = rsds.loc[(rsds.index.year >= baseline_start)
                         & (rsds.index.year <= baseline_end)]
    baseline = reference.groupby(reference.index.month).mean()
    if set(baseline.index) != set(range(1, 13)) or (baseline <= 0).any():
        raise ValueError("Incomplete or nonpositive CORDEX solar baseline")
    result = {}
    for period in pd.DatetimeIndex(forecast_index).to_period("M").unique():
        year, month = period.year, period.month
        window = monthly.loc[(slice(year - trailing_years + 1, year), month)]
        if len(window) != trailing_years or not np.isfinite(window).all():
            raise ValueError(f"Incomplete trailing CORDEX rsds window for {period}")
        result[period] = float(window.mean() / baseline[month])
    return pd.Series(result, name="rsds_trailing_monthly_ratio")


def apply_monthly_solar_signal(
    latent: pd.DataFrame, latitude: float, longitude: float, ratios: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match each month's CORDEX GHI energy through a bounded logit shift.

    Raises when a target is outside the feasible extraterrestrial envelope;
    neither GHI nor DNI is clipped after reconstruction.
    """
    base = decode_solar(latent, latitude, longitude)
    cosz, normal, horizontal = solar_geometry(latent.index, latitude, longitude)
    logit_kt = latent[LATENT_COLUMNS[0]].to_numpy(dtype=float)
    beam = expit(latent[LATENT_COLUMNS[1]].to_numpy(dtype=float))
    new_ghi = np.zeros(len(latent), dtype=float)
    records = []
    periods = latent.index.to_period("M")
    for period in periods.unique():
        where = np.flatnonzero(periods == period)
        if period not in ratios.index:
            raise ValueError(f"Missing CORDEX monthly ratio for {period}")
        ratio = float(ratios.loc[period])
        if not np.isfinite(ratio) or ratio < 0:
            raise ValueError(f"Invalid CORDEX monthly ratio for {period}: {ratio}")
        base_energy = float(base.GHI.to_numpy()[where].sum())
        target = base_energy * ratio
        available = horizontal[where]
        shift = _solve_shift(logit_kt[where], available, target)
        new_ghi[where] = available * expit(logit_kt[where] + shift)
        records.append({
            "month": str(period), "ratio": ratio,
            "base_ghi_sum_wh_m2": base_energy,
            "target_ghi_sum_wh_m2": target,
            "achieved_ghi_sum_wh_m2": float(new_ghi[where].sum()),
            "toa_capacity_wh_m2": float(available.sum()), "logit_shift": shift,
        })
    new_kt = np.divide(new_ghi, horizontal, out=np.zeros_like(new_ghi), where=horizontal > 0)
    dni = np.where(cosz > 0, normal * new_kt * beam, 0.0)
    dhi = new_ghi - dni * np.maximum(cosz, 0)
    return pd.DataFrame({"GHI": new_ghi, "DNI": dni, "DHI": dhi}, index=latent.index), pd.DataFrame(records)


def solar_qa(frame: pd.DataFrame, latitude: float, longitude: float) -> dict[str, float | int]:
    """Measure serialized-series bounds, night values, and component closure."""
    cosz, normal, horizontal = solar_geometry(frame.index, latitude, longitude)
    ghi, dni, dhi = (frame[name].to_numpy(dtype=float) for name in SOLAR_COLUMNS)
    closure = np.abs(ghi - dhi - dni * np.maximum(cosz, 0))
    return {
        "nonfinite_hours": int((~np.isfinite(frame[list(SOLAR_COLUMNS)].to_numpy())).any(axis=1).sum()),
        "negative_hours": int((frame[list(SOLAR_COLUMNS)].to_numpy() < -1e-6).any(axis=1).sum()),
        "night_ghi_gt1_hours": int(((cosz <= 0) & (ghi > 1)).sum()),
        "ghi_above_toa_gt1_hours": int((ghi - horizontal > 1).sum()),
        "dni_above_toa_gt1_hours": int((dni - normal > 1).sum()),
        "closure_gt1_hours": int((closure > 1).sum()),
        "closure_max_w_m2": float(closure.max()),
    }
