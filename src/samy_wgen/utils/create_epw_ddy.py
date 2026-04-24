#!/usr/bin/env python3
"""
Convert TMY CSV files to EPW format and generate DDY files using EnergyPlus.

This script:
1. Reads TMY CSV files from output/TMY directory
2. Converts each TMY to EPW format with proper metadata
3. Uses EnergyPlus to generate DDY files from EPW
4. Saves EPW and DDY files to output/epw directory

Usage:
    python create_epw_ddy.py [--tmy-dir DIR] [--output-dir DIR] [--config-path FILE] [--energyplus-path FILE]

Examples:
    # Use default paths
    python create_epw_ddy.py

    # Specify custom paths
    python create_epw_ddy.py --tmy-dir output/TMY --output-dir output/epw --config-path data/cities_config_final.csv
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba-cache"))

from timezonefinder import TimezoneFinder
import pytz
import argparse
from samy_wgen.utils.calculate_weather_var import calc_dewpoint_from_Q, calc_RH_from_dewpoint, calc_wetbulb


def get_timezone_offset(latitude, longitude):
    """
    Get UTC offset for a given latitude/longitude using timezonefinder.

    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees

    Returns:
        UTC offset in hours (float)
    """
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=latitude, lng=longitude)

    if tz_name is None:
        print(f"  Warning: Could not find timezone for lat={latitude}, lon={longitude}")
        print(f"  Using approximate timezone from longitude")
        return round(longitude / 15.0)

    # Get UTC offset for the timezone (using a date in winter to avoid DST)
    tz = pytz.timezone(tz_name)
    # Use January 15 to avoid DST
    dt = datetime(2020, 1, 15, 12, 0, 0)
    offset = tz.utcoffset(dt).total_seconds() / 3600.0

    return offset


def load_city_metadata(config_path):
    """
    Load city configuration with latitude, longitude, and country info.

    Args:
        config_path: Path to cities_config_final.csv

    Returns:
        Dictionary mapping city names to metadata
    """
    cities_df = pd.read_csv(config_path)
    city_metadata = {}

    for _, row in cities_df.iterrows():
        city_metadata[row['city']] = {
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'country': row['country']
        }

    return city_metadata


def epw_header(city, state, country, lat, lon, tz, elevation_m=0, wmo_code="722950"):
    """
    Generate EPW header lines.

    Args:
        city: City name (e.g., "Los Angeles")
        state: State/Province abbreviation (e.g., "CA")
        country: Country code (e.g., "USA")
        lat: Latitude in degrees
        lon: Longitude in degrees
        tz: UTC offset in hours
        elevation_m: Elevation in meters
        wmo_code: WMO station code

    Returns:
        Header string with newline
    """
    lines = []
    lines.append(
        f"LOCATION,{city},{state},{country},TMY,{wmo_code},{lat:.3f},{lon:.3f},{tz:.1f},{elevation_m:.1f}"
    )
    lines.append("DESIGN CONDITIONS,0")
    lines.append("TYPICAL/EXTREME PERIODS,0")
    lines.append("GROUND TEMPERATURES,0")
    lines.append("HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0")
    lines.append("COMMENTS 1,Generated from TMY data (1991-2020)")
    lines.append("COMMENTS 2,Created by create_epw_ddy.py")
    lines.append("DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31")
    return "\n".join(lines) + "\n"


def convert_tmy_to_epw(tmy_csv_path, output_epw_path, city_name, city_metadata):
    """
    Convert TMY CSV to EPW format.

    Args:
        tmy_csv_path: Path to input TMY CSV file
        output_epw_path: Path to output EPW file
        city_name: Name of the city
        city_metadata: Dictionary with city metadata

    Returns:
        Path to created EPW file
    """
    print(f"\n{city_name}:")
    print(f"  Converting TMY to EPW...")

    # Read TMY CSV
    df = pd.read_csv(tmy_csv_path, index_col=0, parse_dates=True)

    # Detect file type based on columns
    is_stmy = 'specific_humidity' in df.columns
    has_wind_dir = 'WindDir_deg' in df.columns

    if is_stmy:
        print(f"  Detected sTMY format (with specific_humidity, without WindDir_deg)")
    else:
        print(f"  Detected TMY format (with WindDir_deg, without specific_humidity)")

    # Get city metadata
    if city_name not in city_metadata:
        print(f"  Warning: City {city_name} not found in config, using default location")
        lat, lon, country = 34.05, -118.24, "USA"  # Default to LA coordinates
        state = "CA"
        wmo_code = "722950"
        tz = -8.0
    else:
        lat = city_metadata[city_name]['latitude']
        lon = city_metadata[city_name]['longitude']
        country = city_metadata[city_name]['country']
        tz = get_timezone_offset(lat, lon)

        # Determine state from city name (simple mapping)
        state_map = {
            'Los_Angeles': 'CA',
            'Miami': 'FL',
            'Toronto': 'ON',
            'Montreal': 'QC'
        }
        state = state_map.get(city_name, '')

        # WMO code mapping
        wmo_map = {
            'Los_Angeles': '722950',
            'Miami': '722020',
            'Toronto': '716240',
            'Montreal': '716270'
        }
        wmo_code = wmo_map.get(city_name, '000000')

    # Ensure we have 8760 hours (no leap day)
    if len(df) > 8760:
        # Remove Feb 29 if present
        mask_leap = ~((df.index.month == 2) & (df.index.day == 29))
        df = df[mask_leap]

    if len(df) != 8760:
        print(f"  Warning: Expected 8760 hours, got {len(df)}")
        # Try to handle missing hours
        if len(df) < 8760:
            print(f"  Warning: Missing hours will be forward-filled")

    # Map TMY columns to EPW format
    temp_c = df['DryBulb_C']

    # Handle dewpoint: sTMY might have dewpoint directly or need to calculate from specific_humidity
    if 'DewPoint_C' in df.columns:
        dewpoint_c = df['DewPoint_C']
    elif is_stmy and 'specific_humidity' in df.columns:
        # Calculate dewpoint from specific humidity and pressure using utility function
        print(f"  Calculating dewpoint from specific_humidity and pressure...")
        dewpoint_c = pd.Series(index=df.index, dtype=float)

        q = df['specific_humidity']  # kg/kg
        pressure_pa = df['Pressure_Pa']

        valid_mask = q.notna() & pressure_pa.notna() & temp_c.notna()
        if valid_mask.any():
            # Use calc_dewpoint_from_Q utility function
            # Convert pressure from Pa to hPa for the function
            q_valid = q[valid_mask]
            p_valid_hpa = pressure_pa[valid_mask] / 100.0  # Pa to hPa
            temp_valid = temp_c[valid_mask]

            td_calc = calc_dewpoint_from_Q(temp_valid, q_valid, p_valid_hpa)
            dewpoint_c[valid_mask] = td_calc

        n_calculated = valid_mask.sum()
        n_missing = (~valid_mask).sum()
        print(f"    Calculated {n_calculated} dewpoint values from specific_humidity")
        if n_missing > 0:
            print(f"    {n_missing} dewpoint values remain missing")
    else:
        print(f"  Warning: No dewpoint data available")
        dewpoint_c = pd.Series(index=df.index, dtype=float)

    # Relative humidity - directly from CSV or calculate from temp and dewpoint
    if 'RelHum_pct' in df.columns:
        rh = df['RelHum_pct']
        print(f"  Using RelHum_pct from CSV")
    else:
        print(f"  Warning: RelHum_pct not found in CSV, calculating from temperature and dewpoint")
        # Calculate RH from dry bulb temperature and dew point temperature using utility function
        rh = pd.Series(index=df.index, dtype=float)

        valid_mask = temp_c.notna() & dewpoint_c.notna()
        if valid_mask.any():
            T = temp_c[valid_mask]
            Td = dewpoint_c[valid_mask]

            # Use calc_RH_from_dewpoint utility function
            rh_calc = calc_RH_from_dewpoint(T, Td)
            rh[valid_mask] = rh_calc

            n_calculated = valid_mask.sum()
            n_missing = (~valid_mask).sum()
            print(f"    Calculated {n_calculated} RH values from temperature and dewpoint")
            if n_missing > 0:
                print(f"    {n_missing} RH values remain missing (will use EPW missing code 999)")
        else:
            print(f"    Warning: Cannot calculate RH - no valid temperature and dewpoint pairs")
    # Count missing values for reporting
    n_missing_temp = temp_c.isna().sum()
    n_missing_dewpoint = dewpoint_c.isna().sum()
    n_missing_rh = rh.isna().sum()

    if n_missing_temp > 0:
        print(f"  Info: {n_missing_temp} missing Dry Bulb Temperature values (will use EPW missing code 99.9)")
    if n_missing_dewpoint > 0:
        print(f"  Info: {n_missing_dewpoint} missing Dew Point Temperature values (will use EPW missing code 99.9)")
    if n_missing_rh > 0:
        print(f"  Info: {n_missing_rh} missing Relative Humidity values (will use EPW missing code 999)")

    pressure_pa = df['Pressure_Pa']

    # Detect and convert pressure unit if needed (hPa to Pa)
    if pressure_pa.notna().any():
        max_pressure = pressure_pa.max()
        if max_pressure < 2000:  # Likely in hPa (millibar)
            print(f"  Detected pressure in hPa (max={max_pressure:.1f}), converting to Pa...")
            pressure_pa = pressure_pa * 100
            min_p = pressure_pa.min()
            max_p = pressure_pa.max()
            print(f"  Pressure range after conversion: {min_p:.0f} - {max_p:.0f} Pa")
            if min_p < 85000 or max_p > 110000:
                print(f"  Warning: Pressure outside typical range (85000-110000 Pa)")

    n_missing_pressure = pressure_pa.isna().sum()
    if n_missing_pressure > 0:
        print(f"  Info: {n_missing_pressure} missing Atmospheric Pressure values (will use EPW missing code 999999)")

    # Wind variables - handle missing WindDir_deg in sTMY files
    wind_speed = df['WindSpeed_mps']

    if has_wind_dir:
        wind_dir = df['WindDir_deg']
        n_missing_winddir = wind_dir.isna().sum()
        if n_missing_winddir > 0:
            print(f"  Info: {n_missing_winddir} missing Wind Direction values (will use EPW missing code 999)")
    else:
        # sTMY files don't have wind direction, use EPW missing code 999
        print(f"  Info: WindDir_deg not available in sTMY, using EPW missing code 999")
        wind_dir = pd.Series(np.nan, index=df.index, dtype=float)

    n_missing_windspeed = wind_speed.isna().sum()
    if n_missing_windspeed > 0:
        print(f"  Info: {n_missing_windspeed} missing Wind Speed values (will use EPW missing code 999)")

    # Solar radiation - do NOT fill missing values, use EPW missing codes
    # Only clip valid values to prevent extreme values
    ghi = df['GHI_Whm2']
    dni = df['DNI_Whm2']
    dhi = df['DHI_Whm2']

    # Clip only non-NaN values to valid ranges
    ghi_valid_mask = ghi.notna()
    dni_valid_mask = dni.notna()
    dhi_valid_mask = dhi.notna()

    if ghi_valid_mask.any():
        ghi.loc[ghi_valid_mask] = ghi.loc[ghi_valid_mask].clip(lower=0, upper=1500)
    if dni_valid_mask.any():
        dni.loc[dni_valid_mask] = dni.loc[dni_valid_mask].clip(lower=0, upper=1367)  # Solar constant
    if dhi_valid_mask.any():
        dhi.loc[dhi_valid_mask] = dhi.loc[dhi_valid_mask].clip(lower=0, upper=800)

    # Report missing solar radiation values
    n_missing_ghi = ghi.isna().sum()
    n_missing_dni = dni.isna().sum()
    n_missing_dhi = dhi.isna().sum()

    if n_missing_ghi > 0:
        print(f"  Info: {n_missing_ghi} missing GHI values (will use EPW missing code 9999)")
    if n_missing_dni > 0:
        print(f"  Info: {n_missing_dni} missing DNI values (will use EPW missing code 9999)")
    if n_missing_dhi > 0:
        print(f"  Info: {n_missing_dhi} missing DHI values (will use EPW missing code 9999)")

    # Build EPW rows
    rows = []
    for i, ts in enumerate(df.index):
        Y = ts.year
        M = ts.month
        D = ts.day
        H = ts.hour + 1  # EPW uses 1-24
        Minute = 0  # Standard EPW uses 0 for minute field

        # Data Source and Uncertainty Flags (30 characters for 10 fields)
        # Standard format: ?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9*9*9*9
        # Each pair represents [Source][Uncertainty] for:
        # DryBulb, DewPoint, RelHum, Pressure, ExtHorzRad, ExtDirNormRad, HorzIRSky,
        # GlobHorzRad, DirNormRad, DifHorzRad + additional uncertainty flags
        #
        # Our mapping:
        # - C0 for measured/reanalysis data (temp, dewpoint, RH, pressure, wind)
        # - E0 for calibrated solar radiation (GHI, DNI, DHI)
        # - ?9 for missing data
        # - *9 for additional uncertainty flags

        # Determine data source flags based on data availability
        temp_flag = "C0" if pd.notna(temp_c.iloc[i]) else "?9"
        dewpoint_flag = "C0" if pd.notna(dewpoint_c.iloc[i]) else "?9"
        rh_flag = "C0" if pd.notna(rh.iloc[i]) else "?9"
        pressure_flag = "C0" if pd.notna(pressure_pa.iloc[i]) else "?9"

        data_flags = f"{temp_flag}{dewpoint_flag}{rh_flag}{pressure_flag}?9?9?9E0E0E0?9?9?9?9?9*9*9*9*9*9"

        # Use EPW standard missing value codes when data is missing
        # EPW missing value codes:
        # - Temperature (dry bulb, dew point): 99.9
        # - Relative humidity: 999
        # - Pressure: 999999
        # - Wind speed: 999
        # - Wind direction: 999

        temp_val = f"{temp_c.iloc[i]:.1f}" if pd.notna(temp_c.iloc[i]) else "99.9"
        dewpoint_val = f"{dewpoint_c.iloc[i]:.1f}" if pd.notna(dewpoint_c.iloc[i]) else "99.9"
        rh_val = f"{int(rh.iloc[i])}" if pd.notna(rh.iloc[i]) else "999"
        pressure_val = f"{int(pressure_pa.iloc[i])}" if pd.notna(pressure_pa.iloc[i]) else "999999"
        # Wind direction: use 0 for missing (not 999, which is out of range [0, 360])
        winddir_val = f"{int(wind_dir.iloc[i])}" if pd.notna(wind_dir.iloc[i]) else "0"
        windspeed_val = f"{wind_speed.iloc[i]:.1f}" if pd.notna(wind_speed.iloc[i]) else "0.0"

        # Solar radiation values - use EPW missing code 9999 for missing values (not 999999)
        ghi_val = f"{int(ghi.iloc[i])}" if pd.notna(ghi.iloc[i]) else "9999"
        dni_val = f"{int(dni.iloc[i])}" if pd.notna(dni.iloc[i]) else "9999"
        dhi_val = f"{int(dhi.iloc[i])}" if pd.notna(dhi.iloc[i]) else "9999"

        row = [
            Y, M, D, H, Minute, data_flags,
            temp_val,  # [6] Dry bulb temperature (°C)
            dewpoint_val,  # [7] Dew point temperature (°C)
            rh_val,  # [8] Relative humidity (%)
            pressure_val,  # [9] Atmospheric pressure (Pa)
            9999,  # [10] Extraterrestrial horizontal radiation (Wh/m²)
            9999,  # [11] Extraterrestrial direct normal radiation (Wh/m²)
            9999,  # [12] Horizontal infrared radiation (Wh/m²)
            ghi_val,  # [13] Global horizontal irradiance (Wh/m²)
            dni_val,  # [14] Direct normal irradiance (Wh/m²)
            dhi_val,  # [15] Diffuse horizontal irradiance (Wh/m²)
            0,  # [16] Global horizontal illuminance (lux)
            0,  # [17] Direct normal illuminance (lux)
            0,  # [18] Diffuse horizontal illuminance (lux)
            0,  # [19] Zenith luminance (Cd/m²)
            winddir_val,  # [20] Wind direction (degrees)
            windspeed_val,  # [21] Wind speed (m/s)
            99,  # [22] Total sky cover (tenths)
            99,  # [23] Opaque sky cover (tenths)
            9999,  # [24] Visibility (km)
            99999,  # [25] Ceiling height (m)
            9,  # [26] Present weather observation
            9,  # [27] Present weather codes
            999,  # [28] Precipitable water (mm)
            0,  # [29] Aerosol optical depth (use 0 instead of .999)
            999,  # [30] Snow depth (cm)
            99,  # [31] Days since last snowfall
            999,  # [32] Albedo
            999,  # [33] Liquid precipitation depth (mm)
            99,  # [34] Liquid precipitation quantity (hr)
        ]
        rows.append(",".join(map(str, row)))

    # Write EPW file
    output_epw_path.parent.mkdir(parents=True, exist_ok=True)
    # Format city name (replace underscore with space)
    city_display = city_name.replace('_', ' ')
    header = epw_header(city_display, state, country, lat, lon, tz, wmo_code=wmo_code)

    with open(output_epw_path, "w", encoding="utf-8") as f:
        f.write(header)
        for line in rows:
            f.write(line + "\n")

    print(f"  ✓ Created EPW: {output_epw_path.name}")
    return output_epw_path


def _idf_design_day(
    name: str,
    month: int,
    day_of_month: int,
    day_type: str,
    max_db: float,
    daily_range: float,
    humidity_type: str,
    humidity_value: float | None,
    pressure_pa: float,
    wind_speed: float,
    *,
    wind_direction: int = 0,
    sky_clearness: float = 0.0,
    enthalpy_value: float | None = None,
) -> str:
    if humidity_type == "Enthalpy":
        humidity_line = f"  ,                                 !- Wetbulb or DewPoint at Maximum Dry-Bulb {{C}}\n"
        enthalpy_line = f"  {enthalpy_value:.1f},                  !- Enthalpy at Maximum Dry-Bulb {{J/kg}}\n"
    else:
        humidity_line = f"  {humidity_value:.1f},                  !- Wetbulb or DewPoint at Maximum Dry-Bulb {{C}}\n"
        enthalpy_line = "  ,                                 !- Enthalpy at Maximum Dry-Bulb {J/kg}\n"
    return f"""SizingPeriod:DesignDay,
  {name},                           !- Name
  {month},                          !- Month
  {day_of_month},                   !- Day of Month
  {day_type},                       !- Day Type
  {max_db:.1f},                     !- Maximum Dry-Bulb Temperature {{C}}
  {daily_range:.1f},                !- Daily Dry-Bulb Temperature Range {{C}}
  DefaultMultipliers,               !- Dry-Bulb Temperature Range Modifier Type
  ,                                 !- Dry-Bulb Temperature Range Modifier Day Schedule Name
  {humidity_type},                  !- Humidity Condition Type
{humidity_line}  ,                                 !- Humidity Indicating Day Schedule Name
  ,                                 !- Humidity Ratio at Maximum Dry-Bulb {{kgWater/kgDryAir}}
{enthalpy_line}  ,                                 !- Daily Wet-Bulb Temperature Range {{deltaC}}
  {pressure_pa:.0f},                      !- Barometric Pressure {{Pa}}
  {wind_speed:.1f},                    !- Wind Speed {{m/s}}
  {wind_direction},                          !- Wind Direction {{Degrees; N=0, S=180}}
  No,                                !- Rain Indicator
  No,                                !- Snow Indicator
  No,                                !- Daylight Saving Time Indicator
  ASHRAEClearSky,                    !- Solar Model Indicator
  ,                                  !- Beam Solar Day Schedule Name
  ,                                  !- Diffuse Solar Day Schedule Name
  ,                                  !- ASHRAE Clear Sky Optical Depth for Beam Irradiance (taub)
  ,                                  !- ASHRAE Clear Sky Optical Depth for Diffuse Irradiance (taud)
  {sky_clearness:.1f};                 !- Clearness
"""


def _coincident_mean(primary: np.ndarray, other: np.ndarray, pct: float, *, high: bool) -> tuple[float, float]:
    """Return percentile threshold of ``primary`` and mean ``other`` over the extreme tail."""
    threshold = np.percentile(primary, pct)
    mask = primary >= threshold if high else primary <= threshold
    return float(threshold), float(np.mean(other[mask]))


def _estimate_pressure_pa(pressure: pd.Series | None, elevation_m: float) -> float:
    if pressure is not None:
        pressure = pd.to_numeric(pressure, errors="coerce")
        valid = pressure[np.isfinite(pressure)]
        if not valid.empty:
            mean_val = float(valid.mean())
            return mean_val * 100.0 if mean_val < 2000 else mean_val
    return float(101325 * (1 - 2.25577e-5 * elevation_m) ** 5.25588)


def _parse_epw_to_dataframe(epw_path: Path) -> tuple[pd.DataFrame, dict]:
    with open(epw_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 9:
        raise ValueError(f"EPW file is too short: {epw_path}")

    location = lines[0].strip().split(",")
    meta = {
        "city": location[1] if len(location) > 1 else "Unknown",
        "state": location[2] if len(location) > 2 else "",
        "country": location[3] if len(location) > 3 else "",
        "lat": float(location[6]) if len(location) > 6 else 0.0,
        "lon": float(location[7]) if len(location) > 7 else 0.0,
        "tz": float(location[8]) if len(location) > 8 else 0.0,
        "elevation_m": float(location[9]) if len(location) > 9 else 0.0,
    }

    records = []
    for line in lines[8:]:
        parts = line.strip().split(",")
        if len(parts) < 22:
            continue
        try:
            records.append(
                {
                    "year": int(parts[0]),
                    "month": int(parts[1]),
                    "day": int(parts[2]),
                    "hour": int(parts[3]),
                    "temp": float(parts[6]),
                    "dewpoint": float(parts[7]),
                    "relative_humidity": float(parts[8]),
                    "pressure": float(parts[9]),
                    "wind_speed": float(parts[21]),
                }
            )
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No valid hourly rows found in EPW: {epw_path}")
    timestamps = pd.to_datetime(
        {
            "year": df["year"],
            "month": df["month"],
            "day": df["day"],
            "hour": df["hour"].clip(lower=1, upper=24) - 1,
        }
    )
    df = df.drop(columns=["year", "month", "day", "hour"])
    df.index = timestamps
    return df, meta


def compute_ddy_params(df: pd.DataFrame, elevation_m: float = 0.0) -> dict:
    """Extract generalized 18-condition ASHRAE DDY parameters from hourly data."""
    required = {"temp", "dewpoint"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DDY generation requires columns: {sorted(required)}; missing {sorted(missing)}")

    sub = df.copy()
    for col in ["temp", "dewpoint", "pressure", "wind_speed"]:
        if col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")

    # Drop missing-value codes and physically invalid rows.
    temp = sub["temp"]
    dewpoint = sub["dewpoint"]
    valid = (
        temp.between(-50, 60)
        & dewpoint.between(-50, 60)
        & np.isfinite(temp)
        & np.isfinite(dewpoint)
    )
    if "relative_humidity" in sub.columns:
        rh_raw = pd.to_numeric(sub["relative_humidity"], errors="coerce")
        valid &= rh_raw.between(0, 100) & np.isfinite(rh_raw)

    sub = sub.loc[valid].copy()
    if sub.empty:
        raise ValueError("No valid hourly rows remain after DDY filtering")

    TDB = sub["temp"].to_numpy(dtype=float)
    TDP = sub["dewpoint"].to_numpy(dtype=float)
    RH = np.clip(calc_RH_from_dewpoint(TDB, TDP), 0.1, 100.0)
    WB = calc_wetbulb(TDB, RH)
    Pw = 610.7 * np.exp(17.27 * TDP / (237.7 + TDP))
    pressure_pa = _estimate_pressure_pa(sub.get("pressure"), elevation_m)
    humidity_ratio = 0.62198 * Pw / np.maximum(pressure_pa - Pw, 1.0)
    enthalpy = 1006.0 * TDB + humidity_ratio * (2501000.0 + 1860.0 * TDB)

    if "wind_speed" in sub.columns:
        WS = np.nan_to_num(sub["wind_speed"].to_numpy(dtype=float), nan=0.0)
        WS = np.clip(WS, 0.0, None)
    else:
        WS = np.zeros(len(sub), dtype=float)

    if isinstance(sub.index, pd.DatetimeIndex):
        tdb_series = sub["temp"]
        hottest_month = int(tdb_series.groupby(tdb_series.index.month).mean().idxmax())
        hottest_month_data = tdb_series[tdb_series.index.month == hottest_month]
        daily_range = float((hottest_month_data.resample("D").max() - hottest_month_data.resample("D").min()).mean())
    else:
        hottest_month = 7
        daily_range = 10.0

    htg_996_db, _ = _coincident_mean(TDB, TDB, 0.4, high=False)
    htg_99_db, _ = _coincident_mean(TDB, TDB, 1.0, high=False)
    dp_996, mcdb_dp_996 = _coincident_mean(TDP, TDB, 0.4, high=False)
    dp_99, mcdb_dp_99 = _coincident_mean(TDP, TDB, 1.0, high=False)
    ws_996, mcdb_ws_996 = _coincident_mean(WS, TDB, 99.6, high=True)
    ws_99, mcdb_ws_99 = _coincident_mean(WS, TDB, 99.0, high=True)

    clg_db_004, clg_wb_004 = _coincident_mean(TDB, WB, 99.6, high=True)
    clg_db_01, clg_wb_01 = _coincident_mean(TDB, WB, 99.0, high=True)
    clg_db_02, clg_wb_02 = _coincident_mean(TDB, WB, 98.0, high=True)

    clg_wb_peak_004, clg_mdb_at_wb_004 = _coincident_mean(WB, TDB, 99.6, high=True)
    clg_wb_peak_01, clg_mdb_at_wb_01 = _coincident_mean(WB, TDB, 99.0, high=True)
    clg_wb_peak_02, clg_mdb_at_wb_02 = _coincident_mean(WB, TDB, 98.0, high=True)

    clg_dp_004, clg_mdb_at_dp_004 = _coincident_mean(TDP, TDB, 99.6, high=True)
    clg_dp_01, clg_mdb_at_dp_01 = _coincident_mean(TDP, TDB, 99.0, high=True)
    clg_dp_02, clg_mdb_at_dp_02 = _coincident_mean(TDP, TDB, 98.0, high=True)

    clg_h_004, clg_mdb_at_h_004 = _coincident_mean(enthalpy, TDB, 99.6, high=True)
    clg_h_01, clg_mdb_at_h_01 = _coincident_mean(enthalpy, TDB, 99.0, high=True)
    clg_h_02, clg_mdb_at_h_02 = _coincident_mean(enthalpy, TDB, 98.0, high=True)

    cool_wind = float(WS[TDB >= clg_db_004].mean()) if len(WS) else 0.0

    return {
        "pressure_pa": pressure_pa,
        "hottest_month": hottest_month,
        "daily_range": max(daily_range, 0.0),
        "cool_wind": cool_wind,
        "htg_996_db": htg_996_db,
        "htg_99_db": htg_99_db,
        "htg_dp_996": dp_996,
        "htg_mcdb_dp_996": mcdb_dp_996,
        "htg_dp_99": dp_99,
        "htg_mcdb_dp_99": mcdb_dp_99,
        "htg_ws_996": ws_996,
        "htg_mcdb_ws_996": mcdb_ws_996,
        "htg_ws_99": ws_99,
        "htg_mcdb_ws_99": mcdb_ws_99,
        "clg_db_004": clg_db_004,
        "clg_wb_004": clg_wb_004,
        "clg_db_01": clg_db_01,
        "clg_wb_01": clg_wb_01,
        "clg_db_02": clg_db_02,
        "clg_wb_02": clg_wb_02,
        "clg_wb_peak_004": clg_wb_peak_004,
        "clg_mdb_at_wb_004": clg_mdb_at_wb_004,
        "clg_wb_peak_01": clg_wb_peak_01,
        "clg_mdb_at_wb_01": clg_mdb_at_wb_01,
        "clg_wb_peak_02": clg_wb_peak_02,
        "clg_mdb_at_wb_02": clg_mdb_at_wb_02,
        "clg_dp_004": clg_dp_004,
        "clg_mdb_at_dp_004": clg_mdb_at_dp_004,
        "clg_dp_01": clg_dp_01,
        "clg_mdb_at_dp_01": clg_mdb_at_dp_01,
        "clg_dp_02": clg_dp_02,
        "clg_mdb_at_dp_02": clg_mdb_at_dp_02,
        "clg_h_004": clg_h_004,
        "clg_mdb_at_h_004": clg_mdb_at_h_004,
        "clg_h_01": clg_h_01,
        "clg_mdb_at_h_01": clg_mdb_at_h_01,
        "clg_h_02": clg_h_02,
        "clg_mdb_at_h_02": clg_mdb_at_h_02,
    }


def render_ddy_content(params: dict, *, city: str, state: str = "", country: str = "", lat: float = 0.0,
                       lon: float = 0.0, tz: float = 0.0, elevation_m: float = 0.0,
                       label: str = "future") -> str:
    """Render 18 design-day conditions as IDF/DDY text."""
    location_name = "_".join(part for part in [city, state, country, "Design_Conditions"] if part)
    pressure_pa = round(params["pressure_pa"], 0)
    daily_range = round(params["daily_range"], 1)
    cooling_wind = round(params["cool_wind"], 1)

    blocks = [
        _idf_design_day(f"{label} Ann Htg 99.6% Condns DB", 1, 21, "WinterDesignDay",
                        params["htg_996_db"], 0.0, "WetBulb", params["htg_996_db"], pressure_pa, 0.0),
        _idf_design_day(f"{label} Ann Htg 99% Condns DB", 1, 21, "WinterDesignDay",
                        params["htg_99_db"], 0.0, "WetBulb", params["htg_99_db"], pressure_pa, 0.0),
        _idf_design_day(f"{label} Ann Hum_n 99.6% Condns DP=>MCDB", 1, 21, "WinterDesignDay",
                        params["htg_mcdb_dp_996"], 0.0, "DewPoint", params["htg_dp_996"], pressure_pa, 0.0),
        _idf_design_day(f"{label} Ann Hum_n 99% Condns DP=>MCDB", 1, 21, "WinterDesignDay",
                        params["htg_mcdb_dp_99"], 0.0, "DewPoint", params["htg_dp_99"], pressure_pa, 0.0),
        _idf_design_day(f"{label} Ann Htg Wind 99.6% Condns WS=>MCDB", 1, 21, "WinterDesignDay",
                        params["htg_mcdb_ws_996"], 0.0, "WetBulb", params["htg_mcdb_ws_996"], pressure_pa, params["htg_ws_996"]),
        _idf_design_day(f"{label} Ann Htg Wind 99% Condns WS=>MCDB", 1, 21, "WinterDesignDay",
                        params["htg_mcdb_ws_99"], 0.0, "WetBulb", params["htg_mcdb_ws_99"], pressure_pa, params["htg_ws_99"]),
        _idf_design_day(f"{label} Ann Clg .4% Condns DB=>MWB", 7, 21, "SummerDesignDay",
                        params["clg_db_004"], daily_range, "WetBulb", params["clg_wb_004"], pressure_pa, cooling_wind, sky_clearness=1.0),
        _idf_design_day(f"{label} Ann Clg 1% Condns DB=>MWB", 7, 21, "SummerDesignDay",
                        params["clg_db_01"], daily_range, "WetBulb", params["clg_wb_01"], pressure_pa, cooling_wind, sky_clearness=1.0),
        _idf_design_day(f"{label} Ann Clg 2% Condns DB=>MWB", 7, 21, "SummerDesignDay",
                        params["clg_db_02"], daily_range, "WetBulb", params["clg_wb_02"], pressure_pa, cooling_wind, sky_clearness=1.0),
        _idf_design_day(f"{label} Ann Clg .4% Condns WB=>MDB", 7, 21, "SummerDesignDay",
                        params["clg_mdb_at_wb_004"], daily_range, "WetBulb", params["clg_wb_peak_004"], pressure_pa, cooling_wind, sky_clearness=1.0),
        _idf_design_day(f"{label} Ann Clg 1% Condns WB=>MDB", 7, 21, "SummerDesignDay",
                        params["clg_mdb_at_wb_01"], daily_range, "WetBulb", params["clg_wb_peak_01"], pressure_pa, cooling_wind, sky_clearness=1.0),
        _idf_design_day(f"{label} Ann Clg 2% Condns WB=>MDB", 7, 21, "SummerDesignDay",
                        params["clg_mdb_at_wb_02"], daily_range, "WetBulb", params["clg_wb_peak_02"], pressure_pa, cooling_wind, sky_clearness=1.0),
        _idf_design_day(f"{label} Ann Clg .4% Condns DP=>MDB", 7, 21, "SummerDesignDay",
                        params["clg_mdb_at_dp_004"], daily_range, "DewPoint", params["clg_dp_004"], pressure_pa, cooling_wind, sky_clearness=1.0),
        _idf_design_day(f"{label} Ann Clg 1% Condns DP=>MDB", 7, 21, "SummerDesignDay",
                        params["clg_mdb_at_dp_01"], daily_range, "DewPoint", params["clg_dp_01"], pressure_pa, cooling_wind, sky_clearness=1.0),
        _idf_design_day(f"{label} Ann Clg 2% Condns DP=>MDB", 7, 21, "SummerDesignDay",
                        params["clg_mdb_at_dp_02"], daily_range, "DewPoint", params["clg_dp_02"], pressure_pa, cooling_wind, sky_clearness=1.0),
        _idf_design_day(f"{label} Ann Clg .4% Condns Enth=>MDB", 7, 21, "SummerDesignDay",
                        params["clg_mdb_at_h_004"], daily_range, "Enthalpy", None, pressure_pa, cooling_wind, sky_clearness=1.0,
                        enthalpy_value=params["clg_h_004"]),
        _idf_design_day(f"{label} Ann Clg 1% Condns Enth=>MDB", 7, 21, "SummerDesignDay",
                        params["clg_mdb_at_h_01"], daily_range, "Enthalpy", None, pressure_pa, cooling_wind, sky_clearness=1.0,
                        enthalpy_value=params["clg_h_01"]),
        _idf_design_day(f"{label} Ann Clg 2% Condns Enth=>MDB", 7, 21, "SummerDesignDay",
                        params["clg_mdb_at_h_02"], daily_range, "Enthalpy", None, pressure_pa, cooling_wind, sky_clearness=1.0,
                        enthalpy_value=params["clg_h_02"]),
    ]

    header = f""" ! Generalized ASHRAE design-day conditions derived from hourly weather data.
 ! Cooling solar model is generalized to ASHRAEClearSky to avoid city-specific tau constants.

 Site:Location,
  {location_name},                   !- Location Name
  {lat:.2f},                         !- Latitude {{N+ S-}}
  {lon:.2f},                         !- Longitude {{W- E+}}
  {tz:.2f},                          !- Time Zone Relative to GMT {{GMT+/-}}
  {elevation_m:.2f};                 !- Elevation {{m}}

 RunPeriodControl:DaylightSavingTime,
  2nd Sunday in March,               !- Start Date
  2nd Sunday in November;            !- End Date

"""
    return header + "\n\n".join(blocks) + "\n"


def write_ddy_from_dataframe(df: pd.DataFrame, out_path: Path, *, city: str, state: str = "", country: str = "",
                             lat: float = 0.0, lon: float = 0.0, tz: float = 0.0,
                             elevation_m: float = 0.0, label: str = "future") -> Path:
    params = compute_ddy_params(df, elevation_m=elevation_m)
    content = render_ddy_content(
        params,
        city=city,
        state=state,
        country=country,
        lat=lat,
        lon=lon,
        tz=tz,
        elevation_m=elevation_m,
        label=label,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def generate_ddy_from_epw(epw_path, energyplus_path):
    """
    Generate DDY file from EPW by analyzing weather data.

    Analyzes the EPW file to determine design day conditions based on
    ASHRAE guidelines (99.6% and 0.4% temperature conditions).

    Args:
        epw_path: Path to EPW file
        energyplus_path: Path to EnergyPlus executable (not used, kept for compatibility)

    Returns:
        True if successful, False otherwise
    """
    print(f"  Generating DDY from EPW...")

    try:
        epw_df, meta = _parse_epw_to_dataframe(Path(epw_path))
        ddy_path = write_ddy_from_dataframe(
            epw_df,
            Path(epw_path).with_suffix(".ddy"),
            city=meta["city"],
            state=meta["state"],
            country=meta["country"],
            lat=meta["lat"],
            lon=meta["lon"],
            tz=meta["tz"],
            elevation_m=meta["elevation_m"],
            label=meta["city"],
        )
        print(f"  ✓ Created DDY: {ddy_path.name}")
        return True

    except Exception as e:
        print(f"  ✗ Exception during DDY generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to convert TMY files to EPW and generate DDY files."""

    # Configuration
    tmy_dir = Path("output/TMY")
    output_dir = Path("output/epw")
    cities_config = Path("data/cities_config_final.csv")
    energyplus_path = "/Applications/EnergyPlus-25-1-0/energyplus"

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert TMY to EPW/DDY")
    parser.add_argument("--tmy-dir", type=Path, default=tmy_dir, help="Directory with TMY CSV files")
    parser.add_argument("--output-dir", type=Path, default=output_dir, help="Output directory for EPW/DDY files")
    parser.add_argument("--config-path", type=Path, default=cities_config, help="Path to cities config CSV")
    parser.add_argument("--energyplus-path", type=str, default=energyplus_path, help="Path to EnergyPlus executable")
    parser.add_argument("--stmy-mode", action="store_true",
                        help="Process sTMY files from output/sTMY with time-slice subdirectories (e.g., stmy_2025_2045)")
    parser.add_argument("--time-periods", type=str, nargs="+",
                        default=["2025_2045", "2046_2070", "2071_2095"],
                        help="Time periods for sTMY mode (e.g., 2025_2045 2046_2070)")
    args = parser.parse_args()

    # Update configuration from arguments
    tmy_dir = args.tmy_dir
    output_dir = args.output_dir
    cities_config = args.config_path
    energyplus_path = args.energyplus_path
    stmy_mode = args.stmy_mode
    time_periods = args.time_periods

    # Print header
    print("=" * 70)
    if stmy_mode:
        print("sTMY to EPW/DDY Converter (Future Climate Scenarios)")
    else:
        print("TMY to EPW/DDY Converter")
    print("=" * 70)

    if stmy_mode:
        print(f"sTMY mode:          Enabled")
        print(f"Time periods:       {', '.join(time_periods)}")
        print(f"sTMY base dir:      output/sTMY")
        print(f"Output:             To each time period's folder")
    else:
        print(f"TMY directory:      {tmy_dir}")
        print(f"Output directory:   {output_dir}")

    print(f"Cities config:      {cities_config}")
    print(f"EnergyPlus path:    {energyplus_path}")

    # Check if EnergyPlus exists
    if not os.path.exists(energyplus_path):
        print(f"\n✗ Error: EnergyPlus not found at: {energyplus_path}")
        print("Please update the energyplus_path variable in the script")
        return 1

    # Check if cities config exists
    if not cities_config.exists():
        print(f"\n✗ Error: Cities config not found at: {cities_config}")
        return 1

    # Check if TMY/sTMY directory exists
    if not stmy_mode and not tmy_dir.exists():
        print(f"\n✗ Error: TMY directory not found at: {tmy_dir}")
        return 1

    # Create output directory (only for TMY mode)
    if not stmy_mode:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load city metadata
    print(f"\nLoading city metadata...")
    city_metadata = load_city_metadata(cities_config)
    print(f"Loaded metadata for {len(city_metadata)} cities")

    # Find all TMY files based on mode
    if stmy_mode:
        # Process sTMY files from multiple time periods
        tmy_files = []
        stmy_base_dir = Path("output/sTMY")

        if not stmy_base_dir.exists():
            print(f"\n✗ Error: sTMY directory not found at: {stmy_base_dir}")
            return 1

        for period in time_periods:
            period_dir = stmy_base_dir / f"stmy_{period}"
            if not period_dir.exists():
                print(f"\n⚠ Warning: Time period directory not found: {period_dir}")
                continue

            # Find all city subdirectories
            for city_dir in sorted(period_dir.iterdir()):
                if not city_dir.is_dir():
                    continue

                # Find sTMY CSV files (not selection files)
                city_files = list(city_dir.glob(f"*_stmy_{period}.csv"))
                city_files = [f for f in city_files if "_selection" not in f.name]

                for f in city_files:
                    # Store file with metadata (city_name, period, file_path)
                    city_name = f.stem.split(f'_stmy_{period}')[0]
                    tmy_files.append({
                        'file': f,
                        'city': city_name,
                        'period': period,
                        'type': 'stmy'
                    })

        print(f"\nFound {len(tmy_files)} sTMY files across {len(time_periods)} time period(s)")
    else:
        # Original TMY mode
        tmy_file_list = sorted([f for f in tmy_dir.glob("*_tmy_*.csv") if "_selection" not in f.name])
        tmy_files = [{'file': f, 'city': f.stem.split('_tmy_')[0], 'period': '1991_2020', 'type': 'tmy'}
                     for f in tmy_file_list]
        print(f"\nFound {len(tmy_files)} TMY files to process")

    if len(tmy_files) == 0:
        print(f"\n✗ Error: No TMY files found")
        return 1

    print("=" * 70)

    # Process each TMY file
    success_count = 0
    error_count = 0

    for tmy_info in tmy_files:
        try:
            tmy_file = tmy_info['file']
            city_name = tmy_info['city']
            period = tmy_info['period']
            file_type = tmy_info['type']

            # Create output EPW path based on file type
            if file_type == 'stmy':
                # For sTMY: output to output/sTMY/stmy_{period}/{city}/ folder
                stmy_base_dir = Path("output/sTMY")
                city_output_dir = stmy_base_dir / f"stmy_{period}" / city_name
                epw_filename = f"{city_name}_stmy_{period}.epw"
                output_epw = city_output_dir / epw_filename
            else:
                # For TMY: output to output/epw folder (original behavior)
                epw_filename = f"{city_name}_tmy_{period}.epw"
                output_epw = output_dir / epw_filename

            # Convert TMY to EPW
            epw_path = convert_tmy_to_epw(tmy_file, output_epw, city_name, city_metadata)

            # Generate DDY from EPW
            ddy_success = generate_ddy_from_epw(epw_path, energyplus_path)

            if ddy_success:
                success_count += 1
            else:
                error_count += 1

        except Exception as e:
            print(f"\n✗ Error processing {tmy_file.name}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1

    # Print summary
    print("\n" + "=" * 70)
    print("Processing complete!")
    print(f"Successfully processed: {success_count}/{len(tmy_files)}")
    if error_count > 0:
        print(f"Errors:                {error_count}")

    if stmy_mode:
        print(f"Output locations:      output/sTMY/stmy_{{period}}/{{city}}/")
    else:
        print(f"Output directory:      {output_dir.absolute()}")
    print("=" * 70)

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    exit(main())
