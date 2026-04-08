#!/usr/bin/env python3
"""
ISD Data Parser using isdparser library
Process all .gz files in city/station structure and convert to CSV format
Then analyze parsed data for temporal coverage metrics and visualization
"""

import gzip
import os
from pathlib import Path
import pandas as pd
from samy_wgen.utils._isdparser import ISDRecordFactory
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
import argparse

# Library-safe logger: callers configure handlers; CLI uses _setup_logging()
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Mapping from isdparser mandatory-section field names to samy_wgen internal names.
# Values are exactly as returned by ISDRecordFactory (already scaled, e.g. °C, hPa, m/s).
ISD_RENAME: dict[str, str] = {
    "air_temperature_observation_air_temperature":       "temp",        # °C
    "air_temperature_observation_dew_point_temperature": "dewpoint",    # °C
    "atmospheric_pressure_observation_sea_level_pressure": "pressure",  # hPa
    "wind_observation_speed_rate":                       "wind_speed",  # m/s
    "wind_observation_direction_angle":                  "wind_dir",    # degrees
}


def remap_isd_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename ISD raw field names to samy_wgen internal names and coerce to numeric.

    Keeps only the columns present in ISD_RENAME; unknown columns are dropped.
    Missing ISD fields (e.g. wind_dir absent in older records) are silently skipped.

    Args:
        df: DataFrame with isdparser field names as columns (output of
            ``process_station_directory`` after loading into a DataFrame).

    Returns:
        DataFrame with columns: temp, dewpoint, pressure, wind_speed, wind_dir
        (subset, depending on what was present in the input).
    """
    keep = {raw: internal for raw, internal in ISD_RENAME.items() if raw in df.columns}
    if not keep:
        raise ValueError(
            "No recognised ISD mandatory fields found. "
            f"Expected one of: {list(ISD_RENAME)}. Got: {list(df.columns)}"
        )
    out = df[list(keep)].rename(columns=keep)
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# Columns that are circular variables (0–360°) and must not be averaged arithmetically.
_CIRCULAR_COLS: set[str] = {"wind_dir"}


def resample_isd_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a remapped ISD DataFrame to one row per hour.

    Aggregation rules:
    - Continuous variables (temp, dewpoint, pressure, wind_speed): arithmetic mean.
    - Circular variables (wind_dir): mode of observed values within the hour.
      Mode avoids the 0°/360° wrap-around error produced by arithmetic mean,
      and represents the prevailing direction — the standard meteorological convention.

    Args:
        df: DataFrame produced by ``remap_isd_columns``, with a DatetimeIndex.

    Returns:
        Hourly DataFrame with the same columns.
    """
    continuous = [c for c in df.columns if c not in _CIRCULAR_COLS]
    circular   = [c for c in df.columns if c in _CIRCULAR_COLS]

    parts = []
    if continuous:
        parts.append(df[continuous].resample("h").mean())
    if circular:
        def _mode(s: pd.Series) -> float:
            vals = s.dropna()
            return float(vals.mode().iloc[0]) if len(vals) > 0 else float("nan")
        parts.append(df[circular].resample("h").agg(_mode))

    result = pd.concat(parts, axis=1)[df.columns]  # restore original column order
    return result


def _setup_logging(log_dir: str = "logs") -> None:
    """Configure file + console logging for CLI use. Call from main() only."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "parse_ISD.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(),
        ],
    )


def _is_leap_year(y):
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)


def analyze_parsed_csv_temporal_coverage(csv_path: str) -> Dict:
    """
    Analyze temporal coverage of a parsed CSV file to determine hourly ratio and coverage metrics

    Args:
        csv_path: Path to parsed CSV file

    Returns:
        Dict with temporal coverage metrics
    """
    try:
        logger.info(f"Analyzing temporal coverage of {os.path.basename(csv_path)}")

        # Read CSV in chunks to handle large files
        chunk_size = 50000
        year_stats = defaultdict(lambda: {
            'total_records': 0,
            'unique_dates': set(),
            'unique_hours': set(),
            'date_hour_pairs': set()
        })

        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            # Assume first column is datetime
            datetime_col = chunk.columns[0]
            chunk['parsed_datetime'] = pd.to_datetime(chunk[datetime_col], errors='coerce')

            # Filter valid timestamps
            valid_chunk = chunk.dropna(subset=['parsed_datetime'])

            if valid_chunk.empty:
                continue

            # Extract temporal components
            valid_chunk['year'] = valid_chunk['parsed_datetime'].dt.year
            valid_chunk['date'] = valid_chunk['parsed_datetime'].dt.date
            valid_chunk['hour'] = valid_chunk['parsed_datetime'].dt.hour

            # Aggregate by year
            for year in valid_chunk['year'].unique():
                year_data = valid_chunk[valid_chunk['year'] == year]

                year_stats[year]['total_records'] += len(year_data)
                year_stats[year]['unique_dates'].update(year_data['date'].unique())
                year_stats[year]['unique_hours'].update(year_data['hour'].unique())

                # Track date-hour pairs for hourly coverage analysis
                date_hour_pairs = set(zip(year_data['date'], year_data['hour']))
                year_stats[year]['date_hour_pairs'].update(date_hour_pairs)

        # Calculate metrics for each year
        yearly_metrics = {}
        hourly_years = 0
        total_years = len(year_stats)

        for year, stats in year_stats.items():
            unique_dates = len(stats['unique_dates'])
            unique_hours = len(stats['unique_hours'])
            date_hour_pairs = len(stats['date_hour_pairs'])

            # Expected hours in this year
            expected_days = 366 if _is_leap_year(int(year)) else 365
            expected_hours = expected_days * 24

            # Calculate hourly coverage ratio
            hourly_coverage_ratio = date_hour_pairs / expected_hours if expected_hours > 0 else 0

            # Determine if this year has "hourly" data (>= 70% hourly coverage)
            is_hourly_year = hourly_coverage_ratio >= 0.7
            if is_hourly_year:
                hourly_years += 1

            # Average records per day
            avg_records_per_day = stats['total_records'] / unique_dates if unique_dates > 0 else 0

            yearly_metrics[year] = {
                'total_records': stats['total_records'],
                'unique_dates': unique_dates,
                'unique_hours': unique_hours,
                'date_hour_pairs': date_hour_pairs,
                'expected_hours': expected_hours,
                'hourly_coverage_ratio': hourly_coverage_ratio,
                'is_hourly_year': is_hourly_year,
                'avg_records_per_day': avg_records_per_day
            }

        if not yearly_metrics:
            return {
                'station_code': os.path.basename(csv_path).split('_')[0],
                'total_years': 0,
                'hourly_years': 0,
                'hourly_year_ratio': 0,
                'min_year': None,
                'max_year': None,
                'covered_years': [],
                'yearly_metrics': {}
            }

        # Overall metrics
        years = sorted(yearly_metrics.keys())
        hourly_year_ratio = hourly_years / total_years if total_years > 0 else 0

        station_code = os.path.basename(csv_path).split('_')[0]

        return {
            'station_code': station_code,
            'total_years': total_years,
            'hourly_years': hourly_years,
            'hourly_year_ratio': hourly_year_ratio,
            'min_year': min(years),
            'max_year': max(years),
            'covered_years': years,
            'yearly_metrics': yearly_metrics
        }

    except Exception as e:
        logger.error(f"Error analyzing {csv_path}: {e}")
        return {
            'station_code': 'unknown',
            'total_years': 0,
            'hourly_years': 0,
            'hourly_year_ratio': 0,
            'min_year': None,
            'max_year': None,
            'covered_years': [],
            'yearly_metrics': {}
        }


def parse_isd_gz_file(gz_path: str) -> pd.DataFrame:
    """
    Parse a single ISD .gz file and return structured DataFrame

    Args:
        gz_path: Path to the .gz ISD file

    Returns:
        DataFrame with parsed weather data
    """
    try:
        # Read file
        with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        if not lines:
            logger.warning(f"Empty file: {gz_path}")
            return pd.DataFrame()

        # Parse ISD lines
        records = []
        for i, line in enumerate(lines):
            try:
                record = ISDRecordFactory().create(line)
                records.append(record)
            except Exception as e:
                logger.debug(f"Failed to parse line {i+1} in {gz_path}: {e}")
                continue

        if not records:
            logger.warning(f"No valid records found in: {gz_path}")
            return pd.DataFrame()

        # Convert to structured list
        data = []
        for r in records:
            try:
                rec = r.schema()
                row = {
                    "datestamp": rec.get("datestamp"),
                    "identifier": rec.get("identifier"),
                }

                # Flatten all sections -> key:value
                if "sections" in rec:
                    for section in rec["sections"]:
                        if "measures" in section:
                            for m in section["measures"]:
                                name = m.get("measure")
                                value = m.get("value")
                                if name and value is not None:
                                    row[name] = value
                data.append(row)
            except Exception as e:
                logger.debug(f"Failed to process record: {e}")
                continue

        if not data:
            logger.warning(f"No data extracted from: {gz_path}")
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Sort by datestamp if available
        if 'datestamp' in df.columns and not df['datestamp'].isna().all():
            df = df.sort_values("datestamp").reset_index(drop=True)

        logger.info(f"Parsed {len(df)} records from {os.path.basename(gz_path)}")
        return df

    except Exception as e:
        logger.error(f"Error processing {gz_path}: {e}")
        return pd.DataFrame()


def process_station_directory(station_dir: str, output_dir: str = None) -> Optional[str]:
    """
    Process all .gz files in a station directory and combine into one CSV

    Args:
        station_dir: Path to station directory containing .gz files
        output_dir: Optional output directory, defaults to station_dir

    Returns:
        Path to output CSV file if successful, None otherwise
    """
    if not os.path.exists(station_dir):
        logger.error(f"Station directory not found: {station_dir}")
        return None

    # Get all .gz files in the directory
    gz_files = [f for f in os.listdir(station_dir) if f.endswith('.gz')]

    if not gz_files:
        logger.warning(f"No .gz files found in {station_dir}")
        return None

    logger.info(f"Processing {len(gz_files)} .gz files in {os.path.basename(station_dir)}")

    all_dataframes = []

    # Process each .gz file
    for gz_file in sorted(gz_files):
        gz_path = os.path.join(station_dir, gz_file)
        logger.info(f"  Processing {gz_file}...")

        df = parse_isd_gz_file(gz_path)
        if not df.empty:
            all_dataframes.append(df)

    if not all_dataframes:
        logger.warning(f"No data found in any files from {station_dir}")
        return None

    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Sort by datestamp if available
    if 'datestamp' in combined_df.columns and not combined_df['datestamp'].isna().all():
        combined_df = combined_df.sort_values("datestamp").reset_index(drop=True)

    # Generate output filename
    station_name = os.path.basename(station_dir)
    if output_dir:
        out_path = Path(output_dir)

        # Backward compatible behavior:
        # - directory path  -> <dir>/<station_name>_parsed.csv
        # - file path (.csv) -> exactly that file path
        if out_path.suffix.lower() == ".csv":
            out_path.parent.mkdir(parents=True, exist_ok=True)
            output_path = str(out_path)
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            output_path = str(out_path / f"{station_name}_parsed.csv")
    else:
        output_path = os.path.join(station_dir, f"{station_name}_parsed.csv")

    # Save to CSV
    combined_df.to_csv(output_path, index=False)

    logger.info(f"Saved {len(combined_df)} total records to {output_path}")

    # Log available columns
    logger.info(f"Available columns: {list(combined_df.columns)}")

    # Log date range if available
    if 'datestamp' in combined_df.columns and not combined_df['datestamp'].isna().all():
        date_range = f"{combined_df['datestamp'].min()} to {combined_df['datestamp'].max()}"
        logger.info(f"Date range: {date_range}")

    return output_path


def process_city_directory_all_stations(city_dir: str, output_dir: str = None) -> List[str]:
    """
    Process ALL stations in a city directory without filtering

    Args:
        city_dir: Path to city directory containing station subdirectories
        output_dir: Optional output directory, defaults to city_dir

    Returns:
        List of output CSV file paths
    """
    if not os.path.exists(city_dir):
        logger.error(f"City directory not found: {city_dir}")
        return []

    city_name = os.path.basename(city_dir)
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing city: {city_name} (ALL STATIONS)")
    logger.info(f"{'='*60}")

    # Get all station directories
    station_dirs = []
    for item in os.listdir(city_dir):
        item_path = os.path.join(city_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # Check if directory contains .gz files
            gz_files = [f for f in os.listdir(item_path) if f.endswith('.gz')]
            if gz_files:
                station_dirs.append(item_path)

    if not station_dirs:
        logger.warning(f"No station directories with .gz files found in {city_name}")
        return []

    logger.info(f"Found {len(station_dirs)} stations to process")

    if output_dir:
        city_output_dir = os.path.join(output_dir, city_name)
        os.makedirs(city_output_dir, exist_ok=True)
    else:
        city_output_dir = city_dir

    output_files = []

    # Process each station directory
    for station_path in station_dirs:
        station_name = os.path.basename(station_path)
        logger.info(f"\nProcessing station: {station_name}")

        output_path = process_station_directory(station_path, city_output_dir)

        if output_path:
            output_files.append(output_path)
            logger.info(f"✓ Station processed: {os.path.basename(output_path)}")
        else:
            logger.warning(f"✗ Failed to process station: {station_name}")

    logger.info(f"\nCity {city_name} completed: {len(output_files)} stations processed")
    return output_files


def analyze_parsed_data(parsed_data_dir: str, output_dir: str = "output"):
    """
    Analyze all parsed CSV files and create visualizations of temporal coverage metrics

    Args:
        parsed_data_dir: Directory containing parsed CSV files organized by city
        output_dir: Directory for analysis results and visualizations (default: output)
    """
    logger.info(f"Analyzing parsed data in {parsed_data_dir}")

    # Collect all analysis results
    all_results = []

    # Process each city directory
    for city in os.listdir(parsed_data_dir):
        city_path = os.path.join(parsed_data_dir, city)
        if not os.path.isdir(city_path):
            continue

        logger.info(f"Analyzing city: {city}")

        # Find all CSV files in city directory
        csv_files = [f for f in os.listdir(city_path) if f.endswith('_parsed.csv')]

        for csv_file in csv_files:
            csv_path = os.path.join(city_path, csv_file)
            analysis = analyze_parsed_csv_temporal_coverage(csv_path)
            analysis['city'] = city
            analysis['csv_file'] = csv_file
            all_results.append(analysis)

    if not all_results:
        logger.warning("No parsed data found for analysis")
        return

    # Convert to DataFrame for easier analysis
    df_results = pd.DataFrame(all_results)

    # Create visualizations
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed results to CSV
    results_csv_path = os.path.join(output_dir, 'isd_temporal_analysis_results.csv')
    df_results.to_csv(results_csv_path, index=False)
    logger.info(f"Detailed results saved to {results_csv_path}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TEMPORAL COVERAGE ANALYSIS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total stations analyzed: {len(df_results)}")
    logger.info(f"Cities: {', '.join(df_results['city'].unique())}")
    logger.info(f"Average hourly year ratio: {df_results['hourly_year_ratio'].mean():.3f}")
    logger.info(f"Stations with >50% hourly coverage: {(df_results['hourly_year_ratio'] > 0.5).sum()}")
    logger.info(f"Visualization saved to: {os.path.join(output_dir, 'isd_temporal_coverage_analysis.png')}")


def main():
    """
    Main function to process all ISD data and analyze temporal coverage
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Parse ISD weather data from .gz files and analyze temporal coverage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths (data/ISD -> data/ISD_parsed)
  python parse_ISD.py
  
  # Specify custom input directory
  python parse_ISD.py -i data/ISD_custom
  
  # Specify both input and output directories
  python parse_ISD.py -i data/ISD_custom -o data/ISD_output
  
  # Skip analysis phase
  python parse_ISD.py --no-analyze
        """
    )
    parser.add_argument(
        '-i', '--input-dir',
        default='data/ISD',
        help='Input directory containing city/station subdirectories with .gz files (default: data/ISD)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='data/ISD_parsed',
        help='Output directory for parsed CSV files (default: data/ISD_parsed)'
    )
    parser.add_argument(
        '--no-analyze',
        action='store_true',
        help='Skip the temporal coverage analysis phase'
    )

    args = parser.parse_args()

    # Configuration from command line arguments
    ISD_BASE_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir

    if not os.path.exists(ISD_BASE_DIR):
        logger.error(f"ISD base directory not found: {ISD_BASE_DIR}")
        return

    # Get all city directories
    cities = [d for d in os.listdir(ISD_BASE_DIR)
             if os.path.isdir(os.path.join(ISD_BASE_DIR, d))
             and not d.startswith('.')
             and d != 'parser']

    if not cities:
        logger.error(f"No city directories found in {ISD_BASE_DIR}")
        return

    logger.info(f"Found {len(cities)} cities to process: {', '.join(cities)}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_output_files = []

    # Process each city - parse ALL stations
    for city in sorted(cities):
        city_path = os.path.join(ISD_BASE_DIR, city)
        output_files = process_city_directory_all_stations(city_path, OUTPUT_DIR)
        all_output_files.extend(output_files)

    # Analyze and visualize parsed data
    if all_output_files and not args.no_analyze:
        logger.info(f"\n{'='*60}")
        logger.info("STARTING TEMPORAL COVERAGE ANALYSIS")
        logger.info(f"{'='*60}")
        analyze_parsed_data(OUTPUT_DIR, OUTPUT_DIR)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("PROCESSING COMPLETE!")
    logger.info(f"{'='*60}")
    logger.info(f"Total files processed: {len(all_output_files)}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
