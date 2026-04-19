"""samy-download: City-specific weather data downloader for samy-wgen pipeline.

Subcommands
-----------
  list-stations   Print nearby ISD stations for a city (choose one, then download)
  download        Download ISD / NSRDB / CMIP6 / CORDEX-CMIP5 for a city

Credentials
-----------
  NSRDB   : ~/.config/samy_wgen/credentials.yaml
              Run 'samy-download config' to write this file interactively.
  CMIP6   : ~/.cdsapirc  (standard cdsapi config)
  CORDEX  : ~/.cdsapirc
  ISD     : none (public FTP)
"""

from __future__ import annotations

import argparse
import logging
import os
import time

from tqdm import tqdm
from ftplib import FTP
from io import BytesIO, StringIO
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ─── Static configuration ─────────────────────────────────────────────────────

CMIP6_MODEL_VARS: dict[str, dict[str, str]] = {
    "mpi_esm1_2_lr": {
        "near_surface_air_temperature":               "daily",
        "near_surface_specific_humidity":             "daily",
        "near_surface_wind_speed":                    "daily",
        "daily_maximum_near_surface_air_temperature": "daily",
        "daily_minimum_near_surface_air_temperature": "daily",
        "precipitation":                              "daily",
        "sea_level_pressure":                         "daily",
        "surface_downwelling_shortwave_radiation":    "monthly",
    },
    "cnrm_esm2_1": {
        "near_surface_air_temperature":               "daily",
        "near_surface_specific_humidity":             "daily",
        "daily_maximum_near_surface_air_temperature": "daily",
        "daily_minimum_near_surface_air_temperature": "daily",
        "precipitation":                              "daily",
        "sea_level_pressure":                         "daily",
        "surface_downwelling_shortwave_radiation":    "monthly",
        "near_surface_wind_speed":                    "monthly",
    },
}

CMIP6_VAR_SHORT = {
    "near_surface_air_temperature":               "tas",
    "near_surface_specific_humidity":             "huss",
    "near_surface_wind_speed":                    "sfcWind",
    "daily_maximum_near_surface_air_temperature": "tasmax",
    "daily_minimum_near_surface_air_temperature": "tasmin",
    "precipitation":                              "pr",
    "sea_level_pressure":                         "psl",
    "surface_downwelling_shortwave_radiation":    "rsds",
}

CMIP6_EXP_YEARS = {
    "historical": list(range(1970, 2015)),
    "ssp2_4_5":   list(range(2015, 2101)),
    "ssp5_8_5":   list(range(2015, 2101)),
}

CMIP6_EXP_LABEL = {
    "historical": "historical",
    "ssp2_4_5":   "ssp245",
    "ssp5_8_5":   "ssp585",
}

# cordex_domain column value → CDS dataset domain string
CORDEX_DOMAIN_MAP = {
    "NAM-22": "north_america",
    "EUR-11": "europe",
    "SAM-22": "south_america",
    "AFR-22": "africa",
    "WAS-22": "west_asia",
    "EAS-22": "east_asia",
    "SEA-22": "south_east_asia",
    "AUS-22": "australasia",
}

CORDEX_DOMAIN_RES = {
    "north_america": "0_22_degree_x_0_22_degree",
    "europe":        "0_11_degree_x_0_11_degree",
    "south_america": "0_22_degree_x_0_22_degree",
    "africa":        "0_22_degree_x_0_22_degree",
}

CORDEX_VARS = [
    "2m_air_temperature",
    "2m_surface_specific_humidity",
    "10m_wind_speed",
    "maximum_2m_temperature_in_the_last_24_hours",
    "minimum_2m_temperature_in_the_last_24_hours",
    "mean_precipitation_flux",
    "surface_pressure",
    "surface_solar_radiation_downwards",
]

CORDEX_EXP_YEARS = {
    "historical": list(range(1970, 2006)),
    "rcp_8_5":    list(range(2006, 2101)),
}

NSRDB_ENDPOINT = (
    "https://developer.nlr.gov/api/nsrdb/v2/solar"
    "/nsrdb-GOES-aggregated-v4-0-0-download"
)
NSRDB_ATTRS = [
    "air_temperature", "dew_point", "relative_humidity",
    "wind_speed", "wind_direction", "ghi", "dni", "dhi", "surface_pressure",
]

ISD_FTP_HOST    = "ftp.ncei.noaa.gov"
ISD_FTP_DIR     = "/pub/data/noaa/{year}/"
ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"


def _credentials_path() -> Path:
    import sys
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path.home() / ".config"
    return base / "samy_wgen" / "credentials.yaml"

CREDENTIALS_PATH = _credentials_path()


# ─── Shared helpers ───────────────────────────────────────────────────────────


def _load_nsrdb_credentials() -> tuple[str, str]:
    """Return (api_key, email) from ~/.config/samy_wgen/credentials.yaml."""
    if not CREDENTIALS_PATH.exists():
        raise EnvironmentError(
            f"NSRDB credentials not found. Run 'samy-download config' to set them up.\n"
            f"Expected: {CREDENTIALS_PATH}"
        )
    import yaml
    data = yaml.safe_load(CREDENTIALS_PATH.read_text())
    try:
        nsrdb = data["nsrdb"]
        api_key = nsrdb["api_key"]
        email   = nsrdb["email"]
    except (KeyError, TypeError):
        raise EnvironmentError(
            f"Invalid credentials file {CREDENTIALS_PATH}. "
            "Expected keys: nsrdb.api_key and nsrdb.email"
        )
    if not api_key or not email:
        raise EnvironmentError(
            f"nsrdb.api_key or nsrdb.email is empty in {CREDENTIALS_PATH}."
        )
    return str(api_key), str(email)


def _save_nsrdb_credentials(api_key: str, email: str) -> None:
    """Write NSRDB credentials to ~/.config/samy_wgen/credentials.yaml."""
    import yaml
    CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if CREDENTIALS_PATH.exists():
        existing = yaml.safe_load(CREDENTIALS_PATH.read_text()) or {}
    existing["nsrdb"] = {"api_key": api_key, "email": email}
    CREDENTIALS_PATH.write_text(yaml.dump(existing, default_flow_style=False))
    import sys
    if sys.platform != "win32":
        CREDENTIALS_PATH.chmod(0o600)
    logger.info(f"Credentials saved → {CREDENTIALS_PATH}")

def _load_cities_config() -> pd.DataFrame:
    import importlib.resources as ir
    with ir.files("samy_wgen.config").joinpath("cities_config.csv").open() as f:
        return pd.read_csv(f)


def _city_meta(city: str) -> dict:
    """Return lat, lon, cordex_domain for the requested city.

    Matches against both `city` and `city_ascii` columns (case-insensitive).
    """
    cfg = _load_cities_config()
    key = city.lower()
    row = cfg[cfg["city"].str.lower() == key]
    if row.empty and "city_ascii" in cfg.columns:
        row = cfg[cfg["city_ascii"].str.lower() == key]
    if row.empty:
        raise ValueError(
            f"City '{city}' not found in cities_config. "
            f"Available: {cfg['city'].tolist()}"
        )
    r = row.iloc[0]
    return {
        "lat":           float(r["latitude"]),
        "lon":           float(r["longitude"]),
        "cordex_domain": str(r.get("cordex_domain", "NAM-22")),
    }


def _resolve_coords(
    city: str,
    lat: Optional[float],
    lon: Optional[float],
) -> tuple[float, float]:
    """Return (lat, lon), looking up from cities_config when not provided."""
    if lat is not None and lon is not None:
        return lat, lon
    meta = _city_meta(city)
    return meta["lat"], meta["lon"]


def _load_isd_history(cache_path: Path) -> pd.DataFrame:
    """Download and cache isd-history.csv from NOAA."""
    if not cache_path.exists():
        import requests
        logger.info(f"Downloading isd-history.csv → {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(ISD_HISTORY_URL, timeout=120)
        r.raise_for_status()
        cache_path.write_bytes(r.content)
    return pd.read_csv(cache_path, low_memory=False)


def _nc_bytes_to_point_df(nc_bytes: bytes, lat: float, lon: float) -> pd.DataFrame:
    """Extract a time-series DataFrame at the nearest grid point from NetCDF bytes.

    Handles both plain NetCDF and zip-wrapped NetCDF (CDS sometimes uses the
    latter with a .nc extension).
    """
    import numpy as np
    import xarray as xr

    # CDS occasionally wraps .nc inside a .zip with a .nc extension
    if nc_bytes[:2] == b"PK":
        import zipfile
        with zipfile.ZipFile(BytesIO(nc_bytes)) as zf:
            nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
            if not nc_names:
                raise ValueError("No .nc file found inside zip archive")
            nc_bytes = zf.read(nc_names[0])

    ds = xr.open_dataset(BytesIO(nc_bytes))

    # Resolve lat/lon coordinate names (CMIP6: lat/lon; CORDEX: may be rlat/rlon + lat2d/lon2d)
    lat_name = next((c for c in ds.coords if c in ("lat", "latitude")), None)
    lon_name = next((c for c in ds.coords if c in ("lon", "longitude")), None)
    if lat_name is None or lon_name is None:
        raise ValueError(f"lat/lon coordinates not found. Available: {list(ds.coords)}")

    lat_arr = ds[lat_name].values
    lon_arr = ds[lon_name].values

    # Wrap city longitude to match grid convention if grid uses [0, 360]
    if lon_arr.min() >= 0 and lon_arr.max() > 180:
        lon_target = lon % 360
    else:
        lon_target = lon

    # Nearest-neighbour index (handles both 1-D and 2-D coordinate arrays)
    if lat_arr.ndim == 1:
        li = int(np.argmin(np.abs(lat_arr - lat)))
        lj = int(np.argmin(np.abs(lon_arr - lon_target)))
    else:
        dist = (lat_arr - lat) ** 2 + (lon_arr - lon_target) ** 2
        li, lj = np.unravel_index(np.argmin(dist), dist.shape)

    time_index = pd.to_datetime(ds["time"].values)
    series: list[pd.Series] = []

    for var in ds.data_vars:
        if var.endswith(("_bnds", "_bounds")):
            continue
        da = ds[var]
        if "time" not in da.dims:
            continue
        if da.ndim == 3:
            values = da[:, li, lj].values
        elif da.ndim == 1:
            values = da.values
        else:
            continue
        series.append(pd.Series(values, index=time_index, name=var))

    ds.close()
    if not series:
        raise ValueError("No time-series data variables found in NetCDF")

    return pd.concat(series, axis=1).reset_index().rename(columns={"index": "time"})


# ─── ISD ──────────────────────────────────────────────────────────────────────

def list_isd_stations(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: float = 150.0,
    cache_dir: str = "data/ISD",
    city: Optional[str] = None,
) -> pd.DataFrame:
    """Print ISD stations within *radius_km* of the given coordinates.

    Returns a DataFrame with columns: USAF-WBAN, STATION NAME, dist_km, BEGIN, END.
    Use the USAF-WBAN value as --isd-station when downloading.
    """
    from geopy.distance import geodesic

    if city is not None:
        lat, lon = _resolve_coords(city, lat, lon)
    if lat is None or lon is None:
        raise ValueError("Provide either city or both lat and lon.")

    cache_path = Path(cache_dir) / "isd-history.csv"
    stations = _load_isd_history(cache_path)

    valid = stations.dropna(subset=["LAT", "LON"]).copy()
    valid["distance_km"] = valid.apply(
        lambda r: geodesic((lat, lon), (r["LAT"], r["LON"])).km, axis=1
    )
    nearby = valid[valid["distance_km"] <= radius_km].sort_values("distance_km").copy()

    if nearby.empty:
        print(f"No stations found within {radius_km} km of ({lat:.4f}, {lon:.4f})")
        return nearby

    def _year(v: object) -> str:
        try:
            return str(int(v))[:4]
        except Exception:
            return "?"

    display = nearby[["USAF", "WBAN", "STATION NAME", "distance_km", "BEGIN", "END"]].copy()
    display.insert(0, "USAF-WBAN",
        display["USAF"].astype(str).str.zfill(6) + "-"
        + display["WBAN"].astype(str).str.zfill(5)
    )
    display["dist_km"] = display["distance_km"].round(1)
    display["BEGIN"]   = display["BEGIN"].apply(_year)
    display["END"]     = display["END"].apply(_year)

    out = display[["USAF-WBAN", "STATION NAME", "dist_km", "BEGIN", "END"]].reset_index(drop=True)
    print(f"\nISD stations within {radius_km} km of ({lat:.4f}, {lon:.4f}):\n")
    print(out.to_string(index=False))
    print(f"\nTotal: {len(out)} station(s)")
    print("\nPass the USAF-WBAN value to --isd-station when running 'samy-download download'.\n")
    return out


def download_isd(
    station_id: str,
    city: str,
    years: List[int],
    output_dir: str = "data/ISD",
) -> None:
    """Download yearly ISD .gz files for *station_id* (format ``USAF-WBAN``)."""
    if "-" not in station_id:
        raise ValueError("station_id must be USAF-WBAN (e.g. '722020-12839')")
    usaf, wban = station_id.split("-", 1)
    usaf = usaf.zfill(6)
    wban = wban.zfill(5)

    cache_path = Path(output_dir) / "isd-history.csv"
    history = _load_isd_history(cache_path)
    match = history[
        (history["USAF"].astype(str).str.zfill(6) == usaf)
        & (history["WBAN"].astype(str).str.zfill(5) == wban)
    ]
    station_name = (
        match.iloc[0]["STATION NAME"].replace("/", "_").replace(" ", "_")
        if not match.empty else "UNKNOWN"
    )

    station_dir = Path(output_dir) / city / f"{usaf}-{wban}_{station_name}"
    station_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"ISD output → {station_dir}")

    ftp = FTP(ISD_FTP_HOST)
    ftp.login()
    try:
        with tqdm(sorted(years), desc=f"ISD {city}", unit="yr") as pbar:
            for year in pbar:
                pbar.set_postfix(year=year)
                filename = f"{usaf}-{wban}-{year}.gz"
                local    = station_dir / filename
                if local.exists() and local.stat().st_size > 0:
                    logger.debug(f"  {year} already present, skipping")
                    continue
                tmp = local.with_suffix(".gz.tmp")
                try:
                    ftp.cwd(ISD_FTP_DIR.format(year=year))
                    with open(tmp, "wb") as f:
                        ftp.retrbinary(f"RETR {filename}", f.write)
                    tmp.rename(local)
                except Exception as exc:
                    logger.warning(f"  {year} failed: {exc}")
                    if tmp.exists():
                        tmp.unlink()
    finally:
        ftp.quit()

    logger.info(f"ISD download complete ({len(years)} years requested) → {station_dir}")


# ─── NSRDB ────────────────────────────────────────────────────────────────────

def download_nsrdb(
    city: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    years: Optional[List[int]] = None,
    output_dir: str = "data/NSRDB",
    interval: int = 60,
) -> None:
    """Download NSRDB solar irradiance for a point location.

    Coordinates are looked up from cities_config.csv when *lat*/*lon* are omitted.
    Reads credentials from the ``NSRDB_API_KEY`` and ``NSRDB_EMAIL`` env vars.
    """
    import requests

    lat, lon = _resolve_coords(city, lat, lon)
    if years is None:
        years = list(range(1998, 2025))
    api_key, email = _load_nsrdb_credentials()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    city_lower = city.lower().replace(" ", "_")
    valid_years = [y for y in sorted(years) if 1998 <= y <= 2024]

    all_dfs: list[pd.DataFrame] = []
    for i, year in enumerate(tqdm(valid_years, desc=f"NSRDB {city}", unit="yr")):
        if i > 0:
            time.sleep(2)  # respect NREL rate limit
        params = {
            "api_key":    api_key,
            "wkt":        f"POINT({lon} {lat})",
            "attributes": ",".join(NSRDB_ATTRS),
            "names":      str(year),
            "utc":        "false",
            "leap_day":   "false",
            "interval":   interval,
            "email":      email,
        }
        logger.info(f"NSRDB {year} ...")
        try:
            r = requests.get(f"{NSRDB_ENDPOINT}.csv", params=params, timeout=300)
            r.raise_for_status()
            lines = r.text.strip().split("\n")
            header = next((i for i, l in enumerate(lines) if l.startswith("Year,")), None)
            if header is None:
                logger.warning(f"  {year}: unexpected response, skipping")
                continue
            df = pd.read_csv(StringIO("\n".join(lines[header:])))
            all_dfs.append(df)
            logger.info(f"  {year} ✓ ({len(df)} rows)")
        except Exception as exc:
            logger.warning(f"  {year} failed: {exc}")

    if not all_dfs:
        logger.error("No NSRDB data downloaded.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    y0, y1   = int(combined["Year"].min()), int(combined["Year"].max())
    out_path = out_dir / f"nsrdb_{city_lower}_{y0}-{y1}.csv"
    combined.to_csv(out_path, index=False)
    logger.info(f"NSRDB saved → {out_path} ({len(combined)} rows, {y0}–{y1})")


# ─── CMIP6 ────────────────────────────────────────────────────────────────────

def download_cmip6(
    city: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    models: Optional[List[str]] = None,
    experiments: Optional[List[str]] = None,
    output_dir: str = "data/CMIP6",
) -> None:
    """Download CMIP6 projections for a city via CDS, extracting the nearest grid point.

    Coordinates are looked up from cities_config.csv when *lat*/*lon* are omitted.
    Uses ``area: [lat, lon, lat, lon]`` in the CDS request so only the cell
    containing the city is transferred. Reads CDS credentials from ~/.cdsapirc.
    """
    import cdsapi

    lat, lon    = _resolve_coords(city, lat, lon)
    models      = models      or ["mpi_esm1_2_lr"]
    experiments = experiments or ["historical", "ssp2_4_5"]
    all_months  = [f"{i:02d}" for i in range(1, 13)]
    all_days    = [f"{i:02d}" for i in range(1, 32)]
    client      = cdsapi.Client()

    for model in models:
        if model not in CMIP6_MODEL_VARS:
            logger.warning(f"Unknown CMIP6 model '{model}', skipping")
            continue
        var_config = CMIP6_MODEL_VARS[model]
        city_dir   = Path(output_dir) / city
        city_dir.mkdir(parents=True, exist_ok=True)

        for cds_exp in experiments:
            years = CMIP6_EXP_YEARS.get(cds_exp)
            if not years:
                logger.warning(f"No year range configured for experiment '{cds_exp}'")
                continue
            label = CMIP6_EXP_LABEL[cds_exp]

            # Group variables by temporal resolution → separate output CSVs
            groups: dict[str, list[str]] = {}
            for var, res in var_config.items():
                groups.setdefault(res, []).append(var)

            for temporal_res, var_list in groups.items():
                suffix  = "" if temporal_res == "daily" else "_monthly"
                out_csv = city_dir / f"{model}_{label}{suffix}.csv"
                if out_csv.exists():
                    logger.info(f"  {out_csv.name} exists, skipping")
                    continue

                var_dfs: list[pd.DataFrame] = []
                pbar_desc = f"CMIP6 {city} {model} {label}"

                for variable in tqdm(var_list, desc=pbar_desc, unit="var"):
                    short   = CMIP6_VAR_SHORT.get(variable, variable)
                    tmp_nc  = city_dir / f"_tmp_{model}_{cds_exp}_{variable}.nc"
                    logger.info(f"  → {variable} ({short})")
                    try:
                        req: dict = {
                            "temporal_resolution": temporal_res,
                            "experiment":          cds_exp,
                            "variable":            variable,
                            "model":               model,
                            "year":                [str(y) for y in years],
                            "area":                [lat + 1.5, lon - 1.5, lat - 1.5, lon + 1.5],
                        }
                        if temporal_res == "daily":
                            req["month"] = all_months
                            req["day"]   = all_days
                        client.retrieve("projections-cmip6", req, str(tmp_nc))

                        nc_bytes = tmp_nc.read_bytes()
                        df_var   = _nc_bytes_to_point_df(nc_bytes, lat, lon)
                        rename   = {c: CMIP6_VAR_SHORT.get(c, c) for c in df_var.columns if c != "time"}
                        var_dfs.append(df_var.rename(columns=rename))
                        logger.info(f"    ✓ {short}")
                    except Exception as exc:
                        logger.warning(f"    ✗ {variable}: {exc}")
                    finally:
                        if tmp_nc.exists():
                            tmp_nc.unlink()
                    time.sleep(1)

                if not var_dfs:
                    logger.error(f"No variables downloaded for {model} {label} ({temporal_res})")
                    continue

                merged = var_dfs[0]
                for df_v in var_dfs[1:]:
                    merged = pd.merge(merged, df_v, on="time", how="outer")
                merged = merged.sort_values("time").reset_index(drop=True)
                merged.to_csv(out_csv, index=False)
                logger.info(f"CMIP6 saved → {out_csv} ({len(merged)} rows)")


# ─── CORDEX-CMIP5 ─────────────────────────────────────────────────────────────

def download_cordex_cmip5(
    city: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    cordex_domain: Optional[str] = None,
    gcm: str = "mpi_m_mpi_esm_lr",
    rcm: str = "ncar_regcm4",
    ensemble: str = "r1i1p1",
    experiments: Optional[List[str]] = None,
    output_dir: str = "data/CORDEX_CMIP5",
) -> None:
    """Download CORDEX-CMIP5 data for a city via CDS (per-year requests).

    Coordinates and cordex_domain are looked up from cities_config.csv when omitted.
    Uses ``area: [lat, lon, lat, lon]`` to download only the city's grid cell.
    Reads CDS credentials from ~/.cdsapirc.
    """
    import cdsapi

    lat, lon    = _resolve_coords(city, lat, lon)
    experiments = experiments or ["historical", "rcp_8_5"]
    if cordex_domain is None:
        cordex_domain = _city_meta(city)["cordex_domain"]

    cds_domain = CORDEX_DOMAIN_MAP.get(cordex_domain)
    if cds_domain is None:
        raise ValueError(
            f"Unknown cordex_domain '{cordex_domain}'. "
            f"Supported: {list(CORDEX_DOMAIN_MAP)}"
        )
    horizontal_res = CORDEX_DOMAIN_RES.get(cds_domain, "0_22_degree_x_0_22_degree")

    # Short RCM label for directory naming
    rcm_short = "RegCM" if "regcm" in rcm.lower() else rcm.replace("_", "").upper()
    client    = cdsapi.Client()

    for experiment in experiments:
        years = CORDEX_EXP_YEARS.get(experiment)
        if not years:
            logger.warning(f"No year range configured for CORDEX experiment '{experiment}'")
            continue
        exp_label = "historical" if experiment == "historical" else "rcp85"
        out_subdir = Path(output_dir) / f"{cordex_domain}_{rcm_short}_{exp_label}_{ensemble}"
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_csv = out_subdir / f"{city}_{exp_label}.csv"

        if out_csv.exists():
            logger.info(f"  {out_csv} exists, skipping")
            continue

        year_dfs: list[pd.DataFrame] = []
        pbar_desc = f"CORDEX {city} {exp_label}"

        for year in tqdm(years, desc=pbar_desc, unit="yr"):
            tmp_nc = out_subdir / f"_tmp_{city}_{experiment}_{year}.nc"
            try:
                req = {
                    "domain":               cds_domain,
                    "experiment":           experiment,
                    "horizontal_resolution": horizontal_res,
                    "temporal_resolution":  "daily_mean",
                    "variable":             CORDEX_VARS,
                    "gcm_model":            gcm,
                    "rcm_model":            rcm,
                    "ensemble_member":      ensemble,
                    "start_year":           [str(year)],
                    "end_year":             [str(year)],
                    "area":                 [lat, lon, lat, lon],
                }
                client.retrieve(
                    "projections-cordex-domains-single-levels", req, str(tmp_nc)
                )
                nc_bytes = tmp_nc.read_bytes()
                df_year  = _nc_bytes_to_point_df(nc_bytes, lat, lon)
                year_dfs.append(df_year)
                logger.info(f"  {year} ✓")
            except Exception as exc:
                logger.warning(f"  {year} failed: {exc}")
            finally:
                if tmp_nc.exists():
                    tmp_nc.unlink()
            time.sleep(1)

        if not year_dfs:
            logger.error(f"No CORDEX data downloaded for {experiment}")
            continue

        merged = pd.concat(year_dfs, ignore_index=True)
        merged = merged.sort_values("time").reset_index(drop=True)
        merged.to_csv(out_csv, index=False)
        logger.info(f"CORDEX saved → {out_csv} ({len(merged)} rows)")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="samy-download",
        description="Download city-specific weather data for the samy-wgen pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── config ─────────────────────────────────────────────────────────────────
    sub.add_parser("config", help="Save NSRDB credentials to ~/.config/samy_wgen/credentials.yaml")

    # ── list-stations ──────────────────────────────────────────────────────────
    p_list = sub.add_parser("list-stations", help="List nearby ISD stations for a city")
    p_list.add_argument("--city",    required=True,    help="City name (from cities_config.csv)")
    p_list.add_argument("--radius",  type=float, default=150, help="Search radius in km (default: 150)")
    p_list.add_argument("--isd-dir", default="data/ISD", help="Directory to cache isd-history.csv")

    # ── download ───────────────────────────────────────────────────────────────
    p_dl = sub.add_parser("download", help="Download weather data for a city")
    p_dl.add_argument("--city", required=True, help="City name (from cities_config.csv)")
    p_dl.add_argument(
        "--data", nargs="+",
        choices=["isd", "nsrdb", "cmip6", "cordex"],
        default=["isd", "nsrdb", "cmip6"],
        help="Data sources to download (default: isd nsrdb cmip6)",
    )

    # ISD
    p_dl.add_argument("--isd-station", metavar="USAF-WBAN",
                      help="ISD station ID, e.g. 722020-12839 (required when --data includes isd)")
    p_dl.add_argument("--isd-dir",  default="data/ISD")

    # NSRDB
    p_dl.add_argument("--nsrdb-dir", default="data/NSRDB")

    # CMIP6
    p_dl.add_argument("--cmip6-dir",  default="data/CMIP6")
    p_dl.add_argument(
        "--models", nargs="+", default=["mpi_esm1_2_lr"],
        help="CMIP6 models (default: mpi_esm1_2_lr)",
    )
    p_dl.add_argument(
        "--experiments", nargs="+", default=["historical", "ssp2_4_5"],
        help="CMIP6 CDS experiment names (default: historical ssp2_4_5)",
    )

    # CORDEX
    p_dl.add_argument("--cordex-dir",        default="data/CORDEX_CMIP5")
    p_dl.add_argument("--cordex-gcm",        default="mpi_m_mpi_esm_lr")
    p_dl.add_argument("--cordex-rcm",        default="ncar_regcm4")
    p_dl.add_argument("--cordex-ensemble",   default="r1i1p1")
    p_dl.add_argument(
        "--cordex-experiments", nargs="+",   default=["historical", "rcp_8_5"],
        help="CORDEX experiments (default: historical rcp_8_5)",
    )

    # ISD / NSRDB year range
    p_dl.add_argument(
        "--years", nargs=2, type=int, metavar=("START", "END"), default=None,
        help="Year range for ISD / NSRDB (default: full available history)",
    )

    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = _build_parser()
    args   = parser.parse_args()

    if args.command == "config":
        print(f"Saving NSRDB credentials to {CREDENTIALS_PATH}")
        api_key = input("  NSRDB API key : ").strip()
        email   = input("  NSRDB email   : ").strip()
        _save_nsrdb_credentials(api_key, email)
        print("Done. Credentials saved.")
        return

    if args.command == "list-stations":
        meta = _city_meta(args.city)
        list_isd_stations(meta["lat"], meta["lon"], args.radius, args.isd_dir)
        return

    # ── download ───────────────────────────────────────────────────────────────
    meta    = _city_meta(args.city)
    lat     = meta["lat"]
    lon     = meta["lon"]
    sources = set(args.data)

    if "isd" in sources:
        if not args.isd_station:
            parser.error(
                "--isd-station USAF-WBAN is required when downloading ISD data. "
                "Run 'samy-download list-stations --city <city>' to see available stations."
            )
        yr_start, yr_end = args.years if args.years else (1940, 2024)
        download_isd(
            station_id=args.isd_station,
            city=args.city,
            years=list(range(yr_start, yr_end + 1)),
            output_dir=args.isd_dir,
        )

    if "nsrdb" in sources:
        yr_start, yr_end = args.years if args.years else (1998, 2024)
        download_nsrdb(
            city=args.city,
            lat=lat, lon=lon,
            years=list(range(yr_start, yr_end + 1)),
            output_dir=args.nsrdb_dir,
        )

    if "cmip6" in sources:
        download_cmip6(
            city=args.city,
            lat=lat, lon=lon,
            models=args.models,
            experiments=args.experiments,
            output_dir=args.cmip6_dir,
        )

    if "cordex" in sources:
        download_cordex_cmip5(
            city=args.city,
            lat=lat, lon=lon,
            cordex_domain=meta["cordex_domain"],
            gcm=args.cordex_gcm,
            rcm=args.cordex_rcm,
            ensemble=args.cordex_ensemble,
            experiments=args.cordex_experiments,
            output_dir=args.cordex_dir,
        )


if __name__ == "__main__":
    main()
