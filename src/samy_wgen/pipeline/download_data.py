"""samy-download: City-specific weather data downloader for samy-wgen pipeline.

Subcommands
-----------
  list-stations   Print nearby ISD stations for a city (choose one, then download)
  download        Download ISD / NSRDB / ERA5-Land / CMIP6 / CORDEX-CMIP5 for a city
  extract-cmip6   Extract nearest-grid city series from local CMIP6 NetCDF files

Credentials
-----------
  NSRDB   : ~/.config/samy_wgen/credentials.yaml
              Run 'samy-download config' to write this file interactively.
  ERA5-Land: ~/.cdsapirc
  CMIP6   : ~/.cdsapirc  (standard cdsapi config)
  CORDEX  : ~/.cdsapirc
  ISD     : none (public HTTPS archive)
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import functools
import hashlib
import json
import logging
import os
import shutil
import time
import zipfile

from tqdm import tqdm
from io import BytesIO, StringIO
from pathlib import Path
from typing import List, Optional

import pandas as pd
from samy_wgen.utils.city_utils import load_city_table, resolve_city, resolve_city_codes

logger = logging.getLogger(__name__)

# ─── Static configuration ─────────────────────────────────────────────────────

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
    "ssp1_2_6":   list(range(2015, 2101)),
    "ssp2_4_5":   list(range(2015, 2101)),
    "ssp3_7_0":   list(range(2015, 2101)),
    "ssp5_8_5":   list(range(2015, 2101)),
}

CMIP6_EXP_LABEL = {
    "historical": "historical",
    "ssp1_2_6":   "ssp126",
    "ssp2_4_5":   "ssp245",
    "ssp3_7_0":   "ssp370",
    "ssp5_8_5":   "ssp585",
}

# cordex_domain column value → CDS dataset domain string
CORDEX_DOMAIN_MAP = {
    "NAM-22": "north_america",
    "EUR-11": "europe",
    "SAM-22": "south_america",
    "AFR-22": "africa",
    "WAS-22": "south_asia",
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
    "2m_relative_humidity",
    "2m_surface_specific_humidity",
    "10m_wind_speed",
    "maximum_2m_temperature_in_the_last_24_hours",
    "minimum_2m_temperature_in_the_last_24_hours",
    "mean_precipitation_flux",
    "surface_pressure",
    "surface_solar_radiation_downwards",
]
CORDEX_VAR_SHORT = {
    "2m_air_temperature": "tas",
    "2m_relative_humidity": "hurs",
    "2m_surface_specific_humidity": "huss",
    "10m_wind_speed": "sfcWind",
    "maximum_2m_temperature_in_the_last_24_hours": "tasmax",
    "minimum_2m_temperature_in_the_last_24_hours": "tasmin",
    "mean_precipitation_flux": "pr",
    "surface_pressure": "ps",
    "surface_solar_radiation_downwards": "rsds",
}

CORDEX_EXP_YEARS = {
    "historical": list(range(1970, 2006)),
    "rcp_8_5":    list(range(2006, 2101)),
}

NSRDB_SOURCES: dict[str, dict] = {
    "goes_aggregated": {
        "url": "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download",
        "year_range": (1998, 2024),
    },
    "goes_full_disc": {
        "url": "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-full-disc-v4-0-0-download",
        "year_range": (2018, 2024),
    },
    "meteosat": {
        "url": "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-msg-v1-0-0-download",
        "year_range": (2017, 2019),
    },
    # Himawari satellite covers East Asia, Southeast Asia, Australasia
    "himawari7": {
        "url": "https://developer.nlr.gov/api/nsrdb/v2/solar/himawari7-download",
        "year_range": (2011, 2015),
    },
    "himawari8": {
        "url": "https://developer.nlr.gov/api/nsrdb/v2/solar/himawari-download",
        "year_range": (2016, 2020),
    },
    "suny_india": {
        "url": "https://developer.nlr.gov/api/nsrdb/v2/solar/suny-india-download",
        "year_range": (2000, 2014),
    },
}

NSRDB_GROUPS: dict[str, tuple[str, ...]] = {
    "goes_aggregated": ("goes_aggregated",),
    "goes_full_disc": ("goes_full_disc",),
    "meteosat": ("meteosat",),
    "himawari_pair": ("himawari7", "himawari8"),
    "suny_india": ("suny_india",),
}

NSRDB_GROUP_PRIORITY = [
    "goes_aggregated",
    "suny_india",
    "himawari_pair",
    "goes_full_disc",
    "meteosat",
]

NSRDB_GROUP_LABELS = {
    "goes_aggregated": "goes_aggregated",
    "goes_full_disc": "goes_full_disc",
    "meteosat": "meteosat",
    "himawari_pair": "himawari7 + himawari8",
    "suny_india": "suny_india",
}

# CORDEX domain → ordered list of NSRDB source keys covering that region.
# Regions with multiple entries (Himawari) are routed per-year automatically.
CORDEX_TO_NSRDB_SOURCE: dict[str, list[str]] = {
    "NAM-22": ["goes_aggregated"],
    "SAM-22": ["goes_full_disc"],
    "EUR-11": ["meteosat"],
    "AFR-22": ["meteosat"],
    "WAS-22": ["meteosat"],
    "EAS-22": ["himawari7", "himawari8"],
    "SEA-22": ["himawari7", "himawari8"],
    "AUS-22": ["himawari7", "himawari8"],
}

NSRDB_URL_TO_SOURCE_KEY = {
    source["url"]: key for key, source in NSRDB_SOURCES.items()
}

NSRDB_ATTRS = [
    "air_temperature", "dew_point", "relative_humidity",
    "wind_speed", "wind_direction", "ghi", "dni", "dhi", "surface_pressure",
]

ISD_HTTP_URL    = "https://www.ncei.noaa.gov/pub/data/noaa/{year}/{filename}"
ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"

CDS_CMIP6_DATASET = "projections-cmip6"
CDS_CORDEX_DATASET = "projections-cordex-domains-single-levels"
CMIP6_SUPPORTED_VARIABLES = set(CMIP6_VAR_SHORT.keys())
CMIP6_TEMPORAL_PRIORITY = ("daily", "monthly", "fixed")
CORDEX_SUPPORTED_VARIABLES = set(CORDEX_VARS)
CORDEX_DOMAIN_BY_CDS = {value: key for key, value in CORDEX_DOMAIN_MAP.items()}
DEFAULT_ISD_YEAR_RANGE = (1970, 2024)
DEFAULT_NSRDB_YEAR_RANGE = (1998, 2024)
ERA5_LAND_DATASET = "reanalysis-era5-land-timeseries"
ERA5_LAND_AVAILABILITY_LAG_DAYS = 5
ERA5_LAND_VARIABLES = [
    "2m_dewpoint_temperature",
    "2m_temperature",
    "surface_pressure",
    "total_precipitation",
    "surface_solar_radiation_downwards",
]


def _credentials_path() -> Path:
    import sys
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path.home() / ".config"
    return base / "samy_wgen" / "credentials.yaml"

CREDENTIALS_PATH = _credentials_path()


def _cache_dir() -> Path:
    import sys
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path.home() / ".cache"
    return base / "samy_wgen"


CONSTRAINT_CACHE_PATH = _cache_dir() / "cds_constraints.json"


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


def _load_cds_credentials() -> tuple[str, str]:
    """Return (url, key) from ~/.cdsapirc for CDS-compatible clients."""
    path = Path.home() / ".cdsapirc"
    if not path.exists():
        raise EnvironmentError(
            "CDS credentials not found. Create ~/.cdsapirc before requesting CMIP6/CORDEX metadata."
        )

    cfg: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        cfg[key.strip()] = value.strip()

    try:
        return cfg["url"], cfg["key"]
    except KeyError as exc:
        raise EnvironmentError(f"Invalid CDS credentials file {path}. Expected keys: url, key") from exc


@functools.lru_cache(maxsize=1)
def _get_datastores_client():
    """Build an ECMWF datastores client from the local CDS credentials."""
    from ecmwf.datastores import Client

    url, key = _load_cds_credentials()
    return Client(url=url, key=key, progress=False)


@functools.lru_cache(maxsize=4)
def _get_collection(collection_id: str):
    """Return a cached CDS collection object."""
    return _get_datastores_client().get_collection(collection_id)


@functools.lru_cache(maxsize=1)
def _load_constraint_cache() -> dict[str, dict[str, list[str]]]:
    """Load the persistent CDS constraints cache."""
    if not CONSTRAINT_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CONSTRAINT_CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_constraint_cache(cache: dict[str, dict[str, list[str]]]) -> None:
    """Persist the CDS constraints cache."""
    CONSTRAINT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONSTRAINT_CACHE_PATH.write_text(json.dumps(cache, sort_keys=True))
    _load_constraint_cache.cache_clear()


def _constraint_cache_key(collection_id: str, request: dict[str, list[str]]) -> str:
    """Build a stable cache key for a collection constraint request."""
    return json.dumps(
        {"collection_id": collection_id, "request": request},
        sort_keys=True,
        separators=(",", ":"),
    )


def _apply_collection_constraints(collection_id: str, request: dict[str, list[str]]) -> dict[str, list[str]]:
    """Apply collection constraints and return the remaining value space."""
    cache = _load_constraint_cache()
    cache_key = _constraint_cache_key(collection_id, request)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    collection = _get_collection(collection_id)
    constrained = collection.apply_constraints(request)
    if not isinstance(constrained, dict):
        raise TypeError(
            f"Unexpected constraints response for {collection_id}: {type(constrained).__name__}"
        )
    cache[cache_key] = constrained
    _save_constraint_cache(cache)
    return constrained


def _requested_values_available(
    constrained: dict[str, list[str]],
    request: dict[str, list[str]],
    fields: list[str],
) -> bool:
    """Return True when every requested value still appears in the constrained value set."""
    for field in fields:
        requested = request.get(field, [])
        available = constrained.get(field, [])
        if not requested:
            continue
        if not available:
            return False
        if any(value not in available for value in requested):
            return False
    return True


def _get_cmip6_variable_groups(model: str, experiment: str) -> dict[str, list[str]]:
    """Return CMIP6 variables grouped by their highest available CDS temporal resolution."""
    grouped_candidates: dict[str, list[str]] = {}
    selected_resolution: dict[str, str] = {}

    for temporal_resolution in CMIP6_TEMPORAL_PRIORITY:
        request = {
            "model": [model],
            "experiment": [experiment],
            "temporal_resolution": [temporal_resolution],
        }
        constrained = _apply_collection_constraints(CDS_CMIP6_DATASET, request)
        if not _requested_values_available(
            constrained, request, ["model", "experiment", "temporal_resolution"]
        ):
            continue

        variables = sorted(
            var for var in constrained.get("variable", []) if var in CMIP6_SUPPORTED_VARIABLES
        )
        for variable in variables:
            if variable in selected_resolution:
                continue
            grouped_candidates.setdefault(temporal_resolution, []).append(variable)
            selected_resolution[variable] = temporal_resolution

    return {
        temporal_resolution: sorted(variables)
        for temporal_resolution, variables in grouped_candidates.items()
        if variables
    }


def _filter_cmip6_variable_groups(
    groups: dict[str, list[str]],
    requested_variables: list[str] | None,
) -> dict[str, list[str]]:
    """Filter CMIP6 variable groups to the requested subset."""
    if requested_variables is None:
        return groups

    requested_set = set(requested_variables)
    return {
        temporal_resolution: [var for var in variables if var in requested_set]
        for temporal_resolution, variables in groups.items()
        if any(var in requested_set for var in variables)
    }


def _list_cmip6_models(
    experiment: str | None = None,
    variable: str | None = None,
    temporal_resolution: str | None = None,
) -> dict[str, list[str]]:
    """Return the currently available CMIP6 models and compatible request values from CDS."""
    request: dict[str, list[str]] = {}
    if experiment:
        request["experiment"] = [experiment]
    if variable:
        request["variable"] = [variable]
    if temporal_resolution:
        request["temporal_resolution"] = [temporal_resolution]

    constrained = _apply_collection_constraints(CDS_CMIP6_DATASET, request)
    if request and not _requested_values_available(
        constrained, request, list(request.keys())
    ):
        return {"model": [], "experiment": [], "variable": [], "temporal_resolution": []}

    return {
        "model": sorted(constrained.get("model", [])),
        "experiment": sorted(constrained.get("experiment", [])),
        "variable": sorted(constrained.get("variable", [])),
        "temporal_resolution": sorted(constrained.get("temporal_resolution", [])),
    }


def _list_cmip6_model_rows(
    experiment: str | None = None,
    variable: str | None = None,
    temporal_resolution: str | None = None,
    show_progress: bool = False,
) -> list[dict[str, str]]:
    """Return CMIP6 availability rows as model/experiment/resolution/variable tuples."""
    model_info = _list_cmip6_models(
        experiment=experiment,
        variable=variable,
        temporal_resolution=temporal_resolution,
    )
    rows: list[dict[str, str]] = []

    model_iter = model_info["model"]
    if show_progress:
        model_iter = tqdm(model_iter, desc="CDS model details", unit="model")

    for model in model_iter:
        request: dict[str, list[str]] = {"model": [model]}
        if experiment:
            request["experiment"] = [experiment]
        if variable:
            request["variable"] = [variable]
        if temporal_resolution:
            request["temporal_resolution"] = [temporal_resolution]

        detail = _apply_collection_constraints(CDS_CMIP6_DATASET, request)
        experiments = [experiment] if experiment else sorted(detail.get("experiment", []))
        resolutions = [temporal_resolution] if temporal_resolution else sorted(
            detail.get("temporal_resolution", []),
            key=lambda item: CMIP6_TEMPORAL_PRIORITY.index(item)
            if item in CMIP6_TEMPORAL_PRIORITY else len(CMIP6_TEMPORAL_PRIORITY),
        )

        for exp in experiments:
            for resolution in resolutions:
                scoped = _apply_collection_constraints(
                    CDS_CMIP6_DATASET,
                    {
                        "model": [model],
                        "experiment": [exp],
                        "temporal_resolution": [resolution],
                    },
                )
                variables = [
                    var for var in sorted(scoped.get("variable", []))
                    if var in CMIP6_SUPPORTED_VARIABLES and (variable is None or var == variable)
                ]
                if not variables:
                    continue
                rows.append(
                    {
                        "model": model,
                        "experiment": exp,
                        "resolution": resolution,
                        "variable": ",".join(CMIP6_VAR_SHORT[var] for var in variables),
                    }
                )

    return rows


def _normalize_cordex_domain(domain: str | None) -> str | None:
    """Translate user-facing CORDEX domain IDs into CDS domain IDs."""
    if domain is None:
        return None
    if domain in CORDEX_DOMAIN_MAP:
        return CORDEX_DOMAIN_MAP[domain]
    if domain in CORDEX_DOMAIN_BY_CDS:
        return domain
    raise ValueError(
        f"Unknown cordex_domain '{domain}'. Supported: {sorted(CORDEX_DOMAIN_MAP)}"
    )


def _list_cordex_options(
    domain: str | None = None,
    experiment: str | None = None,
    gcm: str | None = None,
    rcm: str | None = None,
    ensemble: str | None = None,
    resolution: str | None = None,
    variable: str | None = None,
) -> dict[str, list[str]]:
    """Return currently available CORDEX request values from CDS."""
    request: dict[str, list[str]] = {}
    cds_domain = _normalize_cordex_domain(domain)
    if cds_domain:
        request["domain"] = [cds_domain]
    if experiment:
        request["experiment"] = [experiment]
    if gcm:
        request["gcm_model"] = [gcm]
    if rcm:
        request["rcm_model"] = [rcm]
    if ensemble:
        request["ensemble_member"] = [ensemble]
    if resolution:
        request["horizontal_resolution"] = [resolution]
    if variable:
        request["variable"] = [variable]

    constrained = _apply_collection_constraints(CDS_CORDEX_DATASET, request)
    if request and not _requested_values_available(constrained, request, list(request.keys())):
        return {
            "domain": [],
            "experiment": [],
            "gcm_model": [],
            "rcm_model": [],
            "ensemble_member": [],
            "horizontal_resolution": [],
            "variable": [],
        }

    return {
        "domain": sorted(CORDEX_DOMAIN_BY_CDS.get(item, item) for item in constrained.get("domain", [])),
        "experiment": sorted(constrained.get("experiment", [])),
        "gcm_model": sorted(constrained.get("gcm_model", [])),
        "rcm_model": sorted(constrained.get("rcm_model", [])),
        "ensemble_member": sorted(constrained.get("ensemble_member", [])),
        "horizontal_resolution": sorted(constrained.get("horizontal_resolution", [])),
        "variable": sorted(
            var for var in constrained.get("variable", []) if var in CORDEX_SUPPORTED_VARIABLES
        ),
    }


def _list_cordex_rows(
    domain: str | None = None,
    experiment: str | None = None,
    gcm: str | None = None,
    rcm: str | None = None,
    ensemble: str | None = None,
    resolution: str | None = None,
    variable: str | None = None,
    show_progress: bool = False,
    expand_ensemble: bool = False,
) -> list[dict[str, str]]:
    """Return CORDEX availability rows as domain/experiment/GCM/RCM/ensemble/resolution tuples."""
    options = _list_cordex_options(
        domain=domain,
        experiment=experiment,
        gcm=gcm,
        rcm=rcm,
        ensemble=ensemble,
        resolution=resolution,
        variable=variable,
    )
    rows: list[dict[str, str]] = []

    if expand_ensemble or ensemble is not None:
        base_rows = [
            {"gcm_model": gcm_id, "rcm_model": rcm_id, "ensemble_member": ens_id}
            for gcm_id in options["gcm_model"]
            for rcm_id in options["rcm_model"]
            for ens_id in options["ensemble_member"]
        ]
    else:
        base_rows = [
            {"gcm_model": gcm_id, "rcm_model": rcm_id}
            for gcm_id in options["gcm_model"]
            for rcm_id in options["rcm_model"]
        ]
    if show_progress:
        base_rows = tqdm(base_rows, desc="CDS CORDEX details", unit="combo")

    for combo in base_rows:
        request: dict[str, list[str]] = {
            "gcm_model": [combo["gcm_model"]],
            "rcm_model": [combo["rcm_model"]],
        }
        if "ensemble_member" in combo:
            request["ensemble_member"] = [combo["ensemble_member"]]
        cds_domain = _normalize_cordex_domain(domain)
        if cds_domain:
            request["domain"] = [cds_domain]
        if experiment:
            request["experiment"] = [experiment]
        if resolution:
            request["horizontal_resolution"] = [resolution]
        if variable:
            request["variable"] = [variable]

        detail = _apply_collection_constraints(CDS_CORDEX_DATASET, request)
        if not _requested_values_available(detail, request, list(request.keys())):
            continue

        available_ensembles = sorted(detail.get("ensemble_member", []))
        if "ensemble_member" in combo:
            ensembles = [combo["ensemble_member"]]
        elif available_ensembles:
            ensembles = [available_ensembles[0]]
        else:
            continue

        domains = [cds_domain] if cds_domain else sorted(detail.get("domain", []))
        experiments = [experiment] if experiment else sorted(detail.get("experiment", []))
        resolutions = [resolution] if resolution else sorted(detail.get("horizontal_resolution", []))

        for cds_domain_value in domains:
            for exp in experiments:
                for ens in ensembles:
                    for res in resolutions:
                        scoped = _apply_collection_constraints(
                            CDS_CORDEX_DATASET,
                            {
                                "domain": [cds_domain_value],
                                "experiment": [exp],
                                "gcm_model": [combo["gcm_model"]],
                                "rcm_model": [combo["rcm_model"]],
                                "ensemble_member": [ens],
                                "horizontal_resolution": [res],
                            },
                        )
                        variables = [
                            var for var in sorted(scoped.get("variable", []))
                            if var in CORDEX_SUPPORTED_VARIABLES and (variable is None or var == variable)
                        ]
                        if not variables:
                            continue
                        rows.append(
                            {
                                "domain": CORDEX_DOMAIN_BY_CDS.get(cds_domain_value, cds_domain_value),
                                "experiment": exp,
                                "gcm_model": combo["gcm_model"],
                                "rcm_model": combo["rcm_model"],
                                "ensemble_member": ens,
                                "resolution": res,
                                "variable": ",".join(variables),
                            }
                        )

    return rows


def _resolve_cordex_download_plan(
    cordex_domain: str,
    gcm: str,
    rcm: str,
    ensemble: str | None,
    experiments: list[str],
    variables: list[str] | None = None,
) -> list[dict[str, str | list[str]]]:
    """Resolve requested CORDEX filters into valid CDS download combinations."""
    cds_domain = _normalize_cordex_domain(cordex_domain)
    assert cds_domain is not None
    requested_variables = variables or sorted(CORDEX_SUPPORTED_VARIABLES)
    plan: list[dict[str, str | list[str]]] = []

    for experiment in experiments:
        request = {
            "domain": [cds_domain],
            "experiment": [experiment],
            "gcm_model": [gcm],
            "rcm_model": [rcm],
        }
        if ensemble is not None:
            request["ensemble_member"] = [ensemble]
        constrained = _apply_collection_constraints(CDS_CORDEX_DATASET, request)
        if not _requested_values_available(constrained, request, list(request.keys())):
            ensemble_msg = ensemble if ensemble is not None else "<auto>"
            logger.warning(
                "CORDEX combination not available in CDS: "
                f"domain='{cordex_domain}', experiment='{experiment}', gcm='{gcm}', rcm='{rcm}', ensemble='{ensemble_msg}'. "
                "Run 'samy-download list-cordex' to inspect currently available combinations."
            )
            continue

        available_ensembles = sorted(constrained.get("ensemble_member", []))
        selected_ensemble = ensemble if ensemble is not None else (available_ensembles[0] if available_ensembles else None)
        if selected_ensemble is None:
            logger.warning(
                "No CORDEX ensemble_member returned for "
                f"domain='{cordex_domain}', experiment='{experiment}', gcm='{gcm}', rcm='{rcm}'."
            )
            continue

        available_resolutions = sorted(constrained.get("horizontal_resolution", []))
        if not available_resolutions:
            logger.warning(
                f"No CORDEX horizontal_resolution returned for domain='{cordex_domain}', experiment='{experiment}'."
            )
            continue

        preferred_resolution = CORDEX_DOMAIN_RES.get(cds_domain)
        resolution = (
            preferred_resolution
            if preferred_resolution in available_resolutions
            else available_resolutions[0]
        )
        scoped = _apply_collection_constraints(
            CDS_CORDEX_DATASET,
            {
                "domain": [cds_domain],
                "experiment": [experiment],
                "gcm_model": [gcm],
                "rcm_model": [rcm],
                "ensemble_member": [selected_ensemble],
                "horizontal_resolution": [resolution],
            },
        )
        available_variables = [
            var for var in sorted(scoped.get("variable", []))
            if var in CORDEX_SUPPORTED_VARIABLES and var in requested_variables
        ]
        if not available_variables:
            logger.warning(
                "No supported CORDEX variables available after applying CDS constraints for "
                f"domain='{cordex_domain}', experiment='{experiment}', gcm='{gcm}', rcm='{rcm}', ensemble='{ensemble}', "
                f"resolution='{resolution}'."
            )
            continue

        plan.append(
            {
                "domain": cds_domain,
                "experiment": experiment,
                "gcm_model": gcm,
                "rcm_model": rcm,
                "ensemble_member": selected_ensemble,
                "horizontal_resolution": resolution,
                "variable": available_variables,
            }
        )

    return plan

def _load_cities_config() -> pd.DataFrame:
    return load_city_table()


def _city_meta(city: str) -> dict:
    """Return lat, lon, cordex_domain for the requested ASCII city code."""
    r = resolve_city(city)
    return {
        "lat": r.latitude,
        "lon": r.longitude,
        "cordex_domain": r.cordex_domain,
        "country_iso2": r.country_iso2,
    }


def _normalize_city_name(city: str, cfg: pd.DataFrame | None = None) -> str:
    """Return the canonical ASCII city code from cities_config.csv for one request."""
    del cfg
    return resolve_city(city).city_ascii


def _normalize_city_args(requested: list[str]) -> list[str]:
    """Resolve city arguments to canonical ASCII city codes while preserving order."""
    return resolve_city_codes(requested)


def _parse_isd_station_args(
    values: list[str] | None,
    cities: list[str],
) -> dict[str, str] | None:
    """Parse `--isd-station` values into a canonical city -> station_id mapping.

    Supported forms:
    - single-city form: `--isd-station 722020-12839`
    - multi/single-city mapped form: `--isd-station Miami=722020-12839`
    """
    if not values:
        return None

    canonical_cities = _normalize_city_args(cities)
    if len(values) == 1 and "=" not in values[0]:
        if len(canonical_cities) != 1:
            raise SystemExit(
                "ERROR: Multi-city --isd-station values must use city_ascii=USAF-WBAN mappings."
            )
        return {canonical_cities[0]: values[0]}

    station_ids: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(
                "ERROR: --isd-station entries must be USAF-WBAN or city_ascii=USAF-WBAN."
            )
        raw_city, station_id = item.split("=", 1)
        city = _normalize_city_name(raw_city.strip())
        station_id = station_id.strip()
        if city not in canonical_cities:
            raise SystemExit(
                f"ERROR: --isd-station city '{city}' is not present in --city {canonical_cities}."
            )
        station_ids[city] = station_id
    return station_ids


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


def _resolve_city_points(cities: list[str]) -> dict[str, tuple[float, float]]:
    """Return canonical city -> (lat, lon) mapping."""
    points: dict[str, tuple[float, float]] = {}
    for city in _normalize_city_args(cities):
        meta = _city_meta(city)
        points[city] = (meta["lat"], meta["lon"])
    return points


def _resolve_nsrdb_source(city: str) -> list[tuple[str, int, int]]:
    """Return a list of (endpoint_url, min_year, max_year) for the city's NSRDB coverage.

    Regions with a single satellite (Americas, Europe/Africa) return one entry.
    Himawari regions (East Asia, SE Asia, Australasia) return two entries that
    together span 2011–2020; callers should route each year to the matching entry.
    Raises ValueError when the city's region has no NSRDB coverage.
    """
    meta = _city_meta(city)
    if meta["country_iso2"] == "IN":
        source_keys = ["suny_india"]
        return [
            (
                NSRDB_SOURCES[key]["url"],
                NSRDB_SOURCES[key]["year_range"][0],
                NSRDB_SOURCES[key]["year_range"][1],
            )
            for key in source_keys
        ]

    cordex_domain = meta["cordex_domain"]
    source_keys = CORDEX_TO_NSRDB_SOURCE.get(cordex_domain, [])
    if not source_keys:
        raise ValueError(
            f"No NSRDB coverage for city '{city}' (CORDEX domain '{cordex_domain}'). "
            "NSRDB is available for: "
            + ", ".join(d for d, keys in CORDEX_TO_NSRDB_SOURCE.items() if keys)
        )
    return [
        (
            NSRDB_SOURCES[key]["url"],
            NSRDB_SOURCES[key]["year_range"][0],
            NSRDB_SOURCES[key]["year_range"][1],
        )
        for key in source_keys
    ]


def _extract_years(values: list[object] | None) -> list[int]:
    years: list[int] = []
    for value in values or []:
        if isinstance(value, bool):
            continue
        try:
            year = int(value)
        except (TypeError, ValueError):
            continue
        years.append(year)
    return sorted(set(years))


def _query_nsrdb_available_years(
    lat: float,
    lon: float,
    api_key: str,
    session=None,
) -> dict[str, list[int]]:
    """Query NSRDB Data Query API and return endpoint_url -> available years."""
    import requests

    session = session or requests.Session()
    response = session.get(
        "https://developer.nlr.gov/api/solar/nsrdb_data_query.json",
        params={"api_key": api_key, "lat": lat, "lon": lon},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    outputs = payload.get("outputs", [])

    years_by_url: dict[str, list[int]] = {}
    for output in outputs:
        links = output.get("links", [])
        available_years = _extract_years(output.get("availableYears"))
        if not available_years:
            continue
        for link in links:
            link_url = str(link.get("link", ""))
            if not link_url:
                continue
            base_url = link_url.split(".csv", 1)[0]
            years_by_url[base_url] = available_years
    return years_by_url


def _nsrdb_years_to_ranges(years: list[int]) -> str:
    if not years:
        return "none"
    ordered = sorted(set(int(year) for year in years))
    ranges: list[str] = []
    start = prev = ordered[0]
    for year in ordered[1:]:
        if year == prev + 1:
            prev = year
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = year
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(ranges)


def _nsrdb_group_sort_key(group: dict) -> tuple[int, int, int]:
    years = group["years"]
    start_year = min(years) if years else 9999
    priority = len(NSRDB_GROUP_PRIORITY) - NSRDB_GROUP_PRIORITY.index(group["group_key"])
    return (len(years), -start_year, priority)


def _nsrdb_source_key_for_url(url: str) -> str | None:
    return NSRDB_URL_TO_SOURCE_KEY.get(url)


def _nsrdb_group_key_for_source(source_key: str) -> str:
    if source_key in {"himawari7", "himawari8"}:
        return "himawari_pair"
    return source_key


def _build_nsrdb_groups_from_source_years(source_years: dict[str, list[int]]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for source_key, years in source_years.items():
        if not years:
            continue
        group_key = _nsrdb_group_key_for_source(source_key)
        payload = grouped.setdefault(
            group_key,
            {
                "group_key": group_key,
                "label": NSRDB_GROUP_LABELS[group_key],
                "source_keys": [],
                "years": set(),
            },
        )
        payload["source_keys"].append(source_key)
        payload["years"].update(int(year) for year in years)

    groups = []
    for payload in grouped.values():
        years = sorted(payload["years"])
        groups.append(
            {
                "group_key": payload["group_key"],
                "label": payload["label"],
                "source_keys": sorted(payload["source_keys"]),
                "years": years,
                "year_ranges": _nsrdb_years_to_ranges(years),
                "coverage_years": len(years),
            }
        )
    return sorted(groups, key=_nsrdb_group_sort_key, reverse=True)


def _build_nsrdb_selected_sources(group: dict, source_years: dict[str, list[int]]) -> list[tuple[str, int, int]]:
    selected: list[tuple[str, int, int]] = []
    for source_key in NSRDB_GROUPS[group["group_key"]]:
        years = sorted(set(int(year) for year in source_years.get(source_key, [])))
        if not years:
            continue
        selected.append((NSRDB_SOURCES[source_key]["url"], min(years), max(years)))
    return selected


def _get_nsrdb_static_source_years(city: str) -> dict[str, list[int]]:
    static_sources = _resolve_nsrdb_source(city)
    source_years: dict[str, list[int]] = {}
    for url, min_year, max_year in static_sources:
        source_key = _nsrdb_source_key_for_url(url)
        if source_key is None:
            continue
        source_years[source_key] = list(range(min_year, max_year + 1))
    return source_years


def _resolve_nsrdb_source_selection(
    city: str,
    lat: float,
    lon: float,
    api_key: str,
    session=None,
) -> dict[str, object]:
    query_success = True
    fallback_used = False
    fallback_reason = ""
    query_error = None

    try:
        raw_years_by_url = _query_nsrdb_available_years(lat=lat, lon=lon, api_key=api_key, session=session)
    except Exception as exc:
        raw_years_by_url = {}
        query_success = False
        fallback_used = True
        fallback_reason = "query failed"
        query_error = str(exc)

    raw_discovered_sources = []
    source_years: dict[str, list[int]] = {}
    for url, years in sorted(raw_years_by_url.items()):
        source_key = _nsrdb_source_key_for_url(url)
        supported = source_key is not None
        ordered_years = sorted(set(int(year) for year in years))
        raw_discovered_sources.append(
            {
                "url": url,
                "source_key": source_key,
                "supported": supported,
                "years": ordered_years,
                "year_ranges": _nsrdb_years_to_ranges(ordered_years),
            }
        )
        if supported and source_key is not None:
            source_years[source_key] = ordered_years

    if not source_years:
        fallback_used = True
        if not fallback_reason:
            fallback_reason = "query returned no supported sources"
        source_years = _get_nsrdb_static_source_years(city)

    candidate_groups = _build_nsrdb_groups_from_source_years(source_years)
    if not candidate_groups:
        raise ValueError(f"No supported NSRDB sources available for city '{city}'.")

    selected_group = candidate_groups[0]
    selected_sources = _build_nsrdb_selected_sources(selected_group, source_years)
    return {
        "query_success": query_success,
        "query_error": query_error,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "raw_discovered_sources": raw_discovered_sources,
        "candidate_groups": candidate_groups,
        "selected_group": selected_group,
        "selected_sources": selected_sources,
    }


def _print_nsrdb_selection_summary(
    city: str,
    lat: float,
    lon: float,
    selection: dict[str, object],
) -> None:
    print(f"\nNSRDB source check for {city} ({lat:.4f}, {lon:.4f})")
    print(f"  Data Query status: {'ok' if selection['query_success'] else 'failed'}")
    if selection["query_error"]:
        print(f"  Query error: {selection['query_error']}")

    discovered = selection["raw_discovered_sources"]
    if discovered:
        print("  Raw query sources:")
        for item in discovered:
            support_label = item["source_key"] if item["supported"] else "unsupported"
            print(f"    - {support_label}: {item['year_ranges']} ({item['url']})")
    else:
        print("  Raw query sources: none")

    print("  Candidate groups:")
    for group in selection["candidate_groups"]:
        members = ", ".join(group["source_keys"])
        print(
            f"    - {group['label']}: {group['year_ranges']} "
            f"({group['coverage_years']} years; members: {members})"
        )

    selected = selection["selected_group"]
    print(
        f"  Selected default: {selected['label']} "
        f"({selected['year_ranges']}; {selected['coverage_years']} years)"
    )
    if selection["fallback_used"]:
        print(f"  Fallback: {selection['fallback_reason']}")
    print()


def _resolve_nsrdb_sources_with_dynamic_years(
    city: str,
    lat: float,
    lon: float,
    api_key: str,
    session=None,
) -> list[tuple[str, int, int]]:
    """Resolve the selected NSRDB source group for download."""
    selection = _resolve_nsrdb_source_selection(
        city=city,
        lat=lat,
        lon=lon,
        api_key=api_key,
        session=session,
    )
    return selection["selected_sources"]


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


def _isd_year(value: object) -> int | None:
    """Extract a 4-digit year from NOAA ISD history BEGIN/END values."""
    try:
        text = str(int(float(value)))
    except Exception:
        text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) < 4:
        return None
    return int(digits[:4])


def _candidate_isd_stations(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: float = 150.0,
    cache_dir: str = "data/ISD",
    city: Optional[str] = None,
) -> pd.DataFrame:
    """Return nearby ISD stations enriched with distance and coverage metadata."""
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
    nearby = valid[valid["distance_km"] <= radius_km].copy()
    if nearby.empty:
        return nearby

    nearby["begin_year"] = nearby["BEGIN"].apply(_isd_year)
    nearby["end_year"] = nearby["END"].apply(_isd_year)
    nearby["coverage_years"] = (
        nearby["end_year"].fillna(-1) - nearby["begin_year"].fillna(-1) + 1
    ).clip(lower=0)
    nearby["USAF-WBAN"] = (
        nearby["USAF"].astype(str).str.zfill(6)
        + "-"
        + nearby["WBAN"].astype(str).str.zfill(5)
    )
    return nearby


def get_isd_station_coverage(
    station_id: str,
    cache_dir: str = "data/ISD",
) -> dict[str, object]:
    """Return metadata-derived year coverage for one ISD station."""
    if "-" not in station_id:
        raise ValueError("station_id must be USAF-WBAN (e.g. '722020-12839')")

    usaf, wban = station_id.split("-", 1)
    usaf = usaf.zfill(6)
    wban = wban.zfill(5)

    cache_path = Path(cache_dir) / "isd-history.csv"
    history = _load_isd_history(cache_path)
    match = history[
        (history["USAF"].astype(str).str.zfill(6) == usaf)
        & (history["WBAN"].astype(str).str.zfill(5) == wban)
    ]
    if match.empty:
        raise ValueError(f"ISD station '{usaf}-{wban}' not found in {cache_path}")

    row = match.iloc[0]
    begin_year = _isd_year(row.get("BEGIN"))
    end_year = _isd_year(row.get("END"))
    coverage_years = None
    if begin_year is not None and end_year is not None:
        coverage_years = max(int(end_year) - int(begin_year) + 1, 0)

    station_name = str(row.get("STATION NAME", "UNKNOWN")).replace("/", "_").replace(" ", "_")
    return {
        "station_id": f"{usaf}-{wban}",
        "station_name": station_name,
        "begin_year": begin_year,
        "end_year": end_year,
        "coverage_years": coverage_years,
    }


def _cmip6_calendar_time_index(values) -> pd.DatetimeIndex:
    """Map decoded CMIP6 calendar dates to civil Gregorian timestamps.

    Standard and noleap calendars retain their month/day labels. For a 360-day
    calendar, daily positions are mapped proportionally across each Gregorian
    year so the seasonal phase is preserved.
    """
    import calendar as _calendar
    import datetime as _datetime

    values = list(values)
    if not values:
        return pd.DatetimeIndex([])
    first = values[0]
    if not hasattr(first, "calendar"):
        return pd.DatetimeIndex(pd.to_datetime(values))

    cal = str(first.calendar).lower()
    converted = []
    for value in values:
        if cal in {"360_day", "360"}:
            source_day = (int(value.month) - 1) * 30 + int(value.day) - 1
            target_days = 366 if _calendar.isleap(int(value.year)) else 365
            target_day = round(source_day * (target_days - 1) / 359)
            base = _datetime.datetime(int(value.year), 1, 1) + _datetime.timedelta(days=target_day)
            converted.append(base.replace(hour=int(value.hour), minute=int(value.minute),
                                           second=int(value.second), microsecond=int(value.microsecond)))
        else:
            converted.append(_datetime.datetime(int(value.year), int(value.month), int(value.day),
                                               int(value.hour), int(value.minute), int(value.second),
                                               int(value.microsecond)))
    return pd.DatetimeIndex(converted)


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

    ds = xr.open_dataset(BytesIO(nc_bytes), engine="h5netcdf")

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

    time_index = _cmip6_calendar_time_index(ds["time"].values)
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


def _nc_bytes_to_point_dfs(
    nc_bytes: bytes,
    city_points: dict[str, tuple[float, float]],
) -> dict[str, pd.DataFrame]:
    """Extract nearest-grid time-series DataFrames for multiple cities from one NetCDF payload."""
    import numpy as np
    import xarray as xr

    if nc_bytes[:2] == b"PK":
        import zipfile
        with zipfile.ZipFile(BytesIO(nc_bytes)) as zf:
            nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
            if not nc_names:
                raise ValueError("No .nc file found inside zip archive")
            nc_bytes = zf.read(nc_names[0])

    ds = xr.open_dataset(BytesIO(nc_bytes), engine="h5netcdf")

    lat_name = next((c for c in ds.coords if c in ("lat", "latitude")), None)
    lon_name = next((c for c in ds.coords if c in ("lon", "longitude")), None)
    if lat_name is None or lon_name is None:
        raise ValueError(f"lat/lon coordinates not found. Available: {list(ds.coords)}")

    lat_arr = ds[lat_name].values
    lon_arr = ds[lon_name].values
    time_index = _cmip6_calendar_time_index(ds["time"].values)

    city_series: dict[str, list[pd.Series]] = {city: [] for city in city_points}
    city_indices: dict[str, tuple[int, int]] = {}
    for city, (lat, lon) in city_points.items():
        if lon_arr.min() >= 0 and lon_arr.max() > 180:
            lon_target = lon % 360
        else:
            lon_target = lon

        if lat_arr.ndim == 1:
            li = int(np.argmin(np.abs(lat_arr - lat)))
            lj = int(np.argmin(np.abs(lon_arr - lon_target)))
        else:
            dist = (lat_arr - lat) ** 2 + (lon_arr - lon_target) ** 2
            li, lj = np.unravel_index(np.argmin(dist), dist.shape)
        city_indices[city] = (li, lj)

    for var in ds.data_vars:
        if var.endswith(("_bnds", "_bounds")):
            continue
        da = ds[var]
        if "time" not in da.dims:
            continue
        for city, (li, lj) in city_indices.items():
            if da.ndim == 3:
                values = da[:, li, lj].values
            elif da.ndim == 1:
                values = da.values
            else:
                continue
            city_series[city].append(pd.Series(values, index=time_index, name=var))

    ds.close()
    result: dict[str, pd.DataFrame] = {}
    for city, series in city_series.items():
        if not series:
            continue
        result[city] = (
            pd.concat(series, axis=1)
            .reset_index()
            .rename(columns={"index": "time"})
        )
    if not result:
        raise ValueError("No time-series data variables found in NetCDF")
    return result


def _cmip6_request_area(city_points: dict[str, tuple[float, float]], padding_deg: float = 1.5) -> list[float]:
    """Build one CDS area box that covers all requested city coordinates."""
    lats = [lat for lat, _ in city_points.values()]
    lons = [lon for _, lon in city_points.values()]
    return [
        round(max(lats) + padding_deg, 2),
        round(min(lons) - padding_deg, 2),
        round(min(lats) - padding_deg, 2),
        round(max(lons) + padding_deg, 2),
    ]


def _merge_cmip6_city_frames(
    city_frames: dict[str, list[pd.DataFrame]],
    city: str,
) -> pd.DataFrame | None:
    """Merge per-variable CMIP6 DataFrames for one city."""
    var_dfs = city_frames.get(city, [])
    if not var_dfs:
        return None
    merged = var_dfs[0]
    for df_v in var_dfs[1:]:
        merged = pd.merge(merged, df_v, on="time", how="outer")
    return merged.sort_values("time").reset_index(drop=True)


def _cmip6_frame_covers_request(
    frame: pd.DataFrame,
    years: list[int],
    variables: list[str],
) -> bool:
    """Return whether a CMIP6 frame has every requested field and year."""
    required_columns = {CMIP6_VAR_SHORT.get(variable, variable) for variable in variables}
    if not required_columns.issubset(frame.columns) or "time" not in frame.columns:
        return False
    try:
        times = pd.to_datetime(frame["time"], errors="coerce")
    except (TypeError, ValueError):
        return False
    if times.empty or times.isna().any() or times.duplicated().any():
        return False
    requested_years = set(years)
    for column in required_columns:
        valid = frame[column].notna()
        if not valid.any() or not requested_years.issubset(set(times[valid].dt.year.unique())):
            return False
    return True


def _cmip6_csv_covers_request(
    path: Path,
    years: list[int],
    variables: list[str],
) -> bool:
    """Return whether an existing CMIP6 CSV satisfies the requested fields/years."""
    if not path.is_file():
        return False
    try:
        return _cmip6_frame_covers_request(pd.read_csv(path), years, variables)
    except (OSError, pd.errors.ParserError, ValueError):
        return False


def _cmip6_variable_covers_request(
    frame: pd.DataFrame, years: list[int], variable: str
) -> bool:
    """Validate one extracted CMIP6 variable independently of its group."""
    return _cmip6_frame_covers_request(frame, years, [variable])


def _cmip6_cache_path(
    output_root: Path, model: str, experiment: str, resolution: str,
    variable: str, city: str, years: list[int], point: tuple[float, float],
) -> Path:
    """Return a coordinate- and request-specific variable cache path."""
    coords = f"{point[0]:.6f},{point[1]:.6f}"
    coord_hash = hashlib.sha256(coords.encode("utf-8")).hexdigest()[:12]
    year_key = f"{min(years)}-{max(years)}_{hashlib.sha256(','.join(map(str, years)).encode()).hexdigest()[:8]}"
    return (
        output_root / ".cmip6_cache" / model / experiment / resolution
        / city / f"{variable}_{year_key}_{coord_hash}.csv"
    )


def _read_cmip6_variable(path: Path, years: list[int], variable: str) -> pd.DataFrame | None:
    """Read a one-variable cache or legacy CSV slice after coverage checks."""
    short = CMIP6_VAR_SHORT.get(variable, variable)
    try:
        frame = pd.read_csv(path, parse_dates=["time"])
    except (OSError, pd.errors.ParserError, ValueError, KeyError):
        return None
    if short not in frame.columns and variable in frame.columns:
        frame = frame.rename(columns={variable: short})
    if not _cmip6_variable_covers_request(frame, years, variable):
        return None
    return frame[["time", short]].copy()


def _write_cmip6_variable(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _cmip6_existing_variable(
    out_csv: Path, years: list[int], variable: str, cache_path: Path,
) -> pd.DataFrame | None:
    """Recover an individually valid field from an older group CSV."""
    frame = _read_cmip6_variable(out_csv, years, variable)
    if frame is not None:
        _write_cmip6_variable(cache_path, frame)
    return frame


def _record_cmip6_failure(
    output_root: Path, *, model: str, experiment: str, resolution: str,
    variable: str, cities: list[str], error: str,
) -> None:
    """Persist a concise JSONL record so failed work is identifiable on restart."""
    report = output_root / ".cmip6_cache" / "failures.jsonl"
    report.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model": model, "experiment": experiment, "temporal_resolution": resolution,
        "variable": variable, "cities": cities, "error": error,
    }
    with report.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_cmip6_final(out_csv: Path, frame: pd.DataFrame, years: list[int], variables: list[str]) -> bool:
    """Atomically replace a group CSV only after its serialized form validates."""
    temporary = out_csv.with_name(f".{out_csv.name}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        if not _cmip6_csv_covers_request(temporary, years, variables):
            temporary.unlink(missing_ok=True)
            return False
        temporary.replace(out_csv)
        return True
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _cmip6_dataset_to_point_dfs(ds, city_points: dict[str, tuple[float, float]]) -> dict[str, pd.DataFrame]:
    """Extract nearest-grid time series from an opened CMIP6 dataset."""
    import numpy as np

    lat_name = next((c for c in ds.coords if c in ("lat", "latitude")), None)
    lon_name = next((c for c in ds.coords if c in ("lon", "longitude")), None)
    if lat_name is None or lon_name is None:
        raise ValueError(f"lat/lon coordinates not found. Available: {list(ds.coords)}")
    if "time" not in ds.coords:
        raise ValueError(f"time coordinate not found. Available: {list(ds.coords)}")

    lat_arr, lon_arr = ds[lat_name].values, ds[lon_name].values
    time_index = _cmip6_calendar_time_index(ds["time"].values)
    city_indices: dict[str, tuple[int, int]] = {}
    for city, (lat, lon) in city_points.items():
        lon_target = lon % 360 if lon_arr.min() >= 0 and lon_arr.max() > 180 else lon
        if lat_arr.ndim == 1 and lon_arr.ndim == 1:
            li, lj = int(np.argmin(abs(lat_arr - lat))), int(np.argmin(abs(lon_arr - lon_target)))
            selected_lat, selected_lon = float(lat_arr[li]), float(lon_arr[lj])
        elif lat_arr.ndim == 2 and lon_arr.ndim == 2:
            li, lj = np.unravel_index(np.argmin((lat_arr - lat) ** 2 + (lon_arr - lon_target) ** 2), lat_arr.shape)
            selected_lat, selected_lon = float(lat_arr[li, lj]), float(lon_arr[li, lj])
        else:
            raise ValueError(f"Unsupported latitude/longitude dimensions: {lat_arr.ndim}D/{lon_arr.ndim}D")
        city_indices[city] = (li, lj)
        logger.info("CMIP6 local extract: city='%s' → grid=(%.4f, %.4f)", city, selected_lat, selected_lon)

    city_series: dict[str, list[pd.Series]] = {city: [] for city in city_points}
    for var, da in ds.data_vars.items():
        if var.endswith(("_bnds", "_bounds")) or "time" not in da.dims:
            continue
        spatial_dims = [dim for dim in da.dims if dim != "time"]
        if not spatial_dims:
            values_by_city = {city: da.values for city in city_points}
        elif len(spatial_dims) == 2:
            values_by_city = {city: da.isel({spatial_dims[0]: li, spatial_dims[1]: lj}).values for city, (li, lj) in city_indices.items()}
        else:
            logger.warning("Skipping CMIP6 variable '%s': unsupported dimensions %s", var, da.dims)
            continue
        for city, values in values_by_city.items():
            city_series[city].append(pd.Series(values, index=time_index, name=var))

    result = {city: pd.concat(series, axis=1).reset_index().rename(columns={"index": "time"}) for city, series in city_series.items() if series}
    if not result:
        raise ValueError("No time-series data variables found in NetCDF")
    return result


def extract_cmip6_from_files(cities: list[str], input_files: list[str | Path], output_dir: str = "data/CMIP6", output_name: str = "extracted_cmip6") -> dict[str, Path]:
    """Extract local CMIP6 files to one nearest-grid CSV per city.

    CDS archives occasionally use a ``.nc`` suffix for a ZIP file containing
    one or more actual NetCDF files.  Both that layout and ordinary NetCDF
    files are accepted.
    """
    import xarray as xr
    import zipfile

    if not input_files:
        raise ValueError("Provide at least one local NetCDF file")
    if Path(output_name).name != output_name or not output_name:
        raise ValueError("output_name must be a filename stem, not a path")
    city_points = _resolve_city_points(_normalize_city_args(cities))
    city_frames: dict[str, list[pd.DataFrame]] = {city: [] for city in city_points}
    for value in input_files:
        input_path = Path(value).expanduser()
        if not input_path.is_file():
            raise FileNotFoundError(f"CMIP6 input file not found: {input_path}")
        logger.info("CMIP6 local extract input → %s", input_path)
        if zipfile.is_zipfile(input_path):
            with zipfile.ZipFile(input_path) as archive:
                nc_names = [name for name in archive.namelist() if name.lower().endswith(".nc")]
                if not nc_names:
                    raise ValueError(f"No NetCDF file found inside CMIP6 ZIP: {input_path}")
                logger.info("  ZIP archive contains %s NetCDF file(s)", len(nc_names))
                for nc_name in nc_names:
                    with archive.open(nc_name) as handle:
                        with xr.open_dataset(BytesIO(handle.read())) as ds:
                            city_dfs = _cmip6_dataset_to_point_dfs(ds, city_points)
                    for city, frame in city_dfs.items():
                        city_frames[city].append(frame)
        else:
            with xr.open_dataset(input_path) as ds:
                city_dfs = _cmip6_dataset_to_point_dfs(ds, city_points)
            for city, frame in city_dfs.items():
                city_frames[city].append(frame)
    outputs: dict[str, Path] = {}
    for city in city_points:
        frames = city_frames[city]
        if not frames:
            raise ValueError(f"No CMIP6 time series extracted for city='{city}'")
        combined = pd.concat(frames, ignore_index=True, sort=False)
        value_columns = [column for column in combined.columns if column != "time"]
        conflicts = []
        for column in value_columns:
            counts = combined.groupby("time")[column].nunique(dropna=True)
            if (counts > 1).any():
                conflicts.append(column)
        if conflicts:
            raise ValueError(f"Conflicting CMIP6 values for duplicate timestamps: {sorted(conflicts)}")
        merged = combined.groupby("time", as_index=False).first().sort_values("time").reset_index(drop=True)
        rename = {col: CMIP6_VAR_SHORT.get(col, col) for col in merged.columns if col != "time"}
        merged = merged.rename(columns=rename)
        if merged.columns.duplicated().any():
            duplicates = merged.columns[merged.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate output CMIP6 columns: {duplicates}")
        city_dir = Path(output_dir) / city
        city_dir.mkdir(parents=True, exist_ok=True)
        output_path = city_dir / f"{output_name}.csv"
        merged.to_csv(output_path, index=False)
        outputs[city] = output_path
        logger.info("CMIP6 local extract saved → %s (%s rows)", output_path, len(merged))
    return outputs


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
    if city is not None:
        lat, lon = _resolve_coords(city, lat, lon)
    if lat is None or lon is None:
        raise ValueError("Provide either city or both lat and lon.")

    nearby = _candidate_isd_stations(
        lat=lat,
        lon=lon,
        radius_km=radius_km,
        cache_dir=cache_dir,
    ).sort_values("distance_km")

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


def list_nsrdb_sources(
    cities: list[str],
) -> None:
    """Print NSRDB Data Query results and the default source selection for one or more cities."""
    api_key, _email = _load_nsrdb_credentials()
    import requests

    session = requests.Session()
    for city in _normalize_city_args(cities):
        meta = _city_meta(city)
        selection = _resolve_nsrdb_source_selection(
            city=city,
            lat=meta["lat"],
            lon=meta["lon"],
            api_key=api_key,
            session=session,
        )
        _print_nsrdb_selection_summary(city, meta["lat"], meta["lon"], selection)


def select_best_isd_station(
    city: str,
    radius_km: float = 50.0,
    cache_dir: str = "data/ISD",
) -> str:
    """Return the station ID with the best temporal coverage inside ``radius_km``."""
    nearby = _candidate_isd_stations(
        city=city,
        radius_km=radius_km,
        cache_dir=cache_dir,
    )
    if nearby.empty:
        raise ValueError(f"No ISD stations found within {radius_km} km of '{city}'.")

    ranked = nearby.sort_values("coverage_years", ascending=False, kind="mergesort")
    best = ranked.iloc[0]
    station_id = str(best["USAF-WBAN"])
    logger.info(
        "Selected ISD station %s for %s (coverage=%s years, distance=%.1f km)",
        station_id,
        city,
        int(best["coverage_years"]),
        float(best["distance_km"]),
    )
    return station_id


def download_isd(
    station_id: str,
    city: str,
    years: List[int],
    output_dir: str = "data/ISD",
) -> None:
    """Download yearly ISD .gz files for *station_id* (format ``USAF-WBAN``)."""
    coverage = get_isd_station_coverage(station_id=station_id, cache_dir=output_dir)
    station_id = str(coverage["station_id"])
    station_name = str(coverage["station_name"])
    begin_year = coverage["begin_year"]
    end_year = coverage["end_year"]
    usaf, wban = station_id.split("-", 1)

    requested_years = sorted(int(year) for year in years)
    valid_years = requested_years
    if begin_year is not None and end_year is not None:
        valid_years = [year for year in requested_years if int(begin_year) <= year <= int(end_year)]
        skipped_years = [year for year in requested_years if year not in valid_years]
        if skipped_years:
            logger.warning(
                "Requested ISD years outside metadata coverage for %s: %s (coverage=%s-%s)",
                station_id,
                skipped_years,
                begin_year,
                end_year,
            )
        if not valid_years:
            logger.warning(
                "No requested ISD years remain for %s after coverage filter (coverage=%s-%s)",
                station_id,
                begin_year,
                end_year,
            )
            return

    station_dir = Path(output_dir) / city / f"{usaf}-{wban}_{station_name}"
    station_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"ISD output → {station_dir}")

    import requests

    with tqdm(valid_years, desc=f"ISD {city}", unit="yr") as pbar:
        for year in pbar:
            pbar.set_postfix(year=year)
            filename = f"{usaf}-{wban}-{year}.gz"
            local = station_dir / filename
            if local.exists() and local.stat().st_size > 0:
                logger.debug(f"  {year} already present, skipping")
                continue
            tmp = local.with_suffix(".gz.tmp")
            response = None
            try:
                url = ISD_HTTP_URL.format(year=year, filename=filename)
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                tmp.replace(local)
            except Exception as exc:
                logger.warning(f"  {year} failed: {exc}")
                if tmp.exists():
                    tmp.unlink()
            finally:
                if response is not None:
                    response.close()

    logger.info(
        "ISD download complete (%s/%s years requested after coverage filter) → %s",
        len(valid_years),
        len(requested_years),
        station_dir,
    )


def download_isd_for_cities(
    cities: list[str],
    years: List[int],
    output_dir: str = "data/ISD",
    isd_station_ids: dict[str, str] | None = None,
    isd_radius_km: float = 50.0,
) -> None:
    """Download ISD data for multiple cities."""
    canonical_cities = _normalize_city_args(cities)
    for city in canonical_cities:
        station_id = (
            isd_station_ids[city]
            if isd_station_ids and city in isd_station_ids
            else select_best_isd_station(
                city=city,
                radius_km=isd_radius_km,
                cache_dir=output_dir,
            )
        )
        download_isd(
            station_id=station_id,
            city=city,
            years=years,
            output_dir=output_dir,
        )


# ─── NSRDB ────────────────────────────────────────────────────────────────────

def _nsrdb_year_cache_path(
    output_dir: Path, city: str, year: int, interval: int,
    lat: float, lon: float, source_url: str,
) -> Path:
    identity = f"{lat:.6f},{lon:.6f},{interval},{source_url},{','.join(NSRDB_ATTRS)}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return output_dir / ".cache" / "NSRDB" / city / f"{year}_{interval}min_{digest}.csv"


def _nsrdb_year_is_complete(frame: pd.DataFrame, year: int, interval: int) -> bool:
    required = {"Year", "Month", "Day", "Hour", "Minute"}
    aliases = {"ghi": "GHI", "dni": "DNI", "dhi": "DHI"}
    required.update(aliases.get(column, column) for column in NSRDB_ATTRS)
    if not required.issubset(frame.columns):
        return False
    try:
        actual_years = pd.to_numeric(frame["Year"], errors="coerce")
        timestamp = pd.to_datetime(
            frame[["Year", "Month", "Day", "Hour", "Minute"]].rename(
                columns={"Year": "year", "Month": "month", "Day": "day", "Hour": "hour", "Minute": "minute"}
            ), errors="coerce",
        )
        if interval <= 0 or 60 % interval != 0:
            return False
        if actual_years.isna().any() or not actual_years.eq(year).all():
            return False
        if timestamp.isna().any() or timestamp.duplicated().any():
            return False
        expected = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq=f"{interval}min", inclusive="left")
        expected = expected[~((expected.month == 2) & (expected.day == 29))]
        if not timestamp.sort_values().reset_index(drop=True).equals(pd.Series(expected)):
            return False
        return not frame[list(required)].isna().any().any()
    except (KeyError, TypeError, ValueError):
        return False


def _read_nsrdb_year(path: Path, year: int, interval: int) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, ValueError):
        return None
    return frame if _nsrdb_year_is_complete(frame, year, interval) else None


def _write_nsrdb_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _era5_land_time_coordinate(dataset):
    return next((dataset[name] for name in ("valid_time", "time") if name in dataset.coords), None)


def _era5_land_file_covers_request(
    path: Path, variable: str, start: pd.Timestamp, end: pd.Timestamp,
) -> bool:
    """Validate the requested field and every requested hourly timestamp."""
    import xarray as xr

    aliases = {
        "2m_dewpoint_temperature": {"d2m", "2m_dewpoint_temperature"},
        "2m_temperature": {"t2m", "2m_temperature"},
        "surface_pressure": {"sp", "surface_pressure"},
        "total_precipitation": {"tp", "total_precipitation"},
        "surface_solar_radiation_downwards": {"ssrd", "surface_solar_radiation_downwards"},
    }
    try:
        with xr.open_dataset(path) as dataset:
            if not (aliases[variable] & set(dataset.data_vars)):
                return False
            time_coordinate = _era5_land_time_coordinate(dataset)
            if time_coordinate is None:
                return False
            actual = pd.DatetimeIndex(pd.to_datetime(time_coordinate.values, errors="coerce"))
            expected = pd.date_range(start.normalize(), end.normalize() + pd.Timedelta(days=1),
                                     freq="h", inclusive="left")
            if actual.hasnans or actual.has_duplicates or len(actual) != len(expected):
                return False
            if not actual.sort_values().equals(expected):
                return False
            values = dataset[next(iter(aliases[variable] & set(dataset.data_vars)))].values
            return bool(pd.notna(values).any())
    except (OSError, ValueError, KeyError, ImportError):
        return False


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

def download_nsrdb(
    city: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    years: Optional[List[int]] = None,
    output_dir: str = "data/NSRDB",
    interval: int = 60,
    max_retries: int = 3,
    retry_backoff_sec: float = 2.0,
) -> dict[str, list]:
    """Download NSRDB solar irradiance for a point location.

    Coordinates are looked up from cities_config.csv when *lat*/*lon* are omitted.
    Retries transient per-year HTTP/TLS failures before skipping the year.
    """
    import requests

    api_key, email = _load_nsrdb_credentials()
    lat, lon = _resolve_coords(city, lat, lon)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    city_lower = city.lower().replace(" ", "_")
    session = requests.Session()
    selection = _resolve_nsrdb_source_selection(
        city=city,
        lat=lat,
        lon=lon,
        api_key=api_key,
        session=session,
    )
    _print_nsrdb_selection_summary(city, lat, lon, selection)
    sources = selection["selected_sources"]
    available_years = sorted({year for _, mn, mx in sources for year in range(mn, mx + 1)})
    if years is None:
        years = available_years
    overall_min = min(available_years) if available_years else None
    overall_max = max(available_years) if available_years else None
    logger.info(f"NSRDB available years for {city}: {overall_min}–{overall_max}")
    valid_years = [y for y in sorted(years) if y in set(available_years)]
    if not valid_years:
        logger.error(
            f"No valid NSRDB years for '{city}' in the requested range. "
            f"Available: {overall_min}–{overall_max}."
        )
        return {"success": [], "failed": [], "incomplete": [
            {"city": city, "years": sorted(years), "reason": "no requested years are available"}
        ]}

    def _url_for_year(yr: int) -> str | None:
        for url, mn, mx in sources:
            if mn <= yr <= mx:
                return url
        return None

    all_dfs: list[pd.DataFrame] = []
    failures: list[dict] = []
    for i, year in enumerate(tqdm(valid_years, desc=f"NSRDB {city}", unit="yr")):
        endpoint_url = _url_for_year(year)
        if endpoint_url is None:
            logger.warning(f"  {year}: no NSRDB source covers this year, skipping")
            failures.append({"city": city, "year": year, "error": "no NSRDB source covers this year"})
            continue
        cache_path = _nsrdb_year_cache_path(out_dir, city, year, interval, lat, lon, endpoint_url)
        cached = _read_nsrdb_year(cache_path, year, interval)
        if cached is None:
            # Reuse a valid year from pre-cache combined outputs when available.
            for legacy_path in sorted(out_dir.glob(f"nsrdb_{city_lower}_*.csv")):
                try:
                    old = pd.read_csv(legacy_path)
                    old_year = pd.to_numeric(old.get("Year"), errors="coerce")
                    candidate = old.loc[old_year.eq(year)].copy()
                except (OSError, pd.errors.ParserError, ValueError, AttributeError):
                    continue
                if _nsrdb_year_is_complete(candidate, year, interval):
                    cached = candidate
                    _write_nsrdb_csv(cache_path, cached)
                    break
        if cached is not None:
            all_dfs.append(cached)
            logger.info("NSRDB %s year %s restored from cache", city, year)
            continue
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
        year_frame = None
        for attempt in range(1, max_retries + 1):
            try:
                if attempt == 1 and i > 0:
                    time.sleep(2)  # respect NREL rate limit between new year requests
                r = session.get(f"{endpoint_url}.csv", params=params, timeout=300)
                r.raise_for_status()
                lines = r.text.strip().split("\n")
                header = next((i for i, l in enumerate(lines) if l.startswith("Year,")), None)
                if header is None:
                    raise ValueError("unexpected response: Year header not found")
                df = pd.read_csv(StringIO("\n".join(lines[header:])))
                if not _nsrdb_year_is_complete(df, year, interval):
                    raise ValueError(f"response failed annual completeness validation for {year}")
                year_frame = df
                _write_nsrdb_csv(cache_path, year_frame)
                logger.info(f"  {year} ✓ ({len(df)} rows; cached)")
                break
            except Exception as exc:
                if attempt >= max_retries:
                    logger.warning(f"  {year} failed after {attempt} attempt(s): {exc}")
                    failures.append({"city": city, "year": year, "error": str(exc)})
                    break
                wait_s = retry_backoff_sec * attempt
                logger.warning(
                    f"  {year} attempt {attempt}/{max_retries} failed: {exc}. "
                    f"Retrying in {wait_s:.1f}s."
                )
                time.sleep(wait_s)
        if year_frame is not None:
            all_dfs.append(year_frame)

    if failures or len(all_dfs) != len(valid_years):
        logger.error("NSRDB incomplete for %s; failed years: %s", city,
                     [item["year"] for item in failures])
        return {"success": [], "failed": failures,
                "incomplete": [{"city": city, "years": valid_years}]}

    if not all_dfs:
        logger.error("No NSRDB data downloaded.")
        return {"success": [], "failed": failures,
                "incomplete": [{"city": city, "years": valid_years}]}

    combined = pd.concat(all_dfs, ignore_index=True).sort_values(
        ["Year", "Month", "Day", "Hour", "Minute"]
    ).reset_index(drop=True)
    y0, y1   = valid_years[0], valid_years[-1]
    out_path = out_dir / f"nsrdb_{city_lower}_{y0}-{y1}.csv"
    temporary = out_path.with_name(f".{out_path.name}.tmp")
    try:
        combined.to_csv(temporary, index=False)
        check = pd.read_csv(temporary)
        if not all(_nsrdb_year_is_complete(check.loc[pd.to_numeric(check["Year"], errors="coerce").eq(year)],
                                            year, interval) for year in valid_years):
            raise ValueError("merged NSRDB CSV failed annual completeness validation")
        temporary.replace(out_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    logger.info(f"NSRDB saved → {out_path} ({len(combined)} rows, {y0}–{y1})")
    return {"success": [{"city": city, "years": valid_years, "path": str(out_path)}],
            "failed": [], "incomplete": []}


def download_nsrdb_for_cities(
    cities: list[str],
    years: Optional[List[int]] = None,
    output_dir: str = "data/NSRDB",
    interval: int = 60,
    max_retries: int = 3,
    retry_backoff_sec: float = 2.0,
) -> dict[str, list]:
    """Download NSRDB data for multiple cities."""
    canonical_cities = _normalize_city_args(cities)
    combined = {"success": [], "failed": [], "incomplete": []}
    for city in canonical_cities:
        result = download_nsrdb(
            city=city,
            years=years,
            output_dir=output_dir,
            interval=interval,
            max_retries=max_retries,
            retry_backoff_sec=retry_backoff_sec,
        )
        for key in combined:
            combined[key].extend(result[key])
    return combined


# ─── ERA5-Land ────────────────────────────────────────────────────────────────

def download_era5_land(
    city: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    start_date: str = "1950-01-01",
    end_date: Optional[str] = None,
    output_dir: str = "data/ERA5_Land",
    variables: Optional[List[str]] = None,
    overwrite: bool = False,
) -> list[Path]:
    """Download hourly ERA5-Land point time series from the CDS.

    One CDS request is made per variable so each response is a predictable
    NetCDF file, even when the selected parameters belong to different CDS
    storage groups. Surface solar radiation is de-accumulated by this CDS
    time-series product and is stored as hourly energy in J m-2.
    """
    import cdsapi

    city = _normalize_city_name(city)
    lat, lon = _resolve_coords(city, lat, lon)
    latest_available_date = date.today() - timedelta(days=ERA5_LAND_AVAILABILITY_LAG_DAYS)
    end_date = end_date or latest_available_date.isoformat()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start < pd.Timestamp("1950-01-01"):
        raise ValueError("ERA5-Land time-series coverage starts on 1950-01-01")
    if start > end:
        raise ValueError("ERA5-Land start_date must be on or before end_date")
    if end.date() > latest_available_date:
        raise ValueError(
            f"ERA5-Land time-series is normally available through {latest_available_date}; "
            "choose an earlier end_date"
        )
    requested_variables = list(ERA5_LAND_VARIABLES if variables is None else variables)
    if not requested_variables:
        raise ValueError("Select at least one ERA5-Land variable")
    unknown = sorted(set(requested_variables) - set(ERA5_LAND_VARIABLES))
    if unknown:
        raise ValueError(f"Unsupported ERA5-Land variables: {unknown}")

    city_dir = Path(output_dir) / city
    city_dir.mkdir(parents=True, exist_ok=True)
    date_label = f"{start:%Y%m%d}-{end:%Y%m%d}"
    client = cdsapi.Client()
    outputs: list[Path] = []
    for variable in requested_variables:
        output_path = city_dir / f"era5_land_{city.lower()}_{variable}_{date_label}.nc"
        outputs.append(output_path)
        cache_identity = f"{lat:.6f},{lon:.6f},{variable},{start:%Y%m%d},{end:%Y%m%d}"
        coord_hash = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()[:16]
        cache_path = (
            Path(output_dir) / ".cache" / "ERA5_Land" / city
            / f"{output_path.stem}_{coord_hash}.nc"
        )
        if output_path.exists() and not overwrite and _era5_land_file_covers_request(
            output_path, variable, start, end
        ):
            logger.info("ERA5-Land exists, skipping: %s", output_path)
            continue
        if not overwrite and _era5_land_file_covers_request(cache_path, variable, start, end):
            _atomic_copy(cache_path, output_path)
            logger.info("ERA5-Land restored from cache → %s", output_path)
            continue
        request = {
            "variable": [variable],
            "location": {"longitude": float(lon), "latitude": float(lat)},
            "date": [f"{start:%Y-%m-%d}/{end:%Y-%m-%d}"],
            "data_format": "netcdf",
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = cache_path.with_suffix(cache_path.suffix + ".part")
        try:
            client.retrieve(ERA5_LAND_DATASET, request, str(partial_path))
            if not partial_path.exists() or partial_path.stat().st_size == 0:
                raise RuntimeError(f"CDS returned an empty ERA5-Land file for {variable}")
            if not _era5_land_file_covers_request(partial_path, variable, start, end):
                raise RuntimeError(f"CDS returned incomplete ERA5-Land data for {variable}")
            partial_path.replace(cache_path)
            _atomic_copy(cache_path, output_path)
            logger.info("ERA5-Land saved → %s", output_path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise
    return outputs


def download_era5_land_for_cities(
    cities: list[str],
    start_date: str = "1950-01-01",
    end_date: Optional[str] = None,
    output_dir: str = "data/ERA5_Land",
    variables: Optional[List[str]] = None,
    overwrite: bool = False,
) -> dict[str, list[Path]]:
    """Download ERA5-Land point time series for several configured cities."""
    canonical_cities = _normalize_city_args(cities)
    return {
        city: download_era5_land(
            city=city,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            variables=variables,
            overwrite=overwrite,
        )
        for city in canonical_cities
    }


# ─── CMIP6 ────────────────────────────────────────────────────────────────────

def download_cmip6(
    city: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    models: Optional[List[str]] = None,
    experiments: Optional[List[str]] = None,
    variables: Optional[List[str]] = None,
    year_range: Optional[tuple] = None,
    output_dir: str = "data/CMIP6",
) -> dict[str, list]:
    """Download CMIP6 projections for a city via CDS, extracting the nearest grid point.

    Coordinates are looked up from cities_config.csv when *lat*/*lon* are omitted.
    Uses ``area: [lat, lon, lat, lon]`` in the CDS request so only the cell
    containing the city is transferred. Reads CDS credentials from ~/.cdsapirc.
    """
    import cdsapi

    lat, lon    = _resolve_coords(city, lat, lon)
    models      = models      or ["mpi_esm1_2_lr"]
    experiments = experiments or ["historical", "ssp2_4_5"]
    variables   = variables   or sorted(CMIP6_SUPPORTED_VARIABLES)
    all_months  = [f"{i:02d}" for i in range(1, 13)]
    all_days    = [f"{i:02d}" for i in range(1, 32)]
    client      = cdsapi.Client()

    summary = {"success": [], "failed": [], "incomplete": []}
    output_root = Path(output_dir)
    for model in models:
        city_dir   = Path(output_dir) / city
        city_dir.mkdir(parents=True, exist_ok=True)

        for cds_exp in experiments:
            variable_groups = _filter_cmip6_variable_groups(
                _get_cmip6_variable_groups(model, cds_exp),
                variables,
            )
            if not variable_groups:
                logger.warning(
                    f"CMIP6 combination not available in CDS: model='{model}', experiment='{cds_exp}'. "
                    "Run 'samy-download list-cmip6' to inspect currently available combinations."
                )
                summary["incomplete"].append({"city": city, "model": model,
                    "experiment": cds_exp, "variables": variables,
                    "reason": "no supported variable group available"})
                continue

            years = CMIP6_EXP_YEARS.get(cds_exp, [])
            if year_range is not None:
                years = [y for y in years if year_range[0] <= y <= year_range[1]]
            if not years:
                logger.warning(f"No years available for experiment '{cds_exp}' with given range, skipping")
                summary["incomplete"].append({"city": city, "model": model,
                    "experiment": cds_exp, "variables": variables, "reason": "no requested years"})
                continue
            label = CMIP6_EXP_LABEL[cds_exp]

            for temporal_res, var_list in variable_groups.items():
                suffix  = "" if temporal_res == "daily" else "_monthly"
                out_csv = city_dir / f"{model}_{label}{suffix}.csv"
                if _cmip6_csv_covers_request(out_csv, years, var_list):
                    logger.info(f"  {out_csv.name} covers requested fields and years, skipping")
                    summary["success"].append({"city": city, "model": model,
                        "experiment": cds_exp, "temporal_resolution": temporal_res,
                        "variables": var_list, "skipped": True})
                    continue
                if out_csv.exists():
                    logger.warning(
                        "  Existing CMIP6 CSV is incomplete for requested fields/years; "
                        "re-downloading: %s",
                        out_csv,
                    )

                city_frames: dict[str, list[pd.DataFrame]] = {city: []}
                pbar_desc = f"CMIP6 {city} {model} {label}"

                for variable in tqdm(var_list, desc=pbar_desc, unit="var"):
                    short   = CMIP6_VAR_SHORT.get(variable, variable)
                    cache_path = _cmip6_cache_path(
                        output_root, model, cds_exp, temporal_res, variable, city, years, (lat, lon)
                    )
                    cached = _read_cmip6_variable(cache_path, years, variable)
                    if cached is None:
                        cached = _cmip6_existing_variable(out_csv, years, variable, cache_path)
                    if cached is not None:
                        city_frames[city].append(cached)
                        continue
                    tmp_nc  = city_dir / f"_tmp_{model}_{cds_exp}_{variable}.nc"
                    logger.info(f"  → {variable} ({short})")
                    try:
                        req: dict = {
                            "temporal_resolution": temporal_res,
                            "experiment":          cds_exp,
                            "variable":            variable,
                            "model":               model,
                            "year":                [str(y) for y in years],
                            "area":                [round(lat + 1.5, 2), round(lon - 1.5, 2), round(lat - 1.5, 2), round(lon + 1.5, 2)],
                        }
                        if temporal_res == "daily":
                            req["month"] = all_months
                            req["day"]   = all_days
                        client.retrieve("projections-cmip6", req, str(tmp_nc))

                        nc_bytes = tmp_nc.read_bytes()
                        df_var   = _nc_bytes_to_point_df(nc_bytes, lat, lon)
                        rename   = {c: CMIP6_VAR_SHORT.get(c, c) for c in df_var.columns if c != "time"}
                        df_var = df_var.rename(columns=rename)[["time", short]]
                        if not _cmip6_variable_covers_request(df_var, years, variable):
                            raise ValueError(f"extracted data failed field/year validation: {short}")
                        _write_cmip6_variable(cache_path, df_var)
                        city_frames[city].append(df_var)
                        logger.info(f"    ✓ {short}")
                    except Exception as exc:
                        summary["failed"].append({"model": model, "experiment": cds_exp,
                            "temporal_resolution": temporal_res, "variable": variable, "cities": [city], "error": str(exc)})
                        summary["incomplete"].append({"city": city, "model": model,
                            "experiment": cds_exp, "temporal_resolution": temporal_res, "variable": variable})
                        _record_cmip6_failure(output_root, model=model, experiment=cds_exp,
                            resolution=temporal_res, variable=variable, cities=[city], error=str(exc))
                        logger.warning(f"    ✗ {variable}: {exc}")
                    finally:
                        if tmp_nc.exists():
                            tmp_nc.unlink()
                    time.sleep(1)

                merged = _merge_cmip6_city_frames(city_frames, city)
                if merged is None or not _cmip6_frame_covers_request(merged, years, var_list):
                    logger.error("CMIP6 incomplete for city=%s model=%s experiment=%s resolution=%s",
                                 city, model, cds_exp, temporal_res)
                    continue
                if not _write_cmip6_final(out_csv, merged, years, var_list):
                    summary["incomplete"].append({"city": city, "model": model,
                        "experiment": cds_exp, "temporal_resolution": temporal_res, "variables": var_list})
                    continue
                summary["success"].append({"city": city, "model": model,
                    "experiment": cds_exp, "temporal_resolution": temporal_res, "variables": var_list})
                logger.info(f"CMIP6 saved → {out_csv} ({len(merged)} rows)")
    logger.info("CMIP6 summary: %d successful outputs, %d failed items, %d incomplete outputs",
                len(summary["success"]), len(summary["failed"]), len(summary["incomplete"]))
    for item in summary["success"]:
        logger.info("CMIP6 success: city=%s model=%s experiment=%s resolution=%s",
                    item["city"], item["model"], item["experiment"], item["temporal_resolution"])
    for item in summary["failed"]:
        logger.error("CMIP6 failed: variable=%s cities=%s model=%s experiment=%s resolution=%s error=%s",
                     item["variable"], item["cities"], item["model"], item["experiment"],
                     item["temporal_resolution"], item["error"])
    for item in summary["incomplete"]:
        logger.error("CMIP6 incomplete: %s", item)
    return summary


def download_cmip6_for_cities(
    cities: list[str],
    models: Optional[List[str]] = None,
    experiments: Optional[List[str]] = None,
    variables: Optional[List[str]] = None,
    year_range: Optional[tuple] = None,
    output_dir: str = "data/CMIP6",
) -> dict[str, list]:
    """Download CMIP6 projections for multiple cities with per-variable recovery."""
    import cdsapi

    city_points = _resolve_city_points(cities)
    models = models or ["mpi_esm1_2_lr"]
    experiments = experiments or ["historical", "ssp2_4_5"]
    variables = variables or sorted(CMIP6_SUPPORTED_VARIABLES)
    all_months = [f"{i:02d}" for i in range(1, 13)]
    all_days = [f"{i:02d}" for i in range(1, 32)]
    area = _cmip6_request_area(city_points)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    summary = {"success": [], "failed": [], "incomplete": []}

    for model in models:
        for cds_exp in experiments:
            variable_groups = _filter_cmip6_variable_groups(
                _get_cmip6_variable_groups(model, cds_exp),
                variables,
            )
            if not variable_groups:
                logger.warning(
                    f"CMIP6 combination not available in CDS: model='{model}', experiment='{cds_exp}'. "
                    "Run 'samy-download list-cmip6' to inspect currently available combinations."
                )
                summary["incomplete"].append({"cities": list(city_points), "model": model,
                    "experiment": cds_exp, "variables": variables,
                    "reason": "no supported variable group available"})
                continue

            years = CMIP6_EXP_YEARS.get(cds_exp, [])
            if year_range is not None:
                years = [y for y in years if year_range[0] <= y <= year_range[1]]
            if not years:
                logger.warning(f"No years available for experiment '{cds_exp}' with given range, skipping")
                summary["incomplete"].append({"cities": list(city_points), "model": model,
                    "experiment": cds_exp, "variables": variables, "reason": "no requested years"})
                continue
            label = CMIP6_EXP_LABEL[cds_exp]

            for temporal_res, var_list in variable_groups.items():
                suffix = "" if temporal_res == "daily" else "_monthly"
                pending_cities = []
                for city in city_points:
                    out_csv = output_root / city / f"{model}_{label}{suffix}.csv"
                    if _cmip6_csv_covers_request(out_csv, years, var_list):
                        logger.info(
                            "CMIP6 output covers requested fields and years: %s",
                            out_csv,
                        )
                        summary["success"].append({"city": city, "model": model,
                            "experiment": cds_exp, "temporal_resolution": temporal_res,
                            "variables": var_list, "skipped": True})
                        continue
                    pending_cities.append(city)
                if not pending_cities:
                    logger.info(
                        "CMIP6 outputs already exist for all cities: model='%s', experiment='%s', resolution='%s'",
                        model, label, temporal_res,
                    )
                    continue

                city_frames: dict[str, list[pd.DataFrame]] = {city: [] for city in pending_cities}
                pbar_desc = f"CMIP6 {model} {label} ({len(pending_cities)} cities)"

                for variable in tqdm(var_list, desc=pbar_desc, unit="var"):
                    short = CMIP6_VAR_SHORT.get(variable, variable)
                    missing_cities = []
                    for city in pending_cities:
                        cache_path = _cmip6_cache_path(
                            output_root, model, cds_exp, temporal_res, variable,
                            city, years, city_points[city],
                        )
                        cached = _read_cmip6_variable(cache_path, years, variable)
                        if cached is None:
                            out_csv = output_root / city / f"{model}_{label}{suffix}.csv"
                            cached = _cmip6_existing_variable(out_csv, years, variable, cache_path)
                        if cached is None:
                            missing_cities.append(city)
                        else:
                            city_frames[city].append(cached)
                    if not missing_cities:
                        continue
                    tmp_nc = output_root / f"_tmp_{model}_{cds_exp}_{variable}.nc"
                    logger.info("  → %s (%d cities)", variable, len(missing_cities))
                    try:
                        request_points = {city: city_points[city] for city in missing_cities}
                        req: dict = {
                            "temporal_resolution": temporal_res,
                            "experiment": cds_exp,
                            "variable": variable,
                            "model": model,
                            "year": [str(y) for y in years],
                            "area": _cmip6_request_area(request_points),
                        }
                        if temporal_res == "daily":
                            req["month"] = all_months
                            req["day"] = all_days
                        client.retrieve(CDS_CMIP6_DATASET, req, str(tmp_nc))

                        city_dfs = _nc_bytes_to_point_dfs(
                            tmp_nc.read_bytes(),
                            request_points,
                        )
                        rename = {}
                        for raw_col in next(iter(city_dfs.values())).columns:
                            if raw_col != "time":
                                rename[raw_col] = CMIP6_VAR_SHORT.get(raw_col, raw_col)
                        city_errors = {}
                        for city in missing_cities:
                            try:
                                df_city = city_dfs.get(city)
                                if df_city is None:
                                    raise ValueError(f"extracted data missing city {city}")
                                df_city = df_city.rename(columns=rename)[["time", short]]
                                if not _cmip6_variable_covers_request(df_city, years, variable):
                                    raise ValueError(f"extracted data for {city} failed field/year validation")
                                cache_path = _cmip6_cache_path(
                                    output_root, model, cds_exp, temporal_res, variable,
                                    city, years, city_points[city],
                                )
                                _write_cmip6_variable(cache_path, df_city)
                                city_frames[city].append(df_city)
                            except Exception as exc:
                                city_errors[city] = str(exc)
                        for city, error in city_errors.items():
                            summary["failed"].append({"model": model, "experiment": cds_exp,
                                "temporal_resolution": temporal_res, "variable": variable,
                                "cities": [city], "error": error})
                            summary["incomplete"].append({"city": city, "model": model,
                                "experiment": cds_exp, "temporal_resolution": temporal_res,
                                "variable": variable})
                            _record_cmip6_failure(output_root, model=model, experiment=cds_exp,
                                resolution=temporal_res, variable=variable, cities=[city], error=error)
                        logger.info("    ✓ %s", CMIP6_VAR_SHORT.get(variable, variable))
                    except Exception as exc:
                        for city in missing_cities:
                            item = {"model": model, "experiment": cds_exp,
                                "temporal_resolution": temporal_res, "variable": variable,
                                "cities": [city], "error": str(exc)}
                            summary["failed"].append(item)
                            summary["incomplete"].append({"city": city, "model": model,
                                "experiment": cds_exp, "temporal_resolution": temporal_res,
                                "variable": variable})
                        _record_cmip6_failure(output_root, model=model, experiment=cds_exp,
                            resolution=temporal_res, variable=variable, cities=missing_cities, error=str(exc))
                        logger.warning("    ✗ %s: %s", variable, exc)
                    finally:
                        if tmp_nc.exists():
                            tmp_nc.unlink()
                    time.sleep(1)

                for city in pending_cities:
                    merged = _merge_cmip6_city_frames(city_frames, city)
                    if merged is None or not _cmip6_frame_covers_request(merged, years, var_list):
                        summary["incomplete"].append({"city": city, "model": model,
                            "experiment": cds_exp, "temporal_resolution": temporal_res,
                            "variables": var_list})
                        logger.error(
                            "CMIP6 data for city='%s', model='%s', experiment='%s', resolution='%s' "
                            "failed field/year validation; no CSV written.",
                            city, model, label, temporal_res,
                        )
                        continue
                    city_dir = output_root / city
                    city_dir.mkdir(parents=True, exist_ok=True)
                    out_csv = city_dir / f"{model}_{label}{suffix}.csv"
                    if not _write_cmip6_final(out_csv, merged, years, var_list):
                        summary["incomplete"].append({"city": city, "model": model,
                            "experiment": cds_exp, "temporal_resolution": temporal_res,
                            "variables": var_list})
                        continue
                    summary["success"].append({"city": city, "model": model,
                        "experiment": cds_exp, "temporal_resolution": temporal_res,
                        "variables": var_list})
                    logger.info("CMIP6 saved → %s (%s rows)", out_csv, len(merged))
    logger.info("CMIP6 summary: %d successful outputs, %d failed items, %d incomplete outputs",
                len(summary["success"]), len(summary["failed"]), len(summary["incomplete"]))
    for item in summary["success"]:
        logger.info("CMIP6 success: city=%s model=%s experiment=%s resolution=%s",
                    item["city"], item["model"], item["experiment"], item["temporal_resolution"])
    for item in summary["failed"]:
        logger.error("CMIP6 failed: variable=%s cities=%s model=%s experiment=%s resolution=%s error=%s",
                     item["variable"], item["cities"], item["model"], item["experiment"],
                     item["temporal_resolution"], item["error"])
    for item in summary["incomplete"]:
        logger.error("CMIP6 incomplete: %s", item)
    return summary


# ─── CORDEX-CMIP5 ─────────────────────────────────────────────────────────────

def _cordex_point_df(nc_bytes: bytes, lat: float, lon: float) -> pd.DataFrame:
    """Merge all NetCDF members returned by a CORDEX request at one city point."""
    if nc_bytes[:2] == b"PK":
        with zipfile.ZipFile(BytesIO(nc_bytes)) as archive:
            nc_names = sorted(name for name in archive.namelist() if name.lower().endswith(".nc"))
            if not nc_names:
                raise ValueError("No NetCDF file found in CORDEX ZIP response")
            frames = [_nc_bytes_to_point_df(archive.read(name), lat, lon) for name in nc_names]
    else:
        frames = [_nc_bytes_to_point_df(nc_bytes, lat, lon)]

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["time"] = pd.to_datetime(combined["time"])
    value_columns = [column for column in combined if column != "time"]
    conflicts = [
        column for column in value_columns
        if (combined.groupby("time")[column].nunique(dropna=True) > 1).any()
    ]
    if conflicts:
        raise ValueError(f"Conflicting CORDEX values for duplicate timestamps: {conflicts}")
    return combined.groupby("time", as_index=False).first().sort_values("time").reset_index(drop=True)


def _cordex_year_point_df(nc_bytes: bytes, lat: float, lon: float, year: int) -> pd.DataFrame:
    """Select one year, including when a retained ZIP covers several years."""
    frame = _cordex_point_df(nc_bytes, lat, lon)
    selected = frame[pd.to_datetime(frame["time"]).dt.year == year]
    if selected.empty:
        raise ValueError(f"CORDEX response has no daily values for {year}")
    return selected.reset_index(drop=True)


def _cordex_csv_covers_request(path: Path, years: list[int], months: list[str],
                               variables: list[str]) -> bool:
    """Check that an existing city CSV covers every requested full month."""
    try:
        frame = pd.read_csv(path, parse_dates=["time"])
        columns = [CORDEX_VAR_SHORT[var] for var in variables]
        if not set(columns).issubset(frame.columns):
            return False
        dates = frame["time"].dt.normalize()
        if dates.duplicated().any() or frame[columns].isna().any().any():
            return False
        counts = frame.groupby([frame["time"].dt.year, frame["time"].dt.month]).size()
        return all(counts.get((year, int(month)), 0) >= 28 for year in years for month in months)
    except (OSError, ValueError, KeyError):
        return False


def download_cordex_cmip5(
    city: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    cordex_domain: Optional[str] = None,
    gcm: str = "mpi_m_mpi_esm_lr",
    rcm: str = "ncar_regcm4",
    ensemble: str = "r1i1p1",
    experiments: Optional[List[str]] = None,
    variables: Optional[List[str]] = None,
    historical_years: Optional[tuple[int, int]] = None,
    rcp85_years: Optional[tuple[int, int]] = None,
    months: Optional[List[str]] = None,
    output_dir: str = "data/CORDEX_CMIP5",
    overwrite: bool = False,
    keep_parts: bool = False,
) -> dict[str, Path]:
    """Download CORDEX-CMIP5 daily city series for the climate-delta reader.

    Output layout is ``<output_dir>/<variant>/<CitySlug>/{historical,rcp85}.csv``.
    The CDS area is a small, nonzero box around the city; the nearest grid cell
    is selected locally from each returned NetCDF file.
    """
    import cdsapi

    city = _normalize_city_name(city)
    lat, lon = _resolve_coords(city, lat, lon)
    experiments = experiments or ["historical", "rcp_8_5"]
    requested_variables = variables
    variables = variables or CORDEX_VARS
    months = months or [f"{month:02d}" for month in range(1, 13)]
    if any(month not in {f"{i:02d}" for i in range(1, 13)} for month in months):
        raise ValueError("CORDEX months must be two-digit values from 01 to 12")
    if any(experiment not in CORDEX_EXP_YEARS for experiment in experiments):
        raise ValueError("CORDEX experiments must be historical and/or rcp_8_5")
    if cordex_domain is None:
        cordex_domain = _city_meta(city)["cordex_domain"]

    cds_domain = _normalize_cordex_domain(cordex_domain)
    if cds_domain is None:
        raise ValueError(
            f"Unknown cordex_domain '{cordex_domain}'. "
            f"Supported: {list(CORDEX_DOMAIN_MAP)}"
        )

    client = cdsapi.Client()

    download_plan = _resolve_cordex_download_plan(
        cordex_domain=cordex_domain,
        gcm=gcm,
        rcm=rcm,
        ensemble=ensemble,
        experiments=experiments,
        variables=variables,
    )
    if not download_plan:
        raise ValueError("No CORDEX download combinations matched the CDS constraints")

    planned_experiments = {str(plan["experiment"]) for plan in download_plan}
    missing_experiments = set(experiments) - planned_experiments
    if missing_experiments:
        raise ValueError(f"CORDEX experiments unavailable in CDS: {sorted(missing_experiments)}")
    if requested_variables is not None:
        for plan in download_plan:
            missing_variables = set(requested_variables) - set(plan["variable"])
            if missing_variables:
                raise ValueError(
                    f"CORDEX variables unavailable for {plan['experiment']}: "
                    f"{sorted(missing_variables)}"
                )

    outputs: dict[str, Path] = {}
    year_ranges = {"historical": historical_years, "rcp_8_5": rcp85_years}
    for plan in download_plan:
        experiment = str(plan["experiment"])
        years = CORDEX_EXP_YEARS.get(experiment)
        if not years:
            logger.warning(f"No year range configured for CORDEX experiment '{experiment}'")
            continue
        selected_range = year_ranges[experiment]
        if selected_range is not None:
            start, end = selected_range
            if start > end:
                raise ValueError(f"Invalid CORDEX year range: {start} > {end}")
            years = [year for year in years if start <= year <= end]
        if not years:
            raise ValueError(f"No CORDEX years selected for {experiment}")
        exp_label = "historical" if experiment == "historical" else "rcp85"
        variant = f"{cordex_domain}_{gcm}_{rcm}_{plan['ensemble_member']}"
        city_dir = Path(output_dir) / variant / city
        city_dir.mkdir(parents=True, exist_ok=True)
        out_csv = city_dir / f"{exp_label}.csv"

        if out_csv.exists() and not overwrite:
            if _cordex_csv_covers_request(out_csv, years, months, list(plan["variable"])):
                logger.info("  %s covers requested years, months and variables; skipping", out_csv)
                outputs[experiment] = out_csv
                continue
            raise FileExistsError(
                f"Existing CORDEX CSV does not cover this request: {out_csv}. "
                "Use --cordex-overwrite or another --cordex-dir."
            )

        year_dfs: list[pd.DataFrame] = []
        part_paths: list[Path] = []
        failed_requests: list[str] = []
        pbar_desc = f"CORDEX {city} {exp_label}"

        for year in tqdm(years, desc=pbar_desc, unit="yr"):
            variable_frames: list[pd.DataFrame] = []
            for variable in plan["variable"]:
                short = CORDEX_VAR_SHORT[str(variable)]
                part_path = city_dir / f"_part_{experiment}_{year}_{short}_{''.join(months)}.zip"
                requested = False
                try:
                    if not part_path.exists():
                        requested = True
                        req = {
                            "format": "zip",
                            "domain": str(plan["domain"]),
                            "experiment": experiment,
                            "horizontal_resolution": str(plan["horizontal_resolution"]),
                            "temporal_resolution": "daily_mean",
                            "variable": str(variable),
                            "gcm_model": str(plan["gcm_model"]),
                            "rcm_model": str(plan["rcm_model"]),
                            "ensemble_member": str(plan["ensemble_member"]),
                            "year": [str(year)],
                            "month": months,
                            "area": [round(lat + 0.5, 2), round(lon - 0.5, 2),
                                     round(lat - 0.5, 2), round(lon + 0.5, 2)],
                        }
                        client.retrieve(CDS_CORDEX_DATASET, req, str(part_path))
                    frame = _cordex_year_point_df(part_path.read_bytes(), lat, lon, year)
                    if short not in frame:
                        raise ValueError(f"CORDEX response missing requested variable '{short}'")
                    variable_frames.append(frame[["time", short]])
                    part_paths.append(part_path)
                    logger.info("  %s %s ✓", year, short)
                except Exception as exc:
                    logger.warning("  %s %s failed: %s", year, short, exc)
                    failed_requests.append(f"{year}/{short}")
                    part_path.unlink(missing_ok=True)
                if requested:
                    time.sleep(1)
            if len(variable_frames) == len(plan["variable"]):
                df_year = variable_frames[0]
                for frame in variable_frames[1:]:
                    df_year = pd.merge(df_year, frame, on="time", how="outer", validate="one_to_one")
                if df_year.isna().any().any():
                    failed_requests.append(f"{year}/incomplete timestamps")
                else:
                    year_dfs.append(df_year)

        if failed_requests:
            raise RuntimeError(
                f"CORDEX {experiment} failed for {failed_requests}; no partial CSV written"
            )

        merged = pd.concat(year_dfs, ignore_index=True)
        if merged["time"].duplicated().any():
            raise ValueError(f"Duplicate CORDEX timestamps in {experiment} response")
        merged = merged.sort_values("time").reset_index(drop=True)
        merged.to_csv(out_csv, index=False)
        if not keep_parts:
            for part_path in part_paths:
                part_path.unlink(missing_ok=True)
        logger.info(f"CORDEX saved → {out_csv} ({len(merged)} rows)")
        outputs[experiment] = out_csv

    return outputs


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="samy-download",
        description="Download city-specific weather data for the samy-wgen pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── config ─────────────────────────────────────────────────────────────────
    sub.add_parser("config", help="Save NSRDB credentials to ~/.config/samy_wgen/credentials.yaml")

    # ── list-cmip6 ─────────────────────────────────────────────────────────────
    p_lm = sub.add_parser("list-cmip6", help="List available CMIP6 model / experiment / resolution / variable combinations from the live CDS catalogue")
    p_lm.add_argument("--detail", action="store_true", help="Expand each model into experiment / resolution / variable rows")
    p_lm.add_argument("--experiment", help="Filter by CDS experiment name, e.g. ssp2_4_5")
    p_lm.add_argument(
        "--var",
        choices=sorted(CMIP6_VAR_SHORT),
        help="Filter by CMIP6 CDS variable name",
    )
    p_lm.add_argument(
        "--resolution",
        choices=["daily", "monthly", "fixed"],
        help="Filter by CDS temporal resolution",
    )

    # ── extract-cmip6 ─────────────────────────────────────────────────────────
    p_ec = sub.add_parser("extract-cmip6", help="Extract nearest-grid city time series from local CMIP6 NetCDF file(s)")
    p_ec.add_argument("--city", required=True, nargs="+", help="ASCII city code(s) from cities_config.csv")
    p_ec.add_argument("--input", required=True, nargs="+", metavar="NETCDF", help="Local CMIP6 NetCDF file(s); shell globs are accepted")
    p_ec.add_argument("--cmip6-dir", default="data/CMIP6", help="Output CMIP6 directory")
    p_ec.add_argument("--name", default="extracted_cmip6", metavar="STEM", help="Output filename stem written as <cmip6-dir>/<city>/<stem>.csv")

    # ── list-cordex ───────────────────────────────────────────────────────────
    p_lc = sub.add_parser("list-cordex", help="List available CORDEX domain / experiment / GCM / RCM / ensemble / resolution / variable combinations from the live CDS catalogue")
    p_lc.add_argument("--detail", action="store_true", help="Expand each GCM/RCM/ensemble combination into domain / experiment / resolution / variable rows")
    p_lc.add_argument("--domain", choices=sorted(CORDEX_DOMAIN_MAP), help="Filter by CORDEX domain, e.g. NAM-22")
    p_lc.add_argument("--experiment", help="Filter by CORDEX CDS experiment name, e.g. historical or rcp_8_5")
    p_lc.add_argument("--gcm", help="Filter by CORDEX driving GCM model")
    p_lc.add_argument("--rcm", help="Filter by CORDEX regional climate model")
    p_lc.add_argument("--ensemble", help="Filter by CORDEX ensemble member")
    p_lc.add_argument("--resolution", help="Filter by CORDEX horizontal resolution")
    p_lc.add_argument("--var", choices=sorted(CORDEX_SUPPORTED_VARIABLES), help="Filter by CORDEX variable")

    # ── list-stations ──────────────────────────────────────────────────────────
    p_list = sub.add_parser("list-stations", help="List nearby ISD stations for a city")
    p_list.add_argument("--city",    required=True,    help="ASCII city code from cities_config.csv")
    p_list.add_argument("--radius",  type=float, default=150, help="Search radius in km (default: 150)")
    p_list.add_argument("--isd-dir", default="data/ISD", help="Directory to cache isd-history.csv")

    # ── list-nsrdb ─────────────────────────────────────────────────────────────
    p_nsrdb = sub.add_parser("list-nsrdb", help="Inspect NSRDB sources and default coverage for one or more cities")
    p_nsrdb.add_argument("--city", required=True, nargs="+", help="ASCII city code(s) from cities_config.csv")

    # ── calibrate-solar ────────────────────────────────────────────────────────
    p_cal = sub.add_parser(
        "calibrate-solar",
        help="Calibrate ERA5-Land solar radiation with overlapping NSRDB observations",
    )
    p_cal.add_argument("--city", required=True, help="ASCII city code from cities_config.csv")
    p_cal.add_argument("--era5", required=True, type=Path, help="ERA5-Land NetCDF or CSV containing surface solar radiation")
    p_cal.add_argument("--nsrdb", required=True, type=Path, help="NSRDB CSV containing GHI, DNI, and DHI")
    p_cal.add_argument("--output", type=Path, default=None, help="Output CSV (default: data/ERA5_Land/calibrated/<city>_solar_calibrated.csv)")
    p_cal.add_argument("--utc-offset", type=float, default=None, help="Override city fixed-standard UTC offset in hours")
    p_cal.add_argument(
        "--nsrdb-time-basis", choices=["auto", "utc", "lst"], default="auto",
        help="NSRDB timestamp basis; auto requires decisive zenith or GHI daylight evidence",
    )

    # ── check-station ──────────────────────────────────────────────────────────
    p_cov = sub.add_parser("check-station", help="Inspect metadata year coverage for one ISD station")
    p_cov.add_argument("--isd-station", required=True, metavar="USAF-WBAN", help="ISD station id to inspect")
    p_cov.add_argument("--isd-dir", default="data/ISD", help="Directory containing cached isd-history.csv")
    p_cov.add_argument(
        "--years", nargs=2, type=int, metavar=("START", "END"), default=None,
        help="Optional requested year range to compare with station metadata",
    )

    # ── download ───────────────────────────────────────────────────────────────
    p_dl = sub.add_parser("download", help="Download weather data for one or more cities")
    p_dl.add_argument("--city", required=True, nargs="+", help="ASCII city code(s) from cities_config.csv")
    p_dl.add_argument(
        "--data", nargs="+",
        choices=["isd", "nsrdb", "cmip6", "cordex", "era5-land"],
        default=["isd", "nsrdb", "cmip6"],
        help="Data sources to download (default: isd nsrdb cmip6)",
    )

    # ISD
    p_dl.add_argument(
        "--isd-station",
        nargs="+",
        metavar="USAF-WBAN|CITY=USAF-WBAN",
        help="ISD station override. Single-city mode accepts a bare USAF-WBAN; multi-city mode accepts City=USAF-WBAN mappings.",
    )
    p_dl.add_argument(
        "--isd-radius", type=float, default=50.0,
        help="Radius in km for automatic ISD station selection (default: 50)",
    )
    p_dl.add_argument("--isd-dir",  default="data/ISD")

    # NSRDB
    p_dl.add_argument("--nsrdb-dir", default="data/NSRDB")

    # ERA5-Land
    p_dl.add_argument("--era5-land-dir", default="data/ERA5_Land")
    p_dl.add_argument(
        "--era5-land-years", nargs=2, type=int, metavar=("START", "END"), default=None,
        help="ERA5-Land year range (default: --years if supplied, otherwise 1950 through the latest available day)",
    )
    p_dl.add_argument(
        "--era5-land-var", nargs="+", choices=ERA5_LAND_VARIABLES, default=None,
        help="ERA5-Land variables (default: dewpoint, temperature, pressure, precipitation, and solar radiation)",
    )
    p_dl.add_argument("--era5-land-overwrite", action="store_true", help="Replace existing ERA5-Land NetCDF files")

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
    p_dl.add_argument(
        "--cmip6-var", nargs="+", default=None, choices=sorted(CMIP6_VAR_SHORT),
        help="CMIP6 CDS variables to download (default: all supported variables)",
    )

    # ISD / NSRDB year range
    p_dl.add_argument(
        "--years", nargs=2, type=int, metavar=("START", "END"), default=None,
        help="Year range for ISD / NSRDB (default: full available history)",
    )

    # CMIP6 year range
    p_dl.add_argument(
        "--cmip6-years", nargs=2, type=int, metavar=("START", "END"), default=None,
        help="Year range for CMIP6 downloads (default: full experiment range)",
    )

    # CORDEX-CMIP5
    p_dl.add_argument("--cordex-dir", default="data/CORDEX_CMIP5")
    p_dl.add_argument("--cordex-overwrite", action="store_true",
                      help="Replace a CORDEX city CSV after a complete successful download")
    p_dl.add_argument("--cordex-domain", choices=sorted(CORDEX_DOMAIN_MAP), default=None)
    p_dl.add_argument("--cordex-gcm", default="mpi_m_mpi_esm_lr")
    p_dl.add_argument("--cordex-rcm", default="ncar_regcm4")
    p_dl.add_argument("--cordex-ensemble", default="r1i1p1")
    p_dl.add_argument(
        "--cordex-experiments", nargs="+", choices=sorted(CORDEX_EXP_YEARS),
        default=["historical", "rcp_8_5"],
    )
    p_dl.add_argument(
        "--cordex-var", nargs="+", choices=sorted(CORDEX_SUPPORTED_VARIABLES),
        default=None, help="CORDEX variables (default: all supported variables)",
    )
    p_dl.add_argument(
        "--cordex-historical-years", nargs=2, type=int, metavar=("START", "END"),
        default=None, help="Historical CORDEX years (default: 1970–2005)",
    )
    p_dl.add_argument(
        "--cordex-rcp85-years", nargs=2, type=int, metavar=("START", "END"),
        default=None, help="RCP8.5 CORDEX years (default: 2006–2100)",
    )
    p_dl.add_argument(
        "--cordex-months", nargs="+", choices=[f"{month:02d}" for month in range(1, 13)],
        default=None, help="Months to download (default: all 12)",
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

    if args.command == "list-cmip6":
        model_info = _list_cmip6_models(
            experiment=args.experiment,
            variable=args.var,
            temporal_resolution=args.resolution,
        )
        models = model_info["model"]
        if not models:
            print("No CMIP6 models matched the requested CDS constraints.")
            return

        if not args.detail:
            print("Model ID")
            print("--------")
            for model in models:
                print(model)
            print(f"\nTotal: {len(models)} models")
            return

        rows = _list_cmip6_model_rows(
            experiment=args.experiment,
            variable=args.var,
            temporal_resolution=args.resolution,
            show_progress=True,
        )
        header = f"{'Model ID':<25} {'Experiment':<14} {'Resolution':<10} Variable"
        print(header)
        print("-" * len(header))
        for row in rows:
            print(f"{row['model']:<25} {row['experiment']:<14} {row['resolution']:<10} {row['variable']}")
        print(f"\nTotal: {len(rows)} rows")
        return

    if args.command == "extract-cmip6":
        extract_cmip6_from_files(args.city, args.input, args.cmip6_dir, args.name)
        return

    if args.command == "calibrate-solar":
        from samy_wgen.utils.solar_calibration import calibrate_era5_land_solar
        from samy_wgen.utils.time_utils import get_utc_offset

        meta = _city_meta(args.city)
        utc_offset = args.utc_offset
        if utc_offset is None:
            utc_offset = get_utc_offset(city=args.city)
        output_path = args.output or Path("data/ERA5_Land/calibrated") / f"{args.city.lower()}_solar_calibrated.csv"
        calibrated = calibrate_era5_land_solar(
            args.era5,
            args.nsrdb,
            latitude=meta["lat"],
            longitude=meta["lon"],
            utc_offset_hours=utc_offset,
            nsrdb_time_basis=args.nsrdb_time_basis,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        calibrated.to_csv(output_path, index=True, index_label="DATE")
        import json

        metadata_path = output_path.with_suffix(".metadata.json")
        metadata_path.write_text(json.dumps(calibrated.attrs, indent=2) + "\n")
        logger.info("Calibrated solar data saved → %s (%s hourly rows)", output_path, len(calibrated))
        logger.info("Solar calibration time basis and sample counts → %s: %s", metadata_path, calibrated.attrs)
        return

    if args.command == "list-cordex":
        rows = _list_cordex_rows(
            domain=args.domain,
            experiment=args.experiment,
            gcm=args.gcm,
            rcm=args.rcm,
            ensemble=args.ensemble,
            resolution=args.resolution,
            variable=args.var,
            show_progress=True,
            expand_ensemble=args.detail or args.ensemble is not None,
        )
        combos = sorted({(row["gcm_model"], row["rcm_model"]) for row in rows})
        if not combos:
            print("No CORDEX combinations matched the requested CDS constraints.")
            return

        if not args.detail:
            header = f"{'GCM Model':<30} {'RCM Model':<20}"
            print(header)
            print("-" * len(header))
            for gcm_id, rcm_id in combos:
                print(f"{gcm_id:<30} {rcm_id:<20}")
            print(f"\nTotal: {len(combos)} combinations")
            return

        header = f"{'Domain':<8} {'Experiment':<12} {'GCM Model':<30} {'RCM Model':<20} {'Ensemble':<10} {'Resolution':<28} Variable"
        print(header)
        print("-" * len(header))
        for row in rows:
            print(
                f"{row['domain']:<8} {row['experiment']:<12} {row['gcm_model']:<30} "
                f"{row['rcm_model']:<20} {row['ensemble_member']:<10} {row['resolution']:<28} {row['variable']}"
            )
        print(f"\nTotal: {len(rows)} rows")
        return

    if args.command == "list-stations":
        meta = _city_meta(args.city)
        list_isd_stations(meta["lat"], meta["lon"], args.radius, args.isd_dir)
        return

    if args.command == "list-nsrdb":
        list_nsrdb_sources(args.city)
        return

    if args.command == "check-station":
        coverage = get_isd_station_coverage(args.isd_station, cache_dir=args.isd_dir)
        print(f"Station: {coverage['station_id']}")
        print(f"Name: {coverage['station_name']}")
        print(f"Coverage: {coverage['begin_year']}–{coverage['end_year']}")
        print(f"Coverage years: {coverage['coverage_years']}")
        if args.years:
            requested_start, requested_end = args.years
            overlap_start = max(int(requested_start), int(coverage['begin_year'])) if coverage['begin_year'] is not None else requested_start
            overlap_end = min(int(requested_end), int(coverage['end_year'])) if coverage['end_year'] is not None else requested_end
            if overlap_start <= overlap_end:
                print(f"Requested overlap: {overlap_start}–{overlap_end}")
            else:
                print(f"Requested overlap: none within {requested_start}–{requested_end}")
        return

    # ── download ───────────────────────────────────────────────────────────────
    cities = _normalize_city_args(args.city)
    sources = set(args.data)

    if "isd" in sources:
        yr_start, yr_end = args.years if args.years else DEFAULT_ISD_YEAR_RANGE
        station_ids = _parse_isd_station_args(args.isd_station, cities)
        download_isd_for_cities(
            cities=cities,
            years=list(range(yr_start, yr_end + 1)),
            output_dir=args.isd_dir,
            isd_station_ids=station_ids,
            isd_radius_km=args.isd_radius,
        )

    if "nsrdb" in sources:
        yr_start, yr_end = args.years if args.years else DEFAULT_NSRDB_YEAR_RANGE
        summary = download_nsrdb_for_cities(
            cities=cities,
            years=list(range(yr_start, yr_end + 1)),
            output_dir=args.nsrdb_dir,
        )
        if summary["failed"] or summary["incomplete"]:
            logger.error("NSRDB required data remains incomplete: %s failed year(s), %s incomplete city output(s)",
                         len(summary["failed"]), len(summary["incomplete"]))
            raise SystemExit(1)

    if "cmip6" in sources:
        summary = download_cmip6_for_cities(
            cities=cities,
            models=args.models,
            experiments=args.experiments,
            variables=args.cmip6_var,
            year_range=tuple(args.cmip6_years) if args.cmip6_years else None,
            output_dir=args.cmip6_dir,
        )
        if summary["failed"] or summary["incomplete"]:
            logger.error("CMIP6 required data remains incomplete: %s failed item(s), %s incomplete output(s)",
                         len(summary["failed"]), len(summary["incomplete"]))
            raise SystemExit(1)

    if "cordex" in sources:
        for city in cities:
            download_cordex_cmip5(
                city=city,
                cordex_domain=args.cordex_domain,
                gcm=args.cordex_gcm,
                rcm=args.cordex_rcm,
                ensemble=args.cordex_ensemble,
                experiments=args.cordex_experiments,
                variables=args.cordex_var,
                historical_years=tuple(args.cordex_historical_years) if args.cordex_historical_years else None,
                rcp85_years=tuple(args.cordex_rcp85_years) if args.cordex_rcp85_years else None,
                months=args.cordex_months,
                output_dir=args.cordex_dir,
                overwrite=args.cordex_overwrite,
            )

    if "era5-land" in sources:
        era5_years = args.era5_land_years or args.years
        if era5_years:
            era5_start = f"{era5_years[0]:04d}-01-01"
            era5_end = min(
                pd.Timestamp(f"{era5_years[1]:04d}-12-31").date(),
                date.today() - timedelta(days=ERA5_LAND_AVAILABILITY_LAG_DAYS),
            ).isoformat()
        else:
            era5_start = "1950-01-01"
            era5_end = (date.today() - timedelta(days=ERA5_LAND_AVAILABILITY_LAG_DAYS)).isoformat()
        download_era5_land_for_cities(
            cities=cities,
            start_date=era5_start,
            end_date=era5_end,
            output_dir=args.era5_land_dir,
            variables=args.era5_land_var,
            overwrite=args.era5_land_overwrite,
        )

if __name__ == "__main__":
    main()
