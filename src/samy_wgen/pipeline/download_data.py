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
import functools
import json
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

NSRDB_SOURCES: dict[str, dict] = {
    "goes_aggregated": {
        "url": "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download",
        "year_range": (1998, 2023),
    },
    "goes_full_disc": {
        "url": "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-full-disc-v4-0-0-download",
        "year_range": (2018, 2023),
    },
    "meteosat": {
        "url": "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-msg-v1-0-0-download",
        "year_range": (2005, 2022),
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

NSRDB_ATTRS = [
    "air_temperature", "dew_point", "relative_humidity",
    "wind_speed", "wind_direction", "ghi", "dni", "dhi", "surface_pressure",
]

ISD_FTP_HOST    = "ftp.ncei.noaa.gov"
ISD_FTP_DIR     = "/pub/data/noaa/{year}/"
ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"

CDS_CMIP6_DATASET = "projections-cmip6"
CDS_CORDEX_DATASET = "projections-cordex-domains-single-levels"
CMIP6_SUPPORTED_VARIABLES = set(CMIP6_VAR_SHORT.keys())
CMIP6_TEMPORAL_PRIORITY = ("daily", "monthly", "fixed")
CORDEX_SUPPORTED_VARIABLES = set(CORDEX_VARS)
CORDEX_DOMAIN_BY_CDS = {value: key for key, value in CORDEX_DOMAIN_MAP.items()}
DEFAULT_ISD_YEAR_RANGE = (1970, 2024)
DEFAULT_NSRDB_YEAR_RANGE = (1998, 2024)


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


def _normalize_city_name(city: str, cfg: pd.DataFrame | None = None) -> str:
    """Return the canonical city name from cities_config.csv for one requested city."""
    cfg = _load_cities_config() if cfg is None else cfg
    key = city.lower()

    row = cfg[cfg["city"].str.lower() == key]
    if row.empty and "city_ascii" in cfg.columns:
        row = cfg[cfg["city_ascii"].str.lower() == key]
    if row.empty:
        raise ValueError(
            f"City '{city}' not found in cities_config. "
            f"Available: {cfg['city'].tolist()}"
        )
    return str(row.iloc[0]["city"])


def _normalize_city_args(requested: list[str]) -> list[str]:
    """Resolve city arguments to canonical config names while preserving order."""
    cfg = _load_cities_config()
    normalized: list[str] = []
    seen: set[str] = set()
    for city in requested:
        resolved = _normalize_city_name(city, cfg)
        if resolved in seen:
            continue
        normalized.append(resolved)
        seen.add(resolved)
    return normalized


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
    cordex_domain = meta["cordex_domain"]
    source_keys = CORDEX_TO_NSRDB_SOURCE.get(cordex_domain, [])
    if not source_keys:
        raise ValueError(
            f"No NSRDB coverage for city '{city}' (CORDEX domain '{cordex_domain}'). "
            "NSRDB is available for: "
            + ", ".join(d for d, keys in CORDEX_TO_NSRDB_SOURCE.items() if keys)
        )
    return [
        (NSRDB_SOURCES[key]["url"], *NSRDB_SOURCES[key]["year_range"])
        for key in source_keys
    ]


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
    time_index = pd.to_datetime(ds["time"].values)

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

def download_nsrdb(
    city: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    years: Optional[List[int]] = None,
    output_dir: str = "data/NSRDB",
    interval: int = 60,
    max_retries: int = 3,
    retry_backoff_sec: float = 2.0,
) -> None:
    """Download NSRDB solar irradiance for a point location.

    Coordinates are looked up from cities_config.csv when *lat*/*lon* are omitted.
    Retries transient per-year HTTP/TLS failures before skipping the year.
    """
    import requests

    lat, lon = _resolve_coords(city, lat, lon)
    sources = _resolve_nsrdb_source(city)
    overall_min = min(mn for _, mn, _ in sources)
    overall_max = max(mx for _, _, mx in sources)
    if years is None:
        years = list(range(overall_min, overall_max + 1))
    api_key, email = _load_nsrdb_credentials()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    city_lower = city.lower().replace(" ", "_")
    valid_years = [y for y in sorted(years) if overall_min <= y <= overall_max]
    if not valid_years:
        logger.error(
            f"No valid NSRDB years for '{city}' in the requested range. "
            f"Available: {overall_min}–{overall_max}."
        )
        return

    def _url_for_year(yr: int) -> str | None:
        for url, mn, mx in sources:
            if mn <= yr <= mx:
                return url
        return None

    all_dfs: list[pd.DataFrame] = []
    session = requests.Session()
    for i, year in enumerate(tqdm(valid_years, desc=f"NSRDB {city}", unit="yr")):
        if i > 0:
            time.sleep(2)  # respect NREL rate limit
        endpoint_url = _url_for_year(year)
        if endpoint_url is None:
            logger.warning(f"  {year}: no NSRDB source covers this year, skipping")
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
        for attempt in range(1, max_retries + 1):
            try:
                r = session.get(f"{endpoint_url}.csv", params=params, timeout=300)
                r.raise_for_status()
                lines = r.text.strip().split("\n")
                header = next((i for i, l in enumerate(lines) if l.startswith("Year,")), None)
                if header is None:
                    logger.warning(f"  {year}: unexpected response, skipping")
                    break
                df = pd.read_csv(StringIO("\n".join(lines[header:])))
                all_dfs.append(df)
                logger.info(f"  {year} ✓ ({len(df)} rows)")
                break
            except Exception as exc:
                if attempt >= max_retries:
                    logger.warning(f"  {year} failed after {attempt} attempt(s): {exc}")
                    break
                wait_s = retry_backoff_sec * attempt
                logger.warning(
                    f"  {year} attempt {attempt}/{max_retries} failed: {exc}. "
                    f"Retrying in {wait_s:.1f}s."
                )
                time.sleep(wait_s)

    if not all_dfs:
        logger.error("No NSRDB data downloaded.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    y0, y1   = int(combined["Year"].min()), int(combined["Year"].max())
    out_path = out_dir / f"nsrdb_{city_lower}_{y0}-{y1}.csv"
    combined.to_csv(out_path, index=False)
    logger.info(f"NSRDB saved → {out_path} ({len(combined)} rows, {y0}–{y1})")


def download_nsrdb_for_cities(
    cities: list[str],
    years: Optional[List[int]] = None,
    output_dir: str = "data/NSRDB",
    interval: int = 60,
    max_retries: int = 3,
    retry_backoff_sec: float = 2.0,
) -> None:
    """Download NSRDB data for multiple cities."""
    canonical_cities = _normalize_city_args(cities)
    for city in canonical_cities:
        download_nsrdb(
            city=city,
            years=years,
            output_dir=output_dir,
            interval=interval,
            max_retries=max_retries,
            retry_backoff_sec=retry_backoff_sec,
        )


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
    variables   = variables   or sorted(CMIP6_SUPPORTED_VARIABLES)
    all_months  = [f"{i:02d}" for i in range(1, 13)]
    all_days    = [f"{i:02d}" for i in range(1, 32)]
    client      = cdsapi.Client()

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
                continue

            years = CMIP6_EXP_YEARS.get(cds_exp, [])
            if year_range is not None:
                years = [y for y in years if year_range[0] <= y <= year_range[1]]
            if not years:
                logger.warning(f"No years available for experiment '{cds_exp}' with given range, skipping")
                continue
            label = CMIP6_EXP_LABEL[cds_exp]

            for temporal_res, var_list in variable_groups.items():
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
                            "area":                [round(lat + 1.5, 2), round(lon - 1.5, 2), round(lat - 1.5, 2), round(lon + 1.5, 2)],
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


def download_cmip6_for_cities(
    cities: list[str],
    models: Optional[List[str]] = None,
    experiments: Optional[List[str]] = None,
    variables: Optional[List[str]] = None,
    year_range: Optional[tuple] = None,
    output_dir: str = "data/CMIP6",
) -> None:
    """Download CMIP6 projections for multiple cities using one CDS request per variable group."""
    import cdsapi

    city_points = _resolve_city_points(cities)
    models = models or ["mpi_esm1_2_lr"]
    experiments = experiments or ["historical", "ssp2_4_5"]
    variables = variables or sorted(CMIP6_SUPPORTED_VARIABLES)
    all_months = [f"{i:02d}" for i in range(1, 13)]
    all_days = [f"{i:02d}" for i in range(1, 32)]
    area = _cmip6_request_area(city_points)
    client = cdsapi.Client()

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
                continue

            years = CMIP6_EXP_YEARS.get(cds_exp, [])
            if year_range is not None:
                years = [y for y in years if year_range[0] <= y <= year_range[1]]
            if not years:
                logger.warning(f"No years available for experiment '{cds_exp}' with given range, skipping")
                continue
            label = CMIP6_EXP_LABEL[cds_exp]

            for temporal_res, var_list in variable_groups.items():
                suffix = "" if temporal_res == "daily" else "_monthly"
                pending_cities = [
                    city for city in city_points
                    if not (Path(output_dir) / city / f"{model}_{label}{suffix}.csv").exists()
                ]
                if not pending_cities:
                    logger.info(
                        "CMIP6 outputs already exist for all cities: model='%s', experiment='%s', resolution='%s'",
                        model, label, temporal_res,
                    )
                    continue

                city_frames: dict[str, list[pd.DataFrame]] = {city: [] for city in pending_cities}
                pbar_desc = f"CMIP6 {model} {label} ({len(pending_cities)} cities)"

                for variable in tqdm(var_list, desc=pbar_desc, unit="var"):
                    tmp_nc = Path(output_dir) / f"_tmp_{model}_{cds_exp}_{variable}.nc"
                    logger.info("  → %s", variable)
                    try:
                        req: dict = {
                            "temporal_resolution": temporal_res,
                            "experiment": cds_exp,
                            "variable": variable,
                            "model": model,
                            "year": [str(y) for y in years],
                            "area": area,
                        }
                        if temporal_res == "daily":
                            req["month"] = all_months
                            req["day"] = all_days
                        client.retrieve(CDS_CMIP6_DATASET, req, str(tmp_nc))

                        city_dfs = _nc_bytes_to_point_dfs(
                            tmp_nc.read_bytes(),
                            {city: city_points[city] for city in pending_cities},
                        )
                        rename = {}
                        for raw_col in next(iter(city_dfs.values())).columns:
                            if raw_col != "time":
                                rename[raw_col] = CMIP6_VAR_SHORT.get(raw_col, raw_col)
                        for city, df_city in city_dfs.items():
                            city_frames[city].append(df_city.rename(columns=rename))
                        logger.info("    ✓ %s", CMIP6_VAR_SHORT.get(variable, variable))
                    except Exception as exc:
                        logger.warning("    ✗ %s: %s", variable, exc)
                    finally:
                        if tmp_nc.exists():
                            tmp_nc.unlink()
                    time.sleep(1)

                for city in pending_cities:
                    merged = _merge_cmip6_city_frames(city_frames, city)
                    if merged is None:
                        logger.error(
                            "No variables downloaded for city='%s', model='%s', experiment='%s', resolution='%s'",
                            city, model, label, temporal_res,
                        )
                        continue
                    city_dir = Path(output_dir) / city
                    city_dir.mkdir(parents=True, exist_ok=True)
                    out_csv = city_dir / f"{model}_{label}{suffix}.csv"
                    merged.to_csv(out_csv, index=False)
                    logger.info("CMIP6 saved → %s (%s rows)", out_csv, len(merged))


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

    cds_domain = _normalize_cordex_domain(cordex_domain)
    if cds_domain is None:
        raise ValueError(
            f"Unknown cordex_domain '{cordex_domain}'. "
            f"Supported: {list(CORDEX_DOMAIN_MAP)}"
        )

    # Short RCM label for directory naming
    rcm_short = "RegCM" if "regcm" in rcm.lower() else rcm.replace("_", "").upper()
    client    = cdsapi.Client()

    download_plan = _resolve_cordex_download_plan(
        cordex_domain=cordex_domain,
        gcm=gcm,
        rcm=rcm,
        ensemble=ensemble,
        experiments=experiments,
        variables=CORDEX_VARS,
    )
    if not download_plan:
        logger.warning("No CORDEX download combinations remained after CDS constraint validation.")
        return

    for plan in download_plan:
        experiment = str(plan["experiment"])
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
                    "domain":               str(plan["domain"]),
                    "experiment":           experiment,
                    "horizontal_resolution": str(plan["horizontal_resolution"]),
                    "temporal_resolution":  "daily_mean",
                    "variable":             list(plan["variable"]),
                    "gcm_model":            str(plan["gcm_model"]),
                    "rcm_model":            str(plan["rcm_model"]),
                    "ensemble_member":      str(plan["ensemble_member"]),
                    "start_year":           [str(year)],
                    "end_year":             [str(year)],
                    "area":                 [round(lat, 2), round(lon, 2), round(lat, 2), round(lon, 2)],
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
    p_list.add_argument("--city",    required=True,    help="City name (from cities_config.csv)")
    p_list.add_argument("--radius",  type=float, default=150, help="Search radius in km (default: 150)")
    p_list.add_argument("--isd-dir", default="data/ISD", help="Directory to cache isd-history.csv")

    # ── download ───────────────────────────────────────────────────────────────
    p_dl = sub.add_parser("download", help="Download weather data for one or more cities")
    p_dl.add_argument("--city", required=True, nargs="+", help="City name(s) from cities_config.csv")
    p_dl.add_argument(
        "--data", nargs="+",
        choices=["isd", "nsrdb", "cmip6"],
        default=["isd", "nsrdb", "cmip6"],
        help="Data sources to download (default: isd nsrdb cmip6)",
    )

    # ISD
    p_dl.add_argument("--isd-station", metavar="USAF-WBAN",
                      help="ISD station ID, e.g. 722020-12839 (default: auto-select best coverage within --isd-radius)")
    p_dl.add_argument(
        "--isd-radius", type=float, default=50.0,
        help="Radius in km for automatic ISD station selection (default: 50)",
    )
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

    # ── download ───────────────────────────────────────────────────────────────
    cities = _normalize_city_args(args.city)
    sources = set(args.data)

    if "isd" in sources:
        yr_start, yr_end = args.years if args.years else DEFAULT_ISD_YEAR_RANGE
        station_ids = None
        if args.isd_station:
            if len(cities) != 1:
                raise SystemExit("ERROR: --isd-station can only be used when exactly one --city is provided.")
            station_ids = {cities[0]: args.isd_station}
        download_isd_for_cities(
            cities=cities,
            years=list(range(yr_start, yr_end + 1)),
            output_dir=args.isd_dir,
            isd_station_ids=station_ids,
            isd_radius_km=args.isd_radius,
        )

    if "nsrdb" in sources:
        yr_start, yr_end = args.years if args.years else DEFAULT_NSRDB_YEAR_RANGE
        download_nsrdb_for_cities(
            cities=cities,
            years=list(range(yr_start, yr_end + 1)),
            output_dir=args.nsrdb_dir,
        )

    if "cmip6" in sources:
        download_cmip6_for_cities(
            cities=cities,
            models=args.models,
            experiments=args.experiments,
            variables=args.cmip6_var,
            year_range=tuple(args.cmip6_years) if args.cmip6_years else None,
            output_dir=args.cmip6_dir,
        )

if __name__ == "__main__":
    main()
