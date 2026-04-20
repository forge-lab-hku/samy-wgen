from __future__ import annotations

import importlib.resources
import re
import unicodedata
from dataclasses import dataclass
from difflib import get_close_matches
from functools import lru_cache

import pandas as pd


@dataclass(frozen=True)
class CityRecord:
    city_id: int
    city_ascii: str
    city: str
    country: str
    country_iso2: str
    latitude: float
    longitude: float
    cordex_domain: str


def _normalize_query(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_value.lower())


def _casefold_code(value: str) -> str:
    return value.strip().lower()


@lru_cache(maxsize=1)
def load_city_table() -> pd.DataFrame:
    ref = importlib.resources.files("samy_wgen.config").joinpath("cities_config.csv")
    with ref.open("r", encoding="utf-8") as f:
        df = pd.read_csv(f)

    required = {
        "city_id",
        "city",
        "city_ascii",
        "country",
        "country_iso2",
        "latitude",
        "longitude",
        "cordex_domain",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "cities_config.csv missing required columns: "
            f"{missing}. Found columns: {list(df.columns)}"
        )

    df = df.copy()
    df["city_id"] = pd.to_numeric(df["city_id"], errors="raise").astype("int64")
    for column in ("city", "city_ascii", "country", "country_iso2", "cordex_domain"):
        df[column] = df[column].astype(str)
        if df[column].str.len().eq(0).any():
            raise ValueError(f"cities_config.csv contains empty values in '{column}'.")
    for column in ("latitude", "longitude"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any():
            raise ValueError(f"cities_config.csv contains invalid numeric values in '{column}'.")

    if df["city_id"].duplicated().any():
        duplicates = df.loc[df["city_id"].duplicated(keep=False), "city_id"].tolist()
        raise ValueError(f"cities_config.csv contains duplicate city_id values: {duplicates}")
    if df["city_ascii"].duplicated().any():
        duplicates = df.loc[df["city_ascii"].duplicated(keep=False), "city_ascii"].tolist()
        raise ValueError(f"cities_config.csv contains duplicate city_ascii values: {duplicates}")

    return df.sort_values("city_ascii").reset_index(drop=True)


def available_city_codes() -> list[str]:
    return load_city_table()["city_ascii"].tolist()


def suggest_city_codes(query: str, *, limit: int = 5) -> list[str]:
    codes = available_city_codes()
    code_lookup = {_normalize_query(code): code for code in codes}
    normalized = _normalize_query(query)
    matches = get_close_matches(normalized, list(code_lookup), n=limit, cutoff=0.5)
    return [code_lookup[match] for match in matches]


def _unknown_city_error(query: str) -> ValueError:
    suggestions = suggest_city_codes(query)
    detail = ""
    if suggestions:
        detail = f" Did you mean: {', '.join(suggestions)}?"
    return ValueError(
        f"City '{query}' not found in cities_config.csv. "
        "Use the ASCII city code from the city_ascii column."
        f"{detail}"
    )


@lru_cache(maxsize=None)
def resolve_city(query: str) -> CityRecord:
    df = load_city_table()
    folded = _casefold_code(query)
    for row in df.itertuples(index=False):
        if _casefold_code(row.city_ascii) == folded:
            return CityRecord(
                city_id=int(row.city_id),
                city_ascii=str(row.city_ascii),
                city=str(row.city),
                country=str(row.country),
                country_iso2=str(row.country_iso2),
                latitude=float(row.latitude),
                longitude=float(row.longitude),
                cordex_domain=str(row.cordex_domain),
            )
    raise _unknown_city_error(query)


def resolve_city_codes(requested: list[str] | None) -> list[str]:
    if not requested:
        return available_city_codes()

    resolved: list[str] = []
    seen: set[int] = set()
    for query in requested:
        record = resolve_city(query)
        if record.city_id in seen:
            continue
        resolved.append(record.city_ascii)
        seen.add(record.city_id)
    return resolved
