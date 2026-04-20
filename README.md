# samy-wgen

Stochastic weather generation for long-term climate projections (2025–2100), producing hourly EPW files via a Regime-Switching VAR (RS-VAR) model with CMIP6 climate deltas.

---

## Installation

> **Dev status** — the package has not yet been published to PyPI and is under active development.
> A stable release is planned for the future.

**With conda (recommended)**

```bash
git clone https://github.com/<org>/samy-wgen.git
cd samy-wgen

conda create -n samy-wgen python=3.11
conda activate samy-wgen

pip install -e .          # core dependencies
pip install -e ".[dev]"   # + dev/test dependencies
```

**With pip only**

```bash
git clone https://github.com/<org>/samy-wgen.git
cd samy-wgen
pip install -e .
```

Python ≥ 3.10 required. All dependencies are declared in `pyproject.toml` and installed automatically.

---

## Credentials and API setup

Configure external data access once after installation.

**NOAA ISD** — no registration required (public FTP).

**NREL NSRDB** (solar irradiance)

1. Register for a free API key at [developer.nlr.gov/signup](https://developer.nlr.gov/signup/)
2. Save credentials once via the interactive prompt:
   ```bash
   samy-download config
   #   NSRDB API key : your_key_here
   #   NSRDB email   : your@email.com
   ```

**Copernicus CDS** (CMIP6 climate projections; optional CORDEX-CMIP5 catalogue queries)

1. Register at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu)
2. Accept the dataset licence for *CMIP6 climate projections* and, if you plan to inspect CORDEX availability through the catalogue tools, *CORDEX regional climate projections* in your CDS profile
3. Create `~/.cdsapirc`:
   ```ini
   # ~/.cdsapirc
   url: https://cds.climate.copernicus.eu/api
   key: <UID>:<API-key>          # shown on your CDS profile page
   ```

---

## Quickstart — one-click pipeline

`samy-run` chains all stages automatically, downloads missing inputs by default, and caches intermediate outputs so re-runs are fast.
All user-facing city values use the canonical ASCII city code from `src/samy_wgen/config/cities_config.csv`.

```bash
# Deterministic forecast — Miami, SSP2-4.5, auto-download missing ISD/NSRDB/CMIP6 inputs
samy-run --city Miami

# Batch one-click run for multiple cities
samy-run --city Miami Los_Angeles

# Custom forecast window and scenario
samy-run --city Miami --start 2025-01-01 --end 2060-12-31 --ssp ssp585

# Keep one-click mode but customize automatic download inputs
samy-run --city Miami --cmip6-models mpi_esm1_2_lr cnrm_esm2_1 --download-years 1991 2024

# Stochastic ensemble — 50 members
samy-run --city Miami --mode ensemble --members 50

# Disable automatic download and use only pre-existing local files
samy-run --city Miami --skip-download

# Force all stages to recompute from scratch
samy-run --city Miami --overwrite
```

Outputs land in `output/<city_ascii>/`:

```
output/Miami/
├── 1_isd_lst.csv              LST-converted ISD  (cached)
├── 2_isd_solar_lst.csv        + NSRDB solar merge (cached)
├── 3_rsvar.xlsx               RS-VAR output       (cached, deterministic)
├── 4_delta_corrected.csv      climate-delta series (cached, deterministic)
├── EPW/
│   ├── Miami_ssp245_2030.epw
│   ├── Miami_ssp245_2050.epw
│   └── Miami_ssp245_2080.epw
└── ensemble/                  (ensemble mode only)
    ├── member_000.parquet
    ├── ...
    └── ensemble_summary.xlsx
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--city` | *(required)* | ASCII city code matching `cities_config.csv` |
| `--mode` | `deterministic` | `deterministic` or `ensemble` |
| `--baseline-start/end` | `1991` / `2010` | Training baseline period |
| `--start` / `--end` | `2025-01-01` / `2100-12-31` | Forecast window |
| `--ssp` | `ssp245` | CMIP6 scenario |
| `--epw-years` | `2030,2050,2080` | Years to export as EPW (deterministic) |
| `--members` | `50` | Ensemble member count |
| `--regime-mode` | `ensemble` | Regime sampling: `soft`/`hard`/`ensemble` |
| `--residual-mode` | `block` | Residual sampling: `deterministic`/`iid`/`block`/`parametric` |
| `--overwrite` | `false` | Ignore all cached intermediate files |
| `--data-dir` | `data` | Root directory for ISD / NSRDB / CMIP6 |
| `--out-dir` | `output/<city_ascii>` | Output directory |
| `--download-data` | `isd nsrdb cmip6` | Missing datasets to auto-download before running |
| `--isd-radius` | `50` | Radius used for automatic ISD station selection |
| `--cmip6-models` | `mpi_esm1_2_lr` | CMIP6 model(s) used for automatic download |
| `--skip-download` | `false` | Require inputs to already exist locally |

---

## Downloading input data

Before running the pipeline, three data sources must be present for each city.
`samy-download` handles all three from the command line.

### 1 · Inspect candidate ISD stations

ISD files are keyed by station ID. You can inspect nearby stations explicitly:

```bash
samy-download list-stations --city Miami
```

If `--isd-station` is omitted, `samy-download download` and `samy-run` now auto-select the station with the best time coverage inside a 50 km radius.

### 2 · Inspect live CMIP6 availability

CMIP6 downloads are backed by the live CDS catalogue. You can inspect which
model / experiment / temporal-resolution / variable combinations are currently
available before downloading:

```bash
# List currently available CMIP6 models
samy-download list-cmip6

# Expand to experiment / resolution / variable rows
samy-download list-cmip6 --detail

# Filter the live catalogue
samy-download list-cmip6 \
  --experiment ssp2_4_5 \
  --resolution daily \
  --var near_surface_air_temperature
```

`samy-download download` uses the same CDS constraint checks before requesting
CMIP6 data, so unavailable model / experiment combinations are skipped early.

### 3 · Download all data

```bash
# ISD + NSRDB + CMIP6 (default set)
samy-download download \
  --city Miami

# Limit the ISD/NSRDB year range (speeds up initial testing)
samy-download download \
  --city Miami \
  --years 1991 2024

# Batch multiple cities, with per-city ISD station overrides
samy-download download \
  --city Montreal Toronto \
  --isd-station Montreal=716270-94792 Toronto=716240-99999

# Download additional CMIP6 models or scenarios
samy-download download \
  --city Miami \
  --data cmip6 \
  --models mpi_esm1_2_lr cnrm_esm2_1 \
  --experiments historical ssp2_4_5 ssp5_8_5
```

Files are written directly to the paths the pipeline expects:

```
data/
├── ISD/Miami/722020-12839_MIAMI_INTERNATIONAL_AIRPORT/
│   ├── 722020-12839-1991.gz
│   └── ...
├── NSRDB/nsrdb_miami_1991-2024.csv
├── CMIP6/Miami/
│   ├── mpi_esm1_2_lr_historical.csv
│   ├── mpi_esm1_2_lr_ssp245.csv
│   └── mpi_esm1_2_lr_ssp245_monthly.csv
```

CMIP6 data are downloaded as city-specific grid-point extractions using the CDS
`area` parameter, so no large global files are stored locally.

### CORDEX-CMIP5 note

This repository does **not** currently provide a built-in CORDEX download path
through `samy-download download`.

You can still use externally prepared CORDEX-CMIP5 data as climate-delta input.
The package only requires that your files follow the expected folder structure
and CSV schema.

Required layout:

```text
<your-cordex-root>/
└── <CitySlug>/
    ├── historical.csv
    └── rcp85.csv
```

Required CSV conventions:

- Each file must include a `time` column.
- Historical and future files must use the same variable columns.
- Variable names must match the climate-delta reader expectations, such as
  `tas`, `tasmax`, `tasmin`, `ps` or `psl`, `sfcWind`, `rsds`, `hurs`, `huss`.

If the folder structure and CSV columns match, the data can be loaded by
`samy-delta` even when the files were downloaded or prepared outside this
repository.

Recommended sources:

- CORDEX official access guide:
  [How to access CORDEX-CMIP5 data](https://cordex.org/data-access/cordex-cmip5-data/how-to-access-cordex-cmip5-data/)
- Copernicus CDS CORDEX dataset page:
  [projections-cordex-domains-single-levels](https://cds.climate.copernicus.eu/datasets/projections-cordex-domains-single-levels?tab=download)

`samy-download list-cordex` remains available to inspect the live CDS CORDEX
catalogue, but actual CORDEX files must currently be downloaded outside the
package and then placed under a compatible directory layout for downstream
climate-delta workflows.

### `samy-download` reference

| Subcommand | Purpose |
|------------|---------|
| `config` | Save NSRDB credentials to `~/.config/samy_wgen/credentials.yaml` |
| `list-cmip6 [--detail]` | List available CMIP6 model / experiment / resolution / variable combinations from the live CDS catalogue |
| `list-cmip6 --experiment <exp> --resolution <res> --var <var>` | Filter live CMIP6 catalogue rows before download |
| `list-cordex [--detail]` | Inspect live CORDEX catalogue combinations from CDS |
| `list-stations --city <city_ascii> [--radius 150]` | Print nearby ISD stations sorted by distance |
| `download --city <city_ascii> [--isd-station <USAF-WBAN>]` | Download ISD + NSRDB + CMIP6 for one city; otherwise auto-select best-coverage ISD station within 50 km |
| `download --city <city_ascii_1> <city_ascii_2> ...` | Batch-download multiple cities; CMIP6 requests are grouped into shared bounding-box fetches |
| `download --city <city_ascii_1> <city_ascii_2> --isd-station <city_ascii_1=id> <city_ascii_2=id>` | Override ISD station IDs per city in batch mode |
| `download --models mpi_esm1_2_lr cnrm_esm2_1` | Specify CMIP6 models |
| `download --experiments historical ssp2_4_5 ssp5_8_5` | Specify CMIP6 CDS experiment names |

---

## Python API

For finer control, call individual pipeline stages directly.

### Parse ISD

```python
from samy_wgen.utils.parse_ISD import process_station_directory

csv_path = process_station_directory(
    station_dir="data/ISD/Miami/722020-12839_MIAMI_INTERNATIONAL_AIRPORT",
    output_dir="output/",
)
```

### Remap fields, UTC → LST, resample to hourly

```python
import pandas as pd
from samy_wgen.utils.parse_ISD import remap_isd_columns, resample_isd_hourly
from samy_wgen.utils.time_utils import utc_to_lst

df = pd.read_csv(csv_path, parse_dates=["datestamp"]).set_index("datestamp").sort_index()
df = remap_isd_columns(df)       # rename ISD fields → temp, dewpoint, pressure, wind_speed, wind_dir
df = utc_to_lst(df, "Miami")     # UTC → LST (fixed offset, no DST)
df = resample_isd_hourly(df)     # hourly mean; wind_dir uses mode to avoid circular-mean error
```

### Merge with NSRDB solar data

```python
nsrdb = pd.read_csv("data/NSRDB/nsrdb_miami.csv")
nsrdb["DATE"] = pd.to_datetime(
    nsrdb[["Year","Month","Day","Hour"]].rename(
        columns={"Year":"year","Month":"month","Day":"day","Hour":"hour"}
    )
)
solar = nsrdb.set_index("DATE")[["GHI","DNI","DHI"]].clip(lower=0).resample("h").mean()

merged = df.join(solar, how="inner").ffill(limit=3).bfill(limit=3)
merged.index.name = "DATE"
merged.to_csv("output/miami_isd_solar.csv")
```

### Train RS-VAR and forecast

```bash
samy-rsvar --isd output/miami_isd_solar.csv --city Miami --output output/rsvar_miami.xlsx
```

```python
from samy_wgen.pipeline.run_rsvar import run_and_save_rsvar

run_and_save_rsvar(
    isd_path="output/miami_isd_solar.csv",
    lat=25.7617,
    output_path="output/rsvar_miami.xlsx",
    city="Miami",
)
```

### Apply CMIP6 climate delta

```bash
samy-delta --rsvar-dir output/ --city Miami --out-dir output/delta/
```

```python
from samy_wgen.pipeline.run_climate_delta import (
    load_cmip6_or_cordex6, compute_cordex_baseline,
    load_rsvar_output, apply_climate_delta,
)

daily_df, _, _ = load_cmip6_or_cordex6("data/CMIP6", city="Miami", ssp="ssp245")
baseline = compute_cordex_baseline(daily_df, baseline_start=1991, baseline_end=2010)
rsvar_df, _ = load_rsvar_output("output/rsvar_miami.xlsx")

output_df, _ = apply_climate_delta(
    rsvar_df=rsvar_df, cordex=daily_df, baseline=baseline,
    gcm="MPI-ESM1-2-LR", rcm=None, lat=25.7617,
)
output_df.to_csv("output/miami_mpi_ssp245.csv")
```

### Generate EPW files

```python
from samy_wgen.pipeline.gen_samy import _write_epw_for_year

for year in [2030, 2050, 2080]:
    _write_epw_for_year(
        df=output_df, year=year,
        out_path=f"output/EPW/Miami_MPI_SSP245_{year}.epw",
        location="Miami-FL-USA", lat=25.7617, lon=-80.1918,
        tz=-5.0, elevation_m=2.0,
    )
```

---

## CLI reference

| Command | Description |
|---------|-------------|
| `samy-download config` | Save NSRDB credentials to `~/.config/samy_wgen/credentials.yaml` |
| `samy-download list-cmip6 [--detail]` | Inspect live CMIP6 CDS catalogue entries |
| `samy-download list-cordex [--detail]` | Inspect live CORDEX CDS catalogue entries |
| `samy-download list-stations --city <city_ascii>` | List nearby ISD stations |
| `samy-download download --city <city_ascii>` | Download ISD / NSRDB / CMIP6 (auto-select ISD station by coverage within 50 km) |
| `samy-run --city <city_ascii>` | Full pipeline with auto-download of missing ISD / NSRDB / CMIP6 inputs |
| `samy-run --city <city_ascii> --mode ensemble` | Full pipeline in ensemble mode |
| `samy-rsvar --isd <csv> --city <city_ascii> --output <xlsx>` | Train RS-VAR + forecast only |
| `samy-delta --rsvar-dir <dir> --city <city_ascii> --projection-type cmip6 --projection-root <dir> --ssp <scenario>` | Apply CMIP6 delta to RS-VAR output |
| `samy-delta --rsvar-dir <dir> --city <city_ascii> --projection-type cordex-cmip5 --projection-root <dir>` | Apply CORDEX-CMIP5 delta from compatible city-folder CSV inputs |
| `samy-ensemble --isd <csv> --city <city_ascii> --members 50 ...` | Generate stochastic ensemble members |
