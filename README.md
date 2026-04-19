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

## Downloading input data

Before running the pipeline, three data sources must be present for each city.
`samy-download` handles all three from the command line.

### 1 · Register API credentials

**NOAA ISD** — no registration required (public FTP).

**NREL NSRDB** (solar irradiance)

1. Register for a free API key at [developer.nlr.gov/signup](https://developer.nlr.gov/signup/)
2. Save credentials once via the interactive prompt:
   ```bash
   samy-download config
   #   NSRDB API key : your_key_here
   #   NSRDB email   : your@email.com
   ```

**Copernicus CDS** (CMIP6 climate projections; CORDEX-CMIP5 catalogue queries)

1. Register at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu)
2. Accept the dataset licence for *CMIP6 climate projections* and, if you plan to inspect CORDEX availability, *CORDEX regional climate projections* in your CDS profile
3. Create `~/.cdsapirc`:
   ```ini
   # ~/.cdsapirc
   url: https://cds.climate.copernicus.eu/api
   key: <UID>:<API-key>          # shown on your CDS profile page
   ```

### 2 · Find and select an ISD station

ISD files are keyed by station ID. Run `list-stations` first to pick the best station for your city:

```bash
samy-download list-stations --city Miami
```

Pass the USAF-WBAN value to --isd-station when running 'samy-download download'.

### 3 · Inspect live CMIP6 availability

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

### 4 · Download all data for a city

```bash
# ISD + NSRDB + CMIP6 (default set)
samy-download download \
  --city Miami \
  --isd-station 722020-12839

# Limit the ISD/NSRDB year range (speeds up initial testing)
samy-download download \
  --city Miami \
  --isd-station 722020-12839 \
  --years 1991 2024

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
Recommended sources:

- CORDEX official access guide:
  [How to access CORDEX-CMIP5 data](https://cordex.org/data-access/cordex-cmip5-data/how-to-access-cordex-cmip5-data/)
- Copernicus CDS CORDEX dataset page:
  [projections-cordex-domains-single-levels](https://cds.climate.copernicus.eu/datasets/projections-cordex-domains-single-levels?tab=download)

`samy-download list-cordex` remains available to inspect the live CDS CORDEX
catalogue, but actual CORDEX files must currently be downloaded outside the
package and then placed under your chosen data directory for downstream climate
delta workflows.

### `samy-download` reference

| Subcommand | Purpose |
|------------|---------|
| `config` | Save NSRDB credentials to `~/.config/samy_wgen/credentials.yaml` |
| `list-cmip6 [--detail]` | List available CMIP6 model / experiment / resolution / variable combinations from the live CDS catalogue |
| `list-cmip6 --experiment <exp> --resolution <res> --var <var>` | Filter live CMIP6 catalogue rows before download |
| `list-cordex [--detail]` | Inspect live CORDEX catalogue combinations from CDS |
| `list-stations --city <name> [--radius 150]` | Print nearby ISD stations sorted by distance |
| `download --city <name> --isd-station <USAF-WBAN>` | Download ISD + NSRDB + CMIP6 |
| `download --models mpi_esm1_2_lr cnrm_esm2_1` | Specify CMIP6 models |
| `download --experiments historical ssp2_4_5 ssp5_8_5` | Specify CMIP6 CDS experiment names |

---

## Quickstart — one-click pipeline

`samy-run` chains all stages automatically and caches intermediate outputs so re-runs are fast.

```bash
# Deterministic forecast — Miami, SSP2-4.5, output EPW for 2030/2050/2080
samy-run --city Miami

# Custom forecast window and scenario
samy-run --city Miami --start 2025-01-01 --end 2060-12-31 --ssp ssp585

# Stochastic ensemble — 50 members
samy-run --city Miami --mode ensemble --members 50

# Force all stages to recompute from scratch
samy-run --city Miami --overwrite
```

Outputs land in `output/<City>/`:

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
| `--city` | *(required)* | City name matching `cities_config.csv` |
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
| `--out-dir` | `output/<city>` | Output directory |

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
| `samy-download list-stations --city <name>` | List nearby ISD stations |
| `samy-download download --city <name> --isd-station <id>` | Download ISD / NSRDB / CMIP6 |
| `samy-run --city <name>` | Full pipeline — ISD parse → RS-VAR → delta → EPW |
| `samy-run --city <name> --mode ensemble` | Full pipeline in ensemble mode |
| `samy-rsvar --isd <csv> --city <name> --output <xlsx>` | Train RS-VAR + forecast only |
| `samy-delta --rsvar-dir <dir> --city <name> --out-dir <dir>` | Apply CMIP6 delta to RS-VAR output |
| `samy-ensemble --isd <csv> --city <name> --members 50 ...` | Generate stochastic ensemble members |
| `samy-delta --list-scenarios` | List available climate scenarios |

---

## Running the POC test (Miami MPI-SSP245)

```bash
# First run — executes all stages and caches intermediate files
pytest -m slow tests/test_miami_pipeline_poc.py -v -s

# Force rerun from scratch
pytest -m slow tests/test_miami_pipeline_poc.py -v -s --overwrite
```

Intermediate outputs are saved to `tests/output/miami_poc_cmip6_ssp245/` for inspection:

```
1_isd_lst.csv               LST-converted ISD
2_isd_solar_lst.csv         + NSRDB solar merge
3_rsvar_miami.xlsx          RS-VAR output
4_cmip6_mpi_ssp245_daily.csv  CMIP6 data
5_cordex_baseline_doy.csv   DOY baseline climatology
6_miami_mpi_ssp245_hourly.csv  delta-corrected forecast
6_delta_stats.csv           per-variable delta statistics
EPW/                        sample EPW files
```

---

## Supported CMIP6 models

Auto-detected from filenames in `data/CMIP6/<City>/`:

| Model | Files |
|-------|-------|
| MPI-ESM1-2-LR | `mpi_esm1_2_lr_historical.csv` + `mpi_esm1_2_lr_<ssp>.csv` |
| CNRM-ESM2-1 | `cnrm_esm2_1_historical.csv` + `cnrm_esm2_1_<ssp>.csv` |

Supported SSPs: `ssp245`, `ssp370`, `ssp585`.
