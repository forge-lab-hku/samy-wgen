# samy-wgen

Stochastic weather generation for long-term climate projections (2025–2100), producing hourly EPW files via a Regime-Switching VAR (RS-VAR) model with CMIP6 climate deltas.

---

## Installation

```bash
# From local source (editable — code changes apply immediately)
pip install -e /path/to/samy-wgen

# With dev/test dependencies
pip install -e "/path/to/samy-wgen[dev]"
```

Python ≥ 3.10 required. All dependencies are declared in `pyproject.toml` and installed automatically.

---

## Data layout expected

```
data/
├── ISD/<City>/<station_id>/        # NOAA ISD .gz files (one per year)
├── NSRDB/nsrdb_<city>_*.csv        # NREL NSRDB solar irradiance (LST)
└── CMIP6/<City>/
    ├── mpi_esm1_2_lr_historical.csv
    └── mpi_esm1_2_lr_ssp245.csv    # or ssp370 / ssp585
```

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
