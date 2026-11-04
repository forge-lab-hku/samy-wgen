# samy-wgen

Building energy simulation needs hourly weather at a specific site. Climate projections are usually supplied at coarser spatial and temporal scales, while a typical meteorological year combines selected historical months into one representative year. This leaves a gap for studies of long, continuous future weather sequences.

The scenario-conditioned Actual Meteorological Year (sAMY) workflow addresses this gap by separating hourly weather structure learned from observations from the imposed climate signal. This package prepares station weather and solar observations, fits a regime-switching vector autoregressive (RS-VAR) model, applies a selected climate projection, and exports hourly weather for building simulation.

**Development status:** This package is under active development. The core functionality has been implemented, but comprehensive testing and validation are still ongoing. APIs and outputs may change before the first stable release.

**Documentation:** [samy-wgen documentation](https://forge-lab-hku.github.io/samy-wgen/)

**Citation:** Guo, H. & He, K. (2026). Scenario-conditioned actual meteorological years (sAMY): A stochastic weather generator using multi-decadal observations. *Energy and Buildings*, **364**, 117508. [doi:10.1016/j.enbuild.2026.117508](https://doi.org/10.1016/j.enbuild.2026.117508)

---

## Installation

**With conda (recommended)**

```bash
git clone https://github.com/forge-lab-hku/samy-wgen.git
cd samy-wgen

conda create -n samy-wgen python=3.11
conda activate samy-wgen

python -m pip install -e .          # core dependencies
# For development and tests, use python -m pip install -e ".[dev]" instead.
```

**With pip only**

```bash
git clone https://github.com/forge-lab-hku/samy-wgen.git
cd samy-wgen
python -m pip install -e .
```

Python ≥ 3.10 required. All dependencies are declared in `pyproject.toml` and installed automatically.

---

## Credentials and API setup

Configure external data access once after installation.

**NOAA ISD** — no registration required (public HTTPS archive).

**NREL NSRDB** (solar irradiance)

1. Register for a free API key at [developer.nlr.gov/signup](https://developer.nlr.gov/signup/)
2. Save credentials once via the interactive prompt:
   ```bash
   samy-download config
   #   NSRDB API key : your_key_here
   #   NSRDB email   : your@email.com
   ```

**Copernicus CDS** (ERA5-Land, CMIP6, and CORDEX-CMIP5)

1. Register at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu)
2. Accept the licence for each CDS dataset you plan to download: *ERA5-Land hourly time-series*, *CMIP6 climate projections*, and/or *CORDEX regional climate projections*
3. Create `~/.cdsapirc`:
   ```ini
   # ~/.cdsapirc
   url: https://cds.climate.copernicus.eu/api
   key: <your-CDS-API-key>
   ```

---

## Quickstart — one-click pipeline

After configuring the NSRDB and CDS credentials above, run this from the repository root:

```bash
samy-run --city Miami
```

`samy-run` downloads missing ISD, NSRDB, and CMIP6 inputs, trains RS-VAR, applies the CMIP6 signal, and writes EPW and DDY files. The first run downloads decades of weather and projection data and can take substantial time. Intermediate deterministic outputs are cached for later runs. This example requires network access and accepted CDS dataset licences; it is not an offline demo. All user-facing city values use the canonical ASCII city code from `src/samy_wgen/config/cities_config.csv`.

Other examples:

```bash
# Batch one-click run for multiple cities
samy-run --city Miami Los_Angeles

# Custom forecast window and scenario
samy-run --city Miami --start 2025-01-01 --end 2060-12-31 --ssp ssp585

# Customize automatic download inputs
samy-run --city Miami --cmip6-models mpi_esm1_2_lr cnrm_esm2_1 --download-years 1998 2024

# Generate stochastic regime-path members
samy-run --city Miami --mode ensemble --members 50

# Use only pre-existing local files
samy-run --city Miami --skip-download

# Recompute deterministic stages and outputs
samy-run --city Miami --overwrite
```

For individual commands, `samy-download` retrieves or inspects input data, `samy-rsvar` fits the weather model, and `samy-delta` applies a climate projection signal. Run a command with `--help` to see its options; detailed command references are in the [documentation](https://forge-lab-hku.github.io/samy-wgen/cli/index.html).

---

## Workflow

The one-click command follows these stages. Each stage can also be inspected in the [workflow guide](https://forge-lab-hku.github.io/samy-wgen/workflows.html).

1. **Download data:** obtain historical station weather and solar observations, plus climate projection data for the selected city and scenario.
2. **Prepare observations:** parse station records, convert timestamps to local standard time, and align the weather and solar variables into an hourly input series.
3. **Fit RS-VAR and generate weather:** learn multivariate hourly behavior from the observations, then generate a continuous hourly trajectory or regime-path members.
4. **Apply the climate signal:** calculate changes from the selected projection relative to a historical baseline and apply them to the generated weather series.
5. **Write weather files:** save the resulting time series and export selected years as EPW files, with DDY files for building energy simulation workflows.

For direct Python use, see the [API reference](https://forge-lab-hku.github.io/samy-wgen/api/index.html); for command-line options, see the [CLI reference](https://forge-lab-hku.github.io/samy-wgen/cli/index.html).
