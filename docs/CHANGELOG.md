# Changelog

## [Unreleased]

### Added
- Added the `samy-download` data-retrieval CLI, automatic missing-input downloads in `samy-run`, and batched multi-city downloads with per-city ISD station overrides.
- Expanded the bundled city catalog for global coverage and added multi-source NSRDB downloads.
- Added hourly ERA5-Land point downloads and NSRDB-based calibration of GHI, DNI, and DHI to extend solar-radiation coverage.
- Added CORDEX-CMIP5 downloads and live catalogue inspection for climate-delta workflows.
- Added `samy-download extract-cmip6` to create city time series from local CMIP6 NetCDF files, including ZIP-wrapped files.
- Added MRI-ESM2-0 support, explicit `samy-run --gcm` selection, and `--rsvar-input` reuse of an existing single-city RS-VAR result.
- Added live NSRDB source and coverage inspection to improve source selection and grouped downloads.

### Changed
- Simplified ensemble generation to regime-path sampling with zero VAR innovations.
- Standardized user-facing city codes as ASCII identifiers and derive local solar time offsets from city coordinates.
- Switched NOAA ISD downloads from FTP to HTTPS.
- Updated CMIP6 handling to preserve daily and monthly inputs together and determine temporal resolution per variable.
- Improved automatic ISD station and NSRDB coverage selection using available station and query coverage.

### Fixed
- Validate cached CMIP6 coverage against the requested years before skipping a download.
- Harden RS-VAR fitting and regime-label handling for sparse assignments and non-finite posterior output.
- Use dewpoint as a fallback when yearly EPW humidity inputs are unavailable, and generalize DDY generation for baseline and future periods.

## [0.1.2] - 2026-04-10

### Added
- `pipeline/run_pipeline.py` — one-click `samy-run` CLI that chains ISD parse → LST remap → NSRDB merge → RS-VAR → CMIP6 delta → EPW in a single command; supports `--mode deterministic` (default) and `--mode ensemble`
- `samy-run` registered as a CLI entry point in `pyproject.toml`
- Key `samy-run` parameters: `--baseline-start/end`, `--start/end`, `--ssp`, `--epw-years`, `--members`, `--regime-mode`, `--residual-mode`, `--overwrite`; all have defaults for zero-config quick runs

### Changed
- `test_miami_pipeline_poc.py`: stage 1 (`_stage1_parse_isd`) no longer writes a CSV to disk; ISD parsing now uses `tempfile.TemporaryDirectory` and returns a `pd.DataFrame` directly
- `test_miami_pipeline_poc.py`: `_stage2_remap_to_lst` signature changed from `(Path, ...)` to `(pd.DataFrame, ...)`; test function checks `1_isd_lst.csv` cache before calling stage 0, avoiding unnecessary ISD re-parsing
- Intermediate cache files in `tests/output/miami_poc_cmip6_ssp245/` renamed from `stageN_*` to `N_*`, numbered by output order (stage 1 raw UTC CSV removed, so numbering starts at 1):
  - `stage2_isd_lst.csv` → `1_isd_lst.csv`
  - `stage3_isd_solar_lst.csv` → `2_isd_solar_lst.csv`
  - `stage4_rsvar_miami.xlsx` → `3_rsvar_miami.xlsx`
  - `stage5_cmip6_*.csv` → `4_cmip6_*.csv`
  - `stage6_cordex_baseline_doy.csv` → `5_cordex_baseline_doy.csv`
  - `stage7_*.csv` → `6_*.csv`

## [0.1.1] - 2026-04-10

### Added
- Added strict deterministic compatibility path in `AnomalyGenerator` so `regime_mode='soft'` + `residual_mode='deterministic'` can reproduce legacy RS-VAR output
- Added regression coverage in `tests/test_ensemble.py` for contiguous residual block extraction and soft-weighted residual mixture sampling
- Added slow validation harness `tests/test_validation_temp_distributions.py` for 2011Q1 temperature distribution comparisons across legacy/ensemble pathways

### Changed
- Updated ensemble CLI/pipeline defaults in `pipeline/run_ensemble.py` to `regime_mode='soft'` and `residual_mode='deterministic'` for safer baseline behavior

### Fixed
- Fixed residual storage in `ForecastTo2100V5` to preserve residual timestamps (`result.resid.copy()`), enabling contiguous-window block sampling
- Fixed block residual sampling to use truly contiguous hourly windows instead of naive reshape on regime-filtered rows
- Fixed soft-regime residual generation to support probability-weighted innovation draws across regimes (`iid`, `block`, `parametric`)

## [0.1.0] - 2026-04-07

### Added
- Initial Python package structure with `src` layout (`samy_wgen`)
- `model/`: RS-VAR + climate-delta forecasting (`ForecastTo2100V5`, `RSVAROnlyPipeline`)
- `pipeline/`: Batch pipeline scripts (`samy-rsvar`, `samy-delta` CLI entry points) and EPW writer (`gen_samy.py`)
- `utils/`: Centralized humidity calculations (`calculate_weather_var`), solar utilities, ISD parser, EPW/DDY converter
- `utils/time_utils`: One-time UTC→LST conversion at ISD ingestion boundary; `assert_is_lst` consistency check
- `config/`: Bundled `cities_config.csv` as package data
- Miami POC dataset support (`data/ISD/Miami`, `data/CMIP6/Miami`, `data/ERA5`, `data/NSRDB`)
- `pyproject.toml` with `hatchling` build backend
- Test suite skeleton: `conftest.py`, unit tests, end-to-end POC test (`@slow`)

### Fixed
- Consolidated Magnus dewpoint formula — removed 3 duplicate inline implementations; all code now imports from `calculate_weather_var`
- Fixed `HGScripts_Dev` import paths → `samy_wgen.model`
- Fixed bare relative import in `create_epw_ddy.py`
- Fixed module-level side effects in `parse_ISD.py` (logging) and `run_climate_delta.py` (config loading)
