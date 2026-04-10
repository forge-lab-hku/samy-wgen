# Changelog

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
- `config/`: Bundled `config.yaml` and `cities_config.csv` as package data
- Miami POC dataset support (`data/ISD/Miami`, `data/CMIP6/Miami`, `data/ERA5`, `data/NSRDB`)
- `pyproject.toml` with `hatchling` build backend
- Test suite skeleton: `conftest.py`, unit tests, end-to-end POC test (`@slow`)

### Fixed
- Consolidated Magnus dewpoint formula — removed 3 duplicate inline implementations; all code now imports from `calculate_weather_var`
- Fixed `HGScripts_Dev` import paths → `samy_wgen.model`
- Fixed bare relative import in `create_epw_ddy.py`
- Fixed module-level side effects in `parse_ISD.py` (logging) and `run_climate_delta.py` (config loading)
