# Changelog

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
