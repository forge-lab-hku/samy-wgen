# Future Climate Simulation Pipeline on EnergyPlus

## Overview
This repository contains the RS-VAR V5 weather generation pipeline and building energy simulation workflow for climate impact assessment across 4 North American cities (Los Angeles, Miami, Montreal, Toronto). The workflow includes stochastic weather generation (sAMY), validation, climate projection, and EnergyPlus building simulations.

---

## Part 1: sAMY Weather Generation & Validation

**Results:** See `docs/multi_city_report.md` for comprehensive validation and projection analysis

### 1. Generate sAMY Weather Files
Generate stochastic Actual Meteorological Year (sAMY) files using RS-VAR model with climate deltas.

**Scripts:**
- `run_rsvar.py` - Fit RS-VAR model on baseline period (1991-2010) and save model parameters
- `run_climate_delta.py` - Apply CORDEX climate deltas to generate future weather files (2025-2100)

**Usage:**
```bash
# Step 1: Fit RS-VAR model and save parameters
python run_rsvar.py 

# Step 2: Apply climate deltas to generate future weather
python run_climate_delta.py 
```

**Key Arguments:**
- `--city`: City names to process (space-separated, default: all cities in config)
- `--config`: Path to cities_config.csv (default: data/cities_config.csv)
- `--isd`: ISD merged CSV path for single-file mode (batch mode auto-uses data/ISD_hguo/merged)
- `--output`: Output Excel file or directory
- `--rsvar-dir`: RS-VAR model output directory (for run_climate_delta.py)
- `--out-dir`: Output directory for climate-delta applied forecasts

**Outputs:** 
- `/output/RSVAR/` - RS-VAR model parameters (rsvar_output_<City>.xlsx)
- `/output/RSVAR_cd/<scenario>/<City>/` - Generated sAMY weather files with climate deltas

### 2. Validation Against ISD Observations
Validate sAMY performance against ISD observations for 2011-2020 period.

**Script:** `val_vs_isd_v2.py`

**Function:** Compares V5 forecast vs. ISD observations and Annex80 baseline, generates validation metrics (bias, RMSE, copula metrics) and summary plots

**Usage:**
```bash
python val_vs_isd_v2.py 
```

**Key Arguments:**
- `--cities`: Comma-separated city names (default: all in config)
- `--cordex-variant-root`: CORDEX variant directory with per-city folders containing historical.csv + rcp85.csv
- `--isd-root`: ISD merged data directory (default: data/ISD_hguo/merged)
- `--isd-template`: Filename template (default: {city_slug}_with_solar.csv)
- `--output-dir`: Output directory for validation results
- `--cities-config`: Path to cities_config.csv
- `--skip-missing`: Skip cities with missing inputs instead of failing

**Outputs:** `/output/val_vs_isd/` - per-city validation results and `_summary/` folder with multi-city metrics, figures, and CSV summaries

### 3. Evaluate sAMY Climate Signal Preservation
Assess sAMY statistical properties and climate signal preservation.

**Script:** `evalute_samy.py`

**Function:** Evaluates generated sAMY files for climate signal fidelity, warming trajectories, and statistical consistency

**Usage:**
```bash
python evalute_samy.py 
```

**Key Arguments:**
- `--cities`: Comma-separated city names (default: auto-discover from forecast-root)
- `--scenarios`: Comma-separated scenario names (default: auto-discover)
- `--forecast-root`: Root directory containing forecast outputs (default: output/RSVAR_cd)
- `--annex80-root`: Annex80 baseline data directory (default: data/annex80)
- `--cordex-variant-root`: CORDEX variant root with per-city folders
- `--isd`: ISD baseline directory (default: data/ISD_hguo/merged)
- `--windows`: Time periods to evaluate (default: 2011-2020,2026-2040,2041-2060,2061-2080,2081-2100)
- `--output-root`: Output root directory (default: output/RSVAR_cd_eval)
- `--debug`: Enable verbose debug output

**Outputs:** `/output/RSVAR_cd_eval/<scenario>/<City>/` - per-city results and `_summary/` folder with warming signal analysis and projection figures

---

## Part 2: EnergyPlus Simulation (WIP)

### Pipeline Workflow

### 1. Building Model Preparation

#### Convert .idf to .epjson
- Update .idf to latest version using EnergyPlus transition tools
```bash
cd /Applications/EnergyPlus-25-1-0/PreProcess/IDFVersionUpdater
./Transition-V22-1-0-to-V22-2-0 /Users/kanxuanhe/Document/20250929_swg/swg/data/epjson/Miami/ASHRAE901_OfficeMedium_STD2019_Miami.idf
./Transition-V22-2-0-to-V23-1-0 /Users/kanxuanhe/Document/20250929_swg/swg/data/epjson/Miami/ASHRAE901_OfficeMedium_STD2019_Miami.idf
./Transition-V23-1-0-to-V23-2-0 /Users/kanxuanhe/Document/20250929_swg/swg/data/epjson/Miami/ASHRAE901_OfficeMedium_STD2019_Miami.idf
./Transition-V23-2-0-to-V24-1-0 /Users/kanxuanhe/Document/20250929_swg/swg/data/epjson/Miami/ASHRAE901_OfficeMedium_STD2019_Miami.idf
./Transition-V24-1-0-to-V24-2-0 /Users/kanxuanhe/Document/20250929_swg/swg/data/epjson/Miami/ASHRAE901_OfficeMedium_STD2019_Miami.idf
./Transition-V24-2-0-to-V25-1-0 /Users/kanxuanhe/Document/20250929_swg/swg/data/epjson/Miami/ASHRAE901_OfficeMedium_STD2019_Miami.idf
```

- Convert .idf to .epjson format
```bash
cd /Applications/EnergyPlus-25-1-0 
./ConvertInputFormat /Users/kanxuanhe/Document/20250929_swg/swg/data/epjson/Miami/ASHRAE901_OfficeMedium_STD2019_Miami.idf
```

#### Generate City-Specific Models
- **Script**: `city_epjson_gen.py`
- **Function**: Merges site location and design day data from DDY files with base epJSON model
- **Output**: City-specific epJSON files with proper location and design conditions

### 2. Climate Data Processing

#### Weather File Generation
- **sMYM Generation**: `gen_samy.py` - Creates stochastic meteorological year files
- **Future Climate Data**: Uses bias-corrected climate projections for 2041, 2081 scenarios
- **Weather File Sources**:
  - Current TMY: Standard weather files for baseline
  - sTMY: Stochastic Typical Meteorological Years files (2001-2100)
  - sMYM: Yearly Multi-Year Meteorological files (2001-2100)
  - Annex80: IEA EBC Annex 80 future weather data

#### Weather File Fixing
- **Script**: `fix_weather_file.py`
- **Function**: Validates and repairs weather files for EnergyPlus compatibility

### 3. HVAC System Sizing

#### Component Autosizing
- **Process**: Run EnergyPlus with current TMY + DDY to autosize HVAC components
- **Script**: `size_freeze.py`
- **Function**: Extracts autosized component capacities and freezes them in the model
- **Output**: Frozen epJSON files with fixed HVAC sizing for consistent comparisons

### 4. Climate Scenario Simulations

#### TMY Runs (sTMY/Annex80)
- **Script**: `run_stmy.py`
- **Function**: Runs simulations for climate scenarios (2001, 2041, 2081) using frozen models
- **Climate Data Sources**:
  - sTMY: Bias-corrected projections from `data/stmy_out_bias_corrected/`
  - Annex80: IEA future weather data from `data/annex80/`
- **Output**: Simulation results for each city-scenario combination

#### Multi-Year Runs (sMYM)
- **Script**: `run_smym.py`
- **Function**: Runs yearly simulations (2001-2100) using pre-generated sMYM EPW files
- **Features**:
  - Handles EnergyPlus crashes from extreme weather
  - Minimal output mode to save disk space
  - Year-by-year trend analysis capability
- **Data Source**: `data/sMYM_epw/{city}/{city}_{year}.epw`

### 5. Results Analysis

#### Automated Results Processing
- **Script**: `analysis.py`
- **Main Class**: `EnergyPlusResultsAnalyzer`
- **Features**:
  - Interactive analysis mode selection
  - Regular analysis for scenario comparisons
  - sMYM analysis for yearly trend analysis

#### Key Metrics Extracted
- Energy consumption per floor area (MJ/m²)
- Heating/cooling unmet degree-hours (°C·hr)
- Monthly peak demand (kW)

#### Visualization Outputs
- **Trend plots**: Year-over-year climate impact trends
- **Comparison plots**: sTMY vs Annex80 data source comparison
- **Heatmaps**: Monthly peak demand by scenario and city (2x2 subplot layout)
- **Summary tables**: CSV exports of all extracted metrics

#### Analysis Types
1. **Regular Analysis**: 
   - Compares current vs future climate scenarios
   - Generates scenario comparison plots
   - Creates peak demand heatmaps with city-specific color schemes

2. **sMYM Analysis**:
   - Processes yearly simulation results (2001-2100)
   - Generates long-term trend analysis
   - Individual city plots and multi-city comparisons
   - Optimized x-axis labeling (decade intervals)

### 6. Output Structure
```
output/
├── {timestamp}/                    # Regular scenario runs
│   ├── {city}_{scenario}/         # Individual simulation results
│   └── _analysis/                 # Analysis outputs
│       ├── *.png                  # Plots and heatmaps
│       ├── *.csv                  # Summary tables
│       └── tables/                # Extracted HTML tables
└── smym_{timestamp}/              # sMYM yearly runs
    ├── {city}_{year}/             # Individual year results
    └── _analysis/                 # Trend analysis outputs
```

### 7. Key Dependencies
- **EnergyPlus 25.1.0**: Building simulation engine
- **Python Libraries**: pandas, numpy, matplotlib, seaborn, BeautifulSoup
- **eppy**: EnergyPlus file handling
- **Climate Data**: Bias-corrected projections and IEA Annex80 datasets

### 8. Usage Examples

#### Run Climate Scenario Simulations
```python
python run_stmy.py
```

#### Run Yearly Time Series Analysis
```python
python run_smym.py
```

#### Analyze Results
```python
python analysis.py
# Select: 1 for regular analysis, 2 for sMYM analysis
```
