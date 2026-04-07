"""samy_wgen: Stochastic future weather EPW generation.

Pipeline:
    ISD (UTC) → utc_to_lst() → RS-VAR training (LST throughout)
    → forecast_with_climate_delta() + CMIP6
    → process_csv_to_yearly_epw() → EPW files
"""

__version__ = "0.1.0"

from samy_wgen.model.rsvar import ForecastTo2100V5
from samy_wgen.pipeline.run_rsvar import RSVAROnlyPipeline
from samy_wgen.pipeline.gen_samy import process_csv_to_yearly_epw
from samy_wgen.utils.calculate_weather_var import (
    calc_dewpoint_from_RH,
    calc_Q_from_dewpoint,
    calc_RH_from_dewpoint,
)
from samy_wgen.utils.time_utils import utc_to_lst, assert_is_lst, get_utc_offset

__all__ = [
    "ForecastTo2100V5",
    "RSVAROnlyPipeline",
    "process_csv_to_yearly_epw",
    "calc_dewpoint_from_RH",
    "calc_Q_from_dewpoint",
    "calc_RH_from_dewpoint",
    "utc_to_lst",
    "assert_is_lst",
    "get_utc_offset",
]
