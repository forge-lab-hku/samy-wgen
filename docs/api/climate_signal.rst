Climate signal
==============

These functions live in ``samy_wgen.pipeline.run_climate_delta``. They are
the main reusable stages behind the deterministic climate-delta path.
Source: `run_climate_delta.py <https://github.com/forge-lab-hku/samy-wgen/blob/main/src/samy_wgen/pipeline/run_climate_delta.py>`_.

.. py:function:: samy_wgen.pipeline.run_climate_delta.load_rsvar_output(rsvar_path: pathlib.Path, verbose: bool = True)

   Read the pre-delta RS-VAR output from Excel or CSV.

   :param rsvar_path: Workbook or CSV path. Excel expects ``RSVAR_Output``.
   :param verbose: Print loading diagnostics.
   :returns: Hourly DataFrame indexed by ``datetime`` and a metadata dictionary.

.. py:function:: samy_wgen.pipeline.run_climate_delta.load_cmip6_projection(base_path: pathlib.Path, city: str, ssp: str, scenario_id: str = None, gcm: str | None = None, verbose: bool = True) -> tuple

   Load a supported GCM's historical and scenario city series.

   :param base_path: CMIP6 root containing city directories.
   :param city: ASCII city code.
   :param ssp: Scenario code such as ``ssp245`` or ``ssp585``.
   :param scenario_id: Optional model/scenario priority hint.
   :param gcm: Explicit supported GCM selector.
   :param verbose: Print selection diagnostics.
   :returns: ``(daily_dataframe, monthly_dataframe, metadata)``.

.. py:function:: samy_wgen.pipeline.run_climate_delta.load_cordex_cmip5_projection(base_path: pathlib.Path, city: str, verbose: bool = True) -> tuple

   Load historical and RCP8.5 CORDEX-CMIP5 city series from a variant root.

   :param base_path: Variant root containing the selected city folder.
   :param city: ASCII city code.
   :param verbose: Print source diagnostics.
   :returns: Daily view, monthly view, and source metadata.

.. py:function:: samy_wgen.pipeline.run_climate_delta.compute_mixed_resolution_baseline(daily_df: pandas.DataFrame, monthly_df: pandas.DataFrame, baseline_start: int, baseline_end: int, var_resolution_config: dict[str, str]) -> dict

   Compute the projection's historical reference climatology from available
   daily and monthly variables.

   :param daily_df: Daily projection series.
   :param monthly_df: Monthly projection series.
   :param baseline_start: First baseline year.
   :param baseline_end: Last baseline year.
   :param var_resolution_config: Variable-to-resolution overrides; the standard pipeline passes an empty dictionary.
   :returns: Baseline information keyed by weather variable.

.. py:function:: samy_wgen.pipeline.run_climate_delta.apply_mixed_resolution_climate_delta(rsvar_df, daily_df, monthly_df, baseline, gcm: str, rcm: str | None, lat: float, baseline_start: int = 1991, baseline_end: int = 2010, isd_path: pathlib.Path = None, debug: bool = False, verbose: bool = True, longitude: float | None = None)

   Apply the projection signal to the hourly RS-VAR forecast. Solar changes
   are handled through the saved bounded-solar coordinates when present.

   :param rsvar_df: Pre-delta hourly forecast.
   :param daily_df: Daily projection variables.
   :param monthly_df: Monthly projection variables.
   :param baseline: Output of ``compute_mixed_resolution_baseline``.
   :param gcm: GCM label for output metadata.
   :param rcm: RCM label, or ``None`` for GCM-only input.
   :param lat: Site latitude used in solar constraints.
   :param baseline_start: First baseline year.
   :param baseline_end: Last baseline year.
   :param isd_path: Optional observed-input path for diagnostics.
   :param debug: Print extra debug information.
   :param verbose: Print progress messages.
   :param longitude: Site longitude for bounded solar reconstruction.
   :returns: ``(adjusted_hourly_dataframe, delta_statistics)``.
