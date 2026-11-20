samy-run
========

Run the full pipeline for one or more ASCII city codes. The default mode
downloads missing ISD/NSRDB/CMIP6 inputs, fits RS-VAR, applies CMIP6 deltas,
and writes EPW/DDY files. Ensemble mode follows the fit branch described in
:doc:`../workflow/rsvar`.

.. parsed-literal::

   **samy-run** --city Miami --ssp ssp245 --epw-years 2030,2050

Required and output options
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Meaning and default
   * - ``--city CITY [CITY ...]``
     - Required ASCII city codes from the bundled city table.
   * - ``--out-dir DIR``
     - Output directory; default ``output/<City>``.
   * - ``--data-dir DIR``
     - Input root; default ``data``.
   * - ``--mode deterministic|ensemble``
     - Output branch; default ``deterministic``.
   * - ``--prepare-only``
     - Stop after writing the prepared hourly ISD and solar CSVs. This path requires no CMIP6 data or station elevation.
   * - ``--overwrite``
     - Recompute cached stages and outputs.

Fit, forecast, and projection options
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Meaning and default
   * - ``--baseline-start YEAR`` / ``--baseline-end YEAR``
     - Observation/projection reference years; defaults 1991 and 2010.
   * - ``--start DATE`` / ``--end DATE``
     - Forecast bounds; defaults ``2025-01-01`` and ``2100-12-31 23:00``.
   * - ``--ssp CODE``
     - CMIP6 scenario; default ``ssp245``.
   * - ``--gcm MODEL``
     - Model used for coverage validation, download, and climate delta. Overrides ``--cmip6-models``.
   * - ``--rsvar-input FILE``
     - Reuse an existing Stage 3 XLSX/CSV for one city. Specify forecast bounds that match the file.
   * - ``--epw-years Y1,Y2,...``
     - Deterministic EPW years; default ``2030,2050,2080``.
   * - ``--elevation METRES``
     - Override selected ISD station elevation. By default, the CLI reads
       ``ISD/isd-history.csv``; if no matching elevation exists, supply this option.
   * - ``--members N``
     - Ensemble member count; default 50.
   * - ``--regime-mode soft|hard|ensemble``
     - Member regime path mode; default ``ensemble``.
   * - ``--seed N``
     - Base member seed; default 0.

Input and automatic-download options
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Meaning and default
   * - ``--isd-dir DIR``
     - Override auto-detected station folder under ``data/ISD/<City>``.
   * - ``--nsrdb FILE``
     - Use an explicit observed/calibrated solar CSV.
   * - ``--cmip6-dir DIR``
     - Override the CMIP6 root under ``data/CMIP6``.
   * - ``--download-data SOURCE [SOURCE ...]``
     - Allow automatic downloads of ``isd``, ``nsrdb``, ``era5-land``, and
       ``cmip6``. All four are enabled by default. ERA5-Land solar is requested
       when the NSRDB record misses months in the fit baseline.
   * - ``--skip-download``
     - Require local input files; disable automatic retrieval.
   * - ``--isd-station ID [CITY=ID ...]``
     - Station override; bare ID for one city, mappings for multiple cities.
   * - ``--isd-radius KM``
     - Automatic station search radius; default 50 km.
   * - ``--download-years START END``
     - Automatic ISD/NSRDB request range.
   * - ``--cmip6-models MODEL [MODEL ...]``
     - The first model is used for a deterministic run without ``--gcm``; default ``mpi_esm1_2_lr``. Use ``samy-download`` to prefetch several models.
   * - ``--cmip6-experiments EXP [EXP ...]``
     - CDS experiments; default historical plus selected SSP.
   * - ``--cmip6-years START END``
     - CMIP6 request year range.

For CORDEX-CMIP5 projections, see :doc:`download` and :doc:`delta`.
