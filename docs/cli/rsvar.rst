samy-rsvar
==========

Fit the observation-based RS-VAR and save a pre-delta forecast. The input
must be an hourly ISD-plus-solar CSV in fixed LST. This command does not
apply CMIP6 or CORDEX changes.

.. parsed-literal::

   **samy-rsvar** --isd output/Miami/2_isd_solar_lst.csv \\
     --city Miami --output output/RSVAR/rsvar_output_Miami.xlsx

.. list-table:: Parameters
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Meaning and default
   * - ``--isd FILE``
     - Prepared CSV; selects single-file mode. Without it, the command uses its batch input layout.
   * - ``--isd-dir DIR``
     - Batch root containing ``<City>_with_solar.csv`` or ``<City>/2_isd_solar_lst.csv``.
   * - ``--baseline-start YEAR`` / ``--baseline-end YEAR``
     - Observation baseline bounds; defaults 1991 and 2010.
   * - ``--start DATE`` / ``--end DATE``
     - Forecast bounds; defaults ``2025-01-01`` and ``2100-12-31 23:00``.
   * - ``--city CITY [CITY ...]``
     - ASCII city code(s); exactly one is required with ``--isd``. In batch mode the default is all configured cities.
   * - ``--output PATH``
     - XLSX file for single-file mode or directory for batch mode; default ``output/RSVAR/rsvar_output_<City>.xlsx``.
   * - ``--config CSV``
     - City table; default is the bundled ``cities_config.csv``.

The output includes ``RSVAR_Output``, ``Metadata``, and ``Seasonal_Profile``
sheets, an adjacent CSV, and a solar calibration CSV. Without ``--isd-dir``,
batch mode checks ``data/ISD_complete_solar/<City>_with_solar.csv`` and
``output/<City>/2_isd_solar_lst.csv``.
