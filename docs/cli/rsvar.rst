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
   * - ``--city CITY [CITY ...]``
     - ASCII city code(s); exactly one is required with ``--isd``. In batch mode the default is all configured cities.
   * - ``--output PATH``
     - XLSX file for single-file mode or directory for batch mode; default ``output/RSVAR/rsvar_output_<City>.xlsx``.
   * - ``--config CSV``
     - City table; default is the bundled ``cities_config.csv``.

The output includes ``RSVAR_Output``, ``Metadata``, and ``Seasonal_Profile``
sheets, an adjacent CSV, and a solar calibration CSV. The standalone command
uses the module's fit and forecast defaults; use ``samy-run`` to set periods
on the one-click path.
