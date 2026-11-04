samy-ensemble
=============

Fit from an already prepared hourly LST CSV and save regime-path members.
This direct command does not apply a climate delta or export EPW/DDY files.

.. parsed-literal::

   **samy-ensemble** --isd output/Miami/2_isd_solar_lst.csv \\
     --city Miami --members 50 --seed 0 --out-dir output/Miami/ensemble

.. list-table:: Parameters
   :header-rows: 1
   :widths: 32 68

   * - Option
     - Meaning and default
   * - ``--isd FILE``
     - Required prepared hourly ISD-plus-solar CSV.
   * - ``--city CITY``
     - Required ASCII city code.
   * - ``--lat DEGREES``
     - Optional latitude override; default from city table.
   * - ``--out-dir DIR``
     - Member destination; default ``output/ensemble``.
   * - ``--members N``
     - Number of members; default 50.
   * - ``--start DATE`` / ``--end DATE``
     - Forecast bounds; defaults ``2025-01-01`` and ``2025-12-31``.
   * - ``--regime-mode soft|hard|ensemble``
     - Path generation mode; default ``ensemble``.
   * - ``--regime-strength VALUE``
     - Regime intensity from 0 to 1; default 1.0.
   * - ``--chunk-years N``
     - Nominal 8760-hour years per centering chunk; default 5.
   * - ``--seed N``
     - Base random seed; default 0.

The result is ``member_*.parquet`` plus ``ensemble_summary.xlsx``. See the
member interpretation in :doc:`../workflow/rsvar`.
