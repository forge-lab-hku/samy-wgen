samy-delta
==========

Apply a selected climate projection to saved RS-VAR forecasts. Use CMIP6
with an SSP, or select a CORDEX-CMIP5 variant root explicitly.

.. parsed-literal::

   **samy-delta** --rsvar-dir output/RSVAR --city Miami \\
     --projection-type cmip6 --projection-root data/CMIP6 --ssp ssp245

.. list-table:: Inputs and selection
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Meaning and default
   * - ``--rsvar-dir DIR``
     - Directory containing saved RS-VAR outputs; default ``output/RSVAR``.
   * - ``--city CITY [CITY ...]``
     - ASCII city code(s); default all configured cities.
   * - ``--config CSV``
     - City metadata CSV; default bundled table.
   * - ``--projection-type cmip6|cordex-cmip5``
     - Projection reader; default ``cmip6``.
   * - ``--projection-root DIR``
     - Per-city projection root; default resolved by projection type.
   * - ``--ssp CODE``
     - Required when using CMIP6, for example ``ssp245``.
   * - ``--gcm NAME`` / ``--rcm NAME``
     - Optional model-name metadata overrides on the standalone delta command.
   * - ``--scenario-label LABEL``
     - Output label in directory and file names.

.. list-table:: Processing and output
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Meaning and default
   * - ``--delta-mode daily|monthly``
     - Daily changes by default; ``monthly`` uses month-mean changes.
   * - ``--baseline-start YEAR`` / ``--baseline-end YEAR``
     - Projection baseline bounds; defaults 1991 and 2010.
   * - ``--out-dir DIR``
     - Output root; default ``output/RSVAR_cd/`` for daily mode.
   * - ``--overwrite``
     - Replace existing delta outputs.
   * - ``--preview-first-year``
     - Save an additional CSV for the first forecast year.
   * - ``--debug``
     - Print baseline and delta diagnostics.

For CORDEX-CMIP5, the variant directory must contain historical and RCP8.5
city series covering the selected baseline:

.. parsed-literal::

   **samy-delta** --rsvar-dir output/RSVAR --city Los_Angeles \\
     --projection-type cordex-cmip5 \\
     --projection-root data/CORDEX_CMIP5/NAM-22_mpi_m_mpi_esm_lr_ncar_regcm4_r1i1p1
