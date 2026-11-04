samy-download
=============

``samy-download`` has subcommands for credentials, discovery, data retrieval,
local extraction, and solar calibration. ``samy-download <subcommand> --help``
shows the complete accepted values in the installed version.

Start with a source check, then request only the data needed for your city:

.. parsed-literal::

   **samy-download** list-stations --city Miami
   **samy-download** list-nsrdb --city Miami
   **samy-download** list-cmip6 --detail
   **samy-download** download --city Miami --data isd nsrdb cmip6

.. list-table:: Subcommands
   :header-rows: 1
   :widths: 30 70

   * - Subcommand
     - Purpose
   * - ``config``
     - Save NSRDB API credentials.
   * - ``list-stations`` / ``check-station``
     - Inspect nearby ISD stations or metadata coverage for one station.
   * - ``list-nsrdb``
     - Inspect NSRDB source and year availability for cities.
   * - ``list-cmip6`` / ``list-cordex``
     - Inspect the live CDS catalogue.
   * - ``download``
     - Download one or more selected data sources.
   * - ``extract-cmip6``
     - Extract city CSVs from local NetCDF/ZIP-wrapped NetCDF.
   * - ``calibrate-solar``
     - Calibrate ERA5-Land solar with overlapping NSRDB observations.

Discovery and extraction parameters
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Subcommand and option
     - Meaning
   * - ``list-stations --city CITY [--radius KM] [--isd-dir DIR]``
     - List nearby stations; radius defaults to 150 km, cache root to ``data/ISD``.
   * - ``check-station --isd-station ID [--years START END] [--isd-dir DIR]``
     - Check reported station years against an optional request.
   * - ``list-nsrdb --city CITY [CITY ...]``
     - Show the available solar products for each city.
   * - ``list-cmip6 [--detail] [--experiment EXP] [--resolution daily|monthly|fixed] [--var VAR]``
     - Expand or filter CDS model availability.
   * - ``list-cordex [--detail] [--domain DOMAIN] [--experiment EXP] [--gcm GCM] [--rcm RCM] [--ensemble ID] [--resolution RES] [--var VAR]``
     - Expand or filter CORDEX domain/model availability.
   * - ``extract-cmip6 --city CITY... --input NETCDF... [--cmip6-dir DIR] [--name STEM]``
     - Extract nearest-grid city series; default root ``data/CMIP6`` and stem ``extracted_cmip6``.

More discovery and local-extraction examples:

.. parsed-literal::

   **samy-download** check-station --isd-station 722020-12839 --years 1991 2024
   **samy-download** list-cordex --domain NAM-22 --detail
   **samy-download** extract-cmip6 --city Miami --input path/to/projection.nc

``config`` opens an interactive prompt for the NSRDB API key and email:

.. parsed-literal::

   **samy-download** config

The ``download`` subcommand
---------------------------

``--city`` is required. ``--data`` selects any of ``isd``, ``nsrdb``,
``cmip6``, ``cordex``, and ``era5-land``; its default is ``isd nsrdb cmip6``.
The options below apply only when the corresponding source is selected.

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - Source
     - Options
     - Meaning and default
   * - ISD
     - ``--isd-station``, ``--isd-radius``, ``--isd-dir``
     - Station override, 50-km auto-selection radius, output root ``data/ISD``.
   * - NSRDB
     - ``--nsrdb-dir``
     - Output root ``data/NSRDB``.
   * - ISD/NSRDB
     - ``--years START END``
     - Requested year range; defaults are 1970–2024 for ISD and 1998–2024 for NSRDB.
   * - CMIP6
     - ``--models``, ``--experiments``, ``--cmip6-var``
     - Model(s), CDS experiment(s), and variable(s); defaults MPI-ESM1-2-LR, historical/SSP2-4.5, all supported variables.
   * - CMIP6
     - ``--cmip6-years START END``, ``--cmip6-dir``
     - Projection years and output root ``data/CMIP6``.
   * - ERA5-Land
     - ``--era5-land-years``, ``--era5-land-var``, ``--era5-land-dir``, ``--era5-land-overwrite``
     - Year range, variables, output root ``data/ERA5_Land``, and replacement switch.
   * - CORDEX
     - ``--cordex-domain``, ``--cordex-gcm``, ``--cordex-rcm``, ``--cordex-ensemble``
     - Domain and model chain; domain defaults from city configuration.
   * - CORDEX
     - ``--cordex-experiments``, ``--cordex-var``, ``--cordex-months``
     - Historical/RCP8.5 by default, all supported variables and months.
   * - CORDEX
     - ``--cordex-historical-years``, ``--cordex-rcp85-years``
     - Historical and RCP8.5 year ranges; defaults 1970–2005 and 2006–2100.
   * - CORDEX
     - ``--cordex-dir``, ``--cordex-overwrite``
     - Output root ``data/CORDEX_CMIP5`` and replacement switch.

.. parsed-literal::

   **samy-download** download --city Miami --data isd nsrdb cmip6
   **samy-download** download --city Miami --data era5-land \\
     --era5-land-years 1950 2024

The ``calibrate-solar`` subcommand
----------------------------------

For a downloaded Miami ERA5-Land solar file and an overlapping NSRDB CSV:

.. parsed-literal::

   **samy-download** calibrate-solar --city Miami \\
     --era5 data/ERA5_Land/Miami/era5_land_miami_surface_solar_radiation_downwards_19500101-20241231.nc \\
     --nsrdb data/NSRDB/nsrdb_miami_1998-2024.csv \\
     --output data/ERA5_Land/calibrated/miami_solar_calibrated.csv

Adjust filenames to match the downloaded year range. The output includes an
adjacent metadata JSON file recording the time-basis decision.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Option
     - Meaning and default
   * - ``--city CITY``
     - Required city code for coordinates and standard UTC offset.
   * - ``--era5 FILE`` / ``--nsrdb FILE``
     - Required ERA5-Land and NSRDB source files.
   * - ``--output CSV``
     - Destination; default under ``data/ERA5_Land/calibrated/``.
   * - ``--utc-offset HOURS``
     - Override the city fixed-standard offset.
   * - ``--nsrdb-time-basis auto|utc|lst``
     - NSRDB timestamp interpretation; default ``auto`` requires decisive solar evidence.
