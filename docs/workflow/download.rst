1. Download data
================

The default ``samy-run`` path needs three inputs: historical station weather
from NOAA ISD, observed solar irradiance from NSRDB, and a CMIP6 historical
plus scenario pair. ``samy-download`` can prepare them in the directory layout
expected by the pipeline. Data access requirements are in :doc:`../getting-started`.

.. list-table:: Available sources
   :header-rows: 1
   :widths: 18 38 44

   * - Source
     - Role
     - Pipeline status
   * - ISD
     - Multi-year station temperature, dew point, pressure, and wind.
     - Required by the default RS-VAR fit.
   * - NSRDB
     - Observed GHI, DNI, and DHI at the city point.
     - Required by the default solar fit; can be replaced by a calibrated CSV.
   * - CMIP6
     - Historical baseline and future SSP change signal.
     - Required by the single-trajectory ``samy-run`` path.
   * - CORDEX-CMIP5
     - Regional historical and RCP8.5 projection series.
     - Optional input to ``samy-delta``; not selected by ``samy-run``.
   * - ERA5-Land
     - Longer hourly reanalysis record; solar can be calibrated to NSRDB.
     - Optional explicit download and calibration path.

ISD and NSRDB
-------------

Inspect stations and NSRDB availability, then download a city. When no ISD
station ID is given, the downloader selects the best coverage within 50 km.
``--years`` controls the ISD/NSRDB request; without it the downloader
requests 1970–2024 for ISD and 1998–2024 for NSRDB. The selected sources
may have shorter actual coverage.

.. code-block:: console

   samy-download list-stations --city Miami
   samy-download list-nsrdb --city Miami
   samy-download download --city Miami --data isd nsrdb --years 1991 2024

To pin a station, pass ``--isd-station USAF-WBAN``. Files are stored under
``data/ISD/Miami/`` and ``data/NSRDB/``. Check the reported station coverage
before fitting a baseline.

CMIP6
-----

Inspect the live CDS catalogue before requesting a GCM/experiment pair.
``samy-download`` extracts city grid-cell series and stores daily and monthly
CSVs under ``data/CMIP6/<City>/``. ``--cmip6-years`` controls the projection
request independently of ``--years``.

.. code-block:: console

   samy-download list-cmip6 --detail
   samy-download download --city Miami --data cmip6 \
     --models mpi_esm1_2_lr --experiments historical ssp2_4_5

For an existing NetCDF or ZIP-wrapped NetCDF archive, use
``samy-download extract-cmip6 --city Miami --input <file.nc>``. The extracted
series still needs the filenames and historical/scenario coverage expected
by :doc:`delta`.

CORDEX-CMIP5
------------

The optional CORDEX path downloads a GCM/RCM city series for later use by
``samy-delta --projection-type cordex-cmip5``. Its default experiments are
historical and RCP8.5. Both must cover the requested baseline period.

.. code-block:: console

   samy-download list-cordex --domain NAM-22 --detail
   samy-download download --city Los_Angeles --data cordex \
     --cordex-gcm mpi_m_mpi_esm_lr --cordex-rcm ncar_regcm4

ERA5-Land solar extension
-------------------------

The ERA5-Land downloader writes hourly point NetCDF files, one per variable.
The solar calibration command compares overlapping ERA5-Land radiation with
NSRDB and writes an hourly LST ``DATE,GHI,DNI,DHI`` CSV. This extends the
time span using calibration relationships; older years are not independent
NSRDB observations.

.. code-block:: console

   samy-download download --city Miami --data era5-land \
     --era5-land-years 1950 2024 \
     --era5-land-var surface_solar_radiation_downwards
   samy-download calibrate-solar --city Miami \
     --era5 data/ERA5_Land/Miami/era5_land_miami_surface_solar_radiation_downwards_19500101-20241231.nc \
     --nsrdb data/NSRDB/nsrdb_miami_1998-2024.csv \
     --output data/ERA5_Land/calibrated/miami_solar_calibrated.csv

These filenames illustrate the requested years; use the actual downloaded
paths if coverage or naming differs. The calibration records time-basis
evidence in an adjacent metadata JSON file; ambiguous NSRDB timestamps
require an explicit ``--nsrdb-time-basis utc`` or ``lst``.

See :doc:`../cli/download` for source-specific options and
:doc:`../api/data_fetch` for Python entry points. Downloads are cached; use a
separate output directory for short experiments rather than replacing
production inputs.
