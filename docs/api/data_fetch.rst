Data fetch
==========

These functions live in ``samy_wgen.utils.city_utils`` and
``samy_wgen.pipeline.download_data``. Downloader calls perform network I/O;
they are suitable for scripts that need more control than ``samy-download``.
Source: `city_utils.py <https://github.com/forge-lab-hku/samy-wgen/blob/main/src/samy_wgen/utils/city_utils.py>`_,
`download_data.py <https://github.com/forge-lab-hku/samy-wgen/blob/main/src/samy_wgen/pipeline/download_data.py>`_.

City lookup
-----------

.. py:function:: samy_wgen.utils.city_utils.resolve_city(query: str) -> CityRecord

   Resolve an ASCII city code against the bundled city table.

   :param query: City code such as ``Miami`` or ``Los_Angeles``.
   :returns: ``CityRecord`` with latitude, longitude, country, and CORDEX domain.
   :raises ValueError: The city code is not in the bundled table.

.. py:function:: samy_wgen.utils.city_utils.available_city_codes() -> list[str]

   Return all supported ASCII city codes. Use this to inspect valid inputs
   before calling a downloader or model stage.

Station and solar source discovery
----------------------------------

.. py:function:: samy_wgen.pipeline.download_data.list_isd_stations(lat=None, lon=None, radius_km=150.0, cache_dir='data/ISD', city=None) -> pandas.DataFrame

   List nearby ISD stations, sorted by distance.

   :param lat: Explicit latitude when ``city`` is omitted.
   :param lon: Explicit longitude when ``city`` is omitted.
   :param radius_km: Search radius in kilometres.
   :param cache_dir: Directory for the ISD station-history cache.
   :param city: City code used to resolve coordinates.
   :returns: Station table including station IDs, distance, and reported coverage.

.. py:function:: samy_wgen.pipeline.download_data.select_best_isd_station(city: str, radius_km: float = 50.0, cache_dir: str = 'data/ISD') -> str

   Select the station with the best reported temporal coverage inside the
   search radius.

   :param city: ASCII city code.
   :param radius_km: Maximum station distance.
   :param cache_dir: ISD metadata cache directory.
   :returns: Station ID in ``USAF-WBAN`` form.
   :raises ValueError: No station is found inside the radius.

.. py:function:: samy_wgen.pipeline.download_data.list_nsrdb_sources(cities: list[str]) -> None

   Print NSRDB source availability and the default selection for each city.

   :param cities: ASCII city codes to inspect.
   :returns: ``None``; results are printed.

Downloads
---------

.. py:function:: samy_wgen.pipeline.download_data.download_isd(station_id: str, city: str, years: list[int], output_dir: str = 'data/ISD') -> None

   Download yearly ISD compressed records for one station.

   :param station_id: ``USAF-WBAN`` station ID.
   :param city: ASCII city code used in the output path.
   :param years: Requested years; unavailable station years are skipped.
   :param output_dir: Root ISD data directory.
   :returns: ``None``; files are saved below ``output_dir``.

.. py:function:: samy_wgen.pipeline.download_data.download_nsrdb(city: str, lat=None, lon=None, years=None, output_dir='data/NSRDB', interval=60, max_retries=3, retry_backoff_sec=2.0) -> dict

   Download point GHI/DNI/DHI from the selected NSRDB source.

   :param city: ASCII city code.
   :param lat: Optional latitude override.
   :param lon: Optional longitude override.
   :param years: Requested list of years.
   :param output_dir: NSRDB output directory.
   :param interval: Request interval in minutes.
   :param max_retries: Retries for transient year-level failures.
   :param retry_backoff_sec: Base retry delay in seconds.
   :returns: Dictionary with ``success``, ``failed``, and ``incomplete`` entries.

.. py:function:: samy_wgen.pipeline.download_data.download_era5_land(city: str, lat=None, lon=None, start_date='1950-01-01', end_date=None, output_dir='data/ERA5_Land', variables=None, overwrite=False) -> list[pathlib.Path]

   Request ERA5-Land hourly point series, one NetCDF per selected variable.

   :param city: ASCII city code.
   :param lat: Optional latitude override.
   :param lon: Optional longitude override.
   :param start_date: First requested date.
   :param end_date: Last requested date; default is the latest available day.
   :param output_dir: ERA5-Land output root.
   :param variables: ERA5-Land variable names; default is the configured set.
   :param overwrite: Replace existing point files.
   :returns: Paths to downloaded NetCDF files.

.. py:function:: samy_wgen.pipeline.download_data.download_cmip6(city: str, lat=None, lon=None, models=None, experiments=None, variables=None, year_range=None, output_dir='data/CMIP6') -> dict

   Request CDS CMIP6 data and extract the nearest city grid cell.

   :param city: ASCII city code.
   :param lat: Optional latitude override.
   :param lon: Optional longitude override.
   :param models: Model identifiers, defaulting to MPI-ESM1-2-LR.
   :param experiments: CDS experiment names, by default historical and SSP2-4.5.
   :param variables: Requested CDS variables; default is all supported variables.
   :param year_range: Optional inclusive year pair.
   :param output_dir: CMIP6 city-series root.
   :returns: Summary of successful, failed, and incomplete requests.

.. py:function:: samy_wgen.pipeline.download_data.extract_cmip6_from_files(cities: list[str], input_files: list[str | pathlib.Path], output_dir: str = 'data/CMIP6', output_name: str = 'extracted_cmip6') -> dict[str, pathlib.Path]

   Extract nearest-grid city CSVs from local NetCDF files or ZIP-wrapped
   NetCDF archives.

   :param cities: City codes to extract.
   :param input_files: Local source file paths.
   :param output_dir: Root CMIP6 directory.
   :param output_name: Stem used for each output CSV.
   :returns: Mapping of city code to written CSV path.

.. py:function:: samy_wgen.pipeline.download_data.download_cordex_cmip5(city: str, lat=None, lon=None, cordex_domain=None, gcm='mpi_m_mpi_esm_lr', rcm='ncar_regcm4', ensemble='r1i1p1', experiments=None, variables=None, historical_years=None, rcp85_years=None, months=None, output_dir='data/CORDEX_CMIP5', overwrite=False, keep_parts=False) -> dict[str, pathlib.Path]

   Download CORDEX-CMIP5 daily city series for the climate-delta reader.

   :param city: ASCII city code.
   :param lat: Optional latitude override.
   :param lon: Optional longitude override.
   :param cordex_domain: CORDEX domain; default from city configuration.
   :param gcm: Driving GCM identifier.
   :param rcm: Regional climate model identifier.
   :param ensemble: Ensemble identifier.
   :param experiments: Historical and/or RCP8.5 experiments.
   :param variables: Requested CORDEX variables.
   :param historical_years: Optional historical year pair.
   :param rcp85_years: Optional RCP8.5 year pair.
   :param months: Two-digit calendar months to request.
   :param output_dir: CORDEX output root.
   :param overwrite: Replace completed city CSVs.
   :param keep_parts: Retain intermediate download parts.
   :returns: Mapping of experiment to city CSV path.
