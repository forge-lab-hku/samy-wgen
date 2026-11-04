Data pre-processing
===================

The functions below turn raw ISD and solar sources into aligned hourly data.
They are used by the standard pipeline, whose output files are described in
:doc:`../workflow/prepare`.
Source: `parse_ISD.py <https://github.com/forge-lab-hku/samy-wgen/blob/main/src/samy_wgen/utils/parse_ISD.py>`_,
`time_utils.py <https://github.com/forge-lab-hku/samy-wgen/blob/main/src/samy_wgen/utils/time_utils.py>`_,
`solar_calibration.py <https://github.com/forge-lab-hku/samy-wgen/blob/main/src/samy_wgen/utils/solar_calibration.py>`_.

.. py:function:: samy_wgen.utils.parse_ISD.process_station_directory(station_dir: str, output_dir: str = None) -> str | None

   Parse every ``.gz`` record in a station folder and write one combined CSV.

   :param station_dir: Directory containing yearly ISD archives.
   :param output_dir: CSV destination; default is the station directory.
   :returns: CSV path, or ``None`` when no station data can be processed.

.. py:function:: samy_wgen.utils.parse_ISD.remap_isd_columns(df: pandas.DataFrame) -> pandas.DataFrame

   Rename recognised ISD weather fields and coerce them to numeric values.

   :param df: Parsed ISD frame with original field names.
   :returns: Frame with the available ``temp``, ``dewpoint``, ``pressure``, ``wind_speed``, and ``wind_dir`` fields.
   :raises ValueError: No recognised weather field exists.

.. py:function:: samy_wgen.utils.time_utils.utc_to_lst(df: pandas.DataFrame, city: str | None = None, latitude: float | None = None, longitude: float | None = None) -> pandas.DataFrame

   Shift a UTC-indexed frame to timezone-naive fixed local standard time.

   :param df: UTC-indexed weather frame.
   :param city: Bundled city code used to find coordinates.
   :param latitude: Explicit latitude, used with ``longitude`` if city is omitted.
   :param longitude: Explicit longitude, used with ``latitude`` if city is omitted.
   :returns: A copy with the shifted DatetimeIndex; no daylight-saving change is applied.

.. py:function:: samy_wgen.utils.time_utils.assert_is_lst(df: pandas.DataFrame, name: str = '') -> None

   Check that an index is timezone-naive, has no duplicates, and shows no
   daylight-saving-type hour jumps.

   :param df: Candidate LST frame.
   :param name: Label to include in an error.
   :raises ValueError: The index fails a structural LST check.

.. py:function:: samy_wgen.utils.parse_ISD.resample_isd_hourly(df: pandas.DataFrame) -> pandas.DataFrame

   Aggregate a remapped weather frame by hour.

   :param df: Time-indexed frame with internal ISD column names.
   :returns: Hourly frame; continuous fields use means and ``wind_dir`` uses a mode.

.. py:function:: samy_wgen.utils.solar_calibration.calibrate_era5_land_solar(era5_path: str | pathlib.Path, nsrdb_path: str | pathlib.Path, *, latitude: float, longitude: float, utc_offset_hours: float, nsrdb_time_basis='auto', min_overlap_samples: int = 24) -> pandas.DataFrame

   Learn ERA5-Land solar calibration relationships from overlapping NSRDB
   observations and return hourly fixed-LST GHI/DNI/DHI.

   :param era5_path: ERA5-Land surface solar radiation NetCDF or CSV.
   :param nsrdb_path: NSRDB CSV with solar components.
   :param latitude: Site latitude in degrees.
   :param longitude: Site longitude in degrees.
   :param utc_offset_hours: Fixed UTC-to-LST offset.
   :param nsrdb_time_basis: ``auto``, ``utc``, or ``lst``.
   :param min_overlap_samples: Minimum usable daylight overlap samples.
   :returns: Calibrated hourly solar DataFrame in LST.
