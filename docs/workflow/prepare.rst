2. Prepare observations
=======================

The preparation stage converts source files into one hourly, fixed local
standard time (LST) table. RS-VAR reads this merged table, not the raw ISD
``.gz`` files.

Parse and align ISD
-------------------

The pipeline parses every yearly file in the selected station directory,
renames available ISD fields to ``temp``, ``dewpoint``, ``pressure``,
``wind_speed``, and ``wind_dir``, then shifts UTC timestamps to the city's
fixed standard offset. It removes rows with no core weather measurements and
resamples to one row per hour. Wind direction uses a within-hour mode rather
than an arithmetic mean across the 0°/360° boundary.

``samy-run`` writes ``1_isd_lst.csv``. The conversion can also be called
directly from Python:

.. code-block:: python

   import pandas as pd
   from samy_wgen.utils.parse_ISD import (
       process_station_directory, remap_isd_columns, resample_isd_hourly,
   )
   from samy_wgen.utils.time_utils import utc_to_lst

   path = process_station_directory(
       "data/ISD/Miami/722020-12839_MIAMI_INTERNATIONAL_AIRPORT"
   )
   raw = pd.read_csv(path, parse_dates=["datestamp"])
   weather = remap_isd_columns(raw.set_index("datestamp").sort_index())
   hourly = resample_isd_hourly(utc_to_lst(weather, city="Miami"))

The example assumes that station folder was downloaded. ``process_station_directory``
writes a combined CSV in the station folder when no output directory is given.

Merge solar observations
------------------------

The next stage reads NSRDB columns ``GHI``, ``DNI``, and ``DHI``. It accepts a
``DATE`` column or ``Year,Month,Day,Hour`` columns, clips negative irradiance
to zero, aggregates to hourly values, joins on matching LST hours, and fills
short gaps of at most three rows in each direction. It drops rows lacking
temperature or the first available solar column. The result is
``2_isd_solar_lst.csv`` with a ``DATE`` index column.

For an alternate calibrated solar CSV, specify its path. It must already be
in the same fixed LST basis as the station record.

.. code-block:: console

   samy-run --city Miami \
     --nsrdb data/ERA5_Land/calibrated/miami_solar_calibrated.csv \
     --skip-download

``--skip-download`` also requires the other inputs to exist. Before fitting,
inspect the merged file's start/end dates, missing hours, and GHI/DNI/DHI
coverage; the requested 1991–2010 fit baseline must be represented in the
actual data. See :doc:`../api/data_preprocessing` for function parameters.
