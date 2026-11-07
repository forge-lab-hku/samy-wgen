5. Export and inspect outputs
=============================

The default output directory is ``output/<City>/``:

.. code-block:: text

   output/Miami/
   ├── 1_isd_lst.csv
   ├── 2_isd_solar_lst.csv
   ├── 3_rsvar.xlsx
   ├── 4_delta_corrected.csv
   ├── EPW/
   └── DDY/

``4_delta_corrected.csv`` is the continuous, climate-adjusted hourly series.
The pipeline writes a historical EPW for 2000 from the merged observation
record, plus future EPWs for selected years of the adjusted series. It also
derives baseline and future DDY design-day files. By default, future EPW years
are 2030, 2050, and 2080.

.. code-block:: console

   samy-run --city Miami --epw-years 2030,2050,2080

Solar calibration and monthly-energy diagnostics can also appear alongside
the stage files. Each EPW is one year extracted from the generated hourly
trajectory; a selected-year EPW alone does not describe the full multi-year
sequence. DDY files summarize design conditions from hourly data and are not
another weather trajectory.

Hourly EPW records use hours 1–24 and minute 60. The exporter omits February 29
and requires every other hour of the selected year; it does not invent missing
weather records. For a leap-year source, EPW data rows use the non-leap 2001
calendar while the filename retains the source year. Unavailable auxiliary
fields use EPW missing-value codes rather than zeros.
The CLI obtains elevation from the selected ISD station unless ``--elevation``
overrides it, and converts its sea-level pressure series to station pressure
for the EPW. For direct TMY/sTMY conversion, specify ``pressure_reference``
as ``sea_level`` in city metadata only when the source pressure is sea-level;
the default ``station`` leaves pressure unchanged.

Check the CSV's first and last timestamps, missing/duplicate hours, source
coverage, and physical QA before using outputs in a simulation. Check the
EPW header's location, timezone, and elevation. File-format and physical
checks do not establish predictive validation.

For a quick structural inspection of an existing output:

.. code-block:: python

   import pandas as pd

   weather = pd.read_csv("output/Miami/4_delta_corrected.csv", parse_dates=["datetime"])
   print(weather["datetime"].min(), weather["datetime"].max())
   print("duplicates:", weather["datetime"].duplicated().sum())
   print("missing temperature:", weather["temp"].isna().sum())

For DDY creation from an hourly DataFrame, see
:py:func:`~samy_wgen.utils.create_epw_ddy.write_ddy_from_dataframe`.
