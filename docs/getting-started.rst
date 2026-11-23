Quick start
===========

Install
-------

Use Python 3.10 or newer. Python 3.11 is recommended. Install from the
``v0.2.0`` tag:

.. code-block:: console

   git clone --branch v0.2.0 --depth 1 https://github.com/forge-lab-hku/samy-wgen.git
   cd samy-wgen
   python -m pip install .

Run the examples below from this checkout. For development, install with
``python -m pip install -e ".[dev]"`` from a checkout of ``main``.

Set up data access
------------------

The default pipeline uses NOAA ISD weather observations, NSRDB solar
observations, and Copernicus CMIP6 projections. It can use calibrated
ERA5-Land radiation when NSRDB coverage misses months in the fit baseline.
ISD needs no account. For NSRDB, obtain an
`API key <https://developer.nlr.gov/signup/>`_ and enter it with
``samy-download config``. For CMIP6 and ERA5-Land, register with the
`Copernicus Climate Data Store <https://cds.climate.copernicus.eu/>`_, accept
the licences for the datasets you need, and configure ``~/.cdsapirc`` as
described by CDS.

Run one city
------------

.. code-block:: console

   samy-run --city Miami

This downloads missing inputs, fits RS-VAR, applies the CMIP6 signal, and
writes results under ``output/Miami/``. The first run downloads years of data
and can take substantial time. Subsequent runs reuse cached stages when the
inputs and run settings match. City names use ASCII codes from
``src/samy_wgen/config/cities_config.csv``.

For a short forecast or an existing local dataset:

.. code-block:: console

   samy-run --city Miami --start 2025-01-01 --end "2030-12-31 23:00" --epw-years 2030
   samy-run --city Miami --skip-download

Include ``23:00`` in the end timestamp to export an EPW for the final year.
``--skip-download`` requires the inputs to be present locally. See
:doc:`workflows` for each processing stage and :doc:`cli/index` for options.
