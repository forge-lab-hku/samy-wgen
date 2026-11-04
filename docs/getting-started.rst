Quick start
===========

Install
-------

Use Python 3.10 or newer. From a repository checkout:

.. code-block:: console

   python -m pip install -e .

The package is not currently available on PyPI. The commands below run from
the repository root.

Set up data access
------------------

The default pipeline uses NOAA ISD weather observations, NREL NSRDB solar
observations, and Copernicus CMIP6 projections. ISD needs no account. For
NSRDB, obtain an API key from `NREL <https://developer.nlr.gov/signup/>`_
and enter it with ``samy-download config``. For CMIP6, register with the
`Copernicus Climate Data Store <https://cds.climate.copernicus.eu/>`_, accept
the dataset licence, and configure ``~/.cdsapirc`` as described by CDS.

Run one city
------------

.. code-block:: console

   samy-run --city Miami

This downloads missing inputs, fits RS-VAR, applies the CMIP6 signal, and
writes results under ``output/Miami/``. The first run downloads years of data
and can take substantial time. Subsequent runs reuse cached stages. City
names use ASCII codes from ``src/samy_wgen/config/cities_config.csv``.

For a short forecast or an existing local dataset:

.. code-block:: console

   samy-run --city Miami --start 2025-01-01 --end 2030-12-31 --epw-years 2030
   samy-run --city Miami --skip-download

``--skip-download`` requires the inputs to be present locally. See
:doc:`workflows` for each processing stage and :doc:`cli/index` for options.
