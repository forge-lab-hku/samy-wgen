CLI reference
=============

The package installs five commands. The guide explains their workflow;
these pages list parameters and defaults from the current CLI parsers.
Run ``<command> --help`` to inspect the installed version before a long job.

.. list-table:: Commands
   :header-rows: 1
   :widths: 28 72

   * - Command
     - Purpose
   * - :doc:`run`
     - End-to-end deterministic weather or regime-path member pipeline.
   * - :doc:`download`
     - Source discovery, downloads, extraction, and solar calibration.
   * - :doc:`rsvar`
     - Fit RS-VAR from a prepared CSV and save a pre-delta forecast.
   * - :doc:`delta`
     - Apply CMIP6 or CORDEX-CMIP5 changes to an RS-VAR forecast.
   * - :doc:`ensemble`
     - Generate regime-path members directly from prepared observations.

.. toctree::
   :maxdepth: 1

   run
   download
   rsvar
   delta
   ensemble
