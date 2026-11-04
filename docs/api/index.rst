Python API reference
====================

The functions are grouped by the task they perform. Each page gives the
actual source-module path, signature, parameters, and return value. The
one-click pipeline is accessed through :doc:`../cli/run`; Python calls below
are useful when preparing or inspecting individual stages.

.. list-table:: Choose a module group
   :header-rows: 1
   :widths: 30 70

   * - Group
     - Use
   * - :doc:`data_fetch`
     - Resolve cities, inspect stations, and retrieve source series.
   * - :doc:`data_preprocessing`
     - Parse ISD, align UTC to fixed LST, and prepare solar inputs.
   * - :doc:`rsvar`
     - Fit RS-VAR and generate conditional regime-path members.
   * - :doc:`climate_signal`
     - Load projections, compute baselines, and apply deltas.
   * - :doc:`weather_files_output`
     - Compute and write EnergyPlus design-day files.

.. toctree::
   :maxdepth: 1

   data_fetch
   data_preprocessing
   rsvar
   climate_signal
   weather_files_output

These are selected public functions used by the documented workflow. Private
pipeline helpers (names starting with ``_``) are implementation details.
The CLI pages describe supported command-line operation.
