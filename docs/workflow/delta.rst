4. Apply climate signal
=======================

The deterministic branch reads the cached RS-VAR forecast and a selected
CMIP6 historical/scenario pair. It computes a 1991–2010 projection baseline
by default, then transfers the projected changes into the hourly weather.
Daily and monthly source variables retain their available resolution; the
pipeline does not create subdaily climate-model observations.

The input set is ``3_rsvar.xlsx`` plus city-specific daily and/or monthly
projection CSVs. The output is ``4_delta_corrected.csv``. A separate
``4_solar_monthly_energy.csv`` records solar targets and achieved energy.

.. code-block:: console

   samy-run --city Miami --ssp ssp585 --gcm MPI-ESM1-2-LR

``--ssp`` chooses the scenario; ``--gcm`` selects a specific supported CMIP6
model. Different SSPs and GCMs are alternative forcings, not samples from a
calibrated probability distribution. For a valid comparison, record the model,
scenario, baseline, source resolution, and coverage.

The RS-VAR result can be reused with another projection without refitting:

.. code-block:: console

   samy-delta --rsvar-dir output/RSVAR --city Miami --ssp ssp245 \
     --projection-root data/CMIP6 --out-dir output/delta

The standalone ``samy-delta`` command supports CMIP6 or compatible
CORDEX-CMIP5 city series. For CORDEX, point ``--projection-root`` at the
variant directory containing the city and select
``--projection-type cordex-cmip5``. The one-click ``samy-run`` path remains
CMIP6-based.

For a direct Python call, use the same sequence as the pipeline:

.. code-block:: python

   from pathlib import Path
   from samy_wgen.pipeline.run_climate_delta import (
       load_cmip6_projection, load_rsvar_output,
       compute_mixed_resolution_baseline,
       apply_mixed_resolution_climate_delta,
   )

   daily, monthly, meta = load_cmip6_projection(
       Path("data/CMIP6"), city="Miami", ssp="ssp245"
   )
   baseline = compute_mixed_resolution_baseline(
       daily, monthly, 1991, 2010, var_resolution_config={}
   )
   rsvar, _ = load_rsvar_output(Path("output/Miami/3_rsvar.xlsx"))
   adjusted, stats = apply_mixed_resolution_climate_delta(
       rsvar, daily, monthly, baseline, gcm=meta["model_name"],
       rcm=None, lat=25.7617, longitude=-80.1918,
   )

The numeric coordinates above are for the Miami example. For another city,
read latitude and longitude from the bundled city table rather than reusing
them. See
:py:func:`~samy_wgen.pipeline.run_climate_delta.load_cmip6_projection`,
:py:func:`~samy_wgen.pipeline.run_climate_delta.compute_mixed_resolution_baseline`,
and :py:func:`~samy_wgen.pipeline.run_climate_delta.apply_mixed_resolution_climate_delta`.
