3. Fit RS-VAR and forecast weather
==================================

The fit stage learns the seasonal-hourly baseline and multivariate weather
anomalies from ``2_isd_solar_lst.csv``. The regime-switching VAR (RS-VAR)
models lagged anomalies under fitted weather regimes. Its forecast is an
hourly sequence before any climate projection is applied.

Single-trajectory forecast
--------------------------

The default ``samy-run`` mode caches the result in ``3_rsvar.xlsx`` and
writes a solar calibration diagnostic beside it. The default fit baseline is
1991–2010, while the default forecast spans 2025–2100. Choose periods that
are covered by the prepared observations and projections.

.. code-block:: console

   samy-run --city Miami --baseline-start 1991 --baseline-end 2010 \
     --start 2025-01-01 --end 2050-12-31 --epw-years 2030,2050

To fit from an already prepared CSV without the one-click pipeline:

.. code-block:: console

   samy-rsvar --isd output/Miami/2_isd_solar_lst.csv --city Miami \
     --output output/RSVAR/rsvar_output_Miami.xlsx

The direct command uses its own forecast defaults. From Python,
:py:func:`~samy_wgen.pipeline.run_rsvar.run_and_save_rsvar` performs the fit
and writes the workbook. ``samy-delta`` can later reuse that workbook.

Ensemble forecast
-----------------

The alternative ``--mode ensemble`` uses the fitted regime model to sample
paths. Current members have zero VAR innovations, so their spread reflects
conditional regime-path variation. They are not climate-forcing or posterior
uncertainty intervals. The one-click ensemble branch checks for CMIP6 inputs
but does not apply CMIP6 deltas to its members.

.. code-block:: console

   samy-run --city Miami --mode ensemble --members 50 \
     --regime-mode ensemble --seed 0

This branch writes ``ensemble/member_*.parquet`` plus
``ensemble/ensemble_summary.xlsx``. It does not write EPW or DDY. From a
prepared CSV, use ``samy-ensemble`` directly:

.. code-block:: console

   samy-ensemble --isd output/Miami/2_isd_solar_lst.csv --city Miami \
     --out-dir output/Miami/ensemble --members 50 --seed 0

For Python use, see :py:func:`~samy_wgen.pipeline.run_ensemble.run_and_save_ensemble`
and :py:class:`~samy_wgen.model.anomaly_generator.AnomalyGenerator`.

Solar treatment
---------------

The current fit encodes observed GHI, DNI, and DHI as bounded clearness and
horizontal beam-share coordinates. It screens hours that violate the chosen
solar envelope or component closure, then reconstructs nonnegative,
closing irradiance after prediction. This is part of the joint model fit,
not a post-hoc clip. The `bounded solar note
<https://github.com/forge-lab-hku/samy-wgen/blob/main/docs/bounded_solar.md>`_
documents equations, calibration, and limits. Check the saved calibration
diagnostic when reviewing an output.
