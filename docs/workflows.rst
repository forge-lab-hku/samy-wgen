Workflow guide
==============

The default ``samy-run`` command carries out these stages in order. Each
page explains the input, transformation, and output of one stage and points
to the corresponding Python functions.

.. toctree::
   :maxdepth: 1

   workflow/download
   workflow/prepare
   workflow/rsvar
   workflow/delta
   workflow/output

The deterministic pipeline writes an hourly climate-adjusted series and
selected EPW/DDY files. The alternative member path is described alongside
the model fit in :doc:`workflow/rsvar`.
