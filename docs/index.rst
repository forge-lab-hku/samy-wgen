samy-wgen
=========

Building energy simulation needs hourly weather at a specific site. Climate
projections are usually supplied at coarser spatial and temporal scales, while
a typical meteorological year combines selected historical months into one
representative year.
This leaves a gap for studies of long, continuous future weather sequences.

The `sAMY study <https://doi.org/10.1016/j.enbuild.2026.117508>`_ addresses
that gap by separating observation-learned hourly structure from the imposed
climate signal. This package implements that workflow: it prepares station
weather and solar observations, fits a regime-switching vector autoregressive
(RS-VAR) model, applies a selected climate projection, and exports hourly
weather for building simulation. The current CLI also offers regime-path
members; their spread has a narrower interpretation than uncertainty across
climate models or scenarios.

The package is under active development and is installed from a repository
checkout. Start with :doc:`getting-started`, then follow the modules below.

Workflow
--------

#. :doc:`workflow/download` — obtain the observational and projection inputs.
#. :doc:`workflow/prepare` — parse observations and align them to local standard time.
#. :doc:`workflow/rsvar` — fit RS-VAR, generate an hourly sequence or regime members.
#. :doc:`workflow/delta` — apply a scenario-conditioned climate signal.
#. :doc:`workflow/output` — inspect CSV, EPW, and DDY outputs.

For direct Python use, see :doc:`api/index`. Command-line options are in
:doc:`cli/index`.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting-started
   workflows
   api/index
   cli/index

Citation
--------

Guo, H. & He, K. (2026). Scenario-conditioned actual meteorological years
(sAMY): A stochastic weather generator using multi-decadal observations.
*Energy and Buildings*, **364**, 117508.
`doi:10.1016/j.enbuild.2026.117508
<https://doi.org/10.1016/j.enbuild.2026.117508>`_.
