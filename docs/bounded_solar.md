# Bounded solar radiation in RS-VAR

`samy-rsvar` and `samy-run` now use this solar representation by default.
`samy-delta` requires the two saved latent columns whenever its RS-VAR input
contains solar radiation. The output still has hourly GHI, DNI, and DHI.

## Historical fit

The input index is fixed local standard time. City latitude **and longitude**
determine the solar zenith using the same approximation as the ERA5/NSRDB
calibration code. Let `I0` be approximate extraterrestrial normal irradiance
and `c = max(cos(zenith), 0)`. At accepted daylight hours:

* `k = GHI / (I0 c)` is the clearness index.
* `b = DNI c / GHI` is the horizontal direct fraction.

The input screen requires finite, nonnegative components, `GHI = DHI + DNI c`
within 1 W/m², GHI below `I0 c + 1 W/m²`, DNI below `I0 + 1 W/m²`, and `b`
within a 0.01 rounding tolerance of `[0, 1]`. Rejected hours are excluded
from the **joint** weather and solar RS-VAR fit. Valid exact 0/1 endpoints are
bounded only to give a finite logit. The model fits `logit(k)` and `logit(b)`
alongside temperature, pressure, wind, and humidity.

After inverse-logit, `GHI = k I0 c`, `DNI = k b I0`, and `DHI = GHI - DNI c`.
All components are zero at night. These equations enforce nonnegativity,
the selected upper envelope, and component closure. Twelve GHI and twelve DNI
logit shifts then match screened historical monthly means using the first
20 generated years (or the explicitly supplied generated calibration years).
The monthly calibration is a fit to historical input, not independent
validation.

## Projection signal

For each forecast month, the default solar change factor is the mean of that
calendar month's projection `rsds` over the trailing 20 years, divided by
its 1991–2010 mean. This separates slowly changing projection radiation from
the generator's hourly weather. One common shift of that month's clearness
index logits is solved so the sum of hourly GHI equals the generated monthly
GHI sum times the factor. If the requested energy exceeds the available `I0 c`
energy, the run raises an error. It does not clip an overlarge GHI afterward.
The saved `*_solar_monthly_energy.csv` lists targets and achieved sums.

Non-solar climate deltas use the existing pipeline. `rsds` is not also
multiplied into hourly GHI by that pipeline. Legacy RS-VAR files containing
GHI/DNI/DHI without `solar_kt_logit` and `solar_beam_logit` are rejected with
an instruction to regenerate Stage 3.

The reusable implementation is in `samy_wgen.utils.solar_latent`, and
`samy_wgen.pipeline.solar_latent.fit_solar_latent_rsvar` can be used directly
from Python. The standard `samy-rsvar`, `samy-delta`, and `samy-run` paths call
these functions automatically.

## Limits

The selected `k <= 1` envelope excludes real short cloud-enhancement events
that may exceed the approximate extraterrestrial horizontal reference. The
solar-position approximation omits the equation of time. Monthly change
factors and baseline mean calibration do not establish future solar extremes,
cloud sequencing, or intervariable dependence. A completed physical QA is
therefore a candidate check, not observational validation of future weather.
