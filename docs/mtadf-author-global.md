# MTADF author-reference global logistic fit

This module covers the recovered author's global quadratic logistic fit and
fresh-cap dose recommendation. It is distinct from the paper-policy logistic
sampler in [`mtadf-logistic.md`](mtadf-logistic.md): the author fit uses a
specific adaptive prior-scale IRLS procedure from `arm::bayesglm.fit`, not an
ordinary Cauchy-prior MAP optimizer.

```python
from mdanderson_stats.mtadf_author_global import (
    mtadf_author_global_decision,
    mtadf_author_global_fit,
)

subjects = [3, 6, 9, 0, 0]
responses = [0, 3, 8, 0, 0]
toxicities = [0, 0, 1, 0, 0]

fit = mtadf_author_global_fit(subjects, responses)
next_dose = mtadf_author_global_decision(subjects, toxicities, responses, current_dose=1, fit=fit)
obd = mtadf_author_global_decision(subjects, toxicities, responses, final=True, fit=fit)
print(fit.coefficients, fit.converged, next_dose.dose, obd.dose)
```

Dose indices are zero-based in Python. The fit uses the full ordinal grid
`x=(1:J-mean(1:J))/(2*sample_sd(1:J))` and columns `(1, x, x²)`. Counts are
grouped internally, but the first IRLS step preserves the source's expanded
Bernoulli initialization. Prior scales use the patient-expanded column
distribution; a constant column keeps its base scale, two distinct values
divide it by their range, and more than two divide it by twice their
patient-weighted sample SD. The scalar default prior scale expands to 2.5 for
all model columns in the inspected implementation, including the intercept.

The fit returns `coefficients`, the full-grid `fitted_efficacy`, prior scales
and their final adaptive working values, deviance, iteration count, and a
convergence flag. The coefficient update follows the cached source's weighted
least-squares IRLS with an intercept prior row centered at the observed design
column means. The finite-df working scales update from centered coefficient
estimates, approximate sampling variances, and the base prior scale. The
decision withholds a recommendation if the fit has not converged.

The decision recomputes the source's fixed beta-binomial toxicity posterior
and unweighted increasing isotonic safety adjustment for each call. It uses
the rightmost maximum of fitted efficacy. Interim movement is one dose toward
that peak, then clipped by the fresh admissible prefix. `final=True` selects
the peak directly before applying the cap; this is a Python convenience over
the source function's single current-dose interface. The private
`_author_global_next_dose` helper accepts a supplied cap so the source
simulator can preserve its lagged-cap ordering.

Inputs are bounded to 2–20 dose levels and 10,000 patients. The deterministic
fit uses at most 100 IRLS iterations and never expands the full patient-level
design in Python. A nonconverged fit remains inspectable but cannot drive this
decision API. The implementation and reference fixtures validate source
algorithm details; they do not claim exact app package-version, emitted
output, or random-stream parity. The
[source audit](../research/mtadf-author-global-audit.md) and
[independent reference](../tools/reference_mtadf_author_global.R) document
the recovered contract and fixture regeneration.

When only one dose has observations, the model's fitted curve is mathematically
flat. Python sets both non-intercept coefficients to exactly zero in this
case, preventing QR roundoff from changing the rightmost-peak decision. This
explicit numerical convention agrees with the inspected one-dose R reference;
it does not apply a tolerance to other peak comparisons.
