# CondiS censored-lifetime imputation

`condis_impute` implements the base CondiS method in MD Anderson catalog entry 157,
[CondiS](https://biostatistics.mdanderson.org/shinyapps/CondiS/).
The method is described by Wang, Flowers, Li and Huang (2022),
*Journal of Biomedical Informatics* 131:104117,
[doi:10.1016/j.jbi.2022.104117](https://doi.org/10.1016/j.jbi.2022.104117).
The [CRAN package](https://cran.r-project.org/package=CondiS) version 0.1.2 was
reviewed and executed for numerical comparisons.
[Source provenance](condis-sources.json) records the archive. Original R programs
and data are not redistributed.

## Calculation

Let c be an observed right-censoring time, S the estimated survival curve, and h
the chosen horizon. For c<h, the imputation is

`c + integral(c, h, S(t) dt) / S(c)`.

Observed event times are preserved exactly. A censoring time at or beyond h is
also preserved. The default horizon is maximum observed follow-up. Explicit
horizons must be nonnegative and no greater than that maximum; no unsupported
tail extrapolation is performed. This is a restricted conditional mean, not an
estimate of an unrestricted lifetime beyond available follow-up. Preserving times
past h follows the native function; this does not truncate every subject to h.

The underlying Kaplan–Meier calculation reuses `exploratory_survival` with exact
ties grouped and tied censorings included in the risk set for tied events.
Inputs are complete vectors with nonnegative finite times and status
`1=event, 0=right-censored`. Input order is retained in the output.

Two interpolation conventions are available:

- `interpolation="linear"` (default): straight lines connect right-continuous
  Kaplan–Meier values at distinct observed times, matching the R function's
  `approxfun` convention. Areas are evaluated analytically by trapezoids.
- `interpolation="step"`: integrate the actual right-continuous Kaplan–Meier
  step function using rectangles. This is an explicit alternative to native
  interpolation; the two methods need not yield the same imputed times.

One sorted curve and reverse cumulative interval integrals handle every censored
observation. Time widths are normalized by the horizon before summation, with
normalization after subtraction to preserve close time differences. Complexity is
O(n log n) for sorting and O(n) storage; no per-subject adaptive integration or
n-by-n matrix is needed. Up to one million observations are supported.

## Usage

```python
import numpy as np
from mdanderson_stats import condis_impute

fit = condis_impute([1, 2, 4, 6], [1, 0, 1, 0])
np.testing.assert_allclose(fit.imputed_time, [1, 4.5, 4, 6])
step = condis_impute([1, 2, 4, 6], [1, 0, 1, 0], interpolation="step")
np.testing.assert_allclose(step.imputed_time, [1, 5, 4, 6])
restricted = condis_impute([1, 2, 4, 6], [1, 0, 1, 0], horizon=3)
np.testing.assert_allclose(restricted.imputed_time, [1, 2.875, 4, 6])
```

`CondiSImputation` retains observed times, binary status, imputed times, added
remaining times, horizon, interpolation mode and the fitted survival curve.
Array results are read-only. Added remaining time is zero for observed events and
for censoring at/beyond the horizon. Degenerate one-time samples and all-censored
samples have well-defined restricted results: all-censored subjects before the
horizon receive that horizon. These degenerate cases are not a native runtime
parity claim.

The estimated curve assumes independent censoring for the population being
analyzed. Imputed targets are deterministic estimates, not observed failures and
not multiple-imputation draws. For predictive evaluation, estimate imputations
within the training data/folds rather than using held-out survival outcomes to
construct training targets. This function does not provide a fitted transform
for applying a training curve to a separate held-out sample.

## Validation and remaining coverage

Three focused tests cover hand-computed linear/step integrals and a partial
horizon, unmodified R 0.1.2 outputs with tied and unsorted inputs, preservation of
events, degenerate samples, input order and time scales from 1e-200 to 1e200.
Native adaptive integration differs from exact segment sums by approximately
1.14e-4 on the selected fixture; the reference comparison allows that integration
error. Python does not reproduce adaptive integration noise. R survival's default
near-tie handling is also not reproduced: Python groups exact ties.

A 100,000-observation example took approximately 0.033 seconds in the development
environment. The imputed times remained between observed times and maximum
follow-up. This is a local benchmark, not a cross-platform guarantee.

**Catalog status is partial.** Base CondiS imputation is implemented. CondiS-X's
eight covariate-refinement learners, their tuning/resampling behavior, interactive
input handling and native graphical reports remain pending.
