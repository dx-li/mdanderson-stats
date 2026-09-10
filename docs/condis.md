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

## Default CondiS-X linear refinement

`condis_linear_refine(imputation, covariates)` adds the native default `glm`
refinement. Its regression uses an intercept, the **censoring status column**,
and every supplied numeric covariate column to predict the base imputed times.
The native source includes status through its `pred_time ~ .` formula. After
fitting on all supplied rows, observed event times are restored unchanged.
Supply covariates in the same row order, with categorical factors explicitly
encoded as numeric columns.

The verified caret model uses a Gaussian family with identity link. There is no
hyperparameter to tune for this model: a scaled least-squares solve reproduces
the final full-sample fit without running repeated cross-validation. Native
resampling metrics are not returned. Column centering/scaling with an intercept
preserves fitted values; a rank-revealing solve handles dependent or constant
columns. Responses are scaled to support very small or large time units. The
implementation accepts up to 500 covariate columns and 2 million covariate cells.

```python
from mdanderson_stats import condis_linear_refine

base = condis_impute(
    [8, 1, 2, 2, 4, 6, 10, 10, 12, 15],
    [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
)
refined = condis_linear_refine(base, np.zeros((10, 1)))
assert refined.below_censoring[-1]
bounded = condis_linear_refine(base, np.zeros((10, 1)), enforce_censoring=True)
assert bounded.refined_time[-1] == 15
```

`CondiSLinearRefinement` retains raw `fitted_time` for all rows, `refined_time`
after event restoration and optional clipping, raw censored-prediction flags
`below_censoring` and `above_horizon`, design rank, residual degrees of freedom,
and training residual RMSE against base imputed targets. This RMSE is not an
out-of-sample prediction error or uncertainty estimate.

As in native CondiS-X, unconstrained fitted times can be negative, below a known
censoring time, or above the base horizon. The default returns those predictions
with diagnostic flags. `enforce_censoring=True` is an explicit Python extension
that clips censored predictions at their observed lower bound. It does not cap
them at the horizon, refit the model, or change the raw predictions/flags. This is
a refinement of the supplied sample, not a deployable survival prediction model:
future censoring status is generally unavailable for new patients.

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

Three further tests cover the executed caret Gaussian model, event restoration,
censoring diagnostics and explicit clipping, and rank/scale invariance. The
comparison isolates the native final fit using exact base targets; it does not
claim reproduction of caret's full resampling pipeline.

**Catalog status is partial.** Base CondiS and the default linear CondiS-X refinement are implemented. The
seven other CondiS-X learners (ridge, lasso, GBM, random forest, SVM, kNN, neural
network), their tuning/resampling behavior, interactive
input handling and native graphical reports remain pending.
