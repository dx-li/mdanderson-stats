# CI of Interaction Index and SYNERGY

This shared numerical core implements the median-effect/Loewe calculations in
Lee and Kong (2009), *Statistics in Biopharmaceutical Research* 1:4–17,
[doi:10.1198/sbr.2009.0001](https://doi.org/10.1198/sbr.2009.0001).
The [CI of Interaction Index readme](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/CIInteractionIndex/ReadmeCI.pdf)
identifies observed-combination and fixed-ray delta-method calculations. The
[SYNERGY catalog entry](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/18)
also cites this paper. [Provenance](interaction-index-sources.json) records the
public readme and paper text; original programs are not redistributed.

## Available calculations

`fit_median_effect(dose, effect)` fits `logit(effect) = intercept + slope*log(dose)`
by centered ordinary least squares. Replicates are individual rows. It returns
coefficient covariance, residual variance and observation count. Assumptions are
independent, homoscedastic normal errors on the transformed response. Doses must
be positive and effects strictly inside (0,1); there is no clipping or pseudocount.
Positive and negative slopes are supported, but a zero slope cannot be inverted.
`MedianEffectFit.effect` predicts effects and `.log_dose` inverts the fitted curve.
A `MedianEffectFit` may also be constructed from externally estimated coefficients
and covariance in intercept/slope order.

`interaction_index(models, doses, effect, effect_variance=...)` computes the
Loewe index as the sum of component dose divided by the corresponding single-drug
dose producing the same effect. The last dose axis enumerates drugs. The
log-delta interval propagates coefficient covariance and effect uncertainty,
assuming independent fits and combination measurements. `effect_variance` is
explicitly the variance of the supplied **mean effect**; divide replicate variance
by replicate count when appropriate. Zero treats the effect as known. No pooled
variance is substituted when replication is absent.

`interaction_index_ray(models, combination, proportions, effect)` estimates the
index along a fixed composition. Fit `combination` against **total dose**; its
coefficient uncertainty is included. Positive proportions are normalized in log
space. Component dose units must be consistent with the fitted total dose.
All curves must have the same response direction.

Both functions return log index, log-index standard error, log confidence limits,
confidence level and residual degrees of freedom. Intervals use Student t with
the sum of `(observations-2)` across the independent regressions involved.
`.index` and `.interval` exponentiate on demand; they raise if the result cannot
be represented, while the log outputs remain available. Fixed-ray intervals are
pointwise, not simultaneous bands. An interval wholly below one supports synergy;
wholly above one supports antagonism. An interval containing one does not prove
additivity. These are approximate delta-method intervals, not exact finite-sample
coverage guarantees.

```python
import numpy as np
from mdanderson_stats import MedianEffectFit, interaction_index, interaction_index_ray

# The paper's decreasing-response three-drug example: median doses 1, 2, 4.
medians = np.array([1.0, 2.0, 4.0])
curves = [MedianEffectFit(np.log(d), -1.0, np.zeros((2, 2)), 10, 0) for d in medians]
result = interaction_index(curves, medians / 3, [0.25, 0.5, 0.75], effect_variance=0)
np.testing.assert_allclose(result.index, [1 / 3, 1, 3])

# An additive fixed ray: component proportions 1:2:4, total median dose 7/3.
combination = MedianEffectFit(np.log(7 / 3), -1.0, np.eye(2) * 0.001, 12, 0.01)
ray = interaction_index_ray(curves, combination, [1, 2, 4], [0.2, 0.5, 0.8])
np.testing.assert_allclose(ray.index, 1)
assert ray.degrees_of_freedom == 34
assert np.all(ray.interval[:, 0] < 1)
assert np.all(ray.interval[:, 1] > 1)
```

## Monte Carlo comparator

`interaction_index_monte_carlo(models, combination, proportions, effect, samples=2000, rng=...)`
implements the normal-coefficient procedure in paper section 3.1. It draws each
intercept/slope pair jointly with its fitted covariance, independently across
curves, and retains the coefficient draws in the result. The reported standard
error is `sqrt(mean((sampled_index - fitted_index)**2))`, as printed in the paper.
It is not centered on the Monte Carlo mean and does not use a samples-minus-one
denominator. The interval is fitted index plus/minus a critical value times this
RMS deviation. `critical_distribution="t"` follows the paper's small-sample
comparisons; `"normal"` uses its original normal critical value.

These untransformed intervals may have negative lower limits, which are retained.
Gaussian slopes are not truncated or resampled: `slope_reversal_fraction` reports
sign reversals for each drug and the combination. Draws near zero slope can create
extreme indices and unstable Monte Carlo uncertainty. Unrepresentable results
raise instead of dropping draws. A finite run does not establish existence of
population moments under this approximation. The existing log-delta interval is
available separately.

Calculation uses bounded sample/effect blocks and log-space squared deviations.
The same coefficient draws are used across the effect grid. Inputs are capped at
20 million sample-effect cells and four million coefficient cells. Fixed seeds
reproduce a fixed call; native software random streams are not reproduced.

## Validation and remaining coverage

Five focused tests cover centered regression against hand-computed coefficients
and covariance, dose-unit rescaling, the paper's three-drug index identity,
finite-difference uncertainty propagation for observed and fixed-ray combinations,
Student-t degrees of freedom, invalid covariance, and indices beyond the range
of ordinary exponentiation. The implementation uses array reductions and
log-sum-exp; covariance quadratic forms use a positive-semidefinite square root.

A local batch of 200,000 three-drug observed-combination calculations, including
confidence limits, took about 0.057 seconds. This is a development-machine
measurement, not a cross-machine guarantee.

Three additional Monte Carlo tests reproduce the printed RMS formula directly
from retained draws, compare small-uncertainty results to the delta method, check
zero covariance and seeded replay, and retain reversed slopes and negative limits.

**Both catalog entries remain partial.** CI of Interaction Index still needs its
median-effect plots, source case-study fixtures and native
workflow audit. SYNERGY additionally needs its other response-surface models,
semiparametric methods and associated workflows. Pooled measurement-error
estimation and native file/report conventions have not been silently inferred.
