# Semiparametric two-drug response surfaces

`fit_synergy_surface` implements the baseline-plus-spline method of
[Kong and Lee (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5096313/), listed
in [MD Anderson SYNERGY](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/18).
It fits the additive baseline from single-drug observations, then estimates
departure from that baseline with a natural bivariate thin-plate spline.
The separate [interaction-index functions](interaction-index.md) provide the
median-effect/Loewe index calculations and their intervals. The recovered
[parametric surfaces](synergy-parametric.md) provide the other four author models.

Supply the already transformed response `Y=g(E)`. The library does not select
a response transformation or back-transform fitted values. Baseline, departure
and total predictions are all on that same scale. When increasing either drug
decreases the response, a negative departure indicates synergy and a positive
departure indicates antagonism; the direction reverses for increasing responses.
Point estimates alone do not provide evidence of statistical significance.

## Fit and predict

```python
import numpy as np
from mdanderson_stats import fit_synergy_surface, predict_synergy_surface

# Single-drug, combination and repeated-dose observations.
index = np.r_[np.arange(9), 0, 2, 4, 8]
dose1 = np.tile([0.0, 0.5, 1.0], 3)[index]
dose2 = np.repeat([0.0, 0.75, 1.5], 3)[index]
response = (
    1.1
    - 0.2 * dose1
    - 0.15 * dose2
    - 0.3 * dose1 * dose2
    + 0.025 * np.sin(np.arange(1, len(index) + 1))
)
fit = fit_synergy_surface(dose1, dose2, response, baseline="raw")
prediction = predict_synergy_surface(fit, [0.2, 0.7], [0.6, 1.2])
print(prediction.baseline)
print(prediction.surface)  # departure from the additive baseline
print(prediction.response)  # baseline plus departure
print(fit.smoothing_parameter, fit.smoothing_at_boundary)
```

`baseline="raw"` fits a common-intercept linear model in the two doses using
only rows where at least one dose is zero. The marginal design must identify
the intercept and both slopes. Repeated observations remain separate rows.

`baseline="log"` fits separate marginal lines in log dose. Each drug needs
at least two distinct positive marginal doses, and the fitted slopes must be
nonzero and have the same sign. The additive combination baseline solves
the paper's implicit varying-relative-potency equation with stable log sums.
There is no guessed dose offset: the log baseline is undefined when both
doses are zero, and predicting at that point raises an error.
Such a training row contributes a zero spline pseudo-response and has `NaN`
for its reported baseline and total fitted response; its observed response
does not enter either marginal log-dose regression.

## Surface and smoothing

The spline is fit to `Y-baseline` at combination doses and zero at marginal
doses, using distinct observed dose pairs as knots. This zeroes the *input
residuals* on the axes; it does not force the fitted surface to be exactly zero
everywhere along them. The thin-plate kernel is
`r**2 * log(r**2) / (16*pi)`, with value zero at `r=0`, and radial coefficients
obey the affine-nullspace constraint. Dose units therefore affect the surface
geometry and smoothing parameter; use a scientifically appropriate scale.

By default, mixed-model REML selects a positive smoothing parameter and BLUP
gives the fitted surface. Selection screens 61 `log(lambda)` values in
`[-30,30]`, refines candidate minima and compares them with the endpoints.
It returns optimizer status, evaluation count and `smoothing_at_boundary`.
This bounded search does not guarantee a global optimum. Check
`optimizer_success` for any local refinement that failed to converge.
A boundary estimate signals that the finite search range affects the result.
Supply `smoothing_parameter=<positive value>` to condition on a chosen value
and skip optimization. That value multiplies the roughness penalty in the
sum-of-squared-residuals criterion, without an extra observation-count factor.
Residuals that are affine to numerical precision have a degenerate residual
variance estimate and do not identify smoothing from the data. This API rejects
them even when lambda is supplied.

The implementation rescales response and affine dose columns for calculation.
Returned predictions are on the supplied response scale. The fit object's
coefficient arrays and `residual_variance_scaled` retain the internal scale;
multiply that variance by `response_scale**2` to obtain response-scale variance.
For raw-dose baseline coefficients, multiply the intercept by `response_scale`
and each slope by `response_scale/dose_scale[j]`. The REML objective is also
reported on the internally scaled response.

## Scope and validation

Fits accept at most 500 observations, subject to combined matrix-storage and
work limits. Prediction uses bounded kernel chunks and checks the requested
grid before expansion. Results are readonly; invalid or numerically unsupported
fits raise errors rather than substituting fabricated values.

Independent base-R calculations solve an augmented penalized spline system
with duplicate dose pairs and profile REML through full covariance matrices.
They agree with the Python surface and smoothing calculations using different
linear algebra. The [source audit](../research/synergy-response-surface-audit.md)
records the reference fixtures and remaining source uncertainties.

SYNERGY remains **partial**. The four 2007 parametric surfaces, wild-bootstrap
intervals, native reports/plots and original case-study reproduction remain
open. The [wild-bootstrap resampling workflow](synergy-surface-bootstrap.md)
implements the documented residual, multiplier and refitting steps. It returns
departure draws and ordinary sample standard deviations as an explicit Python
summary convention. The source interval's exact standard-error centering and
denominator have not been verified, so no native interval is claimed.
