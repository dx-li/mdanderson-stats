# SURVAN Cox proportional-hazards regression

`survan_cox(time, event, x)` fits multivariable proportional hazards using the
original SURVAN **Breslow tied-event partial likelihood**. This differs from the
existing two-arm IPDfromKM Efron fit. The model has no intercept.

```python
import numpy as np
from mdanderson_stats import survan_cox

fit = survan_cox(
    time=[1, 2, 2, 3, 4, 4],
    event=[1, 1, 0, 1, 1, 0],
    x=[0, 1, 0, 1, 0, 1],
)
log_hazard = fit.log_relative_hazard([0, 1])
assert np.isclose(log_hazard[1] - log_hazard[0], fit.coefficients[0])
```

Times must be finite and nonnegative; event is 1 for failure and 0 for right
censoring. A one-dimensional x means a single covariate, otherwise use an
observations-by-covariates matrix. Censors at an event time remain in its risk
set. Only exactly equal times are tied. Covariates are static throughout follow-up.

The result exposes coefficients in input column order, information covariance,
standard errors, one- and two-sided coefficient p-values, negative log likelihood,
null negative log likelihood, likelihood-ratio statistic, df, p-value and Newton
iteration count. The null sets every coefficient to zero. As in SURVAN's report,
`one_sided_pvalues` is the smaller normal tail `Phi(-abs(beta/SE))`, not a test of
a prespecified direction. `two_sided_pvalues` is twice that value. Inference uses
model-based information and asymptotic chi-square/normal approximations.

`log_relative_hazard(new_x)` returns log hazards relative to the training
covariate means. Differences between two returned values are log hazard ratios.
This avoids requiring finite exponentiated hazard ratios. Returned arrays are
immutable.

## Likelihood and numerical treatment

For each failure time t with d tied events, SURVAN contributes

sum(event linear predictors) − d log(sum(risk-set exponential predictors)).

The Python implementation uses this same likelihood and its analytic first and
second derivatives. Covariates are scaled and centered before fitting; estimates
and covariance are transformed back to original units. Constant columns and
unidentified risk-set effects are rejected.

For ordinary predictor ranges, cumulative weighted risk sums and matrix products
compute the score/information. When the predictor range exceeds 500, a blockwise
log-sum-exp and weighted-central-moment calculation rescales each growing risk
set separately. This avoids underflow from using a single global exponential
scale when an early risk set lacks the observation defining that scale.

Before fitting, a sparse linear program checks for monotone likelihood. An
auxiliary maximum for each risk set enforces that each failure's directional
predictor is at least every at-risk predictor, with equality between tied
failures. A positive directional likelihood improvement implies that no finite
maximum exists. This formulation uses O(records × covariates) sparse entries
instead of materializing every event/risk-set pair. The scaled LP objective
uses a numerical separation threshold of 1e-7.

A damped Newton solver then uses analytic information and likelihood-decreasing
steps. It requires squared Newton decrement below 1e-14 and stops with an error
if it cannot converge within 100 iterations. Numerically singular information,
failed line searches/LP checks, or unrepresentable coefficients and covariance
raise errors. Solver paths differ from the original David Gay optimizer.

Limits are 100,000 records, 1..100 covariates and 2,000,000 design entries. There
are no individual case weights, delayed entry, stratified baseline hazards or
time-varying covariates in this interface.

## Validation and remaining baseline workflow

The original PHLD routine was compiled unchanged, with a harness providing its
observation iterator, scratch allocation and small initialization/index helpers.
On generated 100-record, three-covariate tied-event data, the original likelihood
agrees within 1e-10 absolute tolerance, native score components are below 4e-11,
and native information times Python covariance agrees with identity within 1e-9.
The native optimizer itself is not claimed to have been reproduced.

The manual example gives negative log likelihood 100.7191290006, coefficient
−0.59589583 and standard error 0.34840413. The source manual reports −0.595902
and 0.348404, consistent with its optimizer precision. Additional checks cover
risk predictors separated by 2,000 log units, two independent moment-accumulation
paths, monotone likelihood, rank deficiency, and covariate units 1e-100 and 1e100.

SURVAN's separate HZC/FHZPT baseline survivor estimator uses the
Kalbfleisch–Prentice method. It is **still pending**; this interface does not
substitute the common Breslow cumulative-baseline-hazard estimator for it.
See [SURVAN coverage and source provenance](survan.md).
