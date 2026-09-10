# SURVAN logistic regression

`survan_logistic` fits binary logistic regression, reusing the package's existing
STUKEL fixed-zero-shape logistic kernel, analytic derivatives, separation checks
and information covariance. No new optimizer or dependency is introduced.

```python
import numpy as np
from mdanderson_stats import survan_logistic

x = np.r_[np.zeros(19), np.ones(21)]
event = np.r_[np.ones(17), np.zeros(2), np.ones(19), np.zeros(2)]
fit = survan_logistic(x, event)
assert np.allclose(fit.predict([0, 1]), [17 / 19, 19 / 21])
print(fit.coefficients)  # [0.111225..., 2.140066...]: slope, intercept
```

Input x is observations by covariates; a vector means one covariate. Outcomes
must be explicitly zero or one. To use SURVAN's event-list or threshold selection,
form that indicator before fitting, for example `np.isin(status, [1, 3])` or
`response >= threshold`. Other observations are non-events; this logistic model
does not interpret outcomes as right-censored survival data.

`intercept=True` adds an intercept by default. Coefficients, covariance and
standard errors follow the native report order: **slopes first, intercept last**.
The result also exposes fitted probabilities, negative log likelihood, null
negative log likelihood, likelihood-ratio statistic, df, p-value and optimizer
iteration count. Arrays are immutable; `fit.predict(new_x)` uses the saved
scaled design to avoid unnecessarily reconstructing predictors in extreme units.

The native coefficient p-values are `one_sided_pvalues = Phi(-abs(beta / SE))`.
They are the smaller normal tail, not a p-value for a prespecified positive or
negative alternative. `two_sided_pvalues` doubles this value. Covariance uses
fixed binomial dispersion, not Pearson overdispersion. These are asymptotic
information-based summaries.

With an intercept, the likelihood-ratio null is the fitted constant probability.
Without an intercept, the nested null fixes all coefficients to zero, giving
probability 1/2. The native reporting code instead always compares to a fitted
constant probability and takes an absolute likelihood difference, even when the
models are not nested. The Python no-intercept omnibus test corrects that
comparison. Both cases use df equal to the number of covariates.

## Numerical behavior and limits

Covariates are first divided by their maximum absolute values and, when an
intercept is present, centered in those normalized units. The fit's coefficients
and covariance are transformed back to original units. This supports small and
large covariate scales without losing their columns in a raw-design rank check.

The shared kernel uses stable logistic/log-likelihood calculations rather than
SURVAN's exponential cutoff below approximately −18.42. It rejects deficient
designs and complete/quasi-complete separation. Failed optimization, unavailable
information covariance, and coefficients/covariance outside representable ranges
raise errors. The shared optimizer has a ±1e20 bound in scaled coefficient units;
an active bound cannot produce a successful inference result. Numerical solver
paths differ from native DMNH.

Requires positive residual degrees of freedom, 1..100 covariates, at most 100,000
records and 2,000,000 design entries. Inputs must be finite. Predictions must
have the same number of covariates; unrepresentable linear predictors raise an
error. This interface covers individual binary observations, with no case
weights, offsets, grouped-binomial trial counts or automated variable selection.

## Validation

The original LGRLL and LGRGH routines and CKEXP helper were compiled unchanged.
A reference harness supplied observation iteration, a scratch allocator and the
already-binary event indicator. They were evaluated at Python's fitted parameters
on a generated 100-record, three-covariate dataset both with and without an
intercept. Original likelihoods agree within 1e-10 absolute tolerance; native
score components are below 2e-6, and original information times Python covariance
agrees with identity within 1e-9. This checks the original estimating equations,
not parity with the historical optimizer's iteration path.

The manual's two-group example also agrees with the exact binomial-group fit:
intercept log(17/2), slope log(19/17), negative log likelihood 12.99775562100,
and overall p-value approximately 0.9159926865. Additional checks exercise
prediction, covariate units 1e-100 and 1e100, deficient design and separation.

[Source provenance and remaining SURVAN coverage](survan.md).
