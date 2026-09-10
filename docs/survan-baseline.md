# SURVAN Kalbfleisch–Prentice baseline survival

`survan_baseline(time, event, linear_predictor)` implements SURVAN's separate
HZC/FHZPT survivor calculation for fixed Cox coefficients. It estimates each
failure-time survival multiplier alpha, then multiplies these to obtain the
baseline survivor function. This is the Kalbfleisch–Prentice method, rather than
the usual Breslow cumulative-baseline-hazard estimator.

```python
import numpy as np
from mdanderson_stats import survan_baseline, survan_cox

x = np.array([0, 1, 0, 1, 0, 1])
time = [1, 2, 2, 3, 4, 4]
event = [1, 1, 0, 1, 1, 0]
fit = survan_cox(time, event, x)
eta = x * fit.coefficients[0]
curve = survan_baseline(time, event, eta)
profiles = curve.predict([0, fit.coefficients[0]])  # x=0 and x=1
assert profiles.shape == (2, len(curve.time))
```

The supplied predictors define the reference. `x @ beta` gives the native x=0
baseline; `fit.log_relative_hazard(x)` gives a baseline at the training covariate
means. Prediction profiles must use the **same** convention as estimation.
Coefficients can also come from another fitting procedure. No coefficient
estimation or uncertainty intervals are performed by this function.

Times must be finite and nonnegative, event must be zero or one, and predictors
must be finite. Vectors must match, with 1..100,000 records. Censors at a failure
time remain in the risk set. Output times include distinct failure times and an
initial zero when the first failure is later than zero. All-censored data return
a single time-zero row with survival one. No delayed entry, time-dependent
covariates or individual case weights are supported.

## Returned quantities

`curve.alpha` is the per-time survival multiplier at predictor zero;
`curve.hazard_probability` is the native reported 1−alpha, calculated with
`expm1` to avoid cancellation. `curve.survival` is their cumulative product,
computed in logarithmic form. All returned arrays are immutable.

`curve.predict(eta)` returns survival over the whole table with output shape
`eta.shape + (number_of_times,)`. At most 20 million prediction values are
allowed. The log hazard reference and log hazard increments/cumulative hazard
are retained internally as public diagnostic fields. These logarithmic fields
refer to `reference_log_hazard`, the largest supplied predictor, **not** directly
to predictor zero. This keeps prediction informative even when ordinary baseline
survival rounds to zero or one at extreme reference values. The public alpha,
hazard-probability and survival properties apply the reference adjustment.

## Defining equation and numerical treatment

For tied failure risks w_i=exp(eta_i) and total risk-set weight R, the source solves

sum over failures of w_i / (1 − alpha**w_i) = R.

For one failure the solution is alpha=(1−w/R)**(1/w). With equal risks, the
result reduces to Kaplan–Meier, including tied failures and censors. When every
remaining individual fails at a time, alpha is exactly zero.

The Python calculation works with normalized log weights and h=−R log(alpha).
It solves the equivalent equation

sum r_i / expm1(h r_i) = nonfailure risk fraction, where r_i=w_i/R.

The nonfailure risk is accumulated separately from failure risk, avoiding
subtraction of nearly equal totals. A bracketed root search operates on log(h),
with stable small/large-argument formulas; increments accumulate using
log-add-exp. Single-failure cases use the analytic solution. Ordinary survival
probabilities may legitimately round to zero or one, while the stored log
hazards preserve information for prediction at another profile.

This removes the source's approximate-equality shortcut that can set alpha to
zero when one failure's weight is merely close to R. It also avoids the native
search's hard upper endpoint 1−1e-5 and loose root tolerances. Unrepresentable
predictor spreads or inability to bracket a root raise errors.

## Source comparison

Original FHZPT, SETHAZ/HAZF, QMFIND and QRZERD were compiled unchanged; a small
harness provided risk weights and the error-reporting callback. No source-type
repair was required: the original HAZF entry has the correct implicit double
precision. At all 29 failure times in the worked example, using the fitted
Python coefficient, alpha differs by less than 1.2e-6 from the native roots.
The source's risk-sum equation is satisfied to relative 1e-11 by the Python roots.
This comparison isolates the baseline routine from the original optimizer.

Additional checks use an analytic tied-failure quadratic, exact KM reduction
under reference shifts of ±1000, a positive-survival case affected by the native
premature-zero shortcut, and a tiny failure risk relative to the risk set.
Source hashes and original notices are linked from [SURVAN coverage](survan.md).
