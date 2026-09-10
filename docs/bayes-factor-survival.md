# Bayes Factor TTE: posterior monitoring and boundaries

This is an independent Python implementation of the numerical monitoring model in
MD Anderson's [Bayes Factor TTE](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/89).
The [version 1.0.0 guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/BayesFactorTTE/UsersGuide_BayesFactorTTE.pdf)
is dated May 21, 2012. [Source provenance](bayes-factor-survival-source.json) records
the guide used. Original software is not redistributed.

## Model

Patients have independent exponential survival times, with independent censoring.
For `events = d` observed events and `total_time = T` summed follow-up time,
the likelihood of mean survival theta is proportional to
`theta**(-d) * exp(-T/theta)`. Follow-up contributes both for patients with an
event and for censored patients. Total time is neither the calendar duration
of the trial nor a sum restricted to patients with events.

Inputs specify **medians**, as in the desktop program. The exponential mean is
`median/log(2)`. Under H0 the mean is fixed at theta0; under H1 it has a one-sided
iMOM prior on `(theta0,infinity)`, with `k=1`, `nu=2`, and
`tau=1.5*(theta_mode-theta0)**2`. Its normalized density is
`2*tau/(theta-theta0)**3 * exp(-tau/(theta-theta0)**2)`.
Equal prior model probabilities are assumed.

```python
import numpy as np
from mdanderson_stats import bayes_factor_survival, bayes_factor_survival_boundaries

state = bayes_factor_survival(
    events=5,
    total_time=[1, 20, 80],
    null_median=4,
    alternative_median_mode=5.5,
)
np.testing.assert_array_equal(state.decision, ["inferiority", "continue", "superiority"])

boundaries = bayes_factor_survival_boundaries(
    events=[0, 5, 10], null_median=4, alternative_median_mode=5.5
)
np.testing.assert_allclose(
    boundaries.superiority_time, [18.840818, 59.240937, 96.978525], atol=1e-6
)
assert np.isneginf(boundaries.inferiority_time[0])
```

The monitoring function broadcasts event counts and total follow-up. It returns
log Bayes factors, posterior alternative probabilities and decisions. Inferiority
requires posterior probability strictly below `inferiority_cutoff` (default .15);
superiority requires probability strictly above `superiority_cutoff` (default .8).
Decisions use log odds to preserve extreme evidence. Set `final=True` to label
an unresolved result `inconclusive` instead of `continue`. Setting the lower
cutoff to zero or upper cutoff to one disables that stopping direction.

The boundary function solves for follow-up values at which the cutoffs are met.
For a fixed event count, inferiority requires `total_time < inferiority_time`;
superiority requires `total_time > superiority_time`. Negative infinity for the
lower boundary means inferiority is impossible. Positive infinity for the upper
boundary means superiority is disabled. An upper boundary of negative infinity
means superiority already holds at zero exposure. Boundaries are in the same
time unit as the input medians. They retain fractional precision; evaluating
at a numerically solved boundary can differ at the last few floating-point bits.

Event counts are integers from 0 to 500. Follow-up must be finite and nonnegative;
positive event counts require positive follow-up in the monitoring function.
Both medians must be finite and positive, with the alternative above the null.
The Python API permits arbitrary consistent time units and does not impose the
desktop UI's 24-month input ceiling.

## Numerical method and validation

The computation uses dimensionless total exposure `T*log(2)/null_median` and a
logarithmic prior coordinate. The likelihood ratio is evaluated in log space,
with stable `logaddexp`, logistic and `logsumexp` calculations. Composite
Gauss–Legendre orders double from 8 to at most 256 until successive log Bayes
factors differ by less than `2e-10`. An explicit lower bound on the integral
chooses tail truncations with relative omitted contribution at most `2*exp(-40)`
in exact arithmetic. The finite quadrature and floating-point error still require
numerical convergence checks; this is not an all-input error certificate.

A bracketed root solve obtains each exposure boundary and verifies its log-odds
residual. Quadrature uses at most 2,048 unit-width panels to bound memory.
Unresolved integration, unrepresentable ratios/boundaries or failed root checks
raise `ArithmeticError`. Invalid input raises `ValueError`.

Three focused tests compare the model with independent integration in the
original mean-survival coordinate, exercise all displayed guide boundary rows,
verify cutoff residuals and final decisions, check time-unit scaling by `1e-200`
and `1e200`, and exercise log Bayes factors above 300. The guide's integer-day
table differs by up to about 1.3 days from these continuous roots when using
365.25/12 days per month. Tests allow 1.5 days for that rounded native reference;
independent numerical integration is checked much more tightly. The native
program's day discretization/root approximation is not yet established, and
exact integer-day parity is not claimed.

## Remaining catalog coverage

**Catalog status is partial.** Posterior monitoring and continuous exposure
boundaries are implemented. The native calendar simulation, accrual mechanism,
observation-check schedule, stopping at maximum enrollment, text input and HTML
simulation report still require source audit and implementation. The guide lists
an accrual rate but does not fully specify these mechanics. The available model
functions do not silently assume a calendar simulation convention or reproduce
the native Monte Carlo report.

Archive retrieval was checked on September 10, 2026: the [version 1.1 download](https://biostatistics.mdanderson.org/SoftwareDownload/FileDownloader/Index/401) requires email, organization and occupation registration.
