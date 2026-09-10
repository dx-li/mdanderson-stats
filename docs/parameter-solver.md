# Parameter Solver

Independent Python implementation of the numerical workflows in MD Anderson's
[Parameter Solver](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/6)
(version 3.2.2) and its March 2026
[user guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/ParameterSolver/ParameterSolverUsersGuide.pdf).
Source provenance is recorded in [parameter-solver-source.json](parameter-solver-source.json).

All six distribution families support the guide's three input modes: parameters,
mean plus variance (or standard deviation), and two quantiles. Density evaluation
provides the values needed for plots; Windows forms, plot controls and clipboard
operations are not reproduced. No native optimizer iteration parity is claimed.

| Family string | Parameter 1 | Parameter 2 |
| --- | --- | --- |
| `beta` | alpha | beta |
| `gamma` | shape | scale |
| `inverse_gamma` | shape | scale |
| `normal` | mean | variance |
| `lognormal` | mean of log(X) | standard deviation of log(X) |
| `weibull` | shape | scale |

Gamma has mean `shape * scale`. Inverse gamma has density proportional to
`x**(-shape-1) * exp(-scale/x)`. The reciprocal of Gamma(shape, scale) is
InverseGamma(shape, 1/scale). Weibull survival is `exp(-(x/scale)**shape)`.
The normal distribution's second parameter is a **variance**; the lognormal
second parameter is a **standard deviation**, following the guide's equations.

```python
import numpy as np
from mdanderson_stats import (
    ParameterDistribution,
    solve_distribution_moments,
    solve_distribution_quantiles,
)

# Guide: P(X <= 0.1) = 0.025, P(X <= 0.4) = 0.975.
fitted = solve_distribution_quantiles("beta", [0.1, 0.4], [0.025, 0.975])
np.testing.assert_allclose([fitted.parameter1, fitted.parameter2], [6.672, 22.036], atol=0.0006)
np.testing.assert_allclose(fitted.cdf([0.1, 0.4]), [0.025, 0.975], rtol=2e-8)

moments = solve_distribution_moments("beta", mean=0.4, variance=0.01)
np.testing.assert_allclose([moments.parameter1, moments.parameter2], [9.2, 13.8])
normal = solve_distribution_moments("normal", mean=3, standard_deviation=2)
assert normal.parameter2 == 4

specified = ParameterDistribution("beta", 7, 3)
assert specified.mean == 0.7
quantiles = specified.quantile([0.025, 0.5, 0.975])
x = np.linspace(0, 1, 101)
density = specified.pdf(x)  # Can be passed directly to a plotting library.
```

`mean` and `variance` are scalar properties. `pdf`, `cdf`, `sf`, and `quantile`
accept scalars or arrays. `sf` evaluates the upper tail directly. A mathematically
infinite endpoint density is retained. Inverse-gamma mean is infinite for shape
at most one; variance is infinite for shape at most two. Moment fitting requires
finite, positive variance and finite mean. Supply exactly one of `variance` and
`standard_deviation`; beta additionally requires `0 < variance < mean*(1-mean)`.

## Numerical behavior and validation

Moment conversions use analytic identities except Weibull's shape, which uses a
bracketed root. Gamma and inverse-gamma quantile fitting solve for shape from the
log quantile ratio, then recover scale. Beta solves for total concentration with
an inner solve for the shape ratio. Normal, lognormal and Weibull quantile fits
are analytic. These are independent algorithms, not translations of native code.

Numerical shape/concentration searches use log parameters in `[-20, 30]`; beta's
inner logit bracket is `[-700, 700]`. A solution outside these brackets, a failed
root or an unrepresentable positive result raises `ArithmeticError`. The solvers
verify recovered moments or the smaller probability tail to relative tolerance
`2e-8`. Inputs must contain two increasing values and two increasing probabilities
strictly between zero and one. Positive families require positive values and
beta requires values below one. Unsupported or invalid inputs raise `ValueError`.

Calculations use log scales, `log1p`/`expm1`, direct upper tails and a log-gamma
series for concentrated Weibull variances. Inverse incomplete-gamma underflow
during shape search uses the small-quantile log asymptotic only below log-quantile
-25; final fitted tail probabilities must still pass verification. Float64 limits
remain: for example, lognormal forward evaluation requires its scale `exp(log_mean)`
to be representable, and interior quantiles that round to support endpoints are
rejected. Exact endpoint probabilities zero and one are accepted for quantile
queries. Density/CDF inputs must be finite.

Eight focused tests cover the guide's beta quantile and moment examples, both
parameter-recovery modes for all six families, gamma/Weibull exponential identities
at scales from `1e-100` to `1e100`, an inverse-gamma analytic identity, undefined
inverse-gamma moments and Weibull variance `1e-12` at mean one. They test numerical
behavior rather than reproducing desktop UI details.
