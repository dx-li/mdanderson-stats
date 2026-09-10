# Inequality Calculator

Independent Python implementation of MD Anderson's
[Inequality Calculator](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/9),
using the March 2026 version 3.2
[user guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/InequalityCalculator/IneqCalcUsersGuide.pdf).
The original Windows program is not redistributed. Source provenance is in
[inequality-calculator-source.json](inequality-calculator-source.json).

The numerical scope covers all six families (beta, gamma, inverse gamma, normal,
lognormal and Weibull), with any finite additive shift. Variables are assumed
**independent** and must belong to the same family. Parameters use the
[Parameter Solver conventions](parameter-solver.md): normal's second parameter
is variance; lognormal's is the standard deviation of the logarithm.

```python
import numpy as np
from mdanderson_stats import ParameterDistribution, inequality_probability

# Guide's normal example.
x = ParameterDistribution("normal", 0, 1)
y = ParameterDistribution("normal", 1, 1)
result = inequality_probability(x, y)
np.testing.assert_allclose(result.x_greater, 0.2397500610934767)

# Guide's shifted beta example.
x = ParameterDistribution("beta", 6, 6)
y = ParameterDistribution("beta", 3, 3)
result = inequality_probability(x, y, delta=0.1)
np.testing.assert_allclose(result.x_greater, 0.342248, atol=5e-7, rtol=0)
assert abs(result.x_greater + result.shifted_y_greater - 1) < 2e-9
assert result.absolute_error < 1e-9
```

The result contains `x_greater = P(X > Y + delta)` and
`shifted_y_greater = P(Y + delta > X)`, calculated independently. The reverse
probability includes the shift. `absolute_error` is the larger estimated
absolute integration error, including a bound for truncated tails. `method`
identifies the calculation used. The scalar distribution parameters are suitable
for a single comparison; density evaluation on each input accepts NumPy arrays
for plotting. Desktop plot menus and clipboard operations are not reproduced.

## Calculation and numerical limits

Normal comparisons use the exact normal difference distribution, including the
shift. Unshifted lognormal comparisons use the normal difference of logarithms
without exponentiating the log-means. Unshifted gamma and inverse-gamma
comparisons use a beta-distribution identity with stable scale ratios; unshifted
equal-shape Weibulls use a logistic probability. Identical unshifted distributions
return one half. Unshifted beta comparisons reuse the package's existing logit
quadrature. Beta shifts with magnitude at least one have disjoint supports.

All other comparisons integrate the bounded function
`F_Y(Q_X(u) - delta)` over a uniform probability coordinate. Adaptive quadrature
uses breakpoints derived from Y's shifted quantiles and support; breakpoints
within floating-point resolution are coalesced without removing integration
mass. This avoids endpoint density singularities and helps resolve narrow
transitions. Both directions are integrated and checked for complementarity.

`absolute_tolerance` defaults to `1e-9` and must lie in `[1e-12, 1e-3]`.
Each probability-domain integral discards probability tails of length
`tolerance/16` at each end; their combined length bounds the omitted contribution.
Quadrature estimates are not rigorous floating-point error bounds. Closed-form
routes report zero **quadrature** error, not exact arithmetic. The tolerance is
absolute, so it does not guarantee relative accuracy for arbitrarily rare events.
A tail below float64 range can be zero.

Invalid inputs raise `ValueError` or `TypeError`. Unresolved quantiles, failed
quadrature or inconsistent complements raise `ArithmeticError`. Shifted
comparisons inherit Parameter Solver's representability limits, including
lognormal scale exponentiation and beta quantiles that round to endpoints.
There is no claim that every extreme finite input can be resolved, nor that the
Python algorithms reproduce the native application's internal iterations.

## Validation

Nine focused tests cover the guide's normal and shifted beta examples, with
an exact rational-polynomial integral for the beta result; shifted exponential
identities in both gamma and Weibull parameterizations at scales `1e-100`, one
and `1e100`; reciprocal gamma/inverse-gamma identities; normal tails below
`1e-16`; lognormal comparisons with log-means above the exponentiation range;
extreme gamma scale ratios; and beta support separation.

Additional cases compare shifted gamma, inverse gamma, lognormal and Weibull,
and unshifted unequal-shape Weibull, against independent log-density-coordinate
integration. The gamma case also exercises nearly coincident breakpoints.
Together with existing beta comparison and Parameter Solver tests, the focused
validation run passes 22 tests. No new CI workflow is needed.
