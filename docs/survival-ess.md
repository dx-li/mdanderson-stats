# BayesESS survival information criterion

`survival_prior_ess` implements the **analytic expectation of the native
BayesESS `essSurv` criterion** for catalog entry 154. It replaces the nested
Monte Carlo loops with a constant-cost formula plus the source's 50-point
interpolation grid. The reference is
[BayesESS internal.R](https://github.com/github-js/BayesESS/blob/4bbf4df3789912b967774e8ff5c3a2d6d5646cdd/R/internal.R)
from package 0.1.19; [provenance](conjugate-ess-sources.json) already records
this exact file and the app's support PDF.

## Model and source conventions

Survival times are exponential with mean `mu`, where `mu` has an inverse-gamma
prior with shape `a` and scale `b`. Administrative right censoring occurs at a
common time `c`. The prior density is proportional to
`mu**(-a-1) * exp(-b/mu)`. Require `a>1` so the prior mean exists. The Python
implementation permits `a<=1e6` and any positive finite scale/censoring time.
The source hardcodes censoring time 3; Python retains that default and also
allows an explicit alternative in the same units as `mu` and `b`.

**The source evaluates the two curvature quantities at different points.**
Prior curvature is evaluated at the prior mean, `b/(a-1)`, while posterior
curvature is evaluated at each simulated prior draw of `mu`. This differs from
the fixed evaluation point in the general Morita ESS definition. This function
preserves the source criterion's expectation; it does not claim to implement
all definitions of survival prior information or reproduce its random draws.

Let `A=a(a+1)/b²`. The native prior curvature is
`Dp=(a-3)(a-1)²/b²`. Conditional on a sampled mean, expected posterior curvature
for `m` subjects is `[-1+m*(1-exp(-c/mu))]/mu²`. Integrating with respect to
`1/mu ~ Gamma(a, rate=b)` gives

```text
E[Dq(m)] = A * (-1 + m*f)
f = 1 - (b/(b+c))**(a+2)
ess = (1 + Dp/A) / f
```

The calculation uses log probabilities and `expm1` to resolve very small
censoring windows without cancellation. Shared rescaling of scale and censoring
time leaves ESS unchanged. No simulated patient arrays are needed.

## Results

`log_ess` is the logarithm of the unconstrained continuous root. The `ess`
property converts it to ordinary units, raising `ArithmeticError` if it cannot
be represented as a positive float64 value. `grid_ess` minimizes the expected
criterion over 50 equally spaced points from 1 to `max_patients`, matching the
native `approx` default grid. It is a fractional interpolated sample size,
not a rounded integer recommendation. `max_patients` must be 2–100,000.

`at_search_boundary` flags selection of an endpoint; increase the search limit
when necessary before treating a constrained result as an interior optimum.
Read-only `patients` and `normalized_information_distance` expose the grid and
signed difference `(Dp-E[Dq])/A`. This normalization avoids squaring extreme
scale parameters. Grid selection uses the analytic root, avoiding cancellation
in nearly constant distance curves.

```python
from mdanderson_stats import survival_prior_ess

result = survival_prior_ess(4, 2, max_patients=4)
assert 1.45 < result.ess < 1.46
assert abs(result.grid_ess - result.ess) < 3 / 49
assert not result.at_search_boundary
```

Two focused tests compare the analytic result to independent numerical
integration across prior shapes and censoring windows, check native grid
minimization, time-unit invariance, the complete-follow-up limit and extremely
small follow-up. A native R Monte Carlo comparison uses the original `essSurv`
function with inverse-gamma draws supplied as reciprocals of base R gamma draws;
for shape 4, scale 2, maximum 4, seed 154 and 200,000 prior draws, both
select grid ESS 1.428571. Finite-simulation results need not always equal the
analytic expectation. The unconstrained root is approximately 1.455964.

Catalog 154 remains partial: unknown-mean variance models, regression simulation
settings, CRM/TITE-CRM and plots/reports still need coverage. See also the
[conjugate-model implementation](conjugate-ess.md).
