# MULTI beta mixtures

Implemented: the uniform-plus-beta model, density/CDF evaluation, posterior null
probabilities, S STBETA initialization, and EMBETA exponential-family EM fitting.
MULTI remains **partial**: its separate direct likelihood optimizer, automatic
component selection, simulation-based goodness of fit, and remaining
plotting/reporting workflows are not yet ported.

## Specified models

```python
from mdanderson_stats import BetaMixture

model = BetaMixture(null_weight=0.7, weights=[0.3], a=[0.4], b=[8])
p = [0.001, 0.02, 0.2, 0.8]
print(model.logpdf(p))
print(model.cdf(p))
print(model.null_posterior(p))
print(model.log_likelihood(p))
print(model.cramer_von_mises(p))
```

The density is `p0 + sum_j p[j] * BetaPDF(x; a[j], b[j])`. The uniform component
represents the null. `null_weight`, `weights`, `a`, and `b` correspond to P0, P,
R, and S. Shapes must be positive; weights must be nonnegative and sum to one
within 1e-12. Accepted rounding residuals are normalized. Arrays are copied and
made read-only. `BetaMixture(1)` is uniform-only; zero-weight components are
excluded, including at singular endpoints.

The null posterior is **p0 divided by the mixture density** (BPVAL). A source
comment incorrectly describes a CDF denominator. The posterior is conditional
on the specified model and parameters, not a multiple-testing adjusted p-value.
Plugging in estimated parameters does not account for their uncertainty or
establish frequentist error control.

`logpdf`, `cdf`, and `null_posterior` accept scalar or array inputs in [0,1] and
retain their shape. `log_likelihood` sums along the last axis, with leading axes
as batches. `cramer_von_mises` sorts each family on that axis and returns CALCVM:

`mean((F(x[i]) - (i-0.5)/n)**2) + 1/(12*n**2)`.

This is the usual Cramér–von Mises W-squared statistic **divided by n**. It is a
statistic, not a goodness-of-fit p-value. Fitting parameters changes its null
distribution and requires appropriate calibration.

## Endpoints and arithmetic

Default evaluation uses mathematical beta endpoint limits. `logpdf` may return
negative or positive infinity for zero or infinite density. The null posterior
raises at zero mixture density, where it is undefined. At a singular beta
endpoint with positive null weight, the posterior is zero.

`logpdf`, `log_likelihood`, `null_posterior`, and EM fitting accept
`legacy_endpoints=True` for the archived INITLN convention. For x or 1-x at or
below the smallest positive normal double, it substitutes that number's log and
uses **positive** tiny for the complementary logarithm, as the source does.
This approximation differs from true endpoint limits. CDFs always use the actual
x, matching MIXPRB. Log-sum-exp and `xlogy`/`xlog1py` avoid forming underflowed
densities before taking logs or computing posterior ratios.

## Initialization and fitting

```python
from mdanderson_stats import beta_mixture_start, fit_beta_mixture_em

# pvalues is your full vector of observed test p-values.
fit = fit_beta_mixture_em(pvalues)
posterior = fit.model.null_posterior(pvalues)
initial_two = beta_mixture_start(pvalues, previous=fit.model)
fit_two = fit_beta_mixture_em(pvalues, initial=initial_two)
```

STBETA selects observations below 0.05 under the previous model's CDF. Without
a previous model this means p<0.05. It requires at least three selected values,
assigns their sample fraction to the new component, scales existing weights
proportionally, and estimates new shapes by moments using **sample variance**
(denominator count-1). Constant data or invalid shapes raise `BetaMixtureFitError`.

EM defaults to this one-component start or accepts an explicit `BetaMixture`.
The S requirement n>3*k+1 is enforced. Endpoints in nonuniform fits require the
explicit legacy option; uniform-only fits need no iterations. EM updates posterior
component masses, weights, and E[log x]/E[log(1-x)], then solves the original
digamma equations for beta shapes with positive Newton steps. Outer convergence
uses null-weight and log-moment changes, as EMBETA does. Defaults are tolerance
1e-6 and at most 100,000 iterations. Convergence does not certify a global maximum
or choose the number of components.

The inner solver retains the source's absolute step and positive-shape rules,
and adds a residual check based on floating-point rounding. Subtracting nearly
equal digammas cannot reliably meet the original 1e-20 threshold; the new check
avoids cycling for concentrated components. Solver failure, negligible component
mass, nonfinite likelihood, or material likelihood decrease raises an error;
no failed fit is relabeled as uniform.

Results include `model`, `log_likelihood`, `cramer_von_mises`, `iterations`, and
`log_likelihood_history`, including the initial model. Concentrated components
can cause tiny likelihood fluctuations at floating-point precision.

## Validation and performance

`tools/reference_beta_mixture.py` records 15 original model evaluations, three
initializations, and five EM fits in `tests/fixtures/beta_mixture.json`, with
archive hash and compiler provenance. S EMBETA is renamed only to distinguish
its extra likelihood argument from the desktop routine. Shared MOMBET/SWPPAR
numerical bodies were verified identical. Cases cover the published 150 values
with one/two beta components, synthetic data, two convergence tolerances,
endpoint compatibility, and failed initialization.

The published two-component fit agrees with original parameters to roughly eight
significant digits; reference likelihoods agree within 5e-8. Independent checks
cover polynomial beta formulas, Bayes' formula, CVM scaling, endpoint limits,
zero weights, Decimal log-density calculations, single-beta likelihood score
equations, likelihood progress within numerical precision, explicit failures,
input immutability, and posterior calibration under a known generating model.
These do not establish calibration for arbitrary fitted mixtures or complete
the outstanding model-selection workflow.

`uv run python tools/benchmark_beta_mixture.py` reproduces the
[recorded benchmark](beta-mixture-benchmark.json). On that machine, 5,000 posterior
evaluations took about 0.001 seconds in one array call versus 0.314 seconds in
separate scalar calls, about 321x faster. This compares Python call patterns,
not original Fortran throughput or EM speed, and is not a CI performance threshold.
