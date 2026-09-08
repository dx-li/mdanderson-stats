# MULTI beta mixtures

Implemented: the uniform-plus-beta model, density/CDF evaluation, posterior null
probabilities, STBETA initialization, EM/direct likelihood fitting, sequential
component selection, and simulation-based goodness-of-fit checks.
MULTI remains **partial**: remaining desktop and plotting/reporting workflows
still require scope review, implementation, and validation.

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

`logpdf`, `log_likelihood`, `null_posterior`, and both fitters accept
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

## Direct likelihood fitting

`fit_beta_mixture_ml(pvalues, initial=None)` returns the same result type as EM
and uses the same default initialization, sample-size rule, and endpoint policy.
It retains MLBETA's shape bounds [1e-10,1e9] and nonnegative normalized weights.
Successive conditional weight fractions represent the simplex, including exact
zero/one boundaries; log shapes improve numerical scaling. Evaluation and analytic
scores use NumPy arrays and stable logarithmic mixture calculations.

SciPy L-BFGS-B replaces the original David Gay finite-difference optimizer.
Defaults are relative average-negative-likelihood tolerance 1e-9, transformed
projected-gradient tolerance 1e-6, 1,000 iterations, and 15,000 evaluations.
Either convergence criterion can stop the solver. Limits and nonfinite gradients
raise `BetaMixtureFitError`; extreme shapes and boundary weights can exceed
numerical range. Different solver paths can select different local optima.
Convergence does not certify a global maximum or select component count.

Five archived MLBETA fits are recorded alongside EM fixtures. Single-component
likelihoods agree within 4e-5, including legacy endpoints. On the published
two-component data, the new solver reaches likelihood 119.04670 versus the
original direct solver's 118.15971; it agrees with the independently ported EM
solution. Tests also check analytic scores by finite differences at interior
and boundary weights, normalization, likelihood progress, and explicit failures.

## Component selection and simulated model checks

```python
from mdanderson_stats import select_beta_mixture, beta_mixture_bootstrap

selection = select_beta_mixture(pvalues, criterion="data", algorithm="em")
print(selection.status, selection.message)
fit = selection.fit
check = beta_mixture_bootstrap(pvalues, fit.model, algorithm="em", rng=123)
print(check.pvalue, check.attempts, check.failures)
```

The S `betamix` sequence starts with the uniform model (k=0), adds components
using STBETA on the preceding fitted CDF, and tests up to k=10. The default
threshold is 0.05. The three original rules are:

- `data`: when the newly added component's fitted weight is **less than** the
  threshold, select the preceding model.
- `lglk`: when `abs(new_LL-old_LL)/old_LL` is **less than** the threshold, select
  the preceding model. At the initial zero likelihood, positive change is
  treated as infinite; zero change stops. The latter explicitly resolves the
  source's undefined 0/0. This is a heuristic, not a likelihood-ratio test.
- `pcvm`: when the simulated CVM p-value is **greater than** the threshold,
  select the current model, including k=0 when appropriate.

`status="criterion_met"` means the requested rule stopped the sequence.
`fit_failed`, `bootstrap_failed`, and `component_limit` return the preceding
valid fit (or the last fit at the limit) with an explanatory `message`; these
statuses do **not** claim successful selection. A failure at the uniform model
raises. `candidates` retains successful fits, including a candidate rejected by
a stopping rule; `bootstrap_checks` records completed simulation checks.
`algorithm="ml"` uses the new direct optimizer, whose different local optima can
change component selection relative to the archived David Gay solver.

SIMCVM generates samples from the supplied fitted model, refits the **same k**
starting from that model, and calculates CVM on each refitted sample. Its defaults
are 100 successful replicates, at most 200 attempts, and refit tolerance 1e-3.
The result's p-value is the fraction of simulated statistics **strictly greater**
than the observed statistic, with no +1 correction; it can be zero. It is a
Monte Carlo estimate with resolution 1/replicates, not an exact p-value or proof
of fit. Refitting failures are recorded and retried; exhausting the bound raises.
Discarding failed fits can affect the simulated distribution, so inspect failures.
Component selection is not repeated within each replicate.

Sampling uses NumPy categorical/beta draws instead of RANF and numerical CDF
inversion. This preserves the intended distribution, not the original random
stream or inverse-solver rounding. Samples rounded to endpoints are rejected
explicitly rather than silently clipped; extreme shapes may exhaust retries.
An integer seed or Generator provides explicit random state. Bootstrap input is
a model already fitted to the observed data; the function does not establish
that provenance automatically.

`tools/reference_beta_selection.py` records three native sequential STBETA/EMBETA
fits and four native SIMCVM-style EMBETA refits of fixed simulated samples.
The published data select k=1 by weight and k=2 by likelihood change under EM.
Tests compare the native fit chain and refitted CVM values, independently compute
uniform CVM statistics and strict-tail counts, check sampled mixture CDFs, and
exercise all three stopping rules, iteration limits, failed starts, and input
validation. NumPy random draws are not claimed to reproduce archived draws.

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
These do not establish calibration for arbitrary fitted mixtures or frequentist
error control after selection.

`uv run python tools/benchmark_beta_mixture.py` reproduces the
[recorded benchmark](beta-mixture-benchmark.json). On that machine, 5,000 posterior
evaluations took about 0.001 seconds in one array call versus 0.314 seconds in
separate scalar calls, about 321x faster. This compares Python call patterns,
not original Fortran throughput or EM speed, and is not a CI performance threshold.
