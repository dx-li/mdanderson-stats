# U2OET GAO posterior fitting

`fit_u2oet_gao` fits the [2017 GAO model](u2oet-gao.md) to complete ordinal
efficacy/toxicity counts and optional toxicity-only observations. It uses
caller-specified priors and retains posterior draws, dose-pair probabilities
and chain diagnostics. This explicit Python prior convention is separate from
the still-unverified native GAO prior-file interpretation.

```python
import numpy as np
from mdanderson_stats import (
    U2OETCriteria,
    fit_u2oet_gao,
    u2oet_gao_parameter_names,
    u2oet_posterior,
)

names = u2oet_gao_parameter_names(2, 2)
mean = np.array(
    [
        -1,
        0.3,
        0.2,
        -0.1,
        0,  # efficacy coefficients and log-lambda
        -0.4,
        -1.2,
        -0.05,
        0.15,
        0,  # toxicity coefficients and log-lambda
        np.log(0.4),
        np.arctanh(0.55),  # shared log-kappa and Fisher-z
    ]
)
sd = np.zeros(len(names))
sd[0] = 0.5  # One free coordinate keeps this illustrative example short.
initial = np.tile(mean, (2, 1))
initial[:, 0] += [-0.5, 0.5]
counts = np.zeros((2, 2, 2, 2), dtype=int)
counts[0, 0, 1, 0] = 3
partial = np.zeros((2, 2, 2), dtype=int)
partial[1, 0, 1] = 1  # A separate patient with toxicity but no efficacy outcome.
fit = fit_u2oet_gao(
    [1, 3],
    [2, 5],
    counts,
    toxicity_only=partial,
    prior_mean=mean,
    prior_sd=sd,
    initial=initial,
    draws=64,
    warmup=32,
    chains=2,
    rng=np.random.default_rng(7707),
)
assert fit.parameters.shape == (2, 64, 12)
assert fit.joint.shape == (2, 64, 2, 2, 2, 2)
posterior = u2oet_posterior(
    fit.joint.reshape(-1, 2, 2, 2, 2),
    [[20, 0], [100, 50]],
    criteria=U2OETCriteria(efficacy_level=1, toxicity_level=1),
)
assert posterior.mean_utility.shape == (2, 2)
print(fit.parameter_summary.mean[0], posterior.mean_utility)
```

The example demonstrates fitting and downstream utility summaries; its short
chains do not establish statistical precision. Positive prior SDs enable
sampling of any or all coordinates. Use suitable priors, dispersed initial
values and adequate sampling for the actual data and inspect the returned
diagnostics. R-hat and Monte Carlo errors do not certify convergence.

## Priors, data and dose units

`u2oet_gao_parameter_names` gives the exact vector order. For each endpoint,
each successive threshold contributes agent-1 intercept, agent-2 intercept,
agent-1 slope and agent-2 slope, followed by the endpoint's log-lambda.
Efficacy precedes toxicity; shared log-kappa and association Fisher-z come
last. There are 12 coordinates for two binary outcomes and 28 for two
four-category outcomes.

Each retained coordinate has an independent normal prior with the supplied
mean and SD. SD zero fixes it exactly. Exponentiating the link-shape and
interaction coordinates makes those parameters positive; `tanh` converts
Fisher-z to correlation. No transformation Jacobian is added because the
priors are defined in the retained coordinates themselves. These priors do
not import the different 2010 model's interaction or association assumptions.
Unrepresentable transformations, including Fisher-z values whose `tanh`
rounds to exactly -1 or 1, raise an error.

Dose grids contain 2–5 increasing positive raw doses in their original units.
They are not automatically centered or standardized. A unit change requires
consistent slope and prior transformations. Outcomes have 2–4 ordered
categories. Complete counts have axes `(dose1, dose2, efficacy, toxicity)`;
toxicity-only counts have axes `(dose1, dose2, toxicity)`. The two count arrays
must represent disjoint observations: do not count a complete patient again
as toxicity-only. Unobserved patients contribute no outcome likelihood.

## Results, work limits and validation

`parameters` and `log_likelihood` retain chain/draw axes; `joint` adds the
two dose axes and two outcome axes. Flattening only chain/draw axes lets the
existing `u2oet_posterior` and allocation functions consume the fitted
probabilities. The fit also retains names, prior settings, counts and actual
likelihood work. The [GAO calendar trial driver](u2oet-gao-trials.md) connects
this fitter to pending-outcome trial conduct and aggregate summaries. GAO
prior calibration remains open. The [adaptive precision wrapper](u2oet-gao-adaptive-precision.md)
continues these chains toward an explicit per-chain, four-corner utility
MCSE/SD target, with cumulative work and memory limits.

Blocked elliptical slice sampling updates the free normal-prior coordinates.
Shape and retained-storage checks precede large allocations. Likelihood
evaluations and full-grid work have explicit hard-capped budgets; Gaussian
rectangle integration has its own subdivision limit. Integration or sampling
failures propagate. These limits bound computation, not statistical precision.

Independent base-R integration checks two one-coordinate posteriors: a raw
efficacy intercept and a Fisher-z association, including toxicity-only data.
All 38 checked posterior summaries agree within 1.028 estimated Monte Carlo
standard errors; maximum relevant split R-hat was 1.00347. The
[reference generator](../tools/reference_u2oet_gao_fit.R) and
[audit](../research/u2oet-gao-fit-audit.md) record this evidence. It validates
the stated Python prior convention; native prior mapping, calibration and
published trial operating-characteristic reproduction remain open.
