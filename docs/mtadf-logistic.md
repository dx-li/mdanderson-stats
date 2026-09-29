# MTADF: global and local logistic dose finding

These functions implement the global quadratic and local linear efficacy
models in Zang, Lee and Yuan's [2014 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC4239216/),
alongside the existing [isotonic design](mtadf.md). They use complete binary
toxicity and efficacy counts at each dose. Toxicity and efficacy may overlap
within patients. Dose indices are **zero-based**.

## Global quadratic model

The global model is `logit(p_j) = alpha + beta*d_j + gamma*d_j**2`, with
independent Cauchy priors centered at zero and scales `10`, `2.5` and `2.5`.
Supply strictly increasing numerical dose values: the implementation uses
them exactly as given. Centering or changing units changes the prior model,
so select and document dose coding before fitting.

```python
import numpy as np
from mdanderson_stats import mtadf_logistic_posterior, mtadf_logistic_decision

subjects = [10, 10, 10]
responses = [2, 7, 4]
toxicities = [0, 1, 2]
doses = [-1.0, 0.0, 1.0]
fit = mtadf_logistic_posterior(subjects, responses, doses, rng=np.random.default_rng(20260929))
decision = mtadf_logistic_decision(
    subjects, toxicities, responses, doses, current_dose=1, posterior=fit
)
final = mtadf_logistic_decision(
    subjects,
    toxicities,
    responses,
    doses,
    current_dose=None,
    posterior=fit,
    final=True,
)
print(fit.posterior_mean_efficacy, decision.dose, final.dose)
```

The decision function requires a matching fit when efficacy guides assignment.
It ranks admissible doses by posterior mean efficacy, breaks ties at the lowest
dose and moves one level toward that target. Final selection ranks all
admissible doses, including model predictions at untried doses. This differs
from isotonic final selection, which requires observed efficacy at a candidate
dose. Posterior mean and boundary movement are explicit Python choices where
the paper leaves conduct details unspecified.

## Local linear model

The local model is `logit(p) = alpha + beta*d`, with independent Cauchy priors
of scales `10` and `2.5`. By default it uses two adjacent levels ending at the
current dose. At the lowest boundary it uses the first two levels, an explicit
Python boundary convention. Every level in the fitted window must have
observations. `window_length` can request a longer window.

```python
from mdanderson_stats import (
    mtadf_local_logistic_posterior,
    mtadf_local_logistic_decision,
)

local_fit = mtadf_local_logistic_posterior(
    subjects,
    responses,
    doses,
    current_dose=2,
    rng=np.random.default_rng(20260930),
)
local = mtadf_local_logistic_decision(
    subjects,
    toxicities,
    responses,
    doses,
    current_dose=2,
    posterior=local_fit,
)
local_final = mtadf_local_logistic_decision(
    subjects,
    toxicities,
    responses,
    doses,
    current_dose=None,
    final=True,
)
print(local_fit.probability_positive_slope, local.dose, local_final.dose)
```

Initial actions visit the first `window_length` levels in order, subject to
safety. Subsequently, `Pr(beta > 0)` above `efficacy_escalation_cutoff` escalates,
below `efficacy_deescalation_cutoff` de-escalates, and otherwise stays. The
defaults `0.4` and `0.3` illustrate the paper's configuration; they require
calibration for the intended study's operating characteristics.

Before escalating to an already treated next dose, the rule examines the
local window ending at that next dose. It stays if that window's positive-slope
probability is below the de-escalation cutoff. Supply a matching
`bounce_guard_posterior` or pass an explicit NumPy generator in `rng` to fit
that window. The result retains the fit used for this check and its diagnostics.
If `posterior` is omitted, `rng` also supplies the current-window fit. Local
final selection uses the paper's double-sided isotonic rule and needs no MCMC.

## Safety and numerical limits

Both designs reuse the [beta-binomial toxicity model and pooled safety
rule](mtadf.md#statistical-choices). No admissible dose yields a stop with
`dose=None`. An inadmissible current dose triggers a safety action before any
efficacy fit: global conduct drops to the highest admissible dose; local
conduct drops to the highest admissible lower dose or stops if none exists.
These actions need no posterior or RNG. After enrollment, an interim
`current_dose` must have observed subjects.

Posterior sampling uses random-walk Metropolis with proposal adaptation only
during warmup. Chains run serially with explicit randomness. Defaults are four
chains, 1,000 warmup iterations and 2,000 retained draws per chain. Results
include read-only draws, acceptance rates, split R-hat and batch-means Monte
Carlo standard errors. The local positive-slope indicator has its own chain
diagnostics and MCSE. Examine these diagnostics and Monte Carlo uncertainty
near decision thresholds; a completed fit does not establish convergence.

Inputs are limited to 20 doses and 10,000 total subjects. Preflight bounds
limit posterior storage and summary temporaries to two million estimated
array cells and transition work to 20 million dose evaluations. Local decisions
account for both current and next-window fits together. Requests exceeding a
bound raise before fitting. Nonrepresentable quadratic dose coding or initial
posterior arithmetic raises a clear error.

## Evidence and remaining coverage

An independent transformed-Cauchy quadrature checks a local posterior. The
[global reference generator](../tools/reference_mtadf_logistic.py) integrates
the three-parameter model separately from the sampler: 128- and 192-point
rules agree to approximately `3.4e-14` in the checked efficacy means. A seeded
default global fit agreed within 1.82 estimated Monte Carlo standard errors.
See the [method audit](../research/mtadf-logistic-audit.md) and
[source record](mtadf-sources.json).

These are independent implementations of the published models with documented
Python conduct and sampling choices. Native application reports, hidden
settings and random-stream parity remain unverified. The separate
[logistic trial simulator](mtadf-logistic-simulation.md) runs both logistic
designs with complete cohorts, replayable seed pairs and compact summaries.
`simulate_mtadf` continues to use the isotonic design.
