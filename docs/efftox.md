# EffTox bivariate dose finding

Catalog entry 2 is **partial**. Python provides the bivariate binary response
model, posterior fitting with explicit priors, Lp trade-off contours, dose
selection and completed-outcome trial simulation. This is separate from BOP2's
efficacy/toxicity monitoring functions.
The official [EffTox entry](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/2)
lists version 5.2.3, modified June 24, 2026.
[Source provenance](efftox-sources.json) records the inspected references.

## Model and data

For physical dose `d`, the predictor is `x = log(d) - mean(log(doses))`.
When the first dose is zero, the original convention adds the second dose to
every dose before taking logs; `zero_dose_shift=False` rejects zero doses.
Coefficient order is always `(mu_T, beta_T, mu_E, beta_E1, beta_E2, psi)`:

```text
logit(T) = mu_T + beta_T*x
logit(E) = mu_E + beta_E1*x + beta_E2*x*x
Pr(E=a,T=b) = E**a*(1-E)**(1-a)*T**b*(1-T)**(1-b)
              + (-1)**(a+b)*E*(1-E)*T*(1-T)*tanh(psi/2)
```

These are the bivariate equations in
[Thall and Cook (2004)](https://www.johndcook.com/efftox.pdf).
The efficacy curve may turn down at higher doses. `monotone_toxicity=True`
restricts `beta_T` to positive values; disabling it permits either sign.
The toxicity slope enters linearly, without an exponential transformation.

`counts[dose, efficacy, toxicity]` records completed binary outcomes. For
example, `[[4,1],[2,1]]` means four patients with neither outcome, one with
toxicity only, two with efficacy only and one with both. Missing or pending
outcomes are not represented by this table.

## Posterior API

```python
import numpy as np
from mdanderson_stats import EffToxPrior, fit_efftox

# Illustrative coefficient prior, supplied as means and STANDARD DEVIATIONS.
prior = EffToxPrior(
    mean=[-1, .8, .2, 1.1, -.3, 0],
    sd=[.9, .5, 1.1, .7, .2, 1],
)
counts = [
    [[4, 1], [2, 1]],
    [[2, 1], [4, 1]],
    [[1, 2], [4, 3]],
]
fit = fit_efftox([1, 2, 4], counts, prior=prior,
                 draws=1000, warmup=500, chains=2,
                 rng=np.random.default_rng(2026))
print(fit.efficacy_probabilities.mean(axis=(0, 1)))
print(fit.toxicity_probabilities.mean(axis=(0, 1)))
print(fit.summary.split_rhat, fit.summary.batch_mean_mcse)
```

The six independent Gaussian priors must be supplied explicitly. With monotone
toxicity, the second Gaussian is conditioned on a positive slope; its supplied
mean and SD describe the Gaussian **before truncation**. A zero SD fixes a
coefficient, an explicit Python extension useful for reduced models and numerical
validation. It does not estimate that coefficient.

The sampler uses Gaussian-prior elliptical slice updates, treating the slope
constraint as an indicator in the likelihood. With no observations it samples
the prior directly. Chains run sequentially. Results retain parameter draws,
joint and marginal probability draws and log likelihoods, with leading axes
`(chain, draw)`. Arrays are read-only. The parameter summary reports classical
split R-hat, batch-means MCSE, means, medians, SDs and shortest empirical 80%
intervals. R-hat is not meaningful for fixed coefficients. Requested draw counts and
diagnostics do not by themselves establish convergence.

`efftox_standardize`, `efftox_predict`, `efftox_log_joint_probabilities` and
`efftox_log_likelihood` expose the mathematical components. The latter three
take centered dose codes, rather than physical doses. Log probabilities use
factored log-domain expressions that preserve rare cells even when marginal
probabilities round to zero or one. The log likelihood omits the multinomial
coefficient, which is constant in the model parameters.

## Trade-off contour

The [2006 technical report](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/EffTox/NewTradeOffFunctions.pdf)
defines the bivariate desirability curve by three equally desirable points:
`(e0,0)`, `(em,tm)` and `(1,t1)`, with `0<e0<em<1` and `0<tm<t1<1`.
The unique positive exponent solves
`((1-em)/(1-e0))**p + (tm/t1)**p = 1`. Desirability is

```text
1 - (((1-E)/(1-e0))**p + (T/t1)**p)**(1/p)
```

The ideal point `(1,0)` has desirability one. Concave contours with `p<1` are
supported as well as linear and convex contours. Dose selection evaluates
desirability at the posterior mean efficacy and toxicity probabilities; averaging
the desirabilities of individual posterior draws is a different criterion.

## Dose selection

```python
from mdanderson_stats import EffToxContour, efftox_decision

contour = EffToxContour.from_points(.5, .65, .7, .25)
decision = efftox_decision(
    fit, contour,
    efficacy_limit=.45, toxicity_limit=.30,
    efficacy_probability=.10, toxicity_probability=.10,
    starting_dose=1, last_dose=3,
    phase="interim", allow_untried_exploration=True, skip_policy="both",
)
print(decision.action, decision.dose, decision.utility)
```

Dose indices are **one-based**. The ordinary admissible set requires both
`Pr(E > efficacy_limit | data) > efficacy_probability` and
`Pr(T < toxicity_limit | data) > toxicity_probability`; all inequalities are
strict. Tail calculations compare logits to the corresponding thresholds so
rounding of very small or large probabilities does not change endpoint decisions.
Observed counts in the fit determine which doses have been tried. The caller
supplies `last_dose` because aggregated counts do not preserve treatment order.

`allow_untried_exploration=True` additionally admits the lowest untried dose
above the starting dose on the toxicity criterion alone, as in the original
2004 algorithm. The flag applies in both phases. With no observations, the
first interim assignment is the physician's `starting_dose`.

For interim assignments, `skip_policy="both"` prohibits skipping untried
intermediate doses on escalation or de-escalation. This behavior is documented
for native version 4.0.12 by
[Brock et al. (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5520236/).
`skip_policy="escalation"` enforces the restriction only upwards, matching the
original paper's stated constraint. These options describe explicit rules;
they do not assert that all versions of the Windows program behave identically.

`phase="final"` selects the greatest desirability within the admissible set,
without an interim transition constraint, following the 2004 final-selection
rule. Final selection requires observed data. Exact utility ties go to the
lowest dose, an explicit deterministic Python convention. An empty admissible
set stops the trial; a nonempty set with no reachable dose is reported separately.

## Trial simulation

`simulate_efftox` generates completed binary cohorts from a full joint truth
table at each physical dose. Table axes follow the count convention above:
`truth[dose, efficacy, toxicity]`. This retains outcome association instead of
assuming independent efficacy and toxicity. It uses the same starting-dose,
exploration, admissibility and no-skipping rules as `efftox_decision`.

```python
from mdanderson_stats import EffToxPrior, EffToxContour, simulate_efftox

simulation = simulate_efftox(
    [1, 2, 4],
    [
        [[.55, .05], [.35, .05]],
        [[.35, .10], [.45, .10]],
        [[.20, .15], [.40, .25]],
    ],
    prior=EffToxPrior(
        mean=[-1, .8, .2, 1.1, -.3, 0],
        sd=[.9, .5, 1.1, .7, .2, 1],
    ),
    contour=EffToxContour.from_points(.5, .65, .7, .25),
    efficacy_limit=.45, toxicity_limit=.30,
    efficacy_probability=.10, toxicity_probability=.10,
    starting_dose=1, cohorts=2, cohort_size=3,
    trials=4, draws=16, warmup=8, chains=2, rng=2026,
)
print(simulation.selection_probability)  # no selection, dose 1, dose 2, dose 3
print(simulation.mean_patients_per_dose)
print(simulation.max_split_rhat, simulation.max_batch_mean_mcse)
```

This small example demonstrates the API; its replication and draw counts do
not establish precise operating characteristics or adequate posterior sampling.
Each completed cohort receives a fresh posterior fit. The last planned cohort
uses the final-selection rule directly. Interim stops have selected dose zero;
trials with no admissible final dose also have no selection, but reaching the
enrollment cap does not count as an early stop.

`selection_probability` and its binomial `selection_mcse` include no selection
at index zero. `allocation_probability` is the fraction of all simulated patients
treated at each dose, pooled across trials. `observed_joint_probability` pools
outcomes within each dose and is `NaN` for doses never assigned. The result also
retains trial-level counts, completed cohorts, stopping reasons and posterior
fit/evaluation counts. Maximum defined R-hat and batch-means MCSE summarize the
fits in each trial; undefined diagnostics for fixed coefficients remain `NaN`.

With an integer `rng`, outcome generation and posterior sampling use separate
streams. `sampler_rng` can set the latter explicitly. Supplied generators must
have distinct underlying bit generators. NumPy seeds do not reproduce the
Windows program's random draws. Trials, cohorts and chains run sequentially,
and full posterior histories are discarded after each decision.

## Resource limits

Inputs support 2–20 strictly increasing doses and at most 10,000 observed
patients. Prior means have absolute value at most 10,000 and prior SDs lie
in `[0,100]`. The same coefficient range bounds evaluated parameters;
unsupported or nonfinite sampler states raise errors.

Sampling permits 2–4 sequential chains, 8–10,000 retained draws per chain and
0–10,000 warmup iterations. For `C` chains, `D` doses, `S` retained draws and
`W` warmup iterations, retained dose draws satisfy `C*D*S <= 200000` and the
work preflight requires `C*D*(S+W)*6 <= 2000000`. The slice-bracket search is
also bounded. There are no parallel workers, subprocess samplers or dense
parameter grids. Direct prediction batches are limited to 200,000 parameter/dose
pairs and contour utility broadcasts to 200,000 cells, checked before numeric
conversion and broadcast materialization. Evaluated dose codes lie within
`[-1500,1500]`.

Simulation additionally permits at most 2,000 trials, 100 cohorts and 500 patients
per trial. Its conservative work estimate is
`trials*(cohorts+1)*C*D*(S+W)*6`, checked against `max_total_fit_work` before
sampling. The default and maximum budget is 20,000,000; callers may lower it.
This bounds the requested fit dimensions, not elapsed time or the actual number
of slice-likelihood evaluations, which the result reports separately.

## Validation and remaining scope

`tools/reference_efftox.R` generates independent base-R references for joint
probabilities and likelihoods, contour values and posterior integration.
The posterior references include independently varying efficacy/toxicity
intercepts, a random association parameter, and a positive toxicity slope from
a truncated Gaussian whose untruncated mean is negative. Reduced models permit
accurate integration without a large six-dimensional grid. The
[official tutorial](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/EffTox/EfftoxTutorial.html)
also supplies rounded desirabilities for a published contour example.

The 12 focused checks passed in 3.39 seconds (3.49 seconds including the test
runner), with measured peak process RSS of 135.8 MiB and no process swaps on
the validation machine. They include analytical extreme-logit limits and bounded
allocation checks. This measures the focused workload, not every possible input.
Existing CI configuration is unchanged; the full repository suite was not run.

Three focused simulation checks cover fixed-prior final-dose matching, count
and probability conservation, reproducible random streams and stopping after
a completed cohort with a nonfixed-prior posterior update. They verify the
simulation's accounting and decision flow, not large-simulation precision or
native Windows random-number parity.
These checks passed in 1.43 seconds; targeted lint, formatting and type checks
also passed. The small public-API example above ran in 1.12 seconds with a peak
process RSS of 114.0 MiB and no process swaps on the validation machine.

This implementation takes coefficient priors as input. The elicited-probability
and effective-sample-size calibration of
[Thall et al. (2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4229398/)
remains open, as do trinary outcomes, legacy inverse-quadratic contours
and native file/report workflows. The Windows program's integration kernel
has not been run for direct parity checks.
