# EffTox dose finding

Catalog entry 2 is **partial**. Python provides the bivariate binary response
model, elicited-prior calibration, posterior fitting, modern and legacy trade-off
contours, dose selection and completed-outcome trial simulation. A separate
continuation-ratio model fits mutually exclusive efficacy, toxicity and neither
outcomes, with contour elicitation, dose decisions and trial simulation.
This is separate from BOP2's efficacy/toxicity monitoring functions.
The official [EffTox entry](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/2)
lists version 5.2.3, modified June 24, 2026.
[Source provenance](efftox-sources.json) records the inspected references.

## Model and data

For physical dose `d`, the predictor is `x = log(d) - mean(log(doses))`.
When the first dose is zero, the original convention adds the second dose to
every dose before taking logs; `zero_dose_shift=False` rejects zero doses.
Binary coefficient order is `(mu_T, beta_T, mu_E, beta_E1, beta_E2, psi)`:

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

## Calibrating a binary-model prior

`calibrate_efftox_prior` converts elicited efficacy/toxicity probability means
and a target effective sample size (ESS) into an `EffToxPrior`. For each outcome
it matches the means and the average beta-moment ESS, `m*(1-m)/v-1`, with the
SD-difference penalty from
[Thall et al. (2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4229398/).

```python
from mdanderson_stats import calibrate_efftox_prior

calibrated = calibrate_efftox_prior(
    [1, 2, 4, 6.6, 10],
    [0.2, 0.4, 0.6, 0.8, 0.9],
    [0.02, 0.04, 0.06, 0.08, 0.1],
    target_ess=0.9,
    monotone_toxicity=False,
)
print(calibrated.efficacy.mean, calibrated.toxicity.mean)
print(calibrated.efficacy.effective_sample_size.mean())
print(calibrated.toxicity.effective_sample_size.mean())
# Supply calibrated.prior to fit_efftox or simulate_efftox.
```

The example uses an untruncated toxicity slope, whose induced ESS agrees with
the rounded published example. The default `monotone_toxicity=True` calibrates
the positive-conditioned slope used by the default posterior model. This is a
substantive prior choice: the returned prior and calculated moments use the
same choice. `efficacy_target_ess` and `toxicity_target_ess` can override the
common target independently. The quadratic efficacy coefficient is integrated
with its full supplied uncertainty (default mean zero, SD `.2`); the association
prior is supplied separately (default mean zero, SD one).

Inspect achieved moments, objectives, optimizer messages and boundary hits
before using the prior. Optimizer convergence does not mean every target is
matched exactly. Hypermeans are bounded to `[-10000,10000]` and optimized SDs
to `[1e-4,100]`; these Python limits are reported in the result. The adaptive
integrator checks convergence and probability mass, reports estimated absolute
integration error, and raises on unresolved variance. Its `quadrature_order`
setting controls `max(200,4*quadrature_order)` subintervals, not Gaussian nodes.
Optimizer evaluation and overall work limits are checked before fitting.

`efftox_prior_moments(doses, prior, outcome="efficacy")` evaluates an existing
prior without optimization; use `outcome="toxicity"` for the other margin.
A fixed probability has zero variance and limiting ESS infinity. Numerically
unresolved nonfixed priors raise an error. The source comparison and known
native-kernel gaps are recorded in the
[calibration audit](../research/efftox-prior-calibration-audit.md).

## Posterior API

```python
import numpy as np
from mdanderson_stats import EffToxPrior, fit_efftox

# Illustrative coefficient prior, supplied as means and STANDARD DEVIATIONS.
prior = EffToxPrior(
    mean=[-1, 0.8, 0.2, 1.1, -0.3, 0],
    sd=[0.9, 0.5, 1.1, 0.7, 0.2, 1],
)
counts = [
    [[4, 1], [2, 1]],
    [[2, 1], [4, 1]],
    [[1, 2], [4, 3]],
]
fit = fit_efftox(
    [1, 2, 4],
    counts,
    prior=prior,
    draws=1000,
    warmup=500,
    chains=2,
    rng=np.random.default_rng(2026),
)
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

### Original inverse-quadratic contour

`EffToxLegacyContour` implements the original curve `T=a+b/E+c/E**2` and
radial desirability from the 2004 paper. Its score is
`distance(target intersection, ideal)/distance(evaluated point, ideal)-1`,
where the ideal is `(1,0)`. Points on the target have score zero; points closer
to the ideal have positive scores. The ideal itself has limiting score `+inf`.
This numeric score differs from the modern Lp score above.

```python
from mdanderson_stats import EffToxLegacyContour

legacy = EffToxLegacyContour.from_points(
    [0.15, 0.25, 1],
    [0, 0.30, 0.60],  # published Pentostatin targets
)
print(legacy.coefficients)  # a, b, c
print(legacy.utility([0.15, 0.25, 1], [0, 0.30, 0.60]))  # all approximately zero
print(legacy.utility(0.625, 0.15))  # one: twice as close along the middle ray
```

Pass `legacy` as the contour to `efftox_decision` or `simulate_efftox`.
Python interpolates the three supplied points exactly and checks monotonicity
over their efficacy interval. Incompatible or ill-conditioned points raise an
error. This is an explicit convention: the historical Windows fitting routine
could approximate its inputs, and its exact loss and constraints are unavailable.
The [source audit](../research/efftox-legacy-contour-audit.md) records that gap.
Scoring is vectorized with bounded root iteration and a 200,000-cell limit;
unrepresentable finite nonideal scores raise an error.

## Dose selection

```python
from mdanderson_stats import EffToxContour, efftox_decision

contour = EffToxContour.from_points(0.5, 0.65, 0.7, 0.25)
decision = efftox_decision(
    fit,
    contour,
    efficacy_limit=0.45,
    toxicity_limit=0.30,
    efficacy_probability=0.10,
    toxicity_probability=0.10,
    starting_dose=1,
    last_dose=3,
    phase="interim",
    allow_untried_exploration=True,
    skip_policy="both",
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

`simulate_efftox` generates completed cohorts. For the binary model, use a
full joint truth table at each physical dose. Table axes follow the convention above:
`truth[dose, efficacy, toxicity]`. This retains outcome association instead of
assuming independent efficacy and toxicity. It uses the same starting-dose,
exploration, admissibility and no-skipping rules as `efftox_decision`.

```python
from mdanderson_stats import EffToxPrior, EffToxContour, simulate_efftox

simulation = simulate_efftox(
    [1, 2, 4],
    [
        [[0.55, 0.05], [0.35, 0.05]],
        [[0.35, 0.10], [0.45, 0.10]],
        [[0.20, 0.15], [0.40, 0.25]],
    ],
    prior=EffToxPrior(
        mean=[-1, 0.8, 0.2, 1.1, -0.3, 0],
        sd=[0.9, 0.5, 1.1, 0.7, 0.2, 1],
    ),
    contour=EffToxContour.from_points(0.5, 0.65, 0.7, 0.25),
    efficacy_limit=0.45,
    toxicity_limit=0.30,
    efficacy_probability=0.10,
    toxicity_probability=0.10,
    starting_dose=1,
    cohorts=2,
    cohort_size=3,
    trials=4,
    draws=16,
    warmup=8,
    chains=2,
    rng=2026,
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

## Three mutually exclusive outcomes

`EffToxTrinaryPrior` and `fit_efftox_trinary` implement the original
continuation-ratio model. Counts have columns `(neither, efficacy, toxicity)`.
With centered log dose `x`, define `t=logistic(mu_T+beta_T*x)` and
`q=logistic(mu_Q+beta_Q*x)`. The outcome probabilities are
`((1-t)*(1-q), (1-t)*q, t)`. Both slope priors are conditioned positive;
zero prior SD fixes a coefficient, and fixed slopes must be positive.

```python
import numpy as np
from mdanderson_stats import EffToxTrinaryPrior, fit_efftox_trinary

trinary = fit_efftox_trinary(
    [1, 2, 4],
    [[2, 2, 1], [1, 3, 1], [1, 2, 2]],
    prior=EffToxTrinaryPrior(
        mean=[-1, 0.6, 0.4, 0.9],
        sd=[0.7, 0, 0.8, 0],
    ),
    draws=256,
    warmup=128,
    chains=2,
    rng=np.random.default_rng(17),
)
print(trinary.efficacy_probabilities.mean(axis=(0, 1)))
print(trinary.toxicity_probabilities.mean(axis=(0, 1)))
```

The fit retains marginal efficacy `(1-t)*q` and its logit separately from
conditional efficacy `q`. Dose admissibility must use the marginal quantity.
It also returns cell log probabilities, coefficient draws, likelihoods and
chain diagnostics. `efftox_trinary_predict`, `efftox_trinary_log_probabilities`
and `efftox_trinary_log_likelihood` accept standardized dose codes and batched
four-coefficient arrays. The log likelihood omits multinomial constants.

The small example illustrates the API, not a convergence guarantee.
Trinary retained cell probabilities are limited to 200,000 entries;
`chains*(draws+warmup)*doses*4` is limited to two million work units.

For mutually exclusive outcomes, elicit three equally desirable points
`(e0,0)`, `(em,tm)` and `(eh,th)`, where `eh+th=1`. The high point lies on
the probability triangle's edge. `EffToxTrinaryContour` solves the positive
Lp shape and an analytical toxicity-axis scale, which may exceed one.

```python
from mdanderson_stats import EffToxTrinaryContour, efftox_decision

trinary_contour = EffToxTrinaryContour.from_points(
    [0.45, 0.55, 0.84],
    [0, 0.10, 0.16],
)
decision = efftox_decision(
    trinary,
    trinary_contour,
    efficacy_limit=0.2,
    toxicity_limit=0.4,
    efficacy_probability=0.5,
    toxicity_probability=0.5,
    starting_dose=1,
    phase="final",
)
print(decision.action, decision.dose)
```

The score is zero at each target and one at the ideal `(1,0)`. The formula
extends over the unit square, but only pairs with `efficacy+toxicity<=1`
represent trinary probabilities. Dose decisions use marginal efficacy and
toxicity from the fit, with the same exploration and skipping settings as
the binary model.
The contour matches 48 independent R values within `7e-14`; the public
fitting/selection example ran in 0.10 seconds with 114.5 MiB peak memory.

The same simulator accepts `EffToxTrinaryPrior` with `EffToxTrinaryContour`
and a `(dose,3)` truth table in `(neither, efficacy, toxicity)` order:

```python
from mdanderson_stats import EffToxTrinaryPrior, EffToxTrinaryContour, simulate_efftox

trinary_simulation = simulate_efftox(
    [1, 2, 4],
    [[0.55, 0.35, 0.10], [0.40, 0.45, 0.15], [0.30, 0.50, 0.20]],
    prior=EffToxTrinaryPrior(
        mean=[-2, 0.5, 0.5, 0.7],
        sd=[0.3, 0.1, 0.3, 0.1],
    ),
    contour=EffToxTrinaryContour.from_points([0.2, 0.5, 0.8], [0, 0.1, 0.2]),
    efficacy_limit=0.2,
    toxicity_limit=0.4,
    efficacy_probability=0.5,
    toxicity_probability=0.5,
    cohorts=2,
    cohort_size=3,
    trials=4,
    draws=16,
    warmup=8,
    chains=2,
    rng=2026,
)
print(trinary_simulation.outcome_model)  # "trinary"
print(trinary_simulation.selection_probability)
```

Trinary `outcome_counts` have shape `(trial,dose,3)`, and pooled observed
probabilities have shape `(dose,3)`. Binary outputs retain their existing
two outcome axes. The result records `outcome_model`; mismatched prior/contour
types raise before simulation. Both modes share the same allocation, stopping,
random-stream and diagnostic conventions.

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
`trials*(cohorts+1)*C*D*(S+W)*P`, where `P=6` for binary and `P=4` for trinary,
checked against `max_total_fit_work` before
sampling. The default and maximum budget is 20,000,000; callers may lower it.
This bounds the requested fit dimensions, not elapsed time or the actual number
of slice-likelihood evaluations, which the result reports separately.

## Validation and remaining scope

The trinary likelihood matches independent base-R values within `5e-14`,
and its outcome probabilities within `2e-15`. In a four-chain run with 4,000
retained draws per chain, posterior means were within 0.0011 of independent
one-dimensional integration. Five focused checks include nonfixed positive
slope priors, extreme logits, zero counts and output bounds. The
[trinary audit](../research/efftox-trinary-audit.md) describes the reference
construction; no native Windows posterior comparison is claimed.
The public example ran in 0.10 seconds with 114.7 MiB peak process memory
and no swaps; it was checked through the package's public imports.

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

The legacy contour matches 60 independent base-R scores within `4.27e-14`
absolute error. Five contour checks and three simulation checks passed together
in 1.35 seconds, including continuity at very small positive toxicity. Its
public guide example and a 10,000-pair monotonicity/performance check passed;
see the [contour audit](../research/efftox-legacy-contour-audit.md).

Four focused calibration checks passed, including distinct ESS targets and
the difference between conditioned and untruncated slope priors. Its 34-row
independent R reference covers full quadratic uncertainty and nearly fixed
priors, with mean/variance errors below `5.1e-16`/`2.6e-16` and relative ESS
error around `1e-12`.
The public calibration example achieved mean efficacy/toxicity ESS values
`0.90946`/`0.90538` for target `0.9`. Both optimizers converged without hitting
their bounds. It ran in 16.97 seconds with 115.3 MiB peak process memory and
no swaps; see the calibration audit for objectives and comparison details.

Two trinary simulation checks cover independently generated multinomial cells,
patient accounting and a nonfixed posterior update. All five binary/trinary
simulation checks passed after integration. The public four-trial trinary
example completed its 12 posterior fits in 0.045 seconds, using 114.5 MiB peak
process memory with no swaps. These small workloads verify accounting and
integration, not operating-characteristic precision.

[Trinary prior elicitation](efftox-trinary-calibration.md) now evaluates induced
marginal efficacy/toxicity moments and calibrates separate information targets.
Its toxicity-first objective is an explicit Python policy; it preserves
uncertainty in both independent prior blocks and reports achieved residuals
and optimizer convergence.

Remaining scope includes historical approximate contour fitting, legacy
trinary contours, native trinary calibration and native file/report workflows.
The Windows integration kernel has not been run for direct parity checks.
