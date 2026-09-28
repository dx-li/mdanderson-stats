# BARD paper BF-BLRM contract — September 28, 2026

This prepares an explicitly configured component from the
[BARD paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12240483/), not a claim
that the current app implements the same route. The official guide describes
BF-BOIN as its stage-one design. Cached paper/guide evidence is under ignored
`research/raw/BARD`; provenance is recorded in [bard-sources.json](../docs/bard-sources.json).

## Model and prior

The reviewer visually inspected PDF page 9 after the extracted equation raised
a possible log-dose ambiguity. The printed model is exactly

```text
logit(p_j) = log(alpha) + beta * (d_j / d_star),  alpha > 0, beta > 0
(log(alpha), log(beta)) ~ N((mu_alpha, mu_beta), diag(var_alpha, var_beta))
```

There is no logarithm around the dose ratio. Substituting the customary
log-dose BLRM would change the published model. All prior and dose inputs
should be explicit. Page 18's simulation uses means `(-1.1, 0)`, variances
`(4, 1)`, doses `(10, 20, 50, 100, 200)`, reference dose 50, target interval
`(.16, .33)` and overdose cutoff `.30`; these are example values, not verified
native defaults.

## Decisions and backfill

Pages 9–11 specify `PTT_j = Pr(gamma1 < p_j < gamma2 | data)` and
`POD_j = Pr(p_j >= gamma2 | data)`. Among doses with `POD_j < eta`, select the
maximum PTT and move one level toward it from the current dose, or stay if
already there. If all POD values exceed eta, terminate with no MTD. Equality
leaves a gap: all doses at eta satisfy neither strict safety eligibility nor
the strict all-over-toxic rule. A future implementation must distinguish
`no eligible safe dose` and cannot invent an MTD. Tie handling also needs an
explicit convention unless additional evidence establishes it.

Backfill eligibility uses a dose below the current dose and observed response
at that dose or below. It closes when POD is at least eta or the evaluable
patient count reaches the dose cap. Assignment prioritizes filling the
current escalation cohort, then the highest open backfill dose, following the
paper's BF-BOIN scheduler reference. It does not use BF-BOIN's conflict-pooling
rule: the monotone fitted model incorporates all dose data directly.

Final stage-one MTD selection uses all escalation and backfill observations,
requires at least six treated patients, and maximizes PTT among doses with
POD below eta. Existing BF-BOIN calendar components offer scheduling patterns,
while `bard.py` supplies stage two; neither currently fits this model.
The first proposed component is model evaluation and bounded posterior fitting
with target/overdose summaries. No BF-BLRM implementation is claimed by this
source-audit checkpoint.

## Independent posterior references

`tools/reference_bard_blrm.R` evaluates 20 raw-ratio model probabilities and
integrates two explicit normal-prior examples. The fixed-slope case integrates
only log-alpha; the full two-parameter case integrates both independent normal
coordinates. Target/overdose probabilities use analytic log-alpha integration
limits at each log-beta, rather than a discontinuous indicator on a grid.
The likelihood omits binomial coefficients consistently with the proposed
sampler. It generates 22 coefficient, toxicity-probability, target-probability
and overdose-probability summaries in `tests/fixtures/bard-blrm-posterior.csv`.

The integration truncates each free standard-normal coordinate at plus/minus
10. Since the unnormalized Bernoulli likelihood is at most one, omitted prior
mass divided by the computed evidence bounds omitted posterior mass, up to
integration error. This ratio is below `2.6e-19` in both examples. Base-R
integration completed with warnings treated as errors; Python sampler
comparison remains pending at this reference checkpoint. These mild illustrative
priors are unrelated to undocumented native application settings.

## Model and posterior implementation checkpoint

Luna committed the fitted component as `a92e81b`, integrated as `539841b`.
It uses independent Gaussian-prior elliptical slice sampling with dispersed
prior starts, explicit zero-SD coordinates, grouped DLT data and log-odds
target/overdose comparisons. Overflowed predictors fail explicitly; they are
not treated as zero-likelihood proposals, which could truncate a valid
posterior. Work accounting includes all likelihood attempts and retained-dose
predictions, with static limits before random draws.

Read-only review found no substantive error in the reference integration or
the fitted model after numerical review corrections. The reference's
normalization constant excludes binomial coefficients and is unsuitable as
a full binomial marginal likelihood for model comparison.

The Python comparison matches all 20 curve probabilities. Four chains with
1,500 retained draws after 500 warmup draws agree with all 22 posterior
references: the largest discrepancy is 0.9694 MCSE in the fixed-slope case
and 1.9552 MCSE with two free parameters. Maximum split R-hat is 1.00032 and
1.00104, respectively; the fixed coordinate is excluded from this convergence
check. Fits use 11,726 and 15,724 likelihood evaluations. The comparison took
0.923 seconds after import, peaked at 116.88 MiB and reported no swaps.

Three focused worker checks passed in 1.77 seconds, including the exact
fixed-parameter target/overdose boundary and a budget rejection before RNG
consumption. Targeted Ruff/formatting and mypy passed. No full suite, large
Monte Carlo simulation, new dependency or CI change was introduced.
Dose/backfill decisions and the stage-one calendar are separate follow-ups;
the fitted model alone does not complete BARD's full trial workflow.
