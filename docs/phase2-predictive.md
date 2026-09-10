# Phase II Predictive Probability

Catalog **84** implements the Lee–Liu single-arm predictive-probability method,
including specified designs and constrained searches over sample sizes and
cutoff grids. The source is the [2010 software guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/Pid535/PPUserGuide.pdf),
based on Lee and Liu (2008), *A predictive probability design for phase II cancer
clinical trials*, Clinical Trials 5:93–106. Source hash and scope are recorded in
[provenance](phase2-predictive-source.json). No native program is redistributed.

## Specified design

```python
from mdanderson_stats import phase2_predictive_design

design = phase2_predictive_design(
    40,
    null_rate=0.2,
    prior=[1, 1],
    looks=[10, 15, 20, 25, 30, 35, 40],
    theta_lower=0.1,
    theta_final=0.9,
    theta_upper=1,
)
print(design.futility_max, design.final_positive_min)
print(design.monitor(events=2, sample_size=15).decision)
oc = design.operating_characteristics([0.2, 0.4])
print(oc.positive_conclusion, oc.expected_sample_size)
```

Given `x` responses among `n` patients, the posterior is
`Beta(a+x,b+n-x)`. Final efficacy requires `P(p>p0 | final data)>theta_final`.
The predictive probability averages that final decision over the beta-binomial
future-response distribution. The package obtains the same finite sum through
backward recursion, avoiding repeated predictive-distribution evaluations.
At each scheduled interim, stop for futility if `PP<theta_lower`, or efficacy
if `PP>theta_upper`. Otherwise continue. At maximum size the final posterior
rule determines the conclusion.

Both efficacy comparisons are **strict**. In particular, `theta_upper=1`
disables early efficacy even when `PP=1`. This differs from the inclusive BEMPR
convention in `predictive_efficacy_design`; its default behavior is preserved.
The common factory's new `strict_thresholds=True` selects the Phase II PP rules.
The new wrapper sets it automatically.

The returned `BayesianMonitoringDesign` supplies posterior/predictive tables,
count-based decisions, sticky first-stop response-path replay, and exact
Bernoulli operating characteristics. Arrays `stop_low` and `stop_high` give
stage-specific early termination probabilities; `complete_positive` and
`complete_negative` describe final outcomes. `sample_size_probability`, mean,
standard deviation, and quantiles describe enrollment. Early-termination
probability is `stop_low.sum(-1)+stop_high.sum(-1)`.

## Constrained search

```python
from mdanderson_stats import optimize_phase2_predictive

search = optimize_phase2_predictive(
    sample_sizes=[25, 30, 35, 40],
    null_rate=0.2,
    alternative_rate=0.4,
    type1_error=0.1,
    minimum_power=0.8,
    objective="expected_sample_size",
)
best = search.best
print(best.design.max_subjects)
print(best.threshold_pairs)  # Columns: theta_lower, theta_final.
print(best.operating_characteristics.positive_conclusion)
print(best.operating_characteristics.expected_sample_size)
```

For every requested maximum size, the search retains the best boundary meeting
both the null type I error limit and alternative power floor. `objective="power"`
maximizes power, then minimizes null expected enrollment; the alternative
`"expected_sample_size"` reverses those priorities. Across sizes, smaller maximum
size breaks any remaining tie; within a size, the supplied final-cutoff then
lower-cutoff grid order breaks ties. `feasible_designs` contains the best fit for
each feasible size. `searched_sample_sizes` retains the full requested grid.
The search does not stop at the first feasible or infeasible size.

`threshold_pairs` retains all supplied grid pairs yielding the selected boundary.
It does **not** imply that arbitrary pairs inside their rectangular min/max range
are equivalent. This implementation reports discrete pairs instead of claiming
native continuous-range or optimizer tie-breaking parity.

Python defaults are a uniform prior, `theta_lower=0.01,0.02,...,0.20`,
`theta_final=0.80,0.81,...,0.99`, and fixed `theta_upper=1`. Supply
`theta_lowers` and `theta_finals` to change those grids. Lower cutoffs must not
exceed the upper cutoff; all cutoffs lie in `[0,1]`. The first default assessment
is at 10 patients, then every patient. `interim_looks` supplies a common explicit
schedule: each size uses only listed counts below that size, then adds its final
look. An empty schedule requests only the final assessment. Omitted schedules
use `min_subjects` and `cohort_size` at each size.

Specified designs allow up to 1,000 patients; searches allow increasing size
grids in `1..200` and at most 10,000 cutoff pairs per size. Sizes below the default
first assessment require an explicit schedule or smaller `min_subjects`.
`Phase2PredictiveInfeasibleError` distinguishes no feasible grid candidate from
invalid inputs or numerical failures. This is finite-grid optimization, not a
claim of global optimality over all continuous cutoffs and schedules.

Validation compares the predictive table with exact rational beta-binomial sums,
checks strict final-cutoff equality and the absence of early efficacy at an upper
cutoff of one, enumerates all response paths for small designs, and independently
checks both grid objectives and all reported equivalent cutoff pairs. The
existing BEMPR monitoring tests also pass with their original conventions.
