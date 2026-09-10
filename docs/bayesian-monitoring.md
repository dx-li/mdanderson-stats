# Bayesian toxicity and efficacy monitoring

Catalog entries **100 (BTOX)**, **109 (BEMPO)** and **110 (BEMPR)** provide three
related binary-outcome monitoring methods. Their statistical workflows are
implemented through `toxicity_monitoring_design`, `posterior_efficacy_design`
and `predictive_efficacy_design`. Source specifications are the official
[BTOX](https://biostatistics.mdanderson.org/shinyapps/BTOX/BTOX.pdf),
[BEMPO](https://biostatistics.mdanderson.org/shinyapps/BEMPO/BEMPO.pdf) and
[BEMPR](https://biostatistics.mdanderson.org/shinyapps/BEMPR/BEMPR.pdf) manuals,
by Yanhong Zhou and J. Jack Lee, pinned in
[the provenance record](bayesian-monitoring-sources.json).

These are independent Python statistical implementations. Function inputs replace
browser forms, and arrays provide boundary tables, operating-characteristic
summaries/distributions and monitoring histories for analysis or plotting. The
historical CSV input templates, PDF/Excel report layouts and interactive browser
controls are not reproduced. No unavailable Shiny server source is claimed as a
reference implementation.

## Decision rules

With Beta(a,b) prior and r events among n patients, the posterior is
Beta(a+r,b+n-r). The source's distinctions between strict and inclusive tests
are preserved:

| Method | Early low boundary | Early high boundary | Final positive conclusion |
|---|---|---|---|
| BTOX | None | P(toxicity rate > target) >= cutoff | Same excessive-toxicity criterion |
| BEMPO | P(response rate <= futility target) > cutoff | P(response rate > efficacy target) >= cutoff | P(response rate > final target) >= final cutoff |
| BEMPR | Predictive success probability < lower | Predictive success probability >= upper | P(response rate > target) >= final cutoff |

A final positive conclusion means **excessive toxicity** for BTOX and **efficacy**
for BEMPO/BEMPR. The final rule supersedes early rules at maximum sample size.
Contradictory early futility/positive boundaries at a scheduled analysis are
rejected instead of imposing an undocumented priority. The two efficacy methods
allow `stop_futility=False` and/or `stop_efficacy=False` while preserving the final
analysis rule.

## Designs and boundaries

```python
from mdanderson_stats import (
    toxicity_monitoring_design,
    posterior_efficacy_design,
    predictive_efficacy_design,
)

toxicity = toxicity_monitoring_design(15)
print(toxicity.looks)  # [5, 10, 15]
print(toxicity.positive_min)  # [2, 4, 6]

posterior = posterior_efficacy_design(15)
predictive = predictive_efficacy_design(15)
print(predictive.final_positive_min)  # 6 responses at N=15

# Explicit irregular analyses, including the final analysis.
irregular = predictive_efficacy_design(30, looks=[4, 9, 16, 23, 30])
```

Defaults follow the displayed application settings: Beta(0.5,0.5) prior, first
analysis at five patients, cohorts of five, and the corresponding method's rate
and probability cutoffs. Maximum sample size supports 1 through 1,000. For smaller
trials specify `min_subjects` appropriately. Without explicit `looks`, analysis
sizes start at `min_subjects`, increase by `cohort_size`, and include the maximum
sample size even for a shorter last cohort. Explicit `looks` must increase and
end at that maximum. Inputs use probabilities in [0,1].

`futility_max` and `positive_min` give inclusive boundaries at each look. A low
boundary of -1 or high boundary of n+1 indicates that no attainable count triggers
that rule. The separate `final_positive_min` is authoritative at the final look.
All posterior and predictive probability tables have axes (sample size, events),
including n=0; impossible r>n entries are NaN. Stored arrays are read-only.

## Predictive probability

BEMPR evaluates the probability that the final posterior criterion will be met,
assuming the remaining patients are accrued without intervening stopping. This
is different from the operating-characteristic probability of actually declaring
efficacy under the sequential design.

At n=N, the predictive probability is the zero/one final decision. Working
backward,

```text
q(n,r) = (a+r)/(a+b+n) * q(n+1,r+1)
       + (b+n-r)/(a+b+n) * q(n+1,r)
```

This integrates over the beta-binomial future outcomes exactly, without Monte
Carlo. Each row is vectorized; construction takes O(N squared) work and storage.
Probability-one states are preserved exactly so inclusive endpoint cutoffs remain
meaningful. Direct beta CDF and upper-tail evaluations avoid subtracting tiny
posterior tails from one.

## Trial monitoring

```python
state = predictive.monitor(events=1, sample_size=2)
print(state.final_probability)  # approximately 0.7476842
print(state.high_probability)  # predictive probability, approximately 0.6738275
print(state.decision)  # continue: first scheduled look is at n=5

history = predictive.monitor_outcomes([0, 1, 0, 0, 1])
```

`monitor` broadcasts cumulative counts and returns posterior shapes, low/high
monitoring probabilities, the current posterior probability for the final target,
and a decision. An unscheduled observation reports probabilities but does not
trigger a stopping decision. `monitor_outcomes` accepts zero/one outcomes on a
final patient axis and retains the first stopping decision. Subsequent probability
entries, if supplied after stopping, describe the supplied hypothetical data;
they do not reopen a stopped trial. For BEMPR low/high probabilities both refer
to q(n,r), while `final_probability` is the current posterior tail, not q(n,r).

## Exact operating characteristics

```python
oc = toxicity.operating_characteristics([0.1, 0.3, 0.5])
print(oc.positive_conclusion)  # approximately [0.0849, 0.5633, 0.9325]
print(oc.expected_sample_size)  # approximately [14.1700, 9.9627, 6.4551]
print(oc.expected_observed_rate)  # approximately [0.1178, 0.3602, 0.5400]
median_n = oc.sample_size_quantile(0.5)
```

Forward Bernoulli recursion propagates probability over event counts and removes
stopped paths at scheduled analyses. It returns early low/high stopping mass by
look, final positive/negative probabilities, the sample-size distribution,
expected sample size and standard deviation, expected event count, and expected
observed event rate E[R/T]. The latter generally differs from E[R]/E[T] because
stopping depends on outcomes. `positive_conclusion` combines early high stopping
and final positives; final observations are never counted as early stopping.
Scenario probabilities broadcast and are propagated together. No simulation,
probability pruning or approximation to the scheduled stopping rule is used.

## Validation and scope

Thirteen focused tests include the BEMPR predictive-probability example and BTOX's
published operating characteristics. Independent exact rational beta-binomial
sums validate the backward recursion. Exhaustive binary sample paths validate
all three designs' stopping probabilities, final decisions, event counts, observed
rates and sample sizes, including true rates zero and one. Further checks cover
strict/inclusive thresholds, disabled stopping, overlap rejection, irregular
looks, final-rule precedence, batching and persistent stopping decisions.

The implementation covers all three manuals' statistical methods and numerical
outputs. Source-specific file templates and report rendering remain outside this
Python numerical API; callers can export the returned arrays using their existing
analysis tools.
