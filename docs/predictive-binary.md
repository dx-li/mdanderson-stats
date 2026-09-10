# Predictive Probabilities: binary outcomes

Catalog **10** now supports the binary interim-analysis and first-stage planning
workflows. [Time-to-event prediction](predictive-survival.md) is also available.
Sources are the [version 1.5 guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/PredictiveProbabilit/PredictiveProbabilitiesUsersGuide.pdf)
and John Cook's [Predictive Probability Interim Analysis](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/PredictiveProbabilit/PredictiveInterimAnalysis.pdf),
with hashes in [provenance](predictive-probabilities-sources.json).
No native executable or source code is redistributed.

## Interim prediction

```python
from mdanderson_stats import predictive_binary

result = predictive_binary(
    successes=[10, 16],
    failures=[15, 9],
    planned_sizes=[50, 50],
    prior=[[0.6, 0.4], [0.6, 0.4]],
    method="frequentist",
    significance_level=0.05,
)
print(result.arm_a_superior, result.arm_b_superior, result.inconclusive)
# Approximately 0.0000033643, 0.6886101, 0.3113865.
```

All count inputs contain two entries ordered A, B. Planned sample sizes include
already observed patients. Priors are `[[a_A,b_A],[a_B,b_B]]`, defaulting to
independent uniform beta distributions. Current successes and failures update
these shapes before predicting future counts. The calculation sums over every
possible pair of future success counts, weighted by independent beta-binomial
predictive probabilities. It does not substitute posterior mean response rates
into binomial distributions.

`method="frequentist"` predicts the result of the pooled two-proportion z test
with a two-sided `significance_level`. A wins for a z statistic strictly above
the positive critical value; B wins below the negative value. All-success and
all-failure ties are inconclusive. Both planned arm sizes must be positive.

`method="bayesian"` instead uses strict final posterior superiority probability
above `posterior_cutoff`, default `0.95`. The cutoff must be in `[0.5,1)` so
conclusions cannot overlap. Independent beta ordering is computed by the
package's deterministic quadrature. Monotonicity in final B successes permits
binary searches for decision boundaries, avoiding a separate integral at every
terminal count pair. Near a cutoff, integer beta shapes up to 1,000 use an exact
rational beta identity, comparing with the decimal representation of the cutoff;
otherwise an unresolved numerical comparison raises `ArithmeticError`.

`future_success_probability_a` and `_b` contain probabilities for future counts
`0..remaining`. `final_decision` is a matrix indexed by those two future counts,
with values `arm_a_superior`, `arm_b_superior`, or `inconclusive`. All three
predictive probabilities are summed directly, preserving small probabilities
without subtraction from one. Future beta-binomial weights are formed through
normalized log recurrences to avoid cancellation between large log-beta values.

With no remaining patients, prediction is deterministic for the available final
data. Bayesian comparison permits a zero-size arm: its proper prior remains
uncertain and is compared against the other arm's posterior. This is not a
comparison against a fixed known response rate. There is no extra interim
stopping rule; predictions assume accrual continues to the specified totals.
An inconclusive final result is not itself an instruction to stop now.

## First-stage planning

```python
from mdanderson_stats import plan_predictive_binary

plan = plan_predictive_binary(
    interim_sizes=[10, 10],
    planned_sizes=[30, 30],
    prior=[[1, 1], [1, 1]],
    method="bayesian",
    posterior_cutoff=0.95,
)
# Conditional prediction after 3 of 10 responses on A and 7 of 10 on B.
print(plan.arm_a_superior[3, 7], plan.arm_b_superior[3, 7])
```

Rows index A's interim successes and columns B's. The returned axes
`successes_a` and `successes_b` describe the full table; ordinary array slicing
selects narrower success ranges. Each table cell is a conditional predictive
probability, not a probability of reaching that interim outcome. Final decision
boundaries are computed once, and matrix products with each arm's conditional
predictive kernel produce the entire table. This is a planning table, not a
sample-size optimizer or frequentist power calculation at fixed true rates.

Both interfaces support up to 200 planned patients per arm. Input sizes and
counts must be nonnegative integers; priors must be proper. The time-to-event model is documented [separately](predictive-survival.md).
The desktop application's report-history interface is not reproduced. Parameter Solver and Inequality Calculator are separate catalog
entries, not bundled UI utilities in this port.

Validation reproduces Cook's numerical example and beta(2,3) two-attempt
prediction, enumerates small Bayesian terminal tables using exact rational
probabilities, checks every corresponding planning/interim result, tests arm
symmetry, and covers zero future observations and strict-cutoff equality.
