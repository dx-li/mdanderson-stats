# BaCIS trial operating characteristics

`simulate_bacis_oc` generates independent binomial response counts for each
subgroup, applies the existing two-stage BaCIS analysis and summarizes the
resulting classification and efficacy decisions. Its inputs are the true
response probabilities and each subgroup's patient count. It runs one trial
at a time and retains compact results rather than posterior fits.

```python
import numpy as np
from mdanderson_stats import simulate_bacis_oc

# A small single-subgroup workflow example; no between-group borrowing occurs.
result = simulate_bacis_oc(
    [0.3],
    trials_per_group=25,
    replications=32,
    draws=32,
    warmup=0,
    chains=2,
    outcome_rng=153,
    sampler_rng=1153,
)
assert result.successes.shape == (32, 1)
assert result.single_cluster_probability == 1
assert result.familywise_false_positive_probability is None
np.testing.assert_array_equal(result.efficacious[:, 0], result.successes[:, 0] >= 5)
print(result.power, result.power_mcse)
```

The small budgets demonstrate the workflow, not adequate precision for a
trial-design decision. For this special one-group case, the posterior is
exactly Beta(1+y, 1+n-y); under the default efficacy cutoff, y>=5 declares
efficacy. Enumerating binomial outcomes gives power .9095280814 at true
response .3 and false-positive probability .0979936212 at .1. The simulation
estimate varies because response counts are random, even though the
singleton efficacy decision uses the exact posterior tail.

## Model settings

Supply multiple probabilities, for example `[.1, .3, .3, .3, .3]`, to assess
a mixed scenario with between-group borrowing. `trials_per_group` is either
a common integer or one count per subgroup. Defaults use low/high response
centers .1/.3, hierarchical precision shape 50 and **rate** 2, classification
cutoff .5 and efficacy cutoff .92. The rate follows the paper/one-trial
wrapper, while the lower-level `bacis_fit` default follows the app's rate 10.

Set `classification_cutoff=None` for the archived package's adaptive cutoff;
`adaptive_weighting` selects its existing subgroup/patient weighting convention.
An efficacy declaration always means posterior `Pr(p_i > phi_low)` strictly
exceeds `efficacy_cutoff`. It does not also require high-cluster assignment.
No cutoff search or automatic error-rate calibration is performed.

## Reading the results

- `classification_high_probability` estimates assignment to the high cluster.
- `efficacy_probability` estimates an efficacy declaration for each subgroup.
- `false_positive_rate` contains those rejection rates for true probabilities
  at most `phi_low`; other positions are NaN because the label does not apply.
- `power` contains rejection rates for probabilities above `phi_low`, with
  NaN for null groups. Intermediate truths below `phi_high` still test an
  alternative to the stated null.
- `familywise_false_positive_probability` estimates any false rejection
  among null groups; it is `None` when there are no null groups.
- `single_cluster_probability` estimates all groups receiving one common
  cluster label.

Each rate has a binomial Monte Carlo standard error, estimating uncertainty
from the finite number of simulated trials. A reported standard error of
zero when all simulated indicators agree does not establish a zero-error
population probability. These errors do not measure posterior sampler bias.

Response counts and binary classifications/decisions remain available by
replication as read-only arrays. Group-response chain diagnostics are retained
as per-replication finite maxima of split R-hat and batch-means MCSE, plus
flags for nonfinite diagnostics. Inspect these and choose sufficient sampler
budgets for the intended application. A numerical fit failure stops the run;
failed replications are not silently discarded.

```python
ess_result = simulate_bacis_oc(
    [0.3], replications=32, draws=32, warmup=0, chains=2,
    outcome_rng=153, sampler_rng=1153, compute_ess=True,
)
print(ess_result.mean_equivalent_sample_size)
print(ess_result.equivalent_sample_size_mcse)
```

Set `compute_ess=True` to additionally retain the native fixed-response-count
variance-match ESS for every replication and subgroup. The result exposes
`equivalent_sample_size_by_replication`, its per-subgroup arithmetic mean
`mean_equivalent_sample_size`, and the standard error of that mean in
`equivalent_sample_size_mcse`. With one replication the MCSE is `None`. An ESS
failure identifies the replication and subgroup and stops the request; the
summary never silently drops a failed ESS. This averaging rule is an explicit
Python summary convention. The archived one-trial package returns one ESS per
subgroup, and the paper reports subgroup-specific ESS by scenario, but no
native simulation aggregation implementation is available to establish exact
table-generation parity.

Outcome generation and sampler seeds use separate random streams. Supply
distinct seeds or independent generators for reproducibility. Resource and
model settings are validated before random generation; per-fit allocations,
retained results and total sampler work have explicit limits. Large research
runs may require separate independent batches; combine event counts with
their total replication count rather than averaging unequal batch rates.

## Source scope

The [simulation audit](../research/bacis-simulation-audit.md) records the six
published scenarios, independent singleton reference and the paper/software
classification discrepancy. The archived package has no operating-characteristic
driver, and the paper's stated fixed-cutoff model conflicts with its reported
classification rates. This simulator implements the archived model under
explicit settings; it does not certify reproduction of the paper's tables.
Automatic cutoff calibration and native reports remain outside this interface.
Catalog entry 153 remains partial.
