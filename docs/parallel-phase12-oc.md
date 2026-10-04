# Parallel Phase I/II: four-arm operating characteristics

`simulate_parallel_phase12_oc` repeatedly runs the archived four-arm
[beta-binomial design](parallel-phase12.md) and returns compact aggregate
results. It retains the existing phase-I escalation, phase-II allocation,
toxicity closure and efficacy/futility rules. The separate six-dose C++
calendar design is not the model used by this function.

```python
import numpy as np
from mdanderson_stats import simulate_parallel_phase12, simulate_parallel_phase12_oc

truth = dict(
    toxicity_probability=[0.04, 0.09, 0.16, 0.25],
    efficacy_probability=[0.10, 0.20, 0.35, 0.50],
)
summary = simulate_parallel_phase12_oc(**truth, n_trials=4, seed=8503, optimal_arms=(3,))
np.testing.assert_allclose(
    summary.selection_probability.sum() + summary.no_selection_probability,
    1,
)
assert summary.total_enrollment == summary.treated_total.sum()
first = simulate_parallel_phase12(
    **truth,
    rng=np.random.default_rng(summary.per_trial_seeds[0]),
)
assert first.phase == "complete"
```

Four trials illustrate the API, not a precise design evaluation. A supplied
integer seed generates independent per-trial seeds, each of which reproduces
one complete trial with the ordinary simulator. All simulations run
sequentially. Only the seed vector and aggregates are retained; patient
histories from prior trials are released.

## What the summaries mean

Arm arrays use zero-based indices 0–3. Selection probabilities have one entry
per arm, with no selection reported separately. Stopping reasons distinguish
efficacy, futility, no admissible arms and reaching the maximum sample size.
Final admissibility is also summarized. All probabilities use every simulated
trial as their denominator.

`optimal_arms` is optional and must explicitly identify the arms the caller
regards as desirable. Its reported probability means only that a member of
that set was selected. The function does not invent a success criterion from
the supplied toxicity and efficacy rates.

Enrollment, toxicity and response totals are provided by arm, together with
mean counts and Monte Carlo standard errors. Phase-I enrollment includes
patients treated before an early phase-I stop. Mean total enrollment averages
actual trial sizes, including early stops.

The pooled toxicity and response rates divide event totals by treated-patient
totals. These differ from averaging each trial's arm-specific proportion,
particularly when adaptive allocation changes the denominators. Their Monte
Carlo errors use independent trials as clusters. For event count `Y_b`, arm
enrollment `N_b`, pooled rate `r` and `B` trials, the squared standard error is

```text
B / (B - 1) * sum_b (Y_b - r*N_b)**2 / (sum_b N_b)**2.
```

A never-enrolled arm has an undefined rate and standard error (`NaN`). One
trial cannot estimate a Monte Carlo standard error. Reported errors describe
finite simulation uncertainty, not uncertainty in the supplied scenario or
the statistical assumptions of the trial design.

## Source scope and limits

The archived C program writes final selection, a response-tail summary,
total enrollment and per-arm enrollment/toxicity/response/admissibility. The
Python aggregate workflow summarizes the validated trial state and adds
explicit uncertainty estimates and replay seeds. It does not reproduce the
native text-file layout, stale reporting fields, external RNG or the separate
six-dose application's reports.

The single-trial core was checked against 24 original-C control-flow histories
and independent R beta comparisons. Aggregate checks exercise certain-toxicity
stopping, seed replay, all-trial denominators and trial-clustered rate errors.
At most 10,000 trials and one million worst-case patient assignments are
allowed per call. Posterior comparison caches are local to one trial.

The [named scenario report](parallel-phase12-scenario-report.md) captures inputs,
seeds and operating characteristics, and parses the native eight-probability
input order. The separate six-dose native kernel summaries, its DF3+3 comparison
and full published operating-characteristic replication remain open. Entry 85
remains partial; see the [crosswalk](../research/parallel-phase12-scenario-report-audit.md).
