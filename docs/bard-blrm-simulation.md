# BARD BF-BLRM operating-characteristic simulation

`simulate_bard_blrm` runs serial stochastic two-stage BF-BLRM trials and
returns compact operating-characteristic summaries. It composes the existing
BF-BLRM posterior/calendar replay with the existing BARD stage-two allocation
and OBD selection. It does not change the posterior model or sampler.

```python
from mdanderson_stats import (
    BARDLogisticPrior,
    BARDStageTwoDesign,
    BARDBLRMSimulationDesign,
    bard_response_model,
    simulate_bard_blrm,
)

profiles = [[1, 1], [1, 2], [2, 1], [2, 2]]
response = bard_response_model([0.20, 0.35, 0.50], profiles, [0.25] * 4, [[1.0, 1.5], [1.0, 0.8]])
stage_two = BARDStageTwoDesign(
    total_target=6,
    eligible_profiles=[True] * 4,
    prior=[1, 1, 1, 1],
    safety_weights=[1, 1],
    toxicity_limit=0.30,
    efficacy_limit=0.20,
    safety_cutoff=0.99,
    efficacy_cutoff=0.99,
    utilities=[0, 30, 50, 100],
    margin=0.05,
    tie_arm=1,
    stage_two_accrual_rate=2.0,
    dose_pair=(1, 2),
    balanced_factors=(0,),
)
settings = BARDBLRMSimulationDesign(
    doses=[1, 2, 3],
    reference_dose=1,
    prior=BARDLogisticPrior([-2.0, -0.6931471805599453], [0.0, 0.0]),
    target_interval=[0.20, 0.30],
    eta=0.45,
    cohort_size=3,
    max_escalation_patients=9,
    backfill_evaluable_cap=2,
    draws=8,
    warmup=0,
    chains=2,
    max_arrivals=40,
)
summary = simulate_bard_blrm(
    settings,
    true_toxicity=[0.10, 0.22, 0.30],
    response_model=response,
    stage_two=stage_two,
    true_obd_noninferiority=2,
    true_obd_utility=2,
    trials=2,
    rng=165,
)
print(summary.mean_total_enrollment)
print(summary.noninferiority_accuracy.correct_all_trials)
print(summary.utility_accuracy.correct_all_trials)
```

The simulator retains no patient-level ledgers. Selection and correct-selection
probabilities use all valid trials; trials with no MTD, no available dose pair,
or a safety-rejected pair remain in the denominator as no selection. Accuracy
among selected trials is reported separately. Arm-count imbalance includes
valid pairs with an empty arm; factor-level imbalance is defined only when both
arms contain patients and is reported for every modeled factor, whether or not
the factor is used for minimization.

Pass an integer seed for a reproducible local stream or a NumPy `Generator` to
advance caller-owned state. Trials execute serially. The patient-work bound is
preflighted for the whole request. Evaluation and work budgets are also
aggregate: remaining budget is passed into each trial, and exhaustion raises
instead of returning a partial result as complete. Stage-one arrival-tape
exhaustion likewise raises; increase `max_arrivals` rather than treating a
truncated trial as a completed replicate.

The result reports total posterior fits, likelihood evaluations and work
units, maximum observed POD/PTT Monte Carlo errors, the maximum finite split
R-hat, and the number of trials with an undefined or infinite R-hat. These are
diagnostics, not convergence guarantees, and do not filter trials.

The paper's simulation uses five doses, the interval `(0.16, 0.33)`, the
weakly informative prior `N((-1.1, 0), diag(4, 1))` on log coefficients, and
EWOC cutoff 0.30. It says the BF-BLRM stage-one cap was calibrated to match the
mean stage-one sample size of BF-BOIN, but does not give the resulting cap.
Supply the cap explicitly. With the paper prior and raw-ratio model, prior
screening may stop before enrollment; the simulator preserves that result
rather than changing the prior. The paper also does not specify a stage-two
calendar law or a joint DLT-response law. The Python workflow uses its
documented arrival/assessment conventions; when no joint endpoint
probabilities are supplied, it assumes conditional independence of DLT and
response given dose and factor profile. Supply a joint probability table to
model association. These are explicit Python choices, not recovered native
defaults. This implementation reports the BF-BLRM paper study as a
reproducible community workflow under those settings, not exact Table 4 or
native-app replication.

The computational cost is dominated by repeated posterior fits. Use modest
trial counts to inspect a configuration before increasing the count and
aggregate budgets. The paper's 30,000-trial study is not a recommended local
default.
