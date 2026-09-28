# UAROET trial replay and operating characteristics

`run_uaroet_trial` simulates one complete-outcome trial using the
[UAROET posterior and allocation rules](uaroet.md). `simulate_uaroet` runs trials
serially and returns compact operating characteristics. Both require a joint
truth table, utility table, prior parameters, starting dose, analysis schedule
and a nonincreasing utility-tolerance schedule.

Truth axes are `(dose, efficacy_category, toxicity_category)`. Each dose's table
must sum to one. Each patient receives one paired ordinal outcome sampled from
that table; efficacy and toxicity are not generated independently. This supports
associated outcomes even when an illustrative fitted model fixes association.

```python
from mdanderson_stats import run_uaroet_trial

trial = run_uaroet_trial(
    truth=[[[0.56, 0.14], [0.24, 0.06]], [[0.36, 0.24], [0.24, 0.16]]],
    utility=[[0.2, 0], [1, 0.4]],
    prior_mean=[-0.2, 0.3, -1, 0.2],
    prior_sd=[0.7, 0.3, 0.7, 0.3],
    look_sizes=[3, 6, 9],
    utility_tolerance=[1, 0.8, 0.5],
    starting_dose=0,
    toxicity_limit=0.5,
    p_L=0,
    p_U=0.8,
    good_utility_cutoff=0.4,
    association=0,
    draws=64,
    warmup=32,
    chains=2,
    rng=17,
)
print(trial.assigned_dose)
print(trial.selected_dose, trial.stop_reason)
```

These short chains illustrate the interface. Priors, cutoffs, schedule and
sampler precision need study-specific calibration; defaults do not establish
that calibration. Inspect the per-look sampler diagnostics before interpreting
decisions. A completed fit is not evidence of convergence.

## Enrollment and stopping

All patients before the first look receive the supplied starting dose. Each
scheduled analysis fits accumulated complete outcomes. Between looks, each
patient is randomized separately using the latest allocation probabilities.
Those probabilities remain fixed until the next analysis.

The last `look_sizes` entry is maximum planned enrollment. An empty eligible
set before that point stops the trial without selecting a dose. No selection
at the final analysis is distinct from stopping early. Final selection uses
the explicit `final_rule="acceptable"` or `"paper_global"` convention described
in the allocation guide. An interim best dose is not a final selection.

The trial retains patient assignments, assignment probabilities, paired outcome
categories, accumulated counts and compact per-look decision/diagnostic records.
Posterior draw tensors are discarded after each look. Optional `outcome_uniforms`
and `allocation_uniforms` arrays provide one value in `[0,1)` for every planned
patient, indexed by patient position. They support deterministic path checks;
the ordinary interface only requires `rng`.

## Compact simulation

```python
from mdanderson_stats import simulate_uaroet

simulation = simulate_uaroet(
    truth=[[[0.5, 0.1], [0.1, 0.3]]],
    utility=[[0.2, 0], [1, 0.4]],
    prior_mean=[0.2, -0.7],
    prior_sd=[1.1, 0.8],
    look_sizes=[2, 4, 6],
    utility_tolerance=[0, 0, 0],
    starting_dose=0,
    toxicity_limit=1,
    p_L=0,
    p_U=0.8,
    good_utility_cutoff=0.4,
    association=0,
    trials=4,
    draws=32,
    warmup=16,
    chains=2,
    rng=18,
)
print(simulation.selection_probability, simulation.no_selection_probability)
print(simulation.mean_enrollment, simulation.early_stop_probability)
```

Selection probabilities use all simulated trials as their denominator, with
no-selection reported separately. Early stopping means actual enrollment is
less than planned enrollment. Planned allocation shares divide average dose
counts by planned enrollment, so they can sum to less than one when trials
stop early. Outcome and utility means pool observed patients within each dose;
an unassigned dose has an unavailable mean. They are descriptive adaptive-trial
summaries, not unbiased estimates of treatment effects.

Simulation retains per-trial selection, enrollment, dose counts, stopping
reasons and diagnostics, but not every patient path or posterior. Binomial
Monte Carlo standard errors accompany selection and stopping probabilities;
the enrollment mean has its across-trial standard error. Tiny examples above
demonstrate behavior and do not estimate precise operating characteristics.

## Reproducibility and limits

One integer seed or NumPy generator derives separate outcome, allocation and
posterior streams. The result records their seeds. Changing posterior sampling
effort does not directly consume the patient-generation streams; changed
posterior decisions can still change enrollment and outcomes. Native R/C++
random-stream parity is not claimed.

Limits include five doses, two to four categories per endpoint, 10,000 planned
patients, 200 analysis looks and 10,000 simulated trials. Per-fit retained joint
draws are limited to two million cells. `max_fit_evaluations` bounds combined
likelihood/rectangle work per analysis; `max_total_evaluations` limits actual
work across the trial or whole simulation. Minimum planned work and retained
summary sizes are checked before running. Exhausted work or failed posterior
evaluation raises an error rather than producing a substitute decision.

Independent base-R trial references cover paired-outcome mapping, accumulated
counts, terminal selection and the distinction between early stopping and final
no-selection. At fixed zero association, scalar integration provides posterior
utility references. See the [simulation contract](../research/uaroet-simulation-audit.md)
and `tools/reference_uaroet_trial.R`.

This is a complete-outcome Python scheduler. Delayed observations, native prior
files, pseudo-trial/ESS elicitation, adaptive posterior precision and native
report/executable parity remain outside this extension. Catalog entry 92 remains
partial while those workflows are open.
