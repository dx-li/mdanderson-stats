# CiBolus complete-outcome trials

`simulate_cibolus_trial` generates response and toxicity jointly from explicit
truth parameters, fits the [CiBolus model](cibolus.md) after each cohort and
applies its allocation and final-selection rules. Both outcomes are assumed
known at each cohort boundary. Calendar time, pending outcomes and the later
toxicity assessments in the motivating study are separate remaining scope.

## Reproduce a small trial

This example fixes the prior at the truth parameters to isolate the allocation
path. It is an exact replay example, not a calibrated prior. Positive prior
standard deviations enable posterior learning; sampling settings and diagnostics
then determine the accuracy of estimated utilities and screening probabilities.

```python
import numpy as np
from mdanderson_stats import CiBolusPrior, simulate_cibolus_trial

truth = np.log([0.5, 0.7, 0.8, 0.08, 1.4, 1.6, 0.03, 0.9, 0.12, 0.25, 0.2])
trial = simulate_cibolus_trial(
    truth,
    CiBolusPrior(truth, np.zeros(11)),
    concentrations=[0.2, 0.4, 0.8],
    bolus_fractions=[0.1, 0.6],
    endpoints=[0.25, 0.5, 0.75, 1],
    utility=[[100, 0]] * 5 + [[0, 0]],
    n_patients=4,
    cohort_size=2,
    starting=(0, 0),
    toxicity_limit=0.8,
    toxicity_cutoff=0.9,
    efficacy_limit=0.01,
    efficacy_cutoff=0.9,
    draws=8,
    warmup=0,
    chains=2,
    rng=np.random.default_rng(86),
    outcome_uniforms=[0, 0.2, 0.6, 0.95],
)
assert [patient.regimen for patient in trial.patients] == [(0, 0), (0, 0), (1, 1), (1, 1)]
assert trial.final_pair == (2, 1)
assert len(trial.steps) == 2 and not trial.early_stopped
```

Regimen indices are zero-based `(concentration, bolus fraction)` pairs. The
first cohort uses the configured starting pair. Later allocations cannot skip
an untried concentration; bolus fractions are unrestricted. Final selection
removes the concentration restriction, so this example recommends an untried
regimen. Utility ties use the existing lexicographic convention.

Response categories are bolus response, each interval in endpoint order, and
failure by time one. Each category is crossed with toxicity absent/present.
An interval observation uses its upper endpoint for toxicity and delivered
treatment. This preserves the model's response/toxicity dependence.

## Outcomes and posterior randomness

The optional `outcome_uniforms` vector contains one value in `[0,1)` for every
requested patient. If omitted, the supplied generator draws that vector before
posterior sampling. The full vector is returned, including unused values after
an early stop. Subsequent random draws drive the posterior fits. Reusing an
outcome vector permits comparisons without changing the random outcome inputs;
different selected regimens can still produce different observations.

The last cohort can be smaller than `cohort_size` to reach `n_patients` exactly.
One fit is performed after each completed cohort, and the last fit directly
supports final selection. If an interim decision has no eligible regimen,
the trial stops without a recommendation; unrestricted final selection does
not reopen it. `steps` retains the acceptable and eligible masks so the reason
for an exclusion can be inspected.

## Resources and interpretation

The simulator retains patient observations and compact cohort decisions,
including utility uncertainty diagnostics and work counters. It discards the
full posterior fit before the next cohort fit. Required `draws`, `warmup` and
`chains` expose sampling precision; a completed run is not a convergence
guarantee. Fixed parameters have undefined split R-hat.

The existing fitter's 200-patient limit applies. Input grids, retained fit and
decision arrays, cumulative likelihood evaluations and prediction work are
bounded. Inadequate budgets and numerical failures raise errors; sampling
precision is never silently reduced. No parallel workers are started.

Independent base-R quadrature and explicit outcome vectors provide two trial
references, including final selection at an untried regimen and stopping after
an unsafe first-cohort analysis. The [trial audit](../research/cibolus-trial-audit.md)
records the comparison status. Native executable output, native random streams,
large-population validation and prior calibration remain separate.

## Aggregate operating characteristics

`simulate_cibolus_operating_characteristics` repeats the same complete-outcome
trial serially, discards patient and posterior histories between replicates,
and returns selection, stopping, enrollment and observed-outcome summaries.
For example, continuing with `truth` from above:

```python
from mdanderson_stats import simulate_cibolus_operating_characteristics

oc = simulate_cibolus_operating_characteristics(
    truth,
    CiBolusPrior(truth, np.zeros(11)),
    concentrations=[0.2, 0.4, 0.8],
    bolus_fractions=[0.1, 0.6],
    endpoints=[0.25, 0.5, 0.75, 1],
    utility=[[100, 0]] * 5 + [[0, 0]],
    n_patients=4,
    cohort_size=2,
    toxicity_limit=0.8,
    toxicity_cutoff=0.9,
    efficacy_limit=0.01,
    efficacy_cutoff=0.9,
    trials=3,
    draws=8,
    warmup=0,
    chains=2,
    rng=np.random.default_rng(8603),
)
assert oc.selection_count.sum() + oc.no_selection_count == 3
assert oc.assigned_patients.sum() == 12
assert oc.mean_enrollment == 4
print(oc.selection_probability, oc.selection_mcse)
```

Three replicates demonstrate the interface, not precise design evaluation.
Selection, no-selection and stopping probabilities have binomial Monte Carlo
standard errors. `mean_allocation` reports mean patients per trial at each
regimen, with a standard error across trials. Toxicity, response and response
category probabilities pool observed outcomes over assigned patients at that
regimen. Their ratio standard errors use trials as independent clusters, so
varying enrollment and within-trial dependence are retained. Unassigned
regimens have undefined rates and standard errors (`NaN`). With one replicate,
sample-based enrollment, allocation and ratio errors are also undefined.

For pooled counts `Y_b` and assigned patients `N_b`, the reported rate is
`p = sum(Y_b)/sum(N_b)`. Its Monte Carlo variance estimate is
`B/(B-1) * sum((Y_b - p*N_b)**2) / sum(N_b)**2`, for `B > 1`. These are
realized rates under adaptive allocation, not substituted model-truth risks.
Monte Carlo errors do not include bias from inaccurate posterior fitting.

`trial_seeds[i]` recreates replicate `i` by passing
`np.random.default_rng(int(oc.trial_seeds[i]))` to `simulate_cibolus_trial`
with the same design inputs and no explicit outcome-uniform vector. Aggregate
results retain the largest parameter/utility Rhat and utility MCSE, including
infinite Rhat, plus the number of steps with an undefined diagnostic. Fixed
coordinates naturally have undefined Rhat. These summaries are screening
diagnostics, not proof of convergence; individual replicates remain replayable.

Array limits cover aggregate statistics together with one trial's retained
state. Likelihood and work budgets apply across the entire run, including
initial truth validation. They do not reset per replicate. The aggregate
interface adds no calendar assumptions or native random-stream equivalence.
