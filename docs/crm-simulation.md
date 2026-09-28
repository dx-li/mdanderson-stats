# CRM trial replay and simulation

`run_crm_trial` conducts an entire CRM, BMA-CRM or DA-CRM study from specified
patient arrivals and potential toxicity outcomes. `simulate_crm` generates
patients and summarizes repeated trials. They use the same
[calendar decisions](crm-conduct.md), rather than a separate approximation to
the posterior or allocation rules.

## Replay a study

```python
import numpy as np
from mdanderson_stats import run_crm_trial

trial = run_crm_trial(
    [0.1, 0.25, 0.5],
    interarrival=[0, 0.1, 0.1, 0.1, 0.1, 0.1],
    dlt_delays=np.full((6, 3), np.inf),
    window=3,
    target=0.3,
    cohort_size=3,
)
print(trial.assigned_doses)
print(trial.selected_dose, trial.final_time)
```

Each row of `dlt_delays` represents one potential patient and each column a
dose. A finite value is the delay from treatment to DLT; positive infinity
means no DLT throughout the assessment window. The simulation only reveals
the potential outcome at the assigned dose and only when it becomes observed.
Dose indices are zero-based. A trial permits up to 200 patients and 20 doses,
with cohorts of one to four patients and complete planned cohorts.

All time inputs use the same unit. The first enrollment occurs after the
first interarrival gap. Patients within a cohort arrive at their supplied gaps
and receive that cohort's fixed dose. Before a new cohort starts, the runner
analyzes the information available at the prospective patient's arrival.

When that decision says to wait, enrollment is suspended until the next
already-enrolled patient's outcome becomes known. The decision is then
recomputed. The prospective patient is treated as soon as the wait resolves;
later arrival gaps run from actual enrollment. There is no backlog of patients
accumulating during suspension. These are explicit Python scheduling choices;
the native program's internal scheduler is not reproduced.

After maximum enrollment, the runner completes follow-up before selecting an
MTD. A safety stop prevents any further enrollment and returns no MTD, even
if subsequent outcomes would change the recommendation. Already-enrolled
patients are still followed to ascertain their outcomes. `decision_time` is
therefore distinct from `final_time`; `suspension_time` counts enrollment waits
and excludes ordinary final follow-up.

The default method is CRM when one skeleton is supplied and BMA-CRM with
multiple skeletons. [BMS and Occam-window aggregation](crm-model-selection.md)
are also available for this route. Pending outcomes use exact bounded look-ahead. For DA-CRM,
set `method="dacrm"` and supply a matching `da_prior`, explicit
`minimum_observed`, and a NumPy random generator. Completed outcomes use
deterministic CRM integration. The newer CRM Suite decision policy applies;
the [older desktop safety-wait exception](dacrm.md) is a separate behavior.

## Repeated trials

```python
from mdanderson_stats import simulate_crm

simulation = simulate_crm(
    [[0.1, 0.2, 0.35], [0.08, 0.25, 0.5]],
    true_toxicity=[0.05, 0.3, 0.55],
    window=3,
    accrual_rate=2,
    target=0.3,
    cohorts=2,
    cohort_size=3,
    trials=3,
    late_probability=0.7,
    rng=6401,
)
print(simulation.selection_probability)
print(simulation.no_selection_probability)
print(simulation.mean_patients)
```

Repeated simulation uses Weibull toxicity timing and exponential arrival gaps
by default, matching the families described by the
[CRM Suite guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/CRMSuite/BMA-CRMSimulatorHelp.pdf).
The rate is patients per unit of the supplied window: for a window in days,
use a daily rate. No implicit days-to-months conversion is made.

The Weibull calibration satisfies `F(window) = true_toxicity` and
`F(window / 2) = (1 - late_probability) * true_toxicity`. Thus
`late_probability` is conditional on DLT occurring within the window. This
calibration supports zero toxicity but requires probabilities below one;
uniform conditional timing supports both probability endpoints.

```python
from mdanderson_stats import dacrm_uniform_prior

da_simulation = simulate_crm(
    [0.1, 0.25, 0.5],
    [0.05, 0.3, 0.55],
    window=3,
    accrual_rate=2,
    target=0.3,
    method="dacrm",
    da_prior=dacrm_uniform_prior(3),
    minimum_observed=1,
    cohorts=2,
    cohort_size=1,
    trials=2,
    rng=6402,
    sampler_rng=np.random.default_rng(8402),
    draws=128,
    warmup=128,
)
print(da_simulation.max_dose_mcse)
print(da_simulation.max_split_rhat)
```

Patient generation and DA sampling use separate random generators. Use distinct
seeds for these two streams. Changing DA sampling effort does not consume
patient-generation random numbers.
Reproducibility still depends on the supplied generators, numerical libraries
and configuration; native random-sequence parity is not claimed.

Selection probabilities use all simulated trials as their denominator.
No selection is distinct from dose zero. A safety decision after maximum
enrollment contributes to no selection, but not to early termination.
Final follow-up duration and time to the trial decision remain separate
quantities. Small simulation runs illustrate the interface; they do not
provide precise operating-characteristic estimates.

## Numerical work and coverage

Trials run serially. Potential outcomes for one trial and compact decision
summaries are retained; posterior draw arrays are released between decisions.
`CRMTrialStep` records the action, dose, available counts and inference route.
Look-ahead steps report the observed-data posterior probability that the lowest
dose exceeds the target; completion-specific evidence is used inside the
look-ahead decision.
For DA steps, `max_dose_mcse` and `max_split_rhat` summarize the dose-probability
Monte Carlo standard errors and split R-hat. Undefined diagnostics remain NaN,
and deterministic steps have no sampling diagnostics. Assess these values when
choosing sampling effort; a completed run alone does not establish convergence.
The simulation reports the largest DA diagnostics per trial; NaN can indicate
an undefined diagnostic or that no DA sampling was needed for that trial.
Explicit per-decision, per-trial and simulation-wide evaluation budgets bound
work. An exhausted budget raises an error and does not return an incomplete
run disguised as a completed simulation.

Default budgets are 200,000 evaluations per decision, 2 million per trial and
20 million per simulation. Explicit settings can raise these to 2 million,
20 million and 200 million respectively. Simulations permit at most 10,000
trials. These are work bounds, not claims that any particular run will achieve
adequate Monte Carlo precision within them.

This adds scientific simulation coverage for the CRM components of catalog
entries 81 and 132. Native saved files/reports, older-version differences and
the online entry's automatic skeleton calibration and hidden conventions remain open.
See [source provenance](crm-simulation-sources.json).
