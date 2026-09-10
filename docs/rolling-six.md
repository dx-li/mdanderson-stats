# Rolling Six conduct and simulation

Rolling Six is a comparator offered by the MD Anderson TITE-BOIN software
(catalog entry 129), and the BOIN desktop suite (99). The package implements its
patient-by-patient dose assignment and calendar simulation. This does not complete
the desktop suite or imply exact parity with the app's unpublished simulator.

The decision rules follow section 13.2 of the [PBTC-042 protocol](https://cdn.clinicaltrials.gov/large-docs/61/NCT02255461/Prot_000.pdf),
which gives the full Rolling-6 assignment table and cites Skolnik et al. (2008),
DOI 10.1200/JCO.2007.12.7712. The ordinary Rolling 6 column of
[Frankel et al., Table 2](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2765855)
also confirms the single-dose rules. Source hashes are in
[provenance](rolling-six-sources.json).

## Dose decisions

```python
from mdanderson_stats import RollingSixDesign

design = RollingSixDesign()
# Six enrolled, five known DLT-free, one still pending: escalation is allowed.
step = design.next_dose([6, 0], [0, 0], [1, 0], current_dose=1)
assert step.action == "escalate"
assert step.next_dose == 2

strict = RollingSixDesign(require_complete_before_escalation=True)
assert strict.next_dose([6, 0], [0, 0], [1, 0], 1).action == "suspend_pending"
```

Counts are vectors for 2–100 doses, with at most six patients per dose and
`toxicities + pending <= patients`. Dose numbering is one-based. The ordinary
rules are:

* Two observed DLTs exclude that dose and every higher dose.
* Three, four or five fully observed patients with no DLT permit escalation.
* At six patients, escalation requires either all outcomes known with at most
  one DLT, or five DLT-free outcomes and one pending outcome. The explicit strict
  variant disables the latter case.
* If escalation is unavailable and fewer than six patients are enrolled, the next
  patient receives the current dose. At capacity, enrollment waits for outcomes.

After exclusion, the highest remaining dose is filled to six patients and all its
outcomes are resolved before selection. No escalation crosses an exclusion.
Pass the returned `eliminated` mask into subsequent decisions to retain exclusions.
At the highest planned dose, six complete outcomes with at most one DLT produce
`select_highest`, rather than a claim that the MTD has been found. The protocol
itself distinguishes this case from identifying an upper toxicity limit.

`select_mtd(patients,toxicities)` requires complete outcomes and returns `dose`,
`status`, and exclusions. Status is `mtd` when the next higher dose is excluded,
`highest_planned_dose` when the top dose is recommended, `no_safe_dose` when the
lowest is excluded, or `inconclusive` when six outcomes are unavailable at the
highest admissible dose. No Bayesian prior, target-rate estimate or isotonic fit
is substituted for these algorithmic criteria.

## Calendar replay and simulation

```python
import numpy as np
from mdanderson_stats import run_rolling_six_trial, simulate_rolling_six

trial = run_rolling_six_trial(design, [1] * 12, np.full((12, 2), np.inf), window=10)
assert trial.enrollment_times[6] == 15
assert trial.suspension_time == 8
assert trial.final_time == 30
assert trial.selection_status == "highest_planned_dose"

simulation = simulate_rolling_six(
    design,
    [0.05, 0.15, 0.3, 0.45, 0.6],
    window=3,
    accrual_rate=2,
    event_distribution="weibull",
    late_probability=0.8,
    trials=1000,
    rng=6,
)
print(simulation.selection_probability)
print(simulation.duration.mean(), simulation.patients.sum(axis=1).mean())
```

The replay takes potential DLT delays shaped `(planned_patients,doses)` and
interarrival gaps measured from the previous actual enrollment; the first gap
starts at time zero. Delays in `[0,window]` indicate toxicity, and positive infinity
means no DLT during the window. Only assigned, observed outcomes influence a
decision. Enrollment is sequential, with one assignment per decision and no queue
accumulating during suspension.

A waiting candidate is reassessed at the next observed DLT or completed window.
When enrollment stops, all remaining outcomes are followed to ascertainment.
Decisions and stopping are recorded at candidate-arrival/reassessment times,
so the reported duration can exceed the instant at which a decision first became
possible. This is the shared Python scheduler's convention. It has not been
matched to the original software's scheduling and random-number sequence.
The patient cap can be reached before a terminal decision; final selection then
uses complete outcomes and can be inconclusive.

Simulation accepts fixed or exponential arrival gaps and the same uniform,
piecewise-uniform, Weibull and log-logistic scenarios as
[TITE-BOIN](tite-boin.md#calendar-replay-and-simulation). The default total cap is
`min(6*doses,200)`; `max_patients` can specify a smaller or larger cap up to 200,
while the per-dose limit remains six. At most 100,000 trials are accepted per call.
These simulations assume all patients are evaluable; dropout/replacement and
screening queues are not implemented.

Results retain per-trial patient and DLT counts, selection codes, durations,
suspension times, enrollment stop reasons and `selection_status`. Selection code
zero means no recommended dose. Probability bins are `[none,dose 1,...]` and
include a highest-dose recommendation; use `selection_status` when distinguishing
this from a found MTD. Marginal Monte Carlo standard errors accompany the bins.
The replay additionally retains individual enrollment/event times and decisions.

To compare with TITE-BOIN, run both designs under the same toxicity, timing and
arrival scenario and compare selection, allocation, duration and suspension time.
Each design's enrollment rules and sample size remain distinct. A dedicated
comparison report and native-app equivalence checks remain pending.

Validation covers all 84 feasible single-dose count states through six patients,
plus downward completion, exclusions, terminal interpretations, the one-pending
escalation exception, deterministic fast-accrual calendars and simulation capacity.
Existing TITE-BOIN and TITE-Keyboard calendar checks also pass with the shared engine.
