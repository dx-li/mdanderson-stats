# TITE-BOIN imputation and interim conduct

Catalog entry **129**, [TITE-BOIN](https://biostatistics.mdanderson.org/shinyapps/TITE-BOIN/),
is partially implemented. The app snapshot is version 2.5.4.0, updated November 20,
2025. [Provenance](tite-boin-sources.json) records its guides and the authors'
mathematical appendix to *Time-to-event Bayesian Optimal Interval Design to
Accelerate Phase I Trials*, DOI 10.1158/1078-0432.CCR-18-0246.

The implementation provides vectorized single-mean imputation, ordinary follow-up
thresholds, and interim dose decisions. Calendar replay, simulation and optional
3+3 modifications are supported. Dedicated Rolling 6 comparison reports, flowcharts and integrated
protocol reports remain pending. Final selection is available through
`BOINDesign.select_mtd` once all outcomes are ascertained.

## Imputation and follow-up thresholds

```python
from mdanderson_stats import BOINDesign, tite_boin_estimate, tite_boin_decision

design = BOINDesign(target=0.3)
summary = tite_boin_estimate(design, patients=3, toxicities=1, pending=1, stft=0.9)
assert summary.move == 0
print(summary.estimated_rate, summary.deescalate_stft)
```

For enrolled count `n`, observed DLT count `y`, pending count `c`, and standardized
pending follow-up `S`, the observed-outcome posterior has parameters
`a=y+target/2`, `b=n-c-y+1-target/2`. The imputed DLT count is
`y+(a/b)*(c-S)` and the decision statistic is that count divided by `n`.
It is a conservative approximation and can exceed one; it is not clipped or
presented as a fitted probability. With no pending data it reduces to `y/n`.

The result includes the posterior mean, imputed count, rate, ordinary move
(`+1`, `0`, `-1`), and thresholds `c-(n*boundary-y)/(a/b)`. Escalation additionally
requires `y/n < target`; de-escalation requires `y/n > target`. A prohibited
escalation threshold is `inf`, and a prohibited de-escalation threshold is `-inf`.
Thresholds outside `[0,c]` are retained: they describe decisions that always or
never apply over feasible follow-up. The thresholds omit accrual and safety gates.
Floating thresholds are numerical approximations; `move` uses the unrounded
imputed rate. Do not use displayed two-decimal thresholds for borderline decisions.

Counts and follow-up broadcast across scenarios. Each scenario accepts 1–200
patients, `y+c<=n`, and `0<=S<=c`. `S` sums pending follow-up/window ratios under
a uniform timing prior. For informative three-trimester masses, sum the existing
`toxicity_followup_weights` instead; these are conditional timing probabilities,
not overall toxicity probabilities.

## Interim conduct

```python
# One pending patient has completed 90% of the assessment window.
step = tite_boin_decision(design, [3, 0], [1, 0], [[81], []], 1, 90)
assert step.action == "stay"
assert step.next_dose == 1

# The current app requires at least 51% ascertained outcomes by default.
waiting = tite_boin_decision(design, [6, 0], [0, 0], [[45, 45, 45], []], 1, 90)
assert waiting.action == "suspend_pending"
```

Dose numbers are one-based. Patients include pending outcomes, while toxicities
include only observed DLTs. Supply one pending-time vector per dose, with times in
`[0,window)`. Returned diagnostics include per-dose pending counts, weighted STFT,
safety probabilities/exclusions, and the current-dose imputation.

The current app's defaults are `minimum_complete_fraction=0.51` and
`minimum_pending_followup=0.25`. Conduct first applies enrolled-count beta-binomial
overdose control and retained exclusions. Otherwise it suspends if the proportion
of ascertained outcomes is too low, except when the observed DLT rate already
meets the de-escalation cutoff. An actual escalation waits if the shortest pending
follow-up/window ratio is below the specified minimum. This second gate uses
elapsed time, even when the imputation has informative timing weights.

Observed DLTs count as ascertained outcomes. The allowed completion fraction is
`[0.25,1]`; the minimum follow-up fraction is `[0,1]`. The historical 50% rule can
be selected explicitly with `minimum_complete_fraction=0.5`, and the additional
follow-up gate disabled with `minimum_pending_followup=0`. These settings matter:
the appendix's older table permits some assignments the current defaults suspend.

Safety can override suspension. Precision stopping requires that the actual next
dose would be unchanged, consistent with ordinary BOIN. Physical dose limits and
excluded higher doses can make the assignment stay. Pass the returned `eliminated`
mask to subsequent decisions to retain earlier exclusions. Wait for all pending
outcomes before final MTD selection, including after precision stopping.

The optional BOIN modifications are applied after imputation and before the
accrual gates, with safety retaining priority:

* `stay_at_one_of_three=True` changes 1/3 to stay **only with no pending outcomes**.
  This option requires target in `[0.25,0.279]`. Pending 1/3 observations retain
  their ordinary imputation decision.
* `deescalate_at_two_of_six=True` changes 2/6 to de-escalate **with any valid number
  of pending outcomes**. It also overrides the completion-based suspension gate.
  This option requires target in `[0.28,0.33]`.

For forced stay, both escalation and de-escalation thresholds indicate impossible
moves (`inf` and `-inf`). For forced de-escalation, its threshold is `inf`, so every
feasible STFT qualifies. The imputed rate itself remains available unchanged as a
diagnostic. Dose limits, retained exclusions, safety and precision stopping still
apply. Replay and simulation accept these options through `BOINDesign`.

```python
modified = BOINDesign(target=0.3, deescalate_at_two_of_six=True)
step = tite_boin_decision(modified, [0, 6], [0, 2], [[], [80] * 4], 2, 90)
assert step.action == "deescalate"  # Overrides the usual completion-based pause.
assert step.next_dose == 1
```

These details were checked against visible decision rows generated by the live
app version 2.5.4.0. Relevant rows and input settings are preserved in
[the reference capture](../tests/fixtures/tite-boin-modifications.json). The guide's
page-2 example independently confirms the complete-outcome restriction on 1/3.
Full application-output equivalence has not been established.

## Validation

Twelve escalation/de-escalation thresholds agree with the appendix's rounded
Table S1. Independent rational arithmetic checks imputation for all valid
`n<=12` count combinations at a specified follow-up fraction. Complete-data
conduct agrees with ordinary BOIN, including safety. Focused checks cover both
suspension gates, the observed-toxicity exception, precision stopping, informative
weights, coherence despite inflated imputation, and time rescaling by `1e-200`. Modification checks cover pending counts,
complete-data agreement with modified BOIN, safety precedence, and calendar replay.


## Calendar replay and simulation

`run_tite_boin_trial` takes nonnegative interarrival gaps and a matrix of potential
DLT delays shaped `(planned_patients,doses)`. Delays are measured from individual
enrollment; finite values in `[0,window]` indicate toxicity, and positive infinity
means no DLT within the window. Only an assigned patient's observable history
enters an interim decision.

```python
import numpy as np
from mdanderson_stats import run_tite_boin_trial, simulate_tite_boin

trial = run_tite_boin_trial(
    design,
    [0, 0, 80, 15, 15, 15],
    np.full((6, 2), np.inf),
    90,
)
# At day 95 two outcomes are known, but the last patient has only 15 days'
# follow-up. Accrual resumes at day 102.5, when that patient reaches 22.5 days.
assert [step.time for step in trial.steps] == [95, 102.5]
assert trial.suspension_time == 7.5
assert trial.final_time == 222.5

simulation = simulate_tite_boin(
    design,
    [0.05, 0.15, 0.3, 0.45, 0.6],
    window=3,
    accrual_rate=2,
    event_distribution="weibull",
    late_probability=0.8,
    trials=1000,
    rng=129,
)
print(simulation.selection_probability)
print(simulation.duration.mean(), simulation.suspension_time.mean())
```

The scheduler supports staggered enrollment within a cohort assigned one fixed
dose. Decisions occur before the next cohort and, during suspension, when an
outcome becomes ascertained. A minimum-follow-up suspension also schedules a
reassessment when all currently pending patients at that dose reach the specified
follow-up fraction. If an earlier DLT occurs, it is processed first and the
assignment is reconsidered. There are no within-cohort stopping or dose changes;
cohort size one permits decisions before each patient. Suspended accrual creates
no queue: later arrival gaps begin at the preceding actual enrollment time.

After enrollment stops, final analysis waits for all enrolled toxicity outcomes
to become known. A DLT event ascertains its outcome immediately; a DLT-free patient
requires the full assessment window. The final time also includes the last
assignment/stopping decision. Stop reasons describe **why enrollment ended**:
a trial ending at its enrollment cap can subsequently have no admissible MTD
at final analysis. There is no additional enrollment decision after the cap.

The replay result includes enrollment times, dose assignments, DLT event times,
final counts/selection/exclusions, stop reason, final time, suspension time, and
interim histories with full TITE-BOIN diagnostics. The simulation result retains
per-trial counts, selected doses, duration, suspension time and stop reason, plus
selection probabilities and marginal Monte Carlo standard errors. Selection zero
means no MTD; probability bins are `[no MTD,dose 1,...]`. Arrays are read-only.

Simulation uses the same [timing scenarios](tite-keyboard.md#calibrated-weibull-and-log-logistic-scenarios)
as TITE-Keyboard: fixed or exponential arrivals (`arrival`), conditional-uniform
or piecewise-uniform event times (`event_trimester_probabilities`), and calibrated
Weibull/log-logistic scenarios (`event_distribution`, `late_probability`). True
scenario timing and the analysis's `trimester_probabilities` are independent.
The first patient also receives an arrival gap. Units must match the assessment
window. Defaults are 10 cohorts of 3, 1,000 trials, and starting dose 1; planned
enrollment is limited to 200, with at most 100,000 simulated trials per call.

These are explicit Python scheduling conventions, not a claim of exact parity
with the source app's unpublished scheduler or random-number sequence. Checks
cover release before the next outcome, intervening DLTs without future-information
leakage, time scaling, deterministic toxicity extremes and final follow-up. The
shared scheduler also passes the existing TITE-Keyboard calendar and timing checks.

The [Rolling Six comparator](rolling-six.md) now has separate conduct, replay and
simulation APIs using the same scenario-generation functions. Dedicated comparison
report generation remains pending.
