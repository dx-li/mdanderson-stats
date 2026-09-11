# Parallel phase I/II combination trials

Catalog entry **85 remains partial**. The package now implements the complete
four-arm beta-binomial workflow in the archive's `SwatiBiswasCode` C program,
including patient-history replay, phase-I escalation, phase-II adaptive
randomization, toxicity closure, efficacy/futility stopping and simulation.
The later six-dose C++ workflow now has an integrated calendar simulator,
including pending outcomes, blocked arrivals and posterior decisions.

Sources: [MD Anderson's entry](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/85)
and [P12Xuelin archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/P12Xuelin/P12Xuelin_V1.0_.zip),
supporting Huang, Biswas, Oki, Issa and Berry, *A parallel phase I/II clinical
trial design for combination therapies*, Biometrics 63 (2007), 429–436.
The archive includes both the C simulation and a later Windows C++ application
for another trial. The C variant explicitly completes phase I before starting
adaptive randomization. Implementing it does not establish coverage of the
calendar-time C++ design or the full general methodology described in the paper.

## Use

```python
import numpy as np
from mdanderson_stats import parallel_phase12_replay, simulate_parallel_phase12

# Rows: zero-based arm, toxicity (0/1), response (0/1).
next_patient = parallel_phase12_replay([[0, 0, 1], [0, 0, 0], [0, 0, 1]])
print(next_patient.phase, next_patient.probability)

trial = simulate_parallel_phase12(
    toxicity_probability=[0.04, 0.09, 0.16, 0.25],
    efficacy_probability=[0.1, 0.2, 0.35, 0.5],
    rng=np.random.default_rng(8502),
)
print(trial.selected, trial.reason, trial.treated)
```

Native arms are numbered 0–3: the lowest combination, the two adjacent
single-agent escalations, then the highest combination. Counts and assignments
use that order. Histories contain complete binary outcomes; this interface does
not infer pending outcomes or calendar times. A partial three-patient phase-I
cohort keeps the same forced next assignment until the cohort is complete.
Replay rejects an ineligible assignment or additional patients after stopping.

Results retain patient records, per-arm enrollment/toxicity/response counts,
admissibility, escalation flags, next assignment probabilities, phase-I total
enrollment and the stopping reason/selected arm. Arrays are immutable.
`efficacy_probability` records the **last stopping analysis**, not necessarily
the current patient count between five-patient looks. `efficacy_target` identifies
its threshold; before the first eligible analysis it is `None` and probabilities
are `NaN`. Selection is `None` until a completed trial selects an arm.

## Preserved C workflow

The defaults are the fixed values compiled into the archived program; this API
does not introduce a different configurable trial design.

| Component | Source behavior |
|---|---|
| Maximum enrollment | 100 patients |
| Toxicity prior | Independent Beta(1,9) by arm |
| Efficacy prior | Independent Beta(.1,1.9) by arm |
| Phase-II look spacing | Every 5 patients since completion of phase I |
| Toxicity closure | After each phase-II patient, close its arm if Pr(toxicity > .3333334) > .8 |
| Interim response target | .20 |
| Futility | Largest eligible Pr(response > .20) < .05 |
| Early efficacy, multiple arms | Best target probability > .9 and Pr(best response rate > second best) > .8 |
| Early efficacy, one arm | Target probability > .95 |
| Final selection | Largest Pr(response > .05) > .95 at N=100 |

Phase I starts with three patients on arm 0. Zero toxicities clears escalation;
one triggers three more patients; at least two in the first three closes the
trial. After expansion, at least three of six closes the arm, exactly one of
six clears escalation, and two of six leaves the arm admissible without
escalation. Both adjacent arms are evaluated in order (1, then 2), with the
same clearance logic. The highest arm opens only when **both** adjacent arms
clear escalation. It can remain admissible with two toxicities in six patients.
These details are preserved rather than substituted with another 3+3 convention.

With three or four admissible arms, interim stopping requires at least three
arms to have five patients. The C code counts even previously excluded arms
for this gate. With one or two admissible arms, all currently admissible arms
must have five patients. The final analysis bypasses the gate.

Allocation weights are .5 for arm 0 when admissible and Pr(p_i > p_0) for each
other admissible arm. Even if arm 0 closes, other arms continue comparing with
its posterior. Weights normalize, entries below .01 are removed, and weights
normalize again. They refresh every five phase-II patients or immediately after
an arm closes. Closure is permanent. Ties in target probability use native
first-arm order, with the source's additional pairwise tie check retained.

Simulation uses independent Bernoulli toxicity and response outcomes, matching
the source data-generating assumptions. It uses NumPy random state and stores
patient outcomes for replay. Posterior comparisons reuse the existing stable
beta quadrature and cache repeated count comparisons within each trial.
There are no new dependencies or CI workflows.

## Numerical and compatibility details

The beta tail calculation uses the survival function directly, avoiding loss
from `1-CDF` subtraction. Pairwise beta comparisons use logit quadrature at
absolute tolerance 1e-12. If all allocation weights become too small relative to
their integration error estimates to normalize reliably, an explicit arithmetic
error is raised instead of fabricating randomization probabilities.

Two reporting distinctions are deliberate. An immediate toxic stop at the
lowest arm returns an all-false admissibility mask; the C program sets its count
of admissible arms to zero but leaves a stale array flag on that exit path.
At N=100, Python reports the .05-target probabilities actually used for selection;
the C output replaces its single reported best probability with a .20-target
probability after selection. Both .05 selection and .20 interim thresholds are
preserved and identified explicitly in Python. Random streams differ, and R's
external comparison helper is replaced by in-process numerical integration.

## Validation and remaining coverage

The [control-flow audit](parallel-phase12-reference.json) runs the original C
`main` and stopping/allocation functions with rewritten absolute include paths
and explicit numerical/RNG callbacks. The original external R script and
platform-specific private numerical library are absent or unusable on this
platform; callbacks are disclosed, not treated as original numerical parity.
Python replay matches all 24 generated complete histories: final selection,
enrollment, toxicity counts and response counts. Cases include efficacy,
futility, initial toxicity closure and the 100-patient final analysis.

Separately, [independent R integration](parallel-phase12-numerics.json) checks all
179 distinct beta comparisons encountered, with maximum absolute difference
1.14e-10. Four representative comparisons are retained as regression fixtures.
Two focused workflow tests cover partial phase-I cohorts, 2/6 behavior, invalid
assignments, no enrollment after stopping, simulation/replay consistency and
certain toxicity. The existing beta-comparison suite supplies additional
numerical checks.

Reproduce the audits after retrieving the archive with
`tools/reference_parallel_phase12.py`, then `tools/reference_parallel_phase12.R`.
Original source is used only from the ignored research directory; audit tooling
and generated numerical fixtures are bundled, not original code or trial data.

Remaining: native configurable input/report workflows, multi-trial reporting,
native adaptive-importance-sampler
parity and full published operating-characteristic replication. The C++ source explicitly prohibits redistribution
of the original program. No original archive files are shipped; this is an
independent Python expression of the statistical workflows.


## Six-dose C++ model and calendar snapshots

`phase12_snapshot(records, time=...)` accepts rows
`(dose, entry_time, response, response_time, toxicity, toxicity_time)` with
zero-based dose indices 0–5. Event times are absolute. Entries after analysis
time are excluded. Each endpoint contributes only when its own observation time
is at or before analysis time; efficacy may be available without toxicity and
vice versa. The returned `tally` columns are **no response, response, no toxicity,
toxicity**, matching `Kernel::SetData` rather than conflicting comments elsewhere
in the source. Enrollment and pending counts are retained separately.

```python
import numpy as np
from mdanderson_stats import phase12_snapshot, fit_phase12_model

snapshot = phase12_snapshot([[0, 0, 1, 5, 0, 10], [0, 2, 0, 12, 1, 4]], time=5)
print(snapshot.tally[0])  # [0, 1, 0, 1]

fit = fit_phase12_model(
    snapshot.tally,
    draws=1000,
    warmup=500,
    chains=4,
    rng=np.random.default_rng(8524),
)
print(fit.response_summary.mean)
print(fit.coefficient_summary.split_rhat)
```

The source's four-coefficient logistic response model uses the six rows
`(0,-s,-s,-1)`, `(0,-s,-s,1)`, `(-s,s,0,-1)`, `(-s,s,0,1)`,
`(-s,0,s,-1)`, `(-s,0,s,1)`, with `s=.7071067811865` as written in
`SetupTrial`. There is no additional intercept. Coefficient prior means are
`(6.2445,2.0815,2.0815,0)` and independent SDs are `3.16227766` (variance about
10). These are the executable parameters, not an interpretation of an ambiguous
Normal(mean,10) comment. Toxicity uses independent Beta(.1,.9) priors per dose.
Both sets of priors may be supplied explicitly.

`phase12_response_probabilities(coefficients)` evaluates the model;
`phase12_response_loglikelihood(coefficients, tally)` gives the response binomial
log likelihood without combinatorial constants. Both support batched coefficient
vectors. The latter excludes toxicity columns. Log-sigmoid calculations remain
finite for ordinary counts at predictors ±1000, avoiding the source expression's
`log(1-p)` loss near probability one.

`fit_phase12_model` samples the normal-prior response posterior with four-
dimensional elliptical-slice updates. With no observed response outcomes it
samples the prior directly. It returns coefficient and response-probability
chains, classical split R-hat and batch-mean MCSE summaries, plus the source's
posterior comparisons: response at least `.30`, response greater than `.10`,
pairwise response superiority, and superiority to dose zero with its reference
weight fixed at `.5`. Comparisons use predictors to avoid artificial ties from
rounded probabilities at zero or one. Toxicity exceedance at `.33` is evaluated
analytically. These thresholds are explicit arguments.

The posterior is shared across doses through the response regression; it is
not six independent beta efficacy models. This Python sampler differs from the
C++ adaptive mixture importance integrator. Diagnostics do not guarantee
convergence or precise indicator probabilities, especially near a decision
threshold; retained chains allow further precision assessment. The source posterior
decision rules and integrated calendar simulator are supplied below. Native
integration parity remains pending.

Validation uses an [independent R importance calculation](phase12-model-reference.json)
with 200,000 draws from an inflated-Laplace/prior-normal mixture (importance ESS
about 145,977). All four coefficient means and six response means agree within
six combined Monte Carlo errors; four chains with 1,000 warmup and 3,000 retained
draws have maximum split R-hat below 1.04 in the checked dataset. Separate checks
cover prior recovery, the analytic dose contrast, extreme log likelihoods and
independent observation of calendar outcomes. Reference generation is in
`tools/reference_phase12_model.R`.


## Six-dose source decision rules

`phase12_source_decision(fit, enrolled, phase_one_admissible=..., closed=...,
suspended=...)` applies the archived C++ posterior decision logic to a
`Phase12ModelFit`. Masks and enrollment counts each have six entries. Omit
`closed` to initialize it from phase-I inadmissibility; pass the returned masks
on subsequent evaluations. Toxicity closure is permanent; suspension can reverse.
The caller determines when a calendar analysis is due.

The evaluator closes doses when toxicity exceedance probability is strictly
above .95. Reference-superiority weights normalize over nonclosed doses, and
initially admissible, still-open doses suspend below .01 normalized weight.
A stopping analysis is enabled when four doses have at least five patients,
or all nonclosed doses have at least five (`cohort_size` defaults to five).
These are enrollment counts, not counts with observed response outcomes.

Futility uses the maximum efficacy probability among **all phase-I-admissible
doses**, including subsequently closed doses, and stops below .05. Early
selection requires efficacy probability above .90 and pairwise superiority above
.80 against every other dose. The archived selection loop includes closed and
suspended candidates and comparators. This API deliberately reproduces that
behavior: `selected_eligible=False` identifies an ineligible source winner. It
is a source-rule evaluator, not a corrected clinical selection algorithm.

`phase12_source_final_selection(fit, closed=..., suspended=...)` excludes closed
and suspended doses and chooses the first maximum future-study probability
strictly above .90. The archive actually uses its early-selection rule slot here;
the separately configured .80 future-study confidence setting is unused.

Results distinguish the source's trial-termination flag from its arm-closure
flag. Assignment probabilities are zero after either stopping condition, rather
than exposing stale source randomization weights. With every dose closed, the
zero-denominator normalization is skipped while retaining defined source stopping
results. Open-dose weights summing to zero raise an explicit numerical error.

The [decision audit](phase12-decision-reference.json) compiles the unchanged
extracted `EvaluateStoppingRules`, `CalculateToxRate` and `PickWinner` methods
against small in-memory adapters. Across 100 supplied posterior/state cases,
Python matches masks, weights, termination, closure and early/final selections.
Nine cases selected early and nine stopped for futility. Focused checks also
cover reversible suspension, permanent closure, strict cutoffs, enrollment gates,
the ineligible-winner quirk and final tie handling. Run
`tools/reference_phase12_decision.py` after retrieving the archive to reproduce
this audit; original methods are read from ignored research files, not bundled.

The six-dose progression and accrual-readiness primitives are supplied below.
The integrated calendar simulator below schedules these decisions. Native
input/report workflows and complete C++ application parity remain outstanding.


## Six-dose phase-I progression and accrual readiness

`phase12_phase_one(enrolled, toxicities, current_dose=..., opened=..., closed=...,
admissible=...)` applies one `DF3Plus3` transition. Pass six per-dose enrollment
counts, observed positive toxicity counts, and masks retained from the preceding
transition. Initial defaults open only dose zero. Counts per dose cannot exceed
six. The current dose is the actual last assignment (zero initially).

```python
from mdanderson_stats import phase12_phase_one

step = phase12_phase_one([3, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0])
print(step.probability)  # [0, .5, .5, 0, 0, 0]
```

Returned assignment probabilities analytically average the source's fair coins,
so no random generator is needed to evaluate a transition. Sample the next dose
from these weights and pass that actual dose on the next call. `done` ends phase
I; `trial_closed` specifically identifies lowest-dose rejection. A completed
phase I returns zero assignment weights, even where the source retains an unused
next-dose value. The controller must handle the separate rule that phase II does
not start with at most one admissible dose.

The six-dose variant rejects **two or more toxicities**, including 2/6; it differs
from the four-arm C program. Zero of three or at most one of six declares a dose
admissible. One of three forces the next patient onto the same dose. At other
intermediate counts, unresolved paired doses can randomize again. Doses 1/2 and
3/4 form the randomization pairs; this is not three additional patients forced
to one dose after every 1/3 result.

The opening graph is asymmetric. Clearing dose zero opens doses 1 and 2. If dose
2 clears and dose 1 closes, only dose 4 opens. Clearing both 1 and 2 opens both
3 and 4. If dose 1 clears and dose 2 closes, phase I ends. Dose 5 opens only when
both 3 and 4 clear. A pair with no unresolved available dose ends phase I. These
are the archived branches, not a substituted generic combination 3+3 design.

`phase12_accrual_ready(records, time=..., phase_two_start=None)` uses the same
six-column chronological records as `phase12_snapshot`. In phase I it permits
partial cohorts, but at each multiple of three enrolled patients it waits until
all three most recent toxicity observation times have passed. For phase II,
provide the number enrolled before phase II as `phase_two_start`: every multiple
of five phase-II patients waits for the last five response observations. The
first partial cohort of phase II does not wait for earlier phase-I responses.
Equality with the observation time is sufficient. The function also respects
`trial_closed` and the default 80-patient maximum. All supplied patients must have
already entered; this helper does not advance time or generate arrivals.

The [native progression audit](phase12-progression-reference.json) runs 100
complete source phase-I histories, comparing 948 transitions and all four
combinations of up to two fair coins per transition. Open/closed/admissible masks,
phase completion, lowest-dose closure and assignment probabilities agree.
Outcomes are supplied as observed before decisions in this audit; calendar
readiness is checked separately at phase boundaries and exact observation times.
Run `tools/reference_phase12_progression.py` after retrieving the archive to
reproduce it. Original C++ methods remain in ignored research files.

## Integrated six-dose calendar simulation

```python
import numpy as np
from mdanderson_stats import simulate_phase12_calendar

trial = simulate_phase12_calendar(
    [0.05, 0.1, 0.1, 0.15, 0.15, 0.2],
    [0.1, 0.2, 0.2, 0.3, 0.3, 0.5],
    rng=np.random.default_rng(8554),
)
print(trial.reason, trial.early_selected, trial.future_selected)
```

Defaults follow the C++ simulation setup: 80 patients, 72 attempted arrivals per
365 days, response window 84 days and toxicity window 28 days. The first attempt
occurs at zero. Exponential interarrival gaps are rounded with `floor(x + .5)`;
blocked attempts consume time and generate another gap. Zero gaps are allowed.
`max_attempts` raises an error if the simulation cannot finish within its bound.
`max_duration` limits attempted arrival times, with a strict upper bound.

Response and toxicity are independent Bernoulli outcomes conditional on dose.
A positive event receives a rounded uniform delay from one day to its endpoint
window, capped at that window; a negative outcome becomes known at the complete
window. Dose indices and outcome records follow `phase12_snapshot` above.
Phase-I transitions use currently observed toxicities and retained dose masks.
Phase II starts only with at least two admissible doses, initially assigned
uniformly. Posterior analyses occur after every five phase-II enrollments when
the response-cohort readiness rule permits the next attempt. All observed data,
including phase-I data, enter each analysis. Toxicity closure is assessed at
these analyses, not after every intervening patient.

**The archived controller does not complete follow-up before final selection.**
At an enrollment or duration limit, it applies `PickWinner` at the last processed
attempt time. Python preserves that default, including pending outcomes. Set
`complete_followup=True` to move only this final analysis to the latest outcome
observation time. This extension preserves the patient history and does not
reopen an already stopped trial. Final selection uses the retained closed and
suspended masks without another interim closure/futility update. If custom limits
interrupt phase I, the source's posterior closed flags are still initially all
false: final selection can consequently consider doses not declared admissible.

The early-selection eligibility quirk described above is retained. Inspect
`selected_eligible`: false means the early winner was closed or suspended.
For a final winner, true means it passes those masks; it does not certify phase-I
admissibility when custom limits interrupt phase I.

Results retain patient records, per-patient phase (0/1), attempted arrivals
(columns: time, prior enrollment, prior phase, blocked flag), phase-II starting
enrollment, stopping reason/time, final-analysis time and selection. Each analysis
retains its snapshot, decision (`None` for final selection), and maximum
coefficient split-Rhat; `last_fit` retains the latest posterior draws. Independent
returned data and posterior seeds isolate simulation randomness from posterior
sampling. Unchanged tallies reuse the previous fit. No exact native random-stream
or adaptive-importance-sampler equivalence is claimed.

Focused integration checks reconstruct endpoint availability and accrual readiness
from simulated histories, cover early toxicity termination, and compare default
versus complete final follow-up. The [archived Case 1 pilot](phase12-calendar-pilot.json)
is reproducible with `uv run python tools/pilot_phase12_calendar.py`. Its finite
MCMC budget and diagnostic values are recorded; it is a workflow check, not a
published operating-characteristic replication or proof of precision near decision
thresholds. Native input/report workflows, multi-trial reporting and full published
operating-characteristic validation remain pending.
