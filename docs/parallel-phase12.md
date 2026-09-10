# Parallel phase I/II combination trials

Catalog entry **85 remains partial**. The package now implements the complete
four-arm beta-binomial workflow in the archive's `SwatiBiswasCode` C program,
including patient-history replay, phase-I escalation, phase-II adaptive
randomization, toxicity closure, efficacy/futility stopping and simulation.
The later C++ trial implementation is a separate remaining workstream.

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

Remaining: the later C++ six-arm trial, its calendar/pending-outcome behavior,
model fitting, configurable inputs/reports, and full published operating-
characteristic replication. The C++ source explicitly prohibits redistribution
of the original program. No original archive files are shipped; this is an
independent Python expression of the four-arm statistical workflow.
