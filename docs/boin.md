# Bayesian optimal interval dose finding

Catalog entry **120**, the [BOIN application](https://biostatistics.mdanderson.org/shinyapps/BOIN/),
is **partially implemented**: single-agent local BOIN boundaries, cohort decisions,
overdose elimination, final MTD selection, fixed-cohort simulation and accelerated titration are available.
The 3+3 comparison includes both sample-size matching options.
Direct boundary-to-probability inversion is available; protocol generation and
animation remain pending. Desktop entry 99
and BOIN combination/time-to-event variants are separate, unaudited entries.

The application was inspected at version **3.0.20.0**, updated September 4, 2026.
Its technical PDFs and the [BOIN R-package paper](https://doi.org/10.18637/jss.v094.i13)
provide the statistical specification. File hashes and native-reference details
are recorded in [provenance](boin-sources.json). This is an independent Python
implementation; R is used only to generate validation fixtures.

```python
from mdanderson_stats import BOINDesign, simulate_boin

design = BOINDesign(target=0.3)
table = design.boundary_table(max_patients=30)
assert table.escalate_max[2] == 0  # at three patients, escalate with zero DLTs
assert table.deescalate_min[2] == 2
assert table.eliminate_min[2] == 3

next_step = design.next_dose([3, 6, 0], [0, 2, 0], current_dose=2)
assert next_step.next_dose == 2
mtd = design.select_mtd([3, 3, 15, 9, 0], [0, 0, 4, 4, 0])
assert mtd.dose == 3

simulation = simulate_boin(design, [0.05, 0.15, 0.3, 0.45, 0.6], trials=1000, rng=6)
print(simulation.selection_probability)  # no selection, then doses 1 through 5
print(simulation.mean_patients)
```

## Boundaries and conduct

Inputs are complete, evaluable patient and dose-limiting toxicity (DLT) counts.
Dose indices are **one-based**. Untreated doses have zero counts, not synthetic
outcomes. Pending outcomes are outside this method's scope.

The default indifference probabilities are `safe_probability=0.6*target` and
`toxic_probability=1.4*target`. Custom values must satisfy
`0 < safe < target < toxic < 1`, with target in `[.05,.6]`. Log-likelihood crossings
use `log1p` differences to retain precision for nearby probabilities. Escalate at
observed DLT rate **at most** the lower boundary; deescalate at rate **at least**
the upper boundary; otherwise stay. At the end of the dose range, an unavailable
move becomes stay. Decisions never escalate into an eliminated dose.

For safety, use the uniform Beta(1,1) prior and eliminate a dose and all higher
doses when its posterior probability of toxicity exceeding target is **strictly
greater** than `elimination_probability` (default .95), with at least three
patients treated there. `extra_safe=True` applies the lower cutoff
`elimination_probability-safety_offset` at the lowest dose (offset .05 by default).
Eliminating the lowest dose stops the trial with no MTD. **Pass the returned
`eliminated` mask into later decisions and selection** to retain earlier exclusions.
The simulator does this automatically.

Table rows represent patient counts 1 through `max_patients`; escalation is an
inclusive maximum DLT count, while deescalation and elimination are inclusive
minimum counts. An elimination cutoff of `n+1` means elimination is impossible at
that sample size. Safety rules take precedence over the ordinary rate boundaries.

Optional source modifications are `stay_at_one_of_three=True` for targets in
[.25,.279], and `deescalate_at_two_of_six=True` for targets in [.28,.33]. With
`early_stop_patients=m`, stop for precision once the current dose has at least m
patients **and the resulting next assignment stays at the same dose**, including
when the dose range or a safety exclusion prevents escalation/deescalation. This option is off by default.
A stopped decision has `next_dose=None`; final selection is a separate calculation.

## Final estimation and MTD selection

Following the R software's estimation convention, each treated dose has a
Beta(.05,.05) working prior. Its posterior mean is `(y+.05)/(n+.1)` and its variance
is `(y+.05)*(n-y+.05)/((n+.1)**2*(n+1.1))`. The mean is fit by increasing isotonic
regression with inverse-variance weights, using SciPy's compiled pool-adjacent-
violators algorithm. Untreated doses have NaN reporting estimates.

`isotonic_mean` fits all treated doses for reporting. `selection_mean` refits only
treated, noneliminated doses. Select the admissible fitted mean closest to target;
for tied distances choose the highest dose if all tied means are below target,
otherwise the lowest dose. `bound_mtd=True` additionally requires a fitted mean
at or below the deescalation boundary. No admissible dose yields `dose=None`.

`isotonic_interval` applies the same weighted isotonic transformation to the
individual .025 and .975 beta quantiles. It reproduces a source reporting
convention; it is **not a joint 95% posterior credible band**. The unweighted
isotonic `report_overdose_probability` uses the weak working prior, whereas
`safety_overdose_probability` uses Beta(1,1) without isotonic smoothing. These
probabilities have different purposes and should not be interchanged.

## Simulation and source differences

`simulate_boin` generates independent binomial DLT totals at each cohort, with
all outcomes available before the next decision. Defaults are 10 cohorts of 3,
1,000 trials, starting at dose 1. Returned trial-level patient/toxicity counts
allow further operating-characteristic summaries. Selection index zero means no
MTD. Selection probabilities include a no-selection bin followed by all doses,
and `selection_mcse` reports marginal Monte Carlo standard errors. Stop reasons
distinguish safety, precision and the enrollment cap. Probabilities need not be
monotone, allowing evaluation under misspecified dose ordering. NumPy random
streams do not reproduce R draws for equal seeds.

The older Probability PDF prints a strict upper rate inequality; the published
paper and R implementation use the inclusive rule implemented here. The older
Guide says more than three patients for safety, whereas the paper/native rules
use at least three. The latest standalone modification PDF broadens the 1/3
option's target range beyond the older Guide's .25 example. The latest sample-size
PDF requires stay for precision stopping, including unavailable moves at a dose
boundary. This agrees with the R 2.7.2 simulation rule.
The R selection implementation adds a tiny artificial trend to break ties;
Python resolves ties explicitly without perturbing fitted probabilities, so
pathological near-ties or values exactly on a selection bound may differ.

Validation checks six published boundary pairs, the full .30-target cohort table,
exact rational beta/binomial safety identities, 64 original R MTD selections and
reported estimates, and two 10,000-trial original R operating-characteristic runs, with and without titration.
Deterministic safe/unsafe trial paths and the exact one-cohort binomial law check
simulation independently. An exact competing-event calculation checks titration
duration with grade-2 events; deterministic paths check every transition and the
enrollment cap. The reference runner requires BOIN 2.7.2 and its Iso
dependency; neither is a Python runtime dependency.

## Accelerated titration

Set `titration=True` to escalate one patient per dose, starting at `start_dose`.
Stop this phase at the first DLT, second grade-2 event across all titration patients,
or the highest dose. Complete the current cohort with `cohort_size-1` additional
patients, then make a BOIN decision and use full cohorts thereafter.

`titration_cap` defaults to the highest dose. If a lower cap is reached without
those toxicity triggers, begin a full cohort at the next higher dose instead of
expanding the cap dose. A toxicity trigger at the cap takes precedence. As on the
site, titration has no effect with cohort size one or a starting dose at the top.
The total enrollment cap remains `cohorts*cohort_size`, including all titration
patients; the last cohort is shortened when necessary. Even if the cap is reached
during titration, it is never exceeded.

`moderate_toxicity` supplies a probability per dose for a grade-2 event that is
**mutually exclusive with DLT**, not a conditional probability among non-DLT cases.
It defaults to zero, matching the original R simulator's DLT-only titration.
Each grade-2 probability must lie between zero and `1-true_toxicity`. This explicit
scenario input implements the site's documented grade-2 conduct rule; it does not
claim to reproduce an undocumented grade-2 simulation model from the app.

`titration_patients` counts single-patient assignments before cohort completion.
`titration_moderate_toxicities` records grade-2 events only during that phase;
these are never added to the DLT count. `titration_end_reason` records `DLT`,
`grade2`, `highest_dose`, `dose_cap`, `max_patients`, or `disabled` for each trial.

```python
accelerated = simulate_boin(
    design,
    [0.05, 0.15, 0.3, 0.45, 0.6],
    titration=True,
    titration_cap=3,
    moderate_toxicity=[0.05, 0.1, 0.1, 0.15, 0.15],
    trials=1000,
    rng=120,
)
assert (accelerated.patients.sum(axis=1) <= 30).all()
```

## Conventional 3+3 comparison

`simulate_three_plus_three` implements the rules in the site's
[3+3 technical note](https://biostatistics.mdanderson.org/shinyapps/BOIN/plus3.pdf).
At a dose, 0/3 DLTs escalates, 1/3 expands to six patients, and at least two DLTs
eliminates that and higher doses. With at most 1/6 DLTs, escalation continues when
an admissible higher dose exists. After de-escalation, a lower candidate with only
three patients receives three more before selection. The highest dose also needs
six patients before selection. Selection therefore always requires at most one
DLT among six patients. Elimination of the lowest dose produces no MTD.

The simulator advances independent trials in NumPy batches. Its count arrays have
shape `(trials, doses)`; selection index zero means no MTD. Probabilities and Monte
Carlo standard errors have bins `[no MTD, dose 1, ..., dose J]`. Arbitrary true dose
probabilities, including nonmonotone scenarios, are allowed. Dose-finding enrollment
is random, never more than six patients per dose.

`compare_boin_three_plus_three` runs both designs with independent outcome draws
from a reproducible NumPy stream, and implements these `matching` options:

* `none`: BOIN uses the requested cohort count; 3+3 follows its natural stopping rule.
* `expand_three_plus_three`: when both designs select an MTD, add patients at the
  original 3+3 MTD until its total matches realized BOIN enrollment. No patients are
  removed when 3+3 already enrolled more. The expansion produces additional DLT
  outcomes but does not revise the original selected MTD or apply BOIN safety rules.
* `match_boin`: each BOIN trial uses `ceil(realized_3plus3_enrollment/cohort_size)`
  cohorts as its cap. This uses realized 3+3 enrollment, not the theoretical `6*J`
  maximum. BOIN stopping rules can still reduce enrollment below that cap.

Early stopping, absent selections, cohort rounding and a naturally larger 3+3
sample can prevent equal realized sample sizes. The result exposes `boin_max_patients`
and `expansion_patients` per trial. The 3+3 `patients`/`toxicities` arrays include
expansion; `dose_finding_patients`/`dose_finding_toxicities` preserve pre-expansion
counts. Both full result objects remain accessible for custom summaries.
`simulate_boin` also accepts one cohort count per trial to support these varying caps.

```python
from mdanderson_stats import compare_boin_three_plus_three

comparison = compare_boin_three_plus_three(
    design,
    [0.05, 0.15, 0.3, 0.45, 0.6],
    matching="expand_three_plus_three",
    trials=1000,
    rng=120,
)
print(comparison.boin.selection_probability)
print(comparison.three_plus_three.selection_probability)
```

Independent two-dose path calculations validate selection probabilities in three
scenarios with 100,000 simulated trials each. If `a_j=P(0/3)` and `b_j=P(1/3)`,
write `s_j=a_j**2+2*a_j*b_j`. Then selection probabilities are
`P(MTD=1)=(1-s_2)*s_1` and `P(MTD=2)=a_1*(1+b_1)*s_2`; the remaining mass is no MTD.
Deterministic paths check top-dose confirmation and downward expansion, and paired
trial accounting checks both matching modes without assuming they force equality.

## Direct custom boundaries

The site's alternative input mode accepts rate cutoffs instead of indifference
probabilities. `BOINDesign.from_boundaries` provides the same parameterization:

```python
custom = BOINDesign.from_boundaries(
    target=0.3,
    escalation_boundary=0.2,
    deescalation_boundary=0.4,
    extra_safe=True,
)
print(custom.safe_probability, custom.toxic_probability)
# approximately 0.1202446414191221 and 0.5059405275001106
assert custom.escalation_boundary == 0.2
assert custom.deescalation_boundary == 0.4
```

Require `0 < escalation < target < deescalation < 1`, with target in [.05,.6].
All safety, modification, precision-stop and final-selection options are available
as keyword arguments. The factory solves the Bernoulli likelihood-crossing equations
for the two alternatives. It searches log probability below target to resolve tiny
safe alternatives, and representable probability above target. A forward residual
check rejects cutoffs whose alternatives cannot be resolved in double precision;
it does not clip the result to zero or one. For example, target .3 and escalation
.001 imply a safe alternative near `5.37e-156`, which is supported. Escalation
`1e-8` or deescalation `.999` at that target require unrepresentable alternatives
and raise `ArithmeticError`.

The requested cutoffs are retained exactly after forward validation, so inversion
roundoff cannot change decisions at an inclusive boundary. Boundary-table integer
cutoffs are likewise checked against `DLTs/patients`, as used by conduct, rather
than relying exclusively on floating-point multiplication. For example, `.58*50`
rounds below 29, but `29/50 <= .58` is true and the table permits escalation with
29 DLTs. Safety rules continue to take precedence.

Validation includes default-probability round trips across eight targets,
independent 80-digit decimal likelihood identities, tiny alternatives, explicit
resolution failures, and agreement between integer tables and rate comparisons.
