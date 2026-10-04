# Stochastic BARD BF-BLRM trial audit

## Source-defined targets

The cached primary article (`research/raw/BARD/paper.txt`) describes the
two-stage BARD design in Section 2 (printed pages 7–12). For BF-BLRM, stage
one selects a dose only after at least six patients have been treated there,
and among doses with strict `POD < eta` selects the greatest posterior target
probability (page 11, text near lines 438–445). The paper's BF-BOIN simulation
setting uses five doses, up to 30 escalation patients, cohorts of three,
accrual rate 3/month, a one-month DLT window, backfill cap 12 per dose, and
stage-two total target 40; it carries the selected MTD and the dose one level
below when that lower dose exists (printed page 16, lines 695–710). The
BF-BLRM escalation cap is separately calibrated to match BF-BOIN's mean
stage-one sample size; its value is not supplied (lines 739–757), so this API
requires an explicit cap.

Stage two reuses eligible stage-one patients at the selected pair and adds
patients until the combined target is reached (Section 2, printed pages
11–12). Its covariate-adaptive randomization uses Pocock–Simon minimization.
The primary paper's three-binary-factor study balances only its first two
factors in the allocation rule and evaluates all three. The reported operating
characteristics include total sample size, duration, absolute arm-count gap,
per-factor level-1 proportion gap, and correct OBD selection by the efficacy
and utility methods (printed pages 18–19, lines 797–823). The trial result
therefore keeps every factor, outcome and assignment needed to compute those
quantities, while exposing separate stage-one, stage-two and combined counts.

## Implemented composition and explicit conventions

`run_bard_blrm_stochastic_trial` composes the existing BF-BLRM stage-one
calendar and fit, shared dose/profile outcome generator, existing
`continue_bard_trial` minimization allocator, and existing `bard_select_obd`
analyses. Both OBD rules use the same final joint outcome counts; the second
analysis does not randomize another cohort. A single explicit/generated seed
is passed through both phases, or a caller-owned `Generator` is advanced
directly and represented by `seed=None`.

The Python default pair is selected MTD plus its adjacent lower dose. An
explicit ordered pair is accepted only if stage one selected an MTD. Both
doses must have final BF-BLRM safety status `POD < eta`; this deliberately uses
the final strict rule and does not import BF-BOIN persistent elimination.
The underlying BF-BLRM replay's `arrival_schedule_exhausted` is treated as an
incomplete trial and raises before the result can enter an OC denominator.
No-MTD, no-default-lower-dose and unsafe-pair outcomes instead have explicit
no-stage-two statuses and no final OBD labels.

Stage-two eligibility is the configured profile mask. Eligible stage-one
patients on the pair are mandatory carryover, and newly generated profiles
are sampled from the configured distribution conditioned on that mask. The
target includes carryover; if carryover exceeds it, all patients remain and
the trial reports `carryover_exceeds_target` without new enrollment. The
selected factor subset drives minimization; the returned full factor history
retains any omitted factors for their own balance summaries.

The source does not specify the native joint toxicity/response generator,
individual stage-two arrival/assessment schedule, exact random-number stream,
or behavior for every transition boundary. This API uses the shared Python
outcome-tape generator, a configured uniform or exponential stage-two renewal
schedule beginning at complete stage-one follow-up, the shared Weibull DLT
assessment convention, and the configured joint endpoint law (conditional
independence by default). It does not claim native timing or RNG parity. The
API uses caller-supplied priors, dose truths and response models rather than
claiming undocumented app defaults.

Exact reproduction of the paper's BF-BLRM simulation table remains blocked by
the conflict already recorded in [the model audit](bard-blrm-audit.md): the
paper's raw-ratio model/prior implies prior `POD` above the `.30` cutoff at
every dose before patients accrue. The Python replay's initial prior screen
therefore stops with exact prior probabilities; the paper does not resolve
this initialization discrepancy. No
prior, model, or screening rule is silently changed here. The saved stochastic
workflow is useful for explicit feasible inputs but does not certify the
published table's exact scenario parity.
