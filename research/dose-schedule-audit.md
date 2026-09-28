# Dose Schedule Finder coverage audit

Baseline main `ac0f0b9`. Previous goal turn: progress. UAROET's core workflow
was committed, checked against independent references, and verified from the
built wheel. The current catalog has 62 implemented, 59 partial and 17 pending
entries; the full catalog/publication goal remains active and incomplete.

This pass targets pending entry 75. Root uses `feat/dose-schedule-core`; one
Luna agent uses `feat/dose-schedule-luna` in the existing independent checkout.
Only one numerical job runs at a time, with BLAS/OpenMP thread counts set to
one. System memory reported 46% free at the initial check.

The official application page identifies the 2007 Braun/Thall/Nguyen/de Lima
method. The institutional paper's probability, prior, allocation and appendix
sections were read. PDF screenshot calls failed; geometric hazard descriptions,
the extracted equations and the worked prior values provide cross-checks.
No native archive was retrieved or executed. The later 2013 adaptation design
is a different method.

The patient likelihood must use actual administration histories, including
variable dose indices. Censored observations avoid multiplying zero by a log
zero hazard. An event outside all active triangular hazards has a genuine zero
likelihood, unlike a numerical failure. Candidate schedules are nested and use
a constant dose within each regimen.

The paper's moment-elicited prior is approximate, including transformed mean
toxicity and assumed hazard completion. For ordered areas, normal coordinates
describe log positive increments rather than log cumulative areas. Timing can
vary by dose, so finite-horizon risks need not inherit area ordering; evaluate
every candidate directly instead of propagating unsafe labels by index.

Safety has strict inner and outer inequalities. No-skip is an enrollment rule;
final selection uses acceptable regimens. Python tie handling and ambiguous
no-skip history conventions must be explicit, without claiming native parity.

The independent base-R generator ran successfully in 0.19 seconds with warnings
treated as errors. It numerically integrates triangles and provides six actual
history cases. The published approximate prior values match to the printed
precision. In the reduced model, the area likelihood is proportional to
`a^2 * exp(-5.3*a)` with `log(a) ~ Normal(-1, .7^2)` and fixed peak/tail of 2/3.
Direct integration yields mean log area -1.04721190796185 and mean area
.395537892938717; the fully accumulated two-administration risk has mean
.515190870280698. The integrated Python checks below compare these references.

## Integrated checkpoint

Luna commit `67bb24f` was integrated as `43b8f89`. Root added public exports,
independent references, user documentation and the coverage record. Entry 75
moves from pending to partial: 138 catalog entries now comprise 62 implemented,
60 partial and 16 pending. This remains an incomplete catalog/publication goal.

The four modules provide administration hazards, actual-history likelihoods,
explicit and moment-elicited priors, serial posterior sampling, regimen risks
and allocation. The sampler distinguishes a genuinely impossible event time
from numerical failure. Event hazards are summed in log space. Cumulative
hazards use positive components on the falling segment, avoiding subtraction
of nearly equal probabilities. Every regimen is evaluated independently of
area ordering. The fit preserves dose/schedule axes in the risk summaries.

The first assignment bypasses exclusion but still reports the actual safety
mask. Masks are immutable booleans. Final selection requires treatment history;
the caller is responsible for complete follow-up. Two documented no-skip
conventions cover histories with non-dominating tried pairs. These conventions,
tie handling and an optional starting-pair override are not native-equivalence
claims.

Validation used one numerical process at a time and one BLAS/OpenMP thread:

- All seven focused tests passed in 3.97 seconds with warnings treated as
  errors. Measured process elapsed time was 4.076 seconds, peak RSS 133.28 MiB
  and zero process swaps.
- Independent base-R integrations cover 24 triangular-hazard/time cases and
  six actual variable-dose histories. Published prior values are checked
  against separate moment calculations.
- The reduced posterior uses two chains of 1,200 retained draws after 400
  warmup iterations. Posterior log area, area, two-administration risk and
  overdose probability agree with direct integration within the recorded
  Monte Carlo tolerances. Each checked quantity has split R-hat below 1.05.
  This reference does not establish convergence for every data set or prior.
- Ruff and format checks passed on seven affected Python files. Mypy passed
  for all four new source modules with `--follow-imports=silent`.
- The wheel and source distribution built using cached Hatchling. Isolated
  wheel imports confirmed all 11 public exports and exact module/catalog
  bytes. Both documented examples ran; the posterior example used 1,043
  likelihood calls and 27,165 administration/prediction work units.
- Compact packaged checks verified strict safety boundaries, initial safety
  reporting, all-unsafe stopping, empty final-analysis rejection, immutable
  boolean masks, finite event log likelihood when ordinary hazards underflow,
  and resource rejection before random-number consumption. This process took
  2.152 seconds, peaked at 114.77 MiB and had zero process swaps. No native
  binaries, archives or article PDFs are included in the wheel.

No full-suite run, large simulation, dependency installation or CI change was
performed. Full calendar simulation, operating-characteristic calibration,
native input/report workflows and executable parity remain open. CiBolus is
the next pending method, with a primary-paper lead recorded separately. The
earlier GitHub write restriction remains unresolved; this checkpoint is local
and no alternative publication transport was attempted.

## Calendar replay preparation — September 28, 2026

The [2007 primary paper](https://odin.mdacc.tmc.edu/~pfthall/main/ClinTrials%20dose-sched%202007.pdf)
was re-read online; it is not a cached local PDF. Journal page 118 specifies
allocation at each new arrival, starting at the lowest pair, with no-safe-regimen
termination and final analysis after follow-up. Actual past administrations and
the event/censoring history available at the decision time determine the fit.
The source permits within-patient deviations but does not give an automatic
adaptation rule to reproduce. Its example also includes delayed classification
of persistent low-grade toxicity, which a simple immediate-event replay will
not cover. Existing no-skip conventions and direct finite-horizon safety
evaluation remain explicit Python choices.

`tools/reference_dose_schedule_events.R` numerically integrates the triangular
hazards and inverts the sum with base R. Eleven cases cover rising/peak/falling
events, no event by the horizon, overlapping administrations, a hazard-free
gap, and high/low hazard areas. Each is expressed at three time scales,
producing 33 rows in `tests/fixtures/dose-schedule-events.csv`. Three single-dose
solutions also agree with analytic values. The generator ran with warnings
treated as errors. These fixtures specify the inverse-CDF event contract and
do not establish native scheduler or random-stream parity. Python replay
integration and comparison remain a subsequent checkpoint.

## Integrated calendar trial checkpoint — September 28, 2026

The Luna calendar implementation (`7788103`, integrated as `a6f0172`) combines
inverse cumulative-hazard events, observed histories at each arrival,
posterior allocation, permanent safety termination and complete final
follow-up. The public guide is [calendar trials](../docs/dose-schedule-trials.md).
Separate full event and sampler tapes support direct replay after early stops.
Intermediate posterior draws are discarded; compact decisions and diagnostics
are retained under shared numerical work limits.

Review corrected event inversion near zero, final relative follow-up at large
calendar origins, and recommendation behavior after permanent termination.
A final regression reproduced duration rounding from `0.1` to
`0.0999755859375` at origin `1e12`; computing duration from relative elapsed
times fixes this without changing the calendar timestamp contract.

The 33 independent R inversion rows match, including event indicators and
administration counts. An analytic quantile with event uniform `4e-102` gives
time `1e-50` and passes with zero absolute tolerance. That check took 0.070
seconds after import, peaked at 117.81 MiB, and reported no swaps. The public
three-patient example completed in 1.328 seconds including import, peaked at
117.47 MiB with no swaps, and used 337 likelihood calls and 1,979 work units.
After the duration correction, all five focused calendar tests passed in
1.71 seconds. Targeted Ruff and mypy checks passed; formatting was applied to
the duration expression. Numerical processes used one BLAS/OpenMP thread.

This validates the implemented event and replay contracts, not general Monte
Carlo precision or native executable equivalence. Aggregate OCs/calibration,
delayed low-grade-to-DLT classification, within-patient adaptation policies
and native files/reports remain outside the current replay. No broad test
suite, large simulation, dependency installation or CI change was added.
