# CRM trial and operating-characteristic simulation audit

Baseline main: 52b7e9b. The preceding goal turn was progress: exact bounded
look-ahead and calendar snapshots/routing were implemented, validated and
integrated. The full MD Anderson catalog goal remains active.

The current batch implements full trial/cohort replay and serial simulation
using the existing statistical kernels. Root orchestrates, audits sources and
integrates; the sole Luna child implements in the separate mda-efftox-core
checkout. Numerical work remains serial with one BLAS thread. No full-suite
run, new dependencies, large simulation study or new CI job is planned.

Primary source:
https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/CRMSuite/BMA-CRMSimulatorHelp.pdf

The 2018 CRM Suite guide specifies up to 200 patients, cohorts of one to four,
two to twenty doses, and a sample size divisible by cohort size. Its simulation
uses Poisson patient arrivals and Weibull event times calibrated by both the
window toxicity probability and the conditional fraction of toxicities in the
second half of the window. This batch reuses `toxicity_time_quantile`, whose
independent high-precision reference checks already cover this calibration.

Scheduling contract for the independent Python implementation:

- Dose assignments are zero-based and each staggered cohort has a fixed dose.
- The first arrival includes an interarrival gap. A new cohort receives a
  decision at its prospective first arrival. Later members arrive at the
  supplied gaps without a dose change inside that cohort.
- A wait advances to the next ascertainment among patients already enrolled
  at their assigned dose. It does not inspect another potential dose or enroll
  a future patient. A waiting prospective patient is treated when the wait
  resolves; subsequent gaps begin from actual enrollment, without a queue.
- Maximum enrollment is followed by complete ascertainment before final MTD
  selection. A safety stop prevents further enrollment and remains a stop;
  later follow-up never silently restarts the trial or selects an MTD.
- Decision time and final outcome-ascertainment time are separate. Enrollment
  suspension excludes ordinary final follow-up.
- Compact per-decision summaries are retained, never successive MCMC draws.
  Per-decision and whole-run evaluation budgets raise on exhaustion instead
  of reporting an unfinished trial as a completed result.

These choices are explicit because the guide does not expose all internal
scheduling details. No native scheduling or random-stream parity is claimed.
The newer CRM Suite policy is used; the older desktop DA safety-wait exception
remains a distinct, unimplemented version behavior documented in dacrm.md.
All arrival rates and event windows must use the same time unit. The source
interface's separate days/months display does not establish a conversion
convention for this independent API.

Random patient generation must remain separate from DA posterior sampling.
Changes in posterior sampler effort should not consume the generator used for
potential patient outcomes or arrivals. Operating-characteristic summaries
must distinguish no selected dose from dose zero and distinguish a safety stop
after maximum enrollment from an early stop before that maximum.

Trial checkpoint 199cc87 was integrated as 5259237. Root generated an
independent base-R fixture for all four binary outcome histories in a
two-patient, two-dose complete-data trial. The reference integrates the power
model directly and checks the next assignment and terminal selection. It also
provides final posterior means. The three scheduling tests plus one reference
test passed with warnings as errors in 2.84 seconds; peak process memory was
134.70 MiB with zero reported process swaps. Ruff check/format and targeted
mypy passed. Reference source, fixture and test are committed as a0bebe3.

Review corrections passed to Luna for the simulation checkpoint: preserve the
200k/2m default evaluation budgets while allowing explicitly larger hard
limits, and report the lowest-dose safety probability consistently in compact
look-ahead history records. The latter records the observed-data posterior
probability; hypothetical completions are evaluated inside the look-ahead rule.

The fixes and simulation were integrated as 95e2d37 and 62a9516. The combined
six focused tests passed with warnings as errors in 6.81 seconds; peak process
memory was 126.19 MiB with zero reported process swaps. The simulation checks
cover reproducibility, patient-count conservation, immutable results, and the
DA sampler advancing without consuming the scenario generator. Ruff check and
format passed for six affected Python files; targeted mypy passed for both new
modules. The full suite was deliberately not run for this bounded coverage batch.

Additional small local checks distinguished final safety nonselection from an
early safety stop, rejected two wrappers sharing one random bit generator,
rejected a wrongly shaped late-event probability before consuming scenario
randomness, and confirmed budget exhaustion raises. No large simulation study
or extra CI job was added.

The wheel and source distribution built using the cached Hatchling backend.
An isolated wheel import verified all five new exports, byte equality of the
two modules, exports and catalog, and exclusion of raw upstream downloads.
All three documented examples ran from that wheel with warnings as errors in
6.37 seconds. Peak process memory was 115.86 MiB with zero reported process
swaps. These measurements describe the validation processes, not whole-system
memory pressure or the resource needs of every permitted simulation setting.

Catalog entries 81 and 132 remain partial because native reports and the older
desktop conduct differences are still open. Entry 133 remains pending its
model-selection/calibration work. This is progress toward the overall goal,
not completion. GitHub publication remains blocked by the earlier tool error
requiring approval while this session's approval policy is never; no alternate
write transport was attempted.
