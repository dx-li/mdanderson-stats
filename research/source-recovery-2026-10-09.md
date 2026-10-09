# October 9 continuation: native boundaries and replayable studies

The bundled 138-entry snapshot remains **90 implemented / 42 partial / 6 pending**.
A fresh complete official-site inventory is not established because its
metadata endpoints return HTTP 500. Full functional coverage remains the goal.
The [coverage guide](../docs/software-status.md) and
[42-entry contract review](statistical-coverage-priority-2026-10-04.md) track
completed and remaining scope. Six clinical/design calculators still require
source methods or exact fitted constants; they are not replaced with guesses.

## Newly resolved Multc Lean contracts

Entry 12 now provides `multc_legacy_boundaries`, automatic design-duration
replay, native-defined study summaries and bounded legacy Monte Carlo studies.
The native prior screen precedes minimum enrollment. Toxicity vectors count
nontoxicities; disabled endpoints still carry a cap placeholder. The duration
kernel preserves latent counts, clipped/shared follow-up and balked arrivals.
Batch outputs include duration/enrollment/response/toxicity/balk means, the
sample-size PMF, additional MCSEs and replayable Python PCG64 child seeds.

The new [audit](multc-native-boundaries-audit.md) distinguishes native machine
instructions from substituted services: 22 compact-boundary/complement designs
use independent base-R posterior predicates (7,496 saved probabilities), 54
duration replays execute the original kernel with the resulting vectors, and
22 original study-wrapper runs verify means/PMF and prior-screen exits. The
original numerical integrator is not run. All original DLLs/installers remain
ignored local research inputs under their upstream redistribution restriction.

Entry 12 remains partial: native posterior-integrator/full-application parity,
saved files/defaults and protocol/report workflow remain unverified. The
existing saved Python study uses its explicit observation-aware calendar
policy; the new legacy API preserves the recovered historical simulation.

## Validation and delivery

The preceding committed baseline passed the complete **33,726-test** suite
with warnings treated as errors in **881.01 seconds**. This run used a frozen
snapshot of commit `50630e2`; it does not include the additions above. All
**214** focused legacy duration/boundary/simulation tests pass, including native
references, analytic duration/MCSE, replicate replay, immutable inputs,
stream exhaustion and work/storage preflight. Final affected regression and
packaging results are recorded below.

The final affected Multc/WFMM regression passes **519 tests** in 17.08 seconds
with warnings as errors. Ruff, formatting (2,392 files) and mypy (693 source
modules) pass. Wheel/sdist build and isolated-wheel byte checks cover all 693
modules, catalog, license and notices, and exclude original research binaries.
All **11** Python examples in the WFMM and legacy-duration guides execute from
the isolated wheel. Complete collection contains **33,870 tests**; the new
144 tests supplement the frozen full baseline, rather than claiming a second
whole-suite run. The latest continuation/start instructions are saved in the
environment configuration draft; Review/Publish activates that draft.

The baseline is committed and pushed to `codex/native-software-coverage`.
GitHub API read/create requests return `Forbidden`, including the actual
pull-request creation request. No PR is claimed to be open. The prepared branch
can be reviewed and submitted through
[GitHub's PR creation page](https://github.com/dx-li/mdanderson-stats/pull/new/codex/native-software-coverage).

## Next source work

Continue with the recovered original bundles and the primary supplements in
the contract review. Multc Lean's managed/native conversion and study file
reader are the next concrete local leads. WFMM's native MOM/profile optimizer,
automatic proposals, extended wavelet layouts and pass/file workflows remain
separate gaps. SYNERGY's semiparametric bootstrap interval convention and
iBOIN's final isotonic weights/ties/defaults still require primary evidence.
The six pending calculators require their recorded likelihoods/priors or exact
fitted parameters/baselines; repeated blocked downloads provide no new proof.
