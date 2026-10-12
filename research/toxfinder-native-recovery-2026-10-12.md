# ToxFinder recovered native workflow — 2026-10-12

## Source and scope

Official archive:
https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/ToxFinder/ToxFinder_V1.1.1.zip

Archive SHA-256: `eddd41346542f3ee398a07eef8873a3687c3f3a9fa93e2a960ed217d01276d70`.
Original `ToxFinder.exe` SHA-256:
`3951dc95cdfaaad0da0526cfb25d28b7f970d4d82a6836203da4a469282d8070`.
The recovered application algorithm is C with Fortran LAPACK runtime routines;
calling the entire application a Fortran implementation would be inaccurate.
Original binary/source/guide assets remain outside shipped artifacts.

Installed `parameters_tox.wgc` exposes allocation values `1;4`, labels
“Fisher Information and Cancer Killing Potential” and “Fisher Information only”.
Undocumented random/distance branches are not necessary for the two advertised
modes and are not silently mapped to either supported mode.

## Executable evidence

`tools/reference_toxfinder_native.py` takes the hash-pinned executable, fixture
input directory and output directory. It executes bounded x86 routines using
Unicorn and a PE loader. There is no Windows process or native RNG run. Runtime
allocation/free and power services are supplied; the complete information probe
also supplies NumPy LAPACK eigensystem/Cholesky and skips unused postfit covariance.
Controller probes supply explicitly declared predictive/score callbacks.

- MAP `0x403920`: original likelihood, unnormalized Gamma prior and two simplex
  passes, each 250 outer iterations, start at prior means and fixed original
  half-mean steps. Bounds `[.005,30]`; budget exit returns row zero, not sorted best.
  Four complete history fixtures; independent Python agreement within `3.78e-14`
  in the source-recovery experiment. Production tests allow input-conversion
  roundoff separately.
- Information `0x402040`: original observed-history MAP, candidate augmentation,
  Hessian, eigenvalue floor `.1`, determinant and posterior-predictive weighting.
  Three full-chain fixtures include a zero-axis candidate. Independent 40-digit
  differentiation differs by at most `1.78e-15` in Hessian entries and about
  `2e-15` relative in information. Hypothetical outcomes do not change the MAP.
- Allocation `0x401d70`, `0x401580`, `0x4015f0`, `0x4016c0`, `0x401790`,
  `0x401b90`: eight production-grid fixtures, both advertised modes, constant
  objective ties and nonlinear L1. Independent traversal reproduces all pairs.
- Stage one `0x401400`, `0x4010c0`, `0x401220`: eight fixtures distinguish
  highest-ever versus last dose, pre-toxicity interpolation, first-toxic-level
  midpoint expansion, upper-candidate exclusion and strict nearest ties.
  First-candidate crossing has an apparent out-of-bounds fall-through. Python
  deliberately returns the lowest dose; arbitrary predecessor memory is not a
  valid numerical contract.
- Bilinear surface `0x4057d0`: six endpoint/interior fixtures, rows indexed by
  drug 2. The harness explicitly sets x87 precision control for this routine.

## Controller and files

Native `0x402d00` refits once per stage-one cohort and once per pair of stage-two
cohorts. Stage-two first uses branch2, then branch1 from the same posterior.
Original archived rows10/11,12/13 and28/29 share draws and recommendations.
Final report path `0x402fbd` uses the last preallocation fit without incorporating
final outcomes; Python exposes `legacy` and separately named `refit` policies.

Native patient loader `0x408250` excludes unset outcome2. Controller `0x4012e0`
uses filtered observed count for stage and alternating branch. Admitted count
only bounds enrollment capacity. Dates are identifiers, not a time-to-event
likelihood. Python preserves pending records and never codes them as non-toxic.

The original Gem-CTX TMML example ends at normalized `(1,1)` and uses one-based
starting row2. Readers preserve unknown metadata, require explicit physical
scales for CSV, and never infer six prior moments from old `dims.csv` headings.
Saved elicitation/MCMC controls are retained; callers supply a validated prior
and explicit Python inference settings.

`toxfinder-native-archived-summary.json` contains trial0's bounded numeric CSV
excerpt, 60 retained draws and the scenario surface, with independent 60-digit
posterior/report references. Posterior means are
`.28840212760193515, .29872597172271056, .29711611093883050`.
Original prediction agrees within `2e-16`. Both sample/population SD references
are recorded. Original Java later confirms posterior sample SD, but across-trial
GUI columns mislabeled SD use SEM and special sentinels. Python reports correctly
labeled sample SD and MCSE.
The example identifies version1.0.0 (December2004), so it is not a 1.1.1 RNG fixture.

## Remaining limits and classification

The scientific/controller/report workflow is now executable end-to-end with
explicit priors, independent Bayesian inference and recovered native allocation.
Coverage remains **partial** conservatively: native prior-solver and adaptive
MCMC/RNG identity, complete original cohort-file/report round trips, GUI parity
remain unverified; the GUI SD/SEM discrepancy is now established explicitly. These are distinct from the now-resolved
information-method blocker. No missing method was replaced by an invented ridge,
coefficient or standard optimizer, and no broad 138-application completion claim
is justified. Existing distribution/permission decisions remain unchanged.

### Original Java report convention

Original `ToxFinder.jar` SHA-256:
`39e72d4b535595f548b6602496227bc92a6d9107fbb3ce5560973342fd39897a`.
`RawScenarioSummary.calculateSurfaceToxicitySD` calls
`ToxicitySurfaceProvider.calculateSurfaceToxicityVar`, then the no-mean overload
`calculateVariance(double[])`: `(n*sum(x²)-sum(x)²)/(n*(n-1))`.
Posterior SD therefore uses sample variance. The supplied-mean overload divides
by `n`, but is not this report path.

Across-trial `RawScenarioSummary.calcColValue` creates `RawColumnValue` values;
`Column.getValue` with `UseSD=true` directly returns `RawColumnValue.getMeanSD`,
which computes `sqrt((sum2-sum²/n)/(n-1)/n)`: SEM, despite the GUI's SD label.
It returns `1e-5` for constant repeated values and `1e10` for at most one value.
`tools/reference_toxfinder_gui_summary.py` executes these original public Java
methods without GUI startup; its four-case fixture confirms the distinction.
Python's correctly labeled sample SD and MCSE are intentional reporting choices.

### Batch validation

Full warnings-as-errors regression: **34,091 passed in 815.12 seconds**.
Affected ToxFinder tests:75. Fresh Python3.13 installed-wheel tests:75.
All six original x86 fixture sets and original Java summary fixtures regenerate
as identical JSON values. Independent high-precision references pass. Installed
wheel runs all five original 60-patient scenario structures with low-draw smoke
settings, and reproduces final contour pairs exactly for all250 archived trials
using their recorded posterior draws. See `toxfinder-validation-2026-10-12.json`.
