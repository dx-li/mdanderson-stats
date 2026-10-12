# Community completion continuation — October 12, 2026

Latest workflow scope: ARAND62, desktop CRM81/132 and ToxFinder14 now meet their
declared Python workflows. The final status checkpoint passed34,696 full tests
and358 installed-wheel tests; see the native-workflows validation manifest below.
See the final ARAND section below and the [next native contracts](next-native-completion-contracts-2026-10-12.md). Earlier dated sections retain historical counts and evidence.

## Recovered checkout and scope

Repository: https://github.com/dx-li/mdanderson-stats.
The saved checkout was clean on `work` at `c45e678`, mirroring its cached main.
The local `codex/native-software-coverage` reference was older (`511b262`). A
normal fetch recovered the actual feature head
`1f584b8c3b7685e5ec747e55c51bdb97a79ab1a2`, including all six continuation commits
above c45e678. Work continues on `codex/community-completion-2026-10-12`, descended
from that head. No reset, forced update, or removal of unmerged work occurred.
No AGENTS.md files were present in the saved workspace.

The original scope is every desktop and online application, with an intermediate
target of at least 100 implemented and no pending entries. Live membership was
rechecked: both API response hashes match the preserved September inventory,
80 desktop plus 58 online entries. This does not assert a fresh version-specific
audit of every app. The current count is **97 implemented / 35 partial / 6 pending**;
the intermediate and full-project goals remain unmet.

## Completed in this continuation

- Recover and independently implement the official mTPI R final estimator:
  Beta(.005,.005), inverse-variance isotonic weights, source candidate range and
  dose-index offsets. Preserve historical exclusion/stop inputs and keep the
  paper-based controller/simulation API unchanged. Explicitly correct degenerate
  R index ranges. See [guide](../docs/mtpi-legacy-selection.md) and
  [source audit](mtpi-legacy-selection-audit.md).
- Execute unchanged original R helper/final-estimation statements for ten
  synthetic states. Independently check the weighted-isotonic mathematics using
  exact-rational max-min formulas over all 60 (2,3,4)-patient response states.
- Repair 19 missing guide links and stale STPLAN reporting text; add contributor
  instructions and a [138-application status report](per-application-checkpoint-2026-10-12.md).
- Record a [licensing path](community-license-path-2026-10-12.md), including a
  bounded static dependency map for the explicitly restricted adaptation families.
  No terms were removed or relabeled. The complete bundle is not unrestricted
  open source merely because original contributions use MIT.

## Validation

The full warnings-as-errors regression suite passed **33,994 tests in
799.46 seconds** under Python 3.13.5. It included the new estimator and all
17 added tests. An earlier baseline run was deliberately interrupted when
the new tests were ready and is not counted as a pass. Only import ordering,
catalog metadata and documentation changed after the successful run started;
no statistical implementation or test cases changed. Final focused checks
cover the delivered import order. Exact-head GitHub CI remains outstanding.

Focused mTPI tests: **38 passed**, warnings as errors. Independent original-R
fixture regeneration is byte-identical. Lint, formatting and mypy pass; mypy
covers 697 source files. Wheel and sdist builds pass. All packaged Python/JSON
source bytes match the checkout, notices are present, and ignored original
research/binaries are excluded.

A clean Python 3.13.5 wheel installation outside the checkout with `[plot,image]`
passed **66 focused tests**, including mTPI, STPLAN studies and independent TIFF
fixtures, plus **3 BLiP plotting tests**, all with warnings as errors. Python
examples from `mtpi.md`, `mtpi-legacy-selection.md`, and `stplan-study.md` ran from
the installed wheel and verified that the imported package lives in the isolated
environment. Its resolved NumPy 2.5.4 / SciPy 1.18.1 / Matplotlib 3.11.2 / Pillow
12.3.0 installation is independent of the development environment (NumPy 2.5.3).
This is not a clean-installed example audit of all 138 apps.

Local link checks and catalog/status ID checks pass. Existing BayesFactorTTE
managed-control references still substitute declared RNG/Bayes-factor services;
their scope is not broadened to integrator/RNG identity by regression tests.

## Prioritized remaining work

1. Resolve the mTPI controller differences recorded in its audit, including
   distinct safety priors, the missing beta-shape subtraction, top-dose stop
   behavior, mutable-dose control flow and Excel/XLL comparison. Then complete
   tuning and reporting before considering a #72 status promotion.
2. Continue the executable-reference leads for EffTox contour/trinary calibration,
   Adaptive Randomization scheduling/floors, BMA-CRM prior/skeleton calibration,
   CRM Suite timing/precedence, and the separate online BMA-CRM contract. See the
   [earlier pinned archive leads](source-recovery-2026-10-10.md#concrete-next-candidates-toward-100).
3. ToxFinder's archive was downloaded again; SHA-256 matches
   `eddd41346542f3ee398a07eef8873a3687c3f3a9fa93e2a960ed217d01276d70`.
   The Wise payload/Java stage-two information criterion remains unrecovered.
   [WiseUnpacker](https://github.com/mnadareski/WiseUnpacker) points to an
   extractor, [REWise](https://codeberg.org/CYBERDEV/REWise); a direct clone of
   that host failed with CONNECT 403. No proxy/trust/access changes were made.
4. Keep all 39 partial statistical contracts open until independently resolved.
   The [per-application report](per-application-checkpoint-2026-10-12.md) records
   each gap, rather than treating these as merely UI cleanup.
5. Recover exact evidence for six pending applications: ComPAS joint model and
   priors; BayesianSurvival fitted regression/posterior; BLESS four coefficient
   vectors and baseline tables; CNSRISK competing-risk fit/baselines; K-COMPASS
   transformations, Weibull fit and uncertainty; MDS-DPSS dynamic score/calibration.
   Rounded UI examples do not supply these contracts. No approximate substitute
   changes pending status.
6. Resolve distribution scope before release: file-level provenance review and
   an independently implemented open-source core or a separately labeled legacy
   package. The current five restricted families are a minimum set; ambiguous
   retained notices need review. Splitting the public API is a project decision.

## Delivery boundary

No merge, registry release, deployment, third-party contact, permission change,
or environment-settings activation was performed. The earlier PR API Forbidden
response was not retried through another route. A feature-branch push does not
trigger the active GitHub workflow. Local validation does not establish a green
GitHub run for the final commit. No verified URL for this continuation task was
exposed by the execution environment.


## Second implementation batch: native workflow recovery

The initial a4d8e57 checkpoint above is preserved. Subsequent work adds the
[EffTox fitter](../docs/efftox-native-contour.md), fixes pending-data Suite safety
waits, and records [native workflow evidence](native-workflow-recovery-2026-10-12.md).
The ToxFinder MSI payload has now been recovered directly from the .WISE section;
its scientific core is a stripped native Fortran executable, not the GUI JAR.
The [six-model complementary evidence pass](pending-model-evidence-2026-10-12.md)
closes none of the six exact fitted contracts. The [licensing proposal](community-license-path-2026-10-12.md#concrete-split-proposal-not-activated)
now identifies lost APIs, distinct namespaces and compatibility requirements.
No split, relicensing, release, access change or outreach was activated.

EffTox's recovered exported fitting service is complete within its documented
finite/nonsingular input bounds. Whole-application status remains partial because
native files/reports, the legacy utility evaluator and historical trinary behavior
remain unresolved. ARAND and CRM source recovery reveals controller differences
that cannot safely be relabeled as already implemented native behavior.

The earlier 33,994-test full regression applies to a4d8e57. The current batch uses
a focused affected-model regression and installed-wheel verification; collection
alone is not a full regression run. See validation addendum below.

### Second-batch validation

- Affected EffTox, DA-CRM, CRM calendar/trial/simulation and BMA-CRM regression:
  **130 passed**, warnings treated as errors. The new suite initially exposed
  reference-test scope mismatches (raw helper versus adjacent-dose policy) and
  rounding amplification in an exhausted, unselected optimizer start; the final
  tests distinguish these contracts explicitly. Final selected-fit coefficient
  tolerance stays 2e-10; all native iteration/evaluation counts agree.
- **49 passed** from copied tests outside the checkout against the isolated
  installed wheel. A fitted-contour simulation also ran there: four trials,
  16 patients. Installed package resolved under site-packages, not the checkout.
- Original EffTox instruction runner regenerated the fixture JSON byte-for-byte.
  Original CRM Suite x86 runner generated hash-pinned branch fixtures, including
  safety/minimum wait combinations, adjacent and multi-level raw gates, and
  supplied exponential-wait arithmetic.
- Ruff, formatting (2,427 files), mypy (698 source files), wheel/sdist build and
  `git diff --check` passed. Wheel source equality for changed production files
  and exclusion of original binary assets were checked.
- Current collection: **34,030 tests**. This is not a claim that the full suite
  was rerun after this batch; the prior full run was 33,994 passing at a4d8e57.

The feature branch remains the publication boundary. No PR creation retry,
merge, workflow activation, release or author contact was performed.

## CRM desktop and ToxFinder workflow batches

The CRM desktop batch (`4147ad7`) completed native HTML/XML study reading,
scenario simulation and reports, desktop candidate-arrival scheduling, DA prior
endpoint safeguards, and exact-residual original-R mTPI tie verification. Its
whole suite passed **34,044 tests in 806.61 seconds**; isolated-wheel tests,
original saved studies, lint/types and packaging also passed.

The subsequent ToxFinder batch resolves the native information-method blocker
and delivers named observed-history, complete two-stage simulation, repeated-trial
report and native file-reading workflows. The original core is C with Fortran
LAPACK runtime, correcting the earlier whole-core Fortran description above.
See [native recovery](toxfinder-native-recovery-2026-10-12.md) and
[usage guide](../docs/toxfinder-native-workflow.md). Original Java report probes
also establish the native SD/SEM label discrepancy; Python labels them correctly.

Its 75 affected tests and 75 tests in a newly created Python 3.13 environment
passed. Six original x86 fixture sets regenerated exactly as JSON values; native
Java summary fixtures also regenerate. Independent 40-digit Hessians and 60-digit
archived report calculations passed. From the installed wheel, all five original
TMML scenarios completed 60-patient smoke simulations with low draw counts and
settings round-trips. More strongly, all **250 archived trials** reproduced all
three final contour pairs exactly from their recorded posterior draws. This
checks the final decision calculation, not native MCMC/RNG identity.

The full ToxFinder-batch regression is recorded below when complete. Its
production code and tests were frozen before the run; subsequent changes were
source evidence, reference tools and documentation only. Remaining work follows
the [dependency-ordered roadmap](completion-roadmap-2026-10-12.md), not the stale
recovery leads in earlier checkpoint sections. No coverage label was promoted.

### ToxFinder batch final validation

The complete warnings-as-errors suite passed **34,091 tests in 815.12 seconds**
on the frozen production code and test set. Ruff lint/format (2,442 files), mypy
(703 source files), build, local documentation links and all 138 catalog/status
rows passed. The fresh installed-wheel checks and native/independent numerical
references above supplement this full run. No head-specific hosted CI is claimed.

## ToxFinder native stochastic calibration and durable ARAND recovery

The named native ToxFinder elicitor now implements the recovered fixed-mean
formula, 30,000-pair Gamma objective, nine-start precheck, refreshed simplex,
three-attempt retry and native acceptance/failure contract. It uses independent
NumPy random draws and preserves native float32 Gamma-shape narrowing. The
published-equation elicitor remains distinct. Original convergence can accept a
poor objective; a separate assessment therefore exposes tail probabilities,
Monte Carlo errors, residuals and cutoff status without changing the fitted prior.
No failed fit silently substitutes a fallback.

Validation on the frozen production code and tests:

- **34,105 passed in 813.88 seconds**, full suite with warnings as errors.
- **89 affected tests passed**; the same **89 passed in 9.13 seconds** outside
  the checkout in a new Python 3.13 environment against the installed wheel.
- The installed calibration-to-simulation guide example completed successfully.
- All eight original x86 fixture families regenerated as identical JSON values,
  including every call in the 27-attempt, 2,675-evaluation failure path. Independent
  Gamma-density integration checks the seeded Monte Carlo probabilities.
- Ruff lint/format, mypy (704 source files), build, changed documentation links,
  all 138 catalog/status rows and artifact checks passed. Wheel production bytes
  match tested sources; wheel/sdist contain no original EXE/DLL/MSI or raw assets.

ARAND recovery is now durable: a portable, hash-pinned original-instruction
runner regenerates **98 fixtures across 13 families**, including integrated
controller ordering and native TTE censor sufficient statistics. The source audit
also establishes exposed desktop scope, fixed binary delay, TTE scenario defaults
and all-arm posterior ranking. This is reference evidence for the next named
Python workflow, not a completed native controller or a coverage promotion.
See [ARAND recovery](arand-controller-recovery-2026-10-12.md).

Coverage remains **93 implemented / 39 partial / 6 pending**. Remaining work is
listed per application in the [status report](per-application-checkpoint-2026-10-12.md)
and ordered in the [roadmap](completion-roadmap-2026-10-12.md). Exact unavailable
fitted models and distribution licensing still block a complete, unrestricted
community release. No head-specific hosted CI, original RNG identity or verified
task URL is claimed. No PR permission-denial workaround, merge, registry release,
settings activation or third-party contact was performed.

## ARAND native-profile integration continuation

The feature branch was recovered at `e74b1ef82904a8d0587442abd17d53242e0cf183`.
The next [integration checkpoint](arand-native-integration-2026-10-12.md) closes
a named single-endpoint native supplied-calendar/Python-study workflow, including
controller ordering, all-arm posterior integration, censor-aware simulation,
versioned JSON and explicit reports. The existing policy API remains available.
All 98 original fixtures regenerate; all six integrated controller cases agree;
1,200 additional original-instruction comparisons pass. New bounded report-index
and missing-action probes narrow the next native saved-study/report gates.
Coverage remains 93 implemented/39 partial/6 pending. Consult that checkpoint's
final validation and remaining contracts before continuing; do not treat the
source-recovery-only next steps above as still unimplemented Python work.

### ARAND batch final verification

Corrected final code passed **34,179 tests in 818.71 seconds**, warnings as errors.
All 89 affected ARAND tests and 120 isolated-wheel tests (including plot/image
extras) pass; the installed saved-study/simulation/report example also passes.
Lint, formatting, mypy 707-source-file checking, wheel/sdist builds, source-byte
and notice/exclusion audits pass. All 98 prior original fixtures regenerate,
27 additional source probes are preserved, and 1,200 differential comparisons
pass. See the [integration checkpoint](arand-native-integration-2026-10-12.md)
for the interrupted attempt, allocation-population correction, exact evidence
boundaries and next source contracts. Code was pushed at 3732fde7; supplementary
assignment evidence was pushed at 78653d44; final validation follows on the same
feature branch. No merge, release, outreach, PR bypass or license change occurred.

### Continued ARAND native desktop work

After 37a1f7d, constraint switches, per-arm stopping/selection cutoffs and native
batch construction/reset were integrated with independent original-instruction
checks. Original managed serialization now runs in an isolated local Mono tree;
valid synthetic binary/TTE XML plus exact normalized round trips are preserved.
See the continuing section of the integration checkpoint and new reference
runners. No app promotion. Continue through XML/history, per-arm parameter
thresholds/delays, wrapper scenario and native-report contracts; do not stop at
this intermediate checkpoint. Existing license/PR restrictions remain unchanged.

### XML simulation checkpoint and actual report override

Original binary/TTE XML now maps to fresh simulation studies, preserving source
text/history metadata. Per-arm parameter thresholds and response delays are
integrated. Clinical-history import/editing is deliberately still open.
IMPORTANT: base noncovariate report0x57b68 is overridden by native x86 entry
0x18bf90 (slot16); simulation override is0x18d3e0 (slot12), through vtable field0x04000524 assigned at wrapper XMLPopulate IL410.
Do not claim actual report parity from the base percentile fixtures. Shared
accumulation0x59ac0 now has12 full-body original fixtures. See the integration
checkpoint's new section for exact evidence, harness issues and next contracts.
Full final-code tests are running; only completed final results may be reported.

### Per-arm report completion and source-scope correction

Guide-required per-arm early selection/drop reporting is implemented; the actual
native report sums final statuses1+2+3. The explicit legacy-5.2.2 percentile mode
matches actual x86 report probes and suppresses intervals below100. Actual
native replication-loop and original scenario-seed probes are preserved.
CLR FIXUP CORRECTION: source dispatch must resolve the thunk's MethodDef token,
not its neighboring IL body. Native simulation is0x18d3e0/slot12; native report
is0x18bf90/slot16; shared accumulation0x59ac0/slot32. The revised manifest records
all slots. See arand-simulation-scope-2026-10-12.md: the advertised application is
simulation; latent clinical-history conduct is a separate scope, not an automatic
blocker for guide-scope implementation. No promotion before final verification.

### Final ARAND guide-scope completion

ARAND62 is now implemented for its advertised version5.2 simulation workflow:
Binary/TTE models, original simulation XML, native monitoring/stopping and
batch reset, per-arm thresholds/delays, first-candidate timing, scenario seed
offset, saved Python studies and complete per-arm reports. Explicit standard
and legacy percentile profiles remain separate. The recovered original simulator
starts its first candidate at0; the production correction is preserved at
`da61cc310f60f62cb138ffdd32f63f2a70504def`.

Final code: **34,533 tests passed in837.77 seconds**, warnings as errors. All443
ARAND and477 isolated-wheel/extras tests pass, as do installed JSON/replay and
original Binary/TTE XML-to-report examples. All98 baseline fixtures regenerate;
4,200 differential comparisons and extended native/managed probes pass.
Lint/types/build,712 source-file artifact matches,31 root exports and retained
notice/provenance/exclusion checks pass. See [validation](arand-final-validation-2026-10-12.json).

Current counts: **94 implemented / 38 partial / 6 pending**. Only62 is promoted.
The [scope crosswalk](arand-simulation-scope-2026-10-12.md) excludes latent conduct,
RNG/integrator bit identity and GUI formatting from the advertised simulation
contract; clinical validity is not asserted. Next use the
[CRM/BMA-CRM and ToxFinder contract notes](next-native-completion-contracts-2026-10-12.md),
then the remaining per-application roadmap. The six fitted models still require
exact unavailable contracts. Mixed-license distribution and Forbidden PR creation
remain unresolved; no split, release, merge, outreach or bypass is authorized.

## CRM populated-history checkpoint

CRM Suite and older BMA-CRM patient files now have an explicit-date import/edit/
save/conduct workflow, original assessment censoring, current safety checks and
strict-cohort entry replay. Desktop simulation now applies the observed raw-rate
cap after lookahead. Reports expose saved design settings. See
[conduct recovery](crm-desktop-conduct-2026-10-12.md),
[guide crosswalk](crm-guide-scope-2026-10-12.md), and
[validation manifest](crm-conduct-validation-2026-10-12.json).
Both versions reproduce142 original instruction cases and four managed XML files;
four native code regions agree after explicitly mapped pointer relocations.
The catalog remains94/38/6 at this preservation checkpoint; promotion is separate
from code recovery. Next is the already source-recovered ToxFinder saved-prior/
patient/report journey, then the application scope/status checkpoint. Native RNG
identity, clinical validity and mixed-license distribution remain separate claims.


## Completed CRM/ToxFinder scope checkpoint

Desktop entries81/132 and ToxFinder14 are implemented under their explicit Python
profiles. Current catalog: **97 implemented /35 partial /6 pending**. This is not
a claim that all138 workflows or an unrestricted community distribution are done.

Final statistical/application code passed **34,696 tests in863.35 seconds** with
warnings as errors and plot/image extras. A fresh outside-checkout wheel passed
**327 CRM/ToxFinder tests plus31 plot/image tests**. All five original ToxFinder
scenarios completed two60-patient trials each, with final refit, prior assessment,
report, TMML and patient CSV/recommendation checks. Tiny sampler counts are workflow
smoke evidence only. Earlier CRM validation covered all six installer studies.
Original Java, original x86, independent R, artifact/notices and export checks are
recorded in [final validation](native-workflows-validation-2026-10-12.json).

The next EffTox source pass recovered current Lp shape/utility execution and the
nonstandard patient outcome ordering. It does not promote EffTox. Its exact next
contracts are in [current workflow recovery](efftox-current-workflow-contracts-2026-10-12.md).
Other partial applications retain their own per-entry contracts. The six fitted
models still need exact unavailable artifacts; no approximate substitution,
license change, PR bypass, merge, release, deployment or author contact occurred.


## Verified acceptance-review checkpoint and current EffTox recovery

All four acceptance findings are corrected: ARAND retains quadrature errors
through positive-power allocation; CRM final pending consensus retains its
three-subject uncertainty qualification; ToxFinder rejects provably inadequate
work/memory budgets before sampling and preserves per-allocation MAP diagnostics.
Default sampler quality and bounded-work safeguards are retained.

Final production code passed **35,217 tests in927.94 seconds**, warnings as errors.
A fresh outside-checkout wheel passed **1,390 family/extras tests**. Both installed
CRM versions' six saved examples/five scenarios each and all five ToxFinder
examples completed their smoke workflows. A separate installed two60-patient
ToxFinder execution at500 draws/250 warmup/two chains completed with57,240,270
counted evaluations in658.81 seconds, maximum final split R-hat1.007575, and all
MAP passes converged. This is a representative check, not universal mixing or
clinical validity. The independent six-dimensional importance comparison also
passed; see the acceptance review evidence and validation manifest.

EffTox remains partial. Current six-coefficient XML/patient interchange and a
separate desktop decision profile are implemented. Native source now provides
594 bounded instruction cases:500 selection/admissibility,21 pending-loop,
33 cohort-prefix,32 outer-status and8 Rselect. Inference/completion callbacks are
explicit supplied services where noted. Simulation source confirms individual
patient accrual, cohort-boundary analysis, final refit even for partial last
cohorts, and Rselect conditional on selection. Conduct's nonmultiple-cap case
needs a safe explicit correction rather than silently allowing over-enrollment.
Next integrate those contracts with pending enumeration, saved-prior precedence,
simulation and complete diagnostic reports; then continue the35 partial rows.

Counts remain97 implemented/35 partial/6 pending. No status is promoted from
source recovery alone. Six missing fitted-model contracts, mixed-license
restrictions and the earlier Forbidden PR creation remain unchanged. No release,
merge, outreach, licensing split or permissions/settings change is authorized.

## EffTox integrated workflow preservation (validation in progress)

Current XML conduct, bounded pending completion, explicit posterior service,
patient add/edit/ID correction/delete/clear, prior-calibration precedence,
complete-outcome simulation and reports are now implemented. The focused suite
passes **680 tests**. The native-reference set is **738 cases**, including81
complete posterior-integrand cases matching independent Python likelihood/prior
weights within `6.871784649697795e-15` relative error. A recovered negative-slope
toxicity-summary inconsistency is isolated in `desktop_legacy`; scientific
model-based summaries remain the default. See
[workflow guide](../docs/efftox-desktop.md) and
[exact contracts](efftox-current-workflow-contracts-2026-10-12.md).

This is a preservation checkpoint, **not completed final validation or catalog
promotion**. The final behavioral code's full suite is running in
`/tmp/efftox-current-final-suite-v3.log`; installed-wheel verification is in
`/tmp/efftox-current-wheel-tests-v3.log`. Earlier v1/v2 full-suite runs were
terminated when the source-driven scope added the explicit summary profile and
patient deletion/accrual guard; do not mistake those logs for completed checks.
The installed precision comparison is running from
`/tmp/efftox-installed-precision.py`, with output
`/tmp/efftox-current-installed-precision.json` and `.log`. It compares default
model summaries, 3x draws/warmup, and explicit legacy summaries on two39-patient
trials each. All5 original scenarios had previously completed smoke execution;
those low-draw runs do not establish mixing quality. The default-quality model
run reached39patients in both trials, selected zero-based dose3 twice, and had
maximum split R-hat1.074 across fits.

Complete final verification and preserve the evidence, then continue all-scenario/
batch aggregation and precision acceptance. Counts stay97/35/6. The6 unavailable
contracts and distribution/PR restrictions are unchanged. No release, merge,
permission change, outreach or license split has been performed.
