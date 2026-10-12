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

### Latest stable EffTox verification and next source lead

All-scenario/batch reporting and the remaining patient-file mode transitions are
implemented. The original-file exercise caught a newline-padded shutdown zero;
the numeric check is corrected and independently exercised against the original
study. Add switches to TrialConduct; last/all deletion switches to Simulation,
as confirmed in original client IL. Empty saved studies remain editable without
scenarios but cannot run simulation. The focused suite now passes691tests.

The authoritative final full-suite run is now
`/tmp/efftox-current-final-suite-v5.log`, started after all behavioral edits.
Do not count v3 as a valid completed run: it loaded an old Decision class and a
new lazily imported report while edits continued, producing one mixed-revision
AttributeError. V4 was terminated for the subsequently discovered original-file
fix. Installed final-code checks are `/tmp/efftox-current-wheel-tests-v6.log`:
**1567passed in124.44seconds**. Latest built artifact checks use
`/tmp/efftox-current-artifact-audit-v6.log`. The original-file all-scenario/patient
exercise is `/tmp/efftox-current-workflow-examples/validation.json`.
The precision comparison completed and is recorded under the paths above.

While EffTox final verification runs, the next high-confidence source lead is
[BOIN desktop source recovery](boin-desktop-source-recovery-2026-10-12.md).
Its official installer contains active readable R wrappers and managed conduct
code;30 original startup/run-in cases are now preserved. This is research only,
not a BOIN implementation/status promotion. Its original-program redistribution
restriction is recorded; no original assets are bundled.

### EffTox acceptance closed; continuation target BOIN desktop

The final unchanged behavioral revision passed **35,394 tests in862.38seconds**
(`uv run --all-extras pytest -q -W error`). The final installed-wheel suite
passed1,567 in124.44seconds; lint/format/mypy/build and wheel/sdist byte/provenance
checks passed. The rebuilt/installed catalog confirms **98 implemented /34 partial
/6 pending**. [Final EffTox scope](efftox-current-scope-2026-10-12.md) and
[durable manifest](efftox-current-validation-2026-10-12.json) supersede previous
pending-acceptance notes. Default/higher-precision diagnostics remain qualified;
clinical validity and native numerical/RNG identity are not asserted.

BOIN desktop99 is next:78 original startup/run-in/full-wrapper cases and216
interactive boundary rows now preserved. The144 GUI-valid rows (432values)
match the installed existing Python core. Interactive contour conduct modifies
n=3 only; waterfall simulation modifies n=3 and n=6. Recover managed pending/
cohort/titration rules and project/protocol fields next, then integrate these
mode-specific contracts. Keep phantom enrollment, endpoint jump and cross-trial
state defects under explicit compatibility profiles, never scientific defaults.
No BOIN status promotion or original program redistribution occurred.

Six pending fitted models remain externally blocked; licensing/distribution
choices and previous Forbidden PR creation remain parent decisions. No merge,
release, deployment, outreach, permission change or license split occurred.

### Active BOIN implementation after cb99080f

BOIN saved-project reading, single-MTD/contour decisions, managed pending/cohort
conduct and patient edits are now implemented, with4,055 focused tests passing.
See [current scope and exact open contracts](boin-desktop-conduct-2026-10-12.md).
No catalog promotion; full final-code acceptance must follow remaining simulation,
selection, protocol/report and single/TITE saved-workflow integration. Original
controller576 cases and each interactive R mode2304 cases are preserved, with
1728 GUI-valid statistical comparisons per mode. Patient serialization executes
against the original hash-pinned data assembly under isolated Mono.

All four independent review findings remain fixed and explicitly preserved in
[acceptance corrections](acceptance-review-fixes-2026-10-12.md) and its linked
validation manifest: ARAND unresolved-tail rejection, ToxFinder exact resource
preflight, CRM final-consensus uncertainty reporting, and ToxFinder MAP diagnostic
propagation. Their7 targeted regression instances were rerun during this BOIN
work and all passed (ARAND4, CRM1, ToxFinder2). Previous35,394 full-suite acceptance
covers the last EffTox checkpoint, not unfinished BOIN changes.

### BOIN combination workflow checkpoint; continue single/TITE

[Scope and exact remaining contracts](boin-desktop-workflow-checkpoint-2026-10-12.md)
and [validation manifest](boin-desktop-workflow-validation-2026-10-12.json) record
41,052 full-suite tests,5,742 clean installed BOIN tests,28 installed extras tests,
all four original defaults and seven retained independent-review regressions.
10,000 installed safe combination trials cover all five original scenarios.
No catalog promotion:98 implemented/34 partial/6 pending. All original assets
remain excluded, with mixed-license notices retained.

Continue immediately with UseDefaultBounds active/inactive mapping, saved single
simulation/3+3 and desktop TITE/Rolling Six/calendar/protocol contracts. Native
TITE references now include2304 traces; single-agent simulation192; Rolling Six96;
STFT84 executed IL cases. Draft independent kernels exist locally in
/tmp/boin_desktop_tite_calendar.py and /tmp/boin_single_kernel_draft.py; they match
2304 and192 source fixtures respectively but are not integrated production APIs.
/tmp/boin_tite_settings_draft.py is an unvalidated settings-adapter draft. Native
STFT saved follow-up is alwaysdays, and native simulation normalizes timing
weights while the calculator uses raw weights (valid GUI priors sum to one).
Do not silently replace the desktop calendar with the existing web calendar.

### BOIN single/TITE integration after1c6f618

Current implementation supersedes the above unintegrated-draft notes. Saved
single/TITE simulations, active bound flag, native desktop calendar, STFT,
3+3/Rolling Six comparisons, all-scenario batches, bilingual methods and reports
are implemented. [Scope and precise remaining contracts](boin-desktop-univariate-checkpoint-2026-10-12.md).
Catalog remains98/34/6. New pending-table references include7124 rows from64
original requests; a temporary independent probe matched61916 threshold-neighborhood
evaluations, but the public table/flowchart and standalone report crosswalk remain.
The draft probe is `/tmp/boin-pending-table-probe.py`; original call-builder IL is
`/tmp/boin-boundary-ui-il.txt`. Keep source-table two-decimal/strict-boundary rules
separate from unrounded safe mathematics and the simulator's run-in rule.
Continue through available recovery work; these are not external blockers.

The univariate integration checkpoint passed43,912 full-suite tests in879.81s
on unchanged production code. Fresh-wheel evidence:8,518 desktop tests plus481
adverse-history and960 report-numeric tests added during the full run;21 plot,
28 image and7 review regressions; all4 original defaults.40,000 installed quality
trials pass;13 R fixture files regenerate identically. See the linked manifest
for exact qualifications. Newly recovered report sources distinguish native
console versus returned MTD definitions; pending-table and report-summary
integration remain the next authorized work, not a completed application claim.

### BOIN99 complete-workflow acceptance; continue mTPI72

The [complete crosswalk](boin-desktop-complete-checkpoint-2026-10-12.md) closes
pending decision/flowchart, score-table, operating-characteristic summary,
standalone final/calculator and captured-input protocol workflows. All four
modes have installed examples; native defects remain explicit compatibility
profiles. Catalog99 is implemented for this declared scope; counts99/33/6.
The linked validation manifest records full final-code regression and package
acceptance. Clinical validity, native RNG/Iso floating identity and original
GUI/template layout are separate claims. All notices remain unchanged.

Continue mTPI72 immediately. Native XLL decision instructions now match2496
archivedR cases, and495 original cached monitoring cells match Python. A draft
complete replay agrees with source-controlled trial tapes. XLL final offsets
use zero-based index*1e-10 whereas archivedR uses one-based indices; exact ties
also differ. These must not be conflated. Current research lives in
`/tmp/mtpi-next-contracts.md`, with paths to native probes, fixtures and drafts.
The original archive is in ignored `research/raw/mtpi-2026-10-12/`. Integrate
bounded saved inputs, safe/legacy controllers, native final profile and
scenario/report workflow before any72 promotion. No full mTPI acceptance yet.

## Current mTPI workflow checkpoint

Current mTPI72 version2.1 monitoring/simulation/report scope is complete under
explicit paper/legacy profiles; counts100 implemented/32 partial/6 pending.
See mtpi-desktop-complete-checkpoint-2026-10-12.md and its validation manifest.
All original assets remain ignored. Six regenerated fixtures include native
controller/rank/RNG evidence and all20,000 cached trials. Original-TPI tuning
and prior-specific loss recalibration remain separate method-extension limits.
Continue immediately with recovered Dose Schedule Finder75 and preserve the
separate web-app contracts128/129; no arbitrary stopping count.
Six unavailable exact fits, mixed-license distribution decision and Forbidden PR
creation remain separate blockers. No merge/release/deployment/outreach occurred.

## Dose Schedule Finder75 current-workflow acceptance

Current2.2.0 guide-defined scope is closed; counts101 implemented/31 partial/6 pending.
See [acceptance](dose-schedule-desktop-complete-checkpoint-2026-10-12.md) and its
validation manifest. Continue source-recoverable UAROET92 and PRT69 immediately: both
original archives are recovered. UAROET current-guide efficacy futility, calibration,
partial outcomes, cohorts and saved workflows require explicit integration beyond
the paper API. Preserve the separate web-app128/129 and six unavailable-fit blockers.
Mixed-license distribution and Forbidden PR creation remain parent decisions.

UAROET92 current1.8 contracts now have eight portable native reference files.
Toxicity-only posterior inference is integrated and independently checked; status
remains partial, counts101/31/6 unchanged. See uaroet-current-recovery-2026-10-12.md
and the pending-data validation manifest. Next integrate saved-study/prior mapping,
then the complete calibrated-prior, precision, conduct/calendar and report workflow.
Native live-cohort defects and distinct tie rules must remain explicit profiles.

## UAROET92 current-workflow acceptance

Declared current1.8 Python scope is closed; counts102 implemented/30 partial/6 pending.
See [acceptance](uaroet-desktop-complete-checkpoint-2026-10-12.md) and its validation
manifest:70,589 full tests,9,347 installed-wheel tests,15 regenerated native files.
Continue PRT69 through saved study, explicit native profiles, controller/calendar,
simulation and reports before promotion. Native projection/hazard/calendar defects
must remain explicit. Six unavailable fits, separate web contracts, mixed-license
distribution and Forbidden PR creation remain unchanged blockers.

## PRT handoff after UAROET acceptance

UAROET acceptance is preserved at3a6fcd64caf8e570e13317bc698316d69f8e1850.
PRT source recovery now adds64 as-of boundary cases, both native final-clock
paths on3 histories, and90 suspension-arrival/cohort bookkeeping cases. Both
new files regenerate byte-identically;333 combined native projection/hazard/
count/arrival/calendar tests pass. No production Python changed after the
70,589-test acceptance run; PRT69 remains partial.

Immediate remaining contracts: recover statistical suspension and final-choice
controller branches, including finite800-candidate exhaustion; then integrate
versioned original conduct/simulation inputs, explicit projection and hazard
compatibility profiles, complete conduct/calendar/batch/report workflows. Native
max-min projection can exceed1, hazard inversion reuses a variate and shifts
rates, exact-window observations are misclassified, and final time uses only
the last three patients. Do not make these silent scientific defaults. Existing
paper APIs remain distinct. Native posterior pilots agree with independent
quadrature; large-prior pilot MCSE remains inadequate for a precision claim.

Original executables are ignored under research/raw/next-desktop-2026-10-12/PRT_V1.0/PRT.
Source probes pin their hashes. Portable tools and fixtures are committed; scratch
disassembly is /tmp/prt-sims-disassembly.txt and /tmp/prt-main.txt. No task URL
was exposed by this execution environment. PR creation remains Forbidden.

## PRT saved-study and controller recovery acceptance

PRT saved conduct/simulation inputs now round-trip, including both original
bundled studies and88 native count cases. Full production regression passed
71,225 tests; installed wheel passed2,090 PRT and9,347 broader acceptance tests.
Fifteen native files regenerate byte-identically; four R references agree within
8.4e-15. See prt-study-validation-2026-10-12.json. Catalog remains102/30/6.

The source audit now covers final selection, acceptable and negligible conduct
branches, predictive enumeration and endpoint jitter, raw pending-risk means,
simulation pre-screening and late allocation. Native defects include complemented
predictive cutoffs, conduct's hard-coded dose10, unnormalized predictive sums up
to4, ineligible final fallback, exact-window conversion and last-three final time.
These must remain explicit legacy diagnostics, never silent scientific defaults.

Next integrate recovered hazard calibration/inversion and arrivals. A checked draft
is at /tmp/prt_hazard_draft.py:24 native models/288 delays and4800 arrival values
agree; corrected inverse grids achieve intended event probabilities. It is NOT
yet package code. Finish validity-bounded projection/predictive profiles, saved
conduct, complete calendar/batch/reports and precision acceptance before promotion.
Native finite800-arrival exhaustion must be bounded explicitly. Original simulation
window is an integer native field; tighten original-text import accordingly while
keeping any declared continuous Python extension explicit.

Parallel parent research recovered BLESS/BayesianSurvival supplements. Structured
artifacts await executor transfer; see bless-published-recovery-2026-10-12.md and
the updated pending-model source notes. Published-model implementation may proceed
with explicit precision/ambiguity limits once verified. No outreach is authorized.

BLESS transfer update: the three supplied Library records resolved, but the current
supported materialization helper failed to download each file on both the initial
attempt and one bounded fresh-preparation retry. No bytes arrived in this executor.
The parent was informed in progress commentary. No guessed URLs or unsupported
storage downloads were used. Continue independent PRT work while transfer is
resolved; do not claim that the BLESS extraction artifact was locally inspected.

PRT hazard/arrival primitives are integrated after the71,225-test checkpoint:
24 native calibrations and288 event transforms, independent corrected-hazard
inverse grids and4800 native arrival values pass focused checks. Original-text
window validation now enforces the native integer field. This incremental code
has focused verification; another final-code full acceptance is required at the
next major checkpoint. Full desktop controller/calendar/report integration is open.

Library fallback update: complete authenticated Library reads returned all three
artifacts without truncation. Their text omitted exactly one terminal newline;
restoring it matched each authoritative byte count. Local readable files and
Library identity/version xattrs are now present under
/workspace/recovered-models-2026-10-12. This is documented text reconstruction,
not a successful binary download. BLESS JSON SHA256 is
00e01b5a6f522ac02e3e9c9f8fc7c28050192ba1ab1ae58d8b029c8a7b59a855.
Proceed to the parent's requested published-grid BLESS integration after preserving
this PRT increment; do not claim original deployed full-precision parity.

BLESS published-table scope is integrated: all four rounded models, strict exact
seven-day grid, saved inputs/reopen, audited JSON and curve CSV, source-choice
warnings. All499 focused BLESS tests pass against458 independent60-digit Decimal
references; combined current PRT hazard/study and BLESS tests683 pass. Catalog149
is partial, counts102/31/5; unknown daily/full-precision/uncertainty contracts are
not claimed. Full final-code acceptance is now being run.

Parent-authored ComPAS PNOC022 factual specification v1 is reconstructed from
complete Library text (not an original protocol download):26673 bytes, SHA256
b7a5cef6c4a0a8114aa1abd7ba00ca1f96c98dafbca297a123401b0bd1cfe894.
Preserved at research/compas-pnoc022-specification-2026-10-12.json. First author is
Rui Tang. Published core is actionable with explicit group/model weights and
independently validated integration; original PDF is not redistributed.

BLESS final acceptance completed:73,155 full tests passed (985.48s),765 unchanged
production Python files,683 focused and9,347 broader installed-wheel checks,
683 fresh-environment checks plus28 real plotting/TIFF tests, example execution,
15 byte-identical PRT native references and458 independent Decimal BLESS cases.
Ruff/format/mypy/build and retained-notice/raw-asset audits pass. See
bless-published-validation-2026-10-12.json. Implementation b1722d6f524d1ee87410499087f336763022a224
was independently verified at the remote feature branch. ComPAS development is
still isolated in /tmp at this checkpoint and is not included in these claims.

BLESS acceptance is preserved at2247a3225d9711bd5ad730b43557b1d3ce627243, verified
at the remote feature branch. ComPAS explicit published core is now integrated:
proper normalized five-model inference, caller-supplied groups/model priors,
variance calibration plus literal PNOC tuples, saved input/reopen, strict
selected-model decisions, uncertainty holds and reports. All72 focused tests pass
(29.10s). Two-arm independent marginal-prior references agree within5e-13; separate
three-arm Gaussian quadrature evidence converges within7e-12. Catalog140 partial,
counts102/32/4. Dynamic platform/default contracts remain in docs/compas.md.
Full final-code acceptance is running; do not confuse it with the completed
BLESS-only checkpoint. Next recover/integrate PRT controller/calendar/reports.

ComPAS acceptance completed:73,227 full tests (986.11s),767 unchanged production
Python files,755 fresh-wheel focused checks,9,347 broader wheel tests,28 fresh
plotting/TIFF tests and both examples. Independent two-/three-arm references
regenerated byte-identically; lint/types/build/provenance checks pass. Exact
record: compas-published-validation-2026-10-12.json. Implementation commit
ae43f68a80b45fc3004e878bd7520e316e8e2b0e was verified at the remote branch.

PRT work remains isolated outside package code at this acceptance checkpoint.
New native evidence establishes strictly-negative-only moment fallback (zero,
NaN and infinity remain invalid), full conduct branch states including untried
next doses, and the missing decision-text dispatch for code6.0. A draft legacy
module passes2,840 native/reference checks. The bundled guide posterior pilot
has well-mixed chains but the paper projection reaches-0.010342, while the native
lower-floor transform stays in[0,1] in that run; no silent clipping or source/paper
profile substitution is justified. Integrate these explicit diagnostics, then
finish saved conduct, calendar/simulation and reports with qualified precision.

PRT explicit legacy arithmetic is now integrated,2,867 focused tests pass. New
72moment/1,368controller/17dispatch fixtures preserve native defects with validity
flags; paper APIs remain unchanged. Full final-code acceptance is starting.
Entry69 remains partial; saved conduct and native calendar/batch/report are next.

Native arithmetic implementation fe732815d22b4aa1b380a1a9fbe4bab1abdddd0e is verified
at the remote branch. Eighteen references regenerated byte-identically; fresh
wheel5,027 PRT tests,9,347 broader regressions and28 extras pass. The first full
suite was deliberately interrupted before completion to include the saved-conduct
workflow in the same major checkpoint; it is NOT a completed acceptance claim.
Saved conduct is now integrated;2,884 focused tests pass. Final combined full
suite is running against769 frozen production Python files. Remaining PRT work:
generated native calendar, simulation batch and complete OC reports/acceptance.

During final conduct acceptance, original-executable enrollment/calendar evidence
was extended to19 portable fixtures. New18 paths/72 steps establish analysis at
the next candidate arrival, suspension skipping and resumption indexing, final
cohort shrinkage, and x87 boundary rounding (arrival7+delay.1 versus stored7.1).
Independent Fraction/integer checks match this grid; exact-sum arithmetic is not
claimed as a universal x87 replacement. No production code changed during the
combined conduct full suite. See tools/reference_prt_enrollment.py and
check_prt_enrollment.py. This enables the next generated-simulation integration.

Critical reference correction during acceptance: actual CRT startup selects53-bit
x87 precision (control word0x23f), not the earlier probes' forced64-bit0x37f.
Both executables' actual initializer was executed for three initial precisions.
Twenty reference generators are corrected; eight fixture files change numeric
values. All5,044 PRT tests independently pass from the installed wheel using
corrected fixtures; no production change is needed. Arrival7/delay.1 is observed
at stored7.1 under actual startup precision. Preserve the earlier37f evidence in
git history as an alternate arithmetic diagnostic only. New111 precision cases
and six runtime-initializer probes qualify this correction. Full769-file source
suite continues unchanged; corrected fixture regression is separate evidence.

Parent review follow-up executed. ComPAS counterexample not reproduced: log
 evidence-.5/.999 differs2.1e-14; independent six-dimensional marginal-prior
reference agrees1.4e-12 and five thresholds pass. No ComPAS production patch.
BLESS huge-input cancellation confirmed and fixed with400-digit evaluation of
printed decimal coefficients/exact normalized inputs;15 new regressions pass.
Combined5,635 focused tests pass51.89s. Earlier combined full run interrupted at
66,529 passes due this production fix; do not call it accepted. Fresh final full
suite is running with769 source files and all corrected PRT fixtures frozen:
/tmp/review-final-full-suite.log and /tmp/review-final-tested-python.json.
See review-followup-2026-10-12.md. No catalog promotion. A simulation-analysis
refactor draft remains isolated under /tmp/prt-simulation-draft and is not accepted.
