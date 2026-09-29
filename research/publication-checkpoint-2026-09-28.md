# Community publication checkpoint — 2026-09-28

Latest verified package checkpoint: `c44eddd` adds BOP2-DC finite-grid
calibration for randomized binary, Normal and exponential-survival outcomes.
Local `master` contains this validated checkpoint. Fresh read-only
checks still show GitHub `master` and `main` at `45b6e307`; their earlier
changes are already merged locally. The newer statistical additions have not
been confirmed published. See the final section for current package checks.

The user explicitly requested that all completed work be pushed to GitHub and
that stable programs be available on `master`. The public repository is
`dx-li/mdanderson-stats`; its existing default branch is `main`. Preserve that
branch and publish the same validated checkpoint to `master`, without rewriting
remote history. Incomplete source-catalog entries retain their `partial` or
`pending` status; a validated component does not imply complete native software
coverage.

## Earlier remote state and publishing restriction

The read-only GitHub API showed `main` at
`ddf440f76615bdba7f6776e0feec6b8f2b9bb1f7`, last pushed September 12.
The repository is public. No `master` branch or open pull request was present.
At the start of this checkpoint the integrated local branch was 272 commits
ahead of that remote commit, with additional reviewed work arriving from Luna.

After the renewed explicit authorization, a normal push of the development
branch failed because the shell could not resolve `github.com`. Creating that
development branch through the available GitHub connector was rejected with:

> MCP tool call requires approval, but approval policy is never

No remote publication succeeded. No further write attempt, force push, approval
override or alternative publication transport was attempted after that rejection.
User authorization is present; usable publishing permissions are still required.

Once publishing is available, an ordinary fast-forward push can publish the local
checkpoint to both branches and preserve the development branch:

```sh
git push --atomic origin master:refs/heads/master master:refs/heads/main feat/condis-svm:refs/heads/feat/condis-svm
```

If a remote branch has advanced, inspect and reconcile it first; do not force it.
Verify remote branch SHAs after a successful push. Merely preparing local
branches or artifacts is not evidence that GitHub was updated.

## Community documentation and licensing

The software index lists all 138 source entries: 63 implemented, 66 partial,
9 pending. Ninety-five guide links were checked. Entry counts do not estimate
remaining engineering work. README installation instructions cover a local
checkout with Python 3.12 or newer and the optional plotting extra.

`LICENSE.md` grants MIT terms only for original contributions and explicitly
preserves upstream terms for adapted material. Packaging metadata identifies a
mixed-license distribution, including retained commercial-use restrictions;
it does not claim that every component is MIT licensed. Third-party notices
remain included. Removing restrictive legacy adaptations or obtaining different
upstream terms remains necessary before describing the entire distribution as
unrestricted open-source software.

## Validation scope

Existing independent native/R references and focused numerical checks remain
the method-level evidence. This checkpoint integrates the completed TOP
two-endpoint calibration and STPLAN exact inverse-significance workflows.
Review repaired tiny Poisson power bracketing, product underflow handling and
the feasible one-event rejection region. Six focused STPLAN tests and both
guide examples pass. No new broad CI workflow or large simulation is introduced.

The repository-wide type check identified 33 errors in five previously existing
files. A separate small correction passed 26 focused tests, and the integrated
type check now passes across all 511 source modules. Root retained BOP2's original
validation order and checked both count-bound and trial-size failures without
warnings. All public exports are present, without duplicate export names.
The full numerical test suite has not been rerun for this checkpoint,
consistent with the user's request to emphasize useful coverage and keep local
memory use bounded.

## Verified package and local checkpoint

At code revision `0cd36b271267fcf53e41436ab8a0e3487e5a2c97`, repository-wide
Ruff checks pass, all 1,253 checked Python files are formatted, and mypy passes
all 511 source modules. The final TOP integration adds ten passing focused
endpoint/calendar/calibration tests, an exact holdout replay and a runtime
scan-budget exhaustion check distinct from preflight bounds.

Wheel and source-distribution builds passed using cached Hatchling 1.32.4,
without installing dependencies. An isolated interpreter imported the wheel
and checked all 1,400 public exports, seven examples across four guides, all
138 catalog entries, and byte equality for all 513 packaged source/data files
against committed Git contents in both artifacts. The check also verified
preserved licenses/notices and exclusion of ignored native sources and binaries.
It took 10.795 seconds, peaked at 109.17 MiB resident memory and reported zero
swaps. The packaged License field explicitly says mixed license and preserves
the commercial-use qualifications.

This audit-only checkpoint does not change the verified numerical code. Local
`master` is prepared from it, while development continues on feature branches.
The deliverables are the wheel, source distribution and portable all-branches
Git bundle under ignored `dist/`. The bundle captures committed history and
branch references at creation; it does not include ignored native downloads or
uncommitted work in an active implementation checkout. Local artifacts and the
local `master` branch do not resolve the remote publishing restriction above.

## Subsequent UAROET / EasyCellType checkpoint

During the same day's continuation, the remote state changed. A fresh read-only
GitHub API check verified **both `master` and `main` at `e20e582`**, and local
remote-tracking reflogs record that checkpoint as pushed. The earlier restriction
above describes the failed initial attempt; it is no longer evidence that the
previous checkpoint is absent from GitHub. This continuation did not observe
which external action completed that publication.

The next validated code checkpoint is `2d54cff`. It adds complete-outcome UAROET
trial simulation and the observed-score EasyCellType GSEA core, with public APIs,
guides, independent R references and retained upstream license notices. Their
catalog entries remain partial for the explicitly documented remaining workflows.
Counts stay 63 implemented, 66 partial and 9 pending.

The two new modules pass targeted mypy. Focused worker checks and root numerical
reference comparisons are recorded in the individual method audits. Cached
wheel/source builds pass without new dependencies. An isolated wheel check
verifies all 1,409 exports, all 515 packaged source/data files against committed
Git bytes, included licenses/notices, and three examples across the two new
guides. It took 11.15 seconds, peaked at 108.48 MiB and reported zero swaps.
The full numerical suite was not rerun; no new CI pipeline was added.

These validated additions are ready for a fast-forward publication. A push still
needs its own success result and fresh remote-SHA verification; the observed
publication of the earlier checkpoint alone does not establish that the new
additions have reached GitHub.

The subsequent normal atomic push of local `master` to remote `master`/`main`
and the development branch failed with `Could not resolve host: github.com`.
The new additions therefore remain locally prepared, while the earlier
`e20e582` remote checkpoint is verified. The unavailable connector-approval
route was not retried or used to bypass the failed connection.

## PLBARPO allocation checkpoint

The active-arm allocation API, immutable summaries, public example and catalog
coverage are integrated at `47bb9c4`. Eight independent base-R scenarios verify
all four randomization methods, posterior competition after closure, floors,
single active arms and BARN2N's use of all historical enrollment. A separate
read-only review found no material issue. Two focused tests, targeted mypy,
Ruff lint and formatting, and the public example pass.

At `a80f901`, cached wheel and source builds passed. The isolated wheel check
verified all 1,411 public exports, byte equality of all 516 packaged source/data
files against committed Git contents, licenses/notices and the new allocation
example. It took 12.389 seconds, peaked at 104.81 MiB and reported no swaps.
Earlier method examples were not redundantly rerun. Source-catalog counts
remain 63 implemented, 66 partial and 9 pending.

The full no-control PLBARPO platform simulator and full EasyCellType multilevel
inference remain in separate Luna implementation checkouts. Native cumulative
and adaptive-splitting references were added for the latter; review identified
substantive draft corrections before integration. These draft backends are not
included in the validated local `master` checkpoint. Development remains
limited to two implementation workers, one read-only reviewer and single-thread
numerical work, without a new broad CI workflow.

## PLBARPO trial/simulation and full GSEA checkpoint

The no-control PLBARPO trial runner is public at `18eeb27`; serial aggregate
simulation follows at `255c974`. Independent base-R trial references match 14
patient rows, 12 monitoring comparisons and nine terminal arm rows. Recorded
seed replays reconstruct aggregate counts, false declarations, means and
Monte Carlo standard errors. Trial and simulation checks, their public
examples and targeted type/lint checks pass. Control/concurrent-control trial
scheduling is still a separate Luna implementation and is excluded here.

EasyCellType multilevel inference is public at `33f559f`, including signed
normalization, adaptive tail probabilities, uncertainty diagnostics, BH and
the source cutoff. All 66 native cumulative scores, 66 single-set R scores
and 22 splitter statistic pairs match. A small twelve-seed comparison stays
within three combined standard errors of the native splitter; its extreme
tail estimates remain noisy. Five focused worker checks, targeted mypy and
both public GSEA examples pass. The method audit records numerical corrections,
source versions and explicit memory/work bounds. GSEA label processing remains
in its own Luna checkout.

At `9228bc9`, repository-wide Ruff lint passes and all 1,619 files pass format
checking. No full numerical test suite or new large simulation was run.
Cached wheel and source builds pass. An isolated wheel check verifies all
1,420 public exports, all 519 packaged source/data files against committed
Git bytes, licenses/notices, all 138 catalog entries and four examples across
the two changed guides. It took 10.347 seconds, peaked at 103.08 MiB and
reported no swaps. Catalog counts remain 63 implemented, 66 partial, 9 pending.

Fresh read-only GitHub checks verify both `master` and `main` at
`45b6e307a63a4d1c3ea2b083ba9a805e8a76297b`. Those published changes limit CI
execution and format documentation; they do not publish the later statistical
additions. The commits were already present in the shared local Git object
store, so they were merged with the statistical branch, preserving remote
history. A fresh ordinary fetch after discovering the remote update failed
with `Could not resolve host: github.com`. The connector write route still
requires approval unavailable in this session and was not retried. Local
`master`, refreshed packages and the Git bundle are prepared for publication;
none is evidence that this newer checkpoint has reached GitHub.

## Persistent-control trials and GSEA labels checkpoint

GSEA hard/soft label processing is public at `3e7fd65`. Source-backed checks
cover minimum/fifth-position ties, hard-before-soft ordering, missing inference
and malformed reported evidence. The three public GSEA examples pass. The
complete persistent-control replay is public at `772c163`, with actual
comparison windows and independent R evidence for four paths. Seven focused
controller checks and targeted lint/type checks pass; the cap-look conflict
fix also protects the no-control controller. No broad numerical suite or new
CI workflow was added.

Cached builds at `5d6e4c1` pass. The isolated wheel check verifies all 1,425
public exports, byte equality of all 521 packaged source/data files against
Git, licenses/notices, the 138-entry catalog and four examples in the two
changed guides. It takes 10.873 seconds, peaks at 107.41 MiB and reports no
swaps. Catalog counts remain 63 implemented, 66 partial and 9 pending.

Local `master` was fast-forwarded to this verified code. Fresh read-only GitHub
checks still show both branches at `45b6e307`; that commit is an ancestor of
local `master`, with 39 newer local commits at the package checkpoint. The
previous DNS failure and unavailable connector approval remain the publication
barriers; no rejected transport was retried. Control-trial aggregate simulation
and Dose Schedule Finder calendar replay are still in separate Luna checkouts
and are excluded from this validated checkpoint until integration.

## Control OCs, calendar trials and GAO model checkpoint

PLBARPO persistent-control aggregate simulation is public at `1306e9d`. It
reports all-ledger and conditional-on-entry probabilities and Monte Carlo
errors, sample-size/response summaries, error rates with explicit null labels,
and replayable per-trial seeds. It prepares the design once and discards patient
histories between trials. Focused checks and independent control-trial R
references pass; its two public examples pass.

Dose Schedule Finder calendar trials are public at `81ad9cf`, including
as-of-arrival fits, event generation and full follow-up. Thirty-three R event
references and an analytic tiny-probability quantile pass. Review corrected
event inversion and large-calendar-origin timing precision; five focused
calendar checks pass. Multc pending-outcome replay is public at `207c3cc`.
Five independent R scenarios match 18 patient and 12 look records, including
suspension/resumption and decisions made before final outcomes. Review corrected
cap-bound metadata and rejected positive time increments that round away.
The three final focused calendar checks pass.

U2OET GAO probability and likelihood evaluation is public at `02efdac`.
Independent R equations agree on 440 cells and 15 grouped likelihoods, with
maximum absolute probability error `9.99e-16`; three focused checks pass.
The model uses raw doses and a Gaussian copula, with explicit coefficients.
GAO fitting/native prior interpretation remains pending. Targeted lint,
formatting and type checks passed for each added component. No new CI workflow
or broad numerical suite was added.

Cached wheel and source builds at `02efdac` pass. The isolated wheel check
verifies all 1,436 exports, all 525 packaged source/data files against committed
Git bytes, licenses/notices, the 138-entry catalog and five examples in the
three changed guides. It took 14.952 seconds, peaked at 111.20 MiB and reported
no swaps. Source-catalog counts remain 63 implemented, 66 partial, 9 pending;
these are workflow labels, not a percentage of remaining statistical work.

Local `master` was fast-forwarded to this validated code. Fresh read-only GitHub
checks still verify both `master` and `main` at `45b6e307`. The shell DNS failure
and unavailable connector approval remain the publishing barriers; no rejected
write route was retried. Newer local commits, packages and the verified Git
bundle are prepared for publication, but have not been confirmed on GitHub.
The next BARD/BOIN12 source investigations are separate from this validated
checkpoint. Implementation remains limited to two Luna workers and at most one
read-only reviewer, with one bounded numerical process at a time.

## BOIN12 tradeoff and BARD BF-BLRM checkpoint

BOIN12's tradeoff utility mapping is public at `f592a4b`. It preserves the
exact affine relationship to `pi_E - w*pi_T` for arbitrary joint endpoint
probabilities and feeds the existing posterior, decision, final-selection and
simulation APIs. Seventeen focused BOIN12/reference checks pass; a four-trial
comparison of mapped and explicitly supplied utilities replays identically.
The cached app marks multilevel endpoints under development; that is now
distinguished from established native features awaiting a port. Two-stage and
3+3 run-in precedence still need a verified contract.

BARD BF-BLRM fitting is public at `9cddc04`, and its decision helpers at
`e3e907f`. The raw-dose-ratio model matches all 20 independent R probabilities;
22 integrated posterior references agree within two estimated Monte Carlo
standard errors. Maximum split R-hat is below 1.002. Twelve hand-calculated
decision snapshots match exactly, including cutoff equality, unsafe downward
intermediate steps, response before DLT assessment and final minimum-treated
eligibility. Three focused fitter checks and five decision checks pass.
Targeted lint, formatting and module type checks passed. No full numerical
suite, new CI workflow or large simulation was introduced.

Cached wheel and source builds at `e3e907f` pass. The isolated wheel check
verifies all 1,447 public exports, all 527 packaged source/data files against
committed Git bytes, retained licenses/notices, the 138-entry catalog and four
new examples across three changed guides. It took 10.934 seconds, peaked at
113.94 MiB and reported no swaps. Previously checked BOIN12 simulation examples
were not repeated. Catalog counts remain 63 implemented, 66 partial, 9 pending.
These additions close numerical components within partial workflows; they do
not establish complete BOIN12 or BARD native feature coverage.

Local `master` is fast-forwarded to the validated code plus this audit. Fresh
read-only GitHub checks still show both `master` and `main` at `45b6e307`.
The validated code has 61 newer commits than that published point. No rejected
write route was retried: shell GitHub DNS and unavailable connector approval
remain the recorded barriers. Packages and an updated verified all-refs bundle
preserve the local checkpoint, but they do not publish it to the community.

## Stratified interval survival and BARD calendar checkpoint

Shared-coefficient stratified interval-PH fitting is public at `2e39d63`.
It uses independent Turnbull supports and baselines, within-stratum centering
and joint likelihood optimization. Independent R current-status regression
agrees on coefficients and likelihood; ten survival predictions differ by at
most `3.30e-8`. Mixed exact/interval/right-censored data pass direct likelihood,
finite-difference shared-score and KKT checks. Large covariate offsets,
time/covariate scaling and weight scaling preserve the checked results.
Four focused worker tests pass; the independent audit peaks at 116.53 MiB
with zero reported swaps. Native `mets` response-contract parity is not claimed.

BARD BF-BLRM stage-one calendar replay is public at `bc3b2a5`. Independent
fixed-prior R/hand-ledger examples match 16 patient records and all as-of counts,
including early DLTs, late responses, pending backfill beyond the evaluable
cap and partial final cohorts. Fit reuse requires exactly 12, 5 and 2 fits
for the three paths. Four focused worker checks pass. A separate free-prior
replay agrees exactly with four sequential actual fitter calls. Explicit
boundary policies, sticky all-overdose findings and complete follow-up are
documented. Root calendar checks peak at 113.94 MiB with zero reported swaps.

Cached wheel and source builds at `bc3b2a5` pass. The isolated wheel check
verifies all 1,456 public exports, all 529 packaged source/data files against
committed Git bytes, retained licenses/notices, the 138-entry catalog and
three examples across the two new guides. It takes 10.812 seconds, peaks at
116.16 MiB and reports no swaps. Targeted lint/format checks pass; no broad
numerical suite or new CI workflow was added. Catalog counts remain
63 implemented, 66 partial and 9 pending, reflecting broader native workflows.

Local `master` is fast-forwarded to the verified code plus this audit. Fresh
read-only GitHub checks still show `master` and `main` at `45b6e307`; the
verified code has 68 newer local commits. The earlier shell DNS failure and
connector approval rejection still prevent publication. No rejected write
transport was retried. Refreshed packages and the verified all-refs bundle
preserve this checkpoint locally. The next PRT calendar investigation remains
in an isolated Luna checkout and is excluded until implemented and validated.


## CiBolus simulation and PRT calendar checkpoint

CiBolus now generates joint response-category/toxicity observations and runs
complete-outcome cohort trials. Independent base-R quadrature agrees on 36
joint-category rows to `5.27e-16`; six patient records and 18 regimen/look rows
match, with expected utilities within `2.84e-14`. Final selection at an untried
regimen and permanent first-cohort stopping are covered. A free-prior path
reproduces two sequential actual fitter calls exactly. Three focused trial
tests pass; the root reference run takes 0.947 seconds after imports, at
117.50 MiB and zero reported swaps.

CiBolus aggregate simulation adds selection/no-selection and stop rates,
mean enrollment/allocation and pooled observed outcome rates with Monte Carlo
errors. Per-trial seeds permit exact replay. Three focused checks pass in
1.64 seconds, including count conservation, pooled ratio errors and shared
budget rejection before RNG use. Diagnostics retain infinite Rhat rather than
silently discarding it. Full posterior histories are discarded between fits;
resource limits apply cumulatively across the aggregate run.

PRT explicit-input replay integrates the actual posterior/projection with
cohort enrollment, interval follow-up, suspension, queued/declined arrivals
and final selection. All 40 independent R ledger rows agree, including
internal and final endpoint cases. Eleven focused checks pass in 2.16 seconds;
the integrated root ledger/timing check passes in 0.022 seconds after imports,
with 119.23 MiB peak RSS and zero swaps. Review corrected tape-exhaustion
metadata, elapsed-duration rounding and a pre-existing predictive mass-roundoff
failure. Invalid covariance projections still raise. Per-analysis work is
reserved before calculation, with shared likelihood and work caps.

Wheel and source builds at `90537385c0bdb207e117ac79a56db9524bade589` pass.
The wheel verification checks all 1,465 public exports and exact committed
bytes for 532 source/data files, retained licenses/notices, all 138 catalog
entries, and three examples in the new/expanded PRT and CiBolus guides.
It takes 11.586 seconds, peaks at 122.81 MiB and reports zero swaps. Targeted
lint, formatting and type checks pass. The full numerical suite was not rerun;
no CI workflow or large simulation was added. Catalog counts remain
63 implemented / 66 partial / 9 pending: native source and workflow gaps are
still explicit, and no full-coverage claim is made.

Local `master` is fast-forwarded to this verified code plus the present audit.
Fresh read-only GitHub checks still show `master` and `main` at `45b6e307`;
the verified code contains 78 newer commits. Publication remains blocked by
the recorded shell DNS failure and connector approval rejection. The rejected
write route was not retried or bypassed. Updated local packages and a verified
all-refs bundle preserve the checkpoint; they do not make it available on
GitHub. The next OOB survival-forest implementation is delegated to Luna in
an isolated checkout and is excluded from this package until validated.

## Forest diagnostics and BARD continuation checkpoint

Numeric survival forests now retain optional OOB membership and report held-out
KM/NA curves, contributor counts, mortality and native-convention concordance
error. Sixteen unchanged native concordance cases and 264 independently
reconstructed native-kernel curve values agree exactly. Enabled OOB computation
preserves the fitted trees and random stream. Undefined rows remain explicit.
Memory and work are bounded without materializing pairwise comparison matrices.

Permutation importance shuffles each feature within each tree's OOB rows and
scores complete tree blocks. Independent native-kernel comparisons match every
baseline, perturbed, block and mean error across block sizes 7, 3 and 1, including
an omitted tail, an undefined block and a constant feature. The default
whole-forest block is explicitly distinguished from native block size 10;
anti-split importance and native RNG equivalence are not claimed. The portable
reference script rerun takes 1.139 seconds after imports at 117.61 MiB peak RSS
with zero swaps. Nine focused forest/OOB/importance tests and targeted lint,
format and type checks pass.

BARD continuation connects the completed BF-BLRM stage-one ledger to eligible
carryover, combined-history minimization and final OBD selection. The target
includes carryover, and only patients at the supplied dose pair enter the
combined history. Four independent base-R ledgers match eight assignments,
32 count rows, six posterior rows and four target/selection summaries, with
maximum posterior/utility difference `4.27e-14`. Four focused worker tests and
a separate source review pass. Titration, expansion and native per-arm quotas
remain documented gaps; no timing law or hidden native settings are invented.

Wheel and source builds at `9bfefbc78a8dd5f9cc5f406ee376a2e4bfe4d0ce` pass.
The isolated wheel check verifies all 1,471 public exports, exact committed
bytes for 534 source/data files, retained licensing/notices, all 138 catalog
entries, and three executable examples across the OOB/importance and BARD
continuation guides. It takes 13.957 seconds, peaks at 123.67 MiB and reports
zero swaps. No broad numerical suite, new CI workflow or large simulation was
run. Catalog status remains 63 implemented / 66 partial / 9 pending.

Local `master` is fast-forwarded to this verified code plus the present audit.
Fresh read-only GitHub checks still show `master` and `main` at `45b6e307`;
the verified code contains 86 newer commits. Publication remains blocked by
the recorded shell DNS failure and connector approval rejection. The rejected
write route was not retried or bypassed. Refreshed packages and a verified
all-refs bundle preserve this checkpoint locally. Accelerated BARD titration
is being developed in Luna's separate checkout and remains excluded from
`master` and the package until reviewed and validated.


## Titration, GAO fitting and dose-schedule simulation checkpoint

BARD accelerated titration is integrated and documented at `38f904e`.
It supports single-patient escalation, grade-two/DLT triggers, same-dose
cohort top-up and configurable dose caps before the existing BF-BLRM replay.
Source review and hand-ledger checks corrected grade-two-only singleton
transitions and extended the explicit POD boundary policy to titration roles.
Nine focused tests pass; the titration, ordinary calendar and stage-two public
examples pass in 1.492 seconds including imports, with 114.73 MiB peak RSS
and zero swaps. Combining the source titration convention with BF-BLRM is
explicitly described as a Python extension. Expansion and native timing/quota
and calibration gaps remain documented.

U2OET GAO posterior fitting is integrated at `5f14ce8`, with public exports,
examples and reference tooling at `44804a1`. It uses explicitly supplied
independent normal priors on raw coefficients, log link/interaction parameters
and Fisher-z association, preserving chain/draw probabilities and diagnostics.
It does not guess native prior-file meanings. Independent base-R integration
checks two reduced posteriors, including partial toxicity observations: all
38 summaries agree within 1.028 estimated Monte Carlo errors, with maximum
split R-hat 1.00347. The comparison takes 22.530 seconds after imports, peaks
at 112.98 MiB and reports no swaps. Four focused fitter checks pass in 2.68
seconds. Native prior mapping and GAO calibration/calendar integration remain
open.

Dose Schedule Finder aggregate simulations are integrated at `d12d6c6` and
publicly exposed at `44804a1`. The serial driver reports selection/stopping,
allocation, enrollment, observed toxicity and duration with suitable Monte
Carlo errors, separate replayable event/sampler seeds and shared work budgets.
Three focused checks pass in 1.57 seconds, including exact single-trial replay,
clustered pooled-rate errors, RNG-preserving budget rejection and duration
invariance under time-unit scaling by `1e200`. Read-only review confirms the
aggregation and stable duration moments. Automatic calibration, delayed
low-grade toxicity classification and within-patient adaptation remain open.

Cached wheel and source builds at
`44804a1a21164385991acc039a5081c1947b5a7a` pass. The isolated wheel check verifies
all 1,476 public exports, exact committed bytes for 536 source/data files,
retained licenses/notices, the 138-entry catalog and three executable examples
in the new titration, GAO fitting and dose-schedule simulation guides. It takes
13.459 seconds, peaks at 108.06 MiB and reports zero swaps. Targeted Ruff lint,
formatting and worker type checks pass. No broad numerical suite, new CI
workflow or large simulation was added. Catalog labels remain 63 implemented,
66 partial and 9 pending; validated additions within partial programs do not
establish full native feature coverage.

Local `master` is fast-forwarded to the verified code plus this audit. Fresh
read-only GitHub checks still show both `master` and `main` at `45b6e307`;
the verified code contains 95 newer local commits. The recorded shell DNS
failure and connector approval rejection still prevent publication. No
rejected transport was retried or bypassed. Updated packages and a verified
all-refs Git bundle preserve the committed root and worker checkpoints locally;
ignored raw files are excluded. The next BayesChiSquare source triage is
read-only and introduces no unreviewed code into this checkpoint.


## Weibull posterior diagnostics and full-data bootstrap checkpoint

The fixed-shape Weibull posterior diagnostic is integrated at `f6c6f49`.
It adds exact Gamma-prior inference for the transformed rate and Johnson's
complete-data posterior CDF statistic. Centered rate and relative-time power
calculations preserve posterior variation even at extreme common offsets and
neighboring representable times. Independent base-R integration supplies seven
posterior cases and 76 scalar/CDF/diagnostic summaries, all within 1.671
estimated Monte Carlo standard errors. The reference comparison takes 0.047
seconds after imports, peaks at 126.47 MiB and reports no swaps. Six focused
checks pass in 1.42 seconds. Unknown Weibull shape, other family fitting and
censored/rounded-data diagnostic contracts remain explicit gaps.

Proportional Density's full-data Delta_n bootstrap is integrated at `448c9f0`.
It separately resamples observed failures and censor records with fixed
within-arm status counts, refits both curves and computes the exact unit-weight
step integral at the original endpoint. Reproducible index tapes, separate
work/storage bounds and unresolved-fit calibration bounds are supported.
Independent R GLM/KM fits match nine statistics and 104 curve rows across
separate/pooled censoring, changed censor records and failed resamples. Maximum
area and curve discrepancies are `4.30e-14` and `7.46e-14`. The comparison
takes 0.015 seconds after imports, peaks at 118.22 MiB and reports zero swaps.
Six existing/new focused PropDen checks pass in 1.53 seconds. Unequal-censoring
treatment-effect null calibration and parameter uncertainty remain separate
workflows, not consequences of this goodness-of-fit bootstrap.

Public exports, guides and independent reference tooling are committed at
`71db42e3b64b839df3056c5e3066b1d720fd4712`. Cached wheel and source builds pass.
The isolated wheel verification checks all 1,481 public exports, exact
committed bytes for 538 source/data files, licenses/notices, all 138 catalog
entries and the two new executable guide examples. It takes 10.233 seconds,
peaks at 122.83 MiB and reports no swaps. Targeted Ruff, formatting and worker
type checks pass. No broad numerical suite, new CI workflow or dependency was
introduced. Catalog labels remain 63 implemented, 66 partial and 9 pending.
A separate source audit corrects BOP2 scope: its six advertised endpoint
families are covered; two-arm/joint survival is an extension, while native
optimizer/report equivalence remains open.

Local `master` is fast-forwarded to the validated code plus this audit. Fresh
read-only GitHub checks still show `master` and `main` at `45b6e307`, with 100
newer commits at the verified code checkpoint. The recorded shell DNS failure
and connector approval rejection remain publication barriers. No rejected
transport was retried or bypassed. Packages and a refreshed, verified all-refs
bundle preserve committed root and Luna work locally; ignored raw files are
excluded. BOP2-DC survival workflow triage remains read-only at this checkpoint.

## Survival trial and lognormal diagnostic checkpoint

BOP2-DC survival calendar replay and simulation are integrated at `25f688c`.
They use first-stop decisions, as-of censoring, relative risk times and explicit
fixed/Poisson arrival schedules. Outcome probabilities, Monte Carlo errors,
compact trial summaries and replay seeds are returned. Eight independent
base-R histories match 18 looks, including no events, all terminal outcomes,
boundary events, calendar shifts and extreme time-unit changes. An analytic
one-patient OC case matches 16,000 simulations within 1.447 estimated Monte
Carlo errors. The reference check takes 0.0141 seconds after imports, peaks at
124.31 MiB and reports no swaps. Four focused new checks plus the existing
survival reference pass. Calibration remains a separate open feature.

The lognormal Bayesian diagnostic is integrated at `f9704d5`. It jointly fits
log-location and log-variance with an explicit proper Normal-Inverse-Gamma
prior. Centered coordinates and separate posterior weights preserve time-unit
invariance and tiny, material prior contributions. Six independent base-R
cases verify posterior parameters, predictive CDFs and integrated joint
Johnson diagnostics. All 88 summaries agree within 2.851 estimated Monte Carlo
errors. The final comparison takes 0.0682 seconds after imports, peaks at
123.59 MiB and reports zero swaps. Two focused checks cover conjugate/CDF
identities, extreme unit changes, weak-prior means and pre-RNG resource bounds.
Native prior defaults, censoring and rounded observations remain open.

Public interfaces, guides and reference tooling are committed at
`7c68ec39dd5a25c75df2bb0f272460e5cab48432`. Cached wheel and source builds pass.
The isolated wheel check verifies all 1,487 public exports, exact committed
bytes for 540 source/data files, licenses/notices, all 138 catalog entries
and three executable examples across the two affected guides. It takes
16.643 seconds, peaks at 111.88 MiB and reports zero swaps. Targeted Ruff,
format and type checks pass. No broad numerical suite, additional CI workflow
or dependency was introduced. Coverage labels remain 63 implemented,
66 partial and 9 pending; these additions expand two partial programs.

Local `master` is fast-forwarded to the verified code plus this audit. Fresh
read-only GitHub checks still show `master` and `main` at `45b6e307`, with
104 newer commits at the verified code checkpoint. The recorded shell DNS
failure and connector approval rejection still block publication. No rejected
transport was retried or bypassed. Updated packages and a verified all-refs
bundle preserve committed root and Luna checkpoints locally. Ignored raw
files and the next in-progress survival calibration work are not included in
the validated package or local `master`.

## Survival calibration and unknown-shape Weibull checkpoint

BOP2-DC survival calibration is integrated at `7b8a48c`. It selects a design
on explicit four-parameter grids using shared paths and source CGR/futile-ESS
objectives, then reports a separate holdout without reselection. An independent
base-R replay verifies 18 candidates and 760 probability, sample-size and
Monte Carlo error summaries across both objectives. The two objectives select
different candidates. A passing 4% calibration false-go limit at `2/64` fails
on held-out data at `3/48`, and this remains visible as validation failure.
Two focused checks include a nondegenerate analytic event/censoring partition
and pre-RNG budget/input rejection. The R comparison takes 0.6084 seconds
after Python imports; summing separate Python/R peak resident memory gives a
conservative 203.91 MiB combined upper bound. No Python swaps were reported.

Unknown-shape Weibull fitting is integrated at `a1be193`. Both shape and scale
are sampled jointly under an explicit correlated Gaussian prior on their logs,
using serial elliptical slice updates. Centered likelihoods and explicit
evaluation/work/storage bounds preserve stable unit transformations. Two
independent bivariate quadrature cases verify 23 posterior mean, variance,
covariance and CDF summaries within 2.281 batch-means Monte Carlo errors;
maximum classical split R-hat is 1.00175. Reference sensitivity to quadrature
order/domain is below `9.47e-10`. The Python comparison takes 1.032 seconds
after imports, peaks at 120.91 MiB and reports zero swaps. Three focused checks,
Ruff, formatting and mypy pass.

Public interfaces, guides and reference tooling are committed at
`73847dd35753c7972f73e837d12df96475514c8f`. Cached wheel and source builds pass.
The isolated wheel verification checks all 1,493 public exports, exact
committed bytes for 542 source/data files, licenses/notices, all 138 catalog
entries and both new executable guide examples. It takes 13.011 seconds,
peaks at 109.81 MiB and reports zero swaps. No dependency, broad numerical
suite or additional CI workflow was introduced. Coverage remains
63 implemented, 66 partial and 9 pending. A source audit records the substantive
continuous Normal and randomized-comparison gaps for BOP2-DC #156 separately
from native report differences and the distinct BOP2 #112 software.

Local `master` is fast-forwarded to this verified code plus the audit. Fresh
read-only GitHub checks show both `master` and `main` still at `45b6e307`;
the verified code contains 108 newer local commits. Shell DNS failure and
connector approval rejection remain publication barriers, with no retry or
bypass of rejected write routes. Updated packages and the verified all-refs
bundle preserve committed root/worker checkpoints locally; in-progress paired
calibration and continuous Normal work remains outside this package and master.

## Normal endpoint and exact paired calibration checkpoint

The Normal endpoint is integrated at `3de513f`. It uses the source NIG prior
and Student-t mean posterior, with complete-outcome replay and serial Normal
trial simulation. Review found and repaired a material large-offset error:
centering observations, prior mean and thresholds together preserves posterior
tails when adding `1e15` to exactly representable inputs. The centered mean
and offset are retained separately. Independent base-R calculations verify
eight cases and 16 posterior analyses, then replay 64 fixed simulation paths
with 154 reached looks. All 898 numeric summaries agree, including all four
terminal decisions, equality, unit changes and Monte Carlo errors. The check
takes .1083 seconds after imports, peaks at 119.19 MiB and reports no swaps.

Exact paired calibration is integrated at `4ba6669`, with the result-copy
memory bound corrected at `c142e2c`. Explicit endpoint-specific grids retain
joint truth distributions and their association. A base-R oracle enumerates
256 paths per truth for 48 candidate/scenario combinations. Both objectives,
meaningful false-decision constraints, all decision/stopping/sample-size
probabilities and expected enrollment agree across 1,920 numeric summaries.
The Python comparison takes .0971 seconds after imports, peaks at 120.63 MiB
and reports no swaps. R and Python numerical work were run sequentially.

Public interfaces, guides and reference tooling are committed at
`55eb2dc8c62cdcaf01fdf62ef1eb7af61fec42f0`. Seven focused tests pass in the
integrated checkout, as do both guide examples, targeted Ruff/format/mypy
checks and cached wheel/source builds. Isolated wheel verification checks
1,504 exports, exact committed bytes for 544 source/data files, notices and
licenses, all 138 catalog entries and both affected examples. It takes
11.855 seconds, peaks at 110.19 MiB and reports zero swaps. No additional CI
workflow, dependency or broad numerical test run was introduced. Coverage
labels remain 63 implemented, 66 partial and 9 pending: these are substantive
additions to one partial program, not a claim of full BOP2-DC app parity.

Local `master` is fast-forwarded to this verified code plus the audit. Fresh
read-only GitHub checks still show `master` and `main` at `45b6e307`, with
114 newer commits at the verified code checkpoint. Shell DNS failure and the
connector's approval rejection continue to prevent publication. No rejected
write route was retried or bypassed. Updated packages and the verified
all-refs bundle preserve committed root/Luna checkpoints locally. Uncommitted
Normal calibration and randomized-comparison work is outside the validated
package and local `master`.

### Normal simulation precision follow-up

Before closing this batch, review identified that adding a large absolute
truth mean during simulation could erase residual variation before the
otherwise centered monitor received it. Commit `d21a011` generates residuals
and shifts the design instead. The parent verified that the previous code
changed decisions/enrollment for two of 32 trials at seed 0 after a `1e15`
model offset, while the corrected version agrees exactly. Commit `365871a`
records that discriminating regression and documents the simulation coordinates.
The five focused Normal tests pass; the parent reran the corrected regression
and all 898 independent-R summaries successfully.

The final wheel/source build is verified at
`365871ae63440de8cca2855686d69ab94283fdf8`: all 1,504 exports, 544 source/data
files, catalog/license checks and both guide examples pass. Verification takes
10.888 seconds, peaks at 109.12 MiB and reports zero swaps. Local `master`
advances to this checkpoint plus the present audit. The verified code is
117 commits ahead of the last verified GitHub heads at `45b6e307`; publication
restrictions are unchanged. In-progress Normal calibration and randomized
comparison implementations remain on worker branches pending independent
reference checks.


## Normal calibration and randomized binary checkpoint

Normal endpoint calibration is integrated at `ec854a8`. It requires explicit
truths, proper NIG priors and finite grids, shares centered simulated paths
across candidates, and reports full candidate evidence plus independent
holdout feasibility without reselection. Independent base-R replay verifies
18 candidates and 836 probability/enrollment/MCSE summaries for both objectives.
A selected candidate passes a 4% calibration false-go limit at `2/64`, fails
holdout at `3/48`, and retains that failure visibly. A common `1e15` location
shift leaves the candidate/holdout results exactly unchanged. The check takes
.6839 seconds after imports, with a conservative Python-plus-R peak-memory
upper bound of 198.78 MiB. A dense 81-candidate default-count smoke with
5,000 trials per stage also passes in .2041 seconds after imports; the combined
smoke/guide process peaks at 110.48 MiB. No Python swaps were reported.

Randomized binary comparisons are integrated at `bc8c622`, with numerical
decision guards at `45adea4` and shared randomized policy at `1d0e38d`. Explicit
independent Beta priors and a fixed allocation tape support signed risk
margins, unequal allocation, optional O'Brien-Fleming graduation, absorbing
replay and exact conditional operating characteristics. Reported quadrature
error must not change the combined strict action; ambiguous decisions fail
explicitly. Independent base-R Beta-polynomial integration and exhaustive
paths verify 100 reached analyses and 336 numeric summaries across four
designs and 12 truth scenarios. Maximum posterior discrepancy is 1.094e-10,
within reported numerical error. The latest post-extraction comparison takes
.3416 seconds after imports, peaks at 115.70 MiB and reports zero swaps.

Public APIs, guides and reference tooling are committed at
`b3058d9d26a48e04365e64120e25305bb7a26ff1`. Seven focused tests pass, along with
targeted Ruff/format/mypy checks and both executable guides. Cached wheel and
source builds pass. Isolated wheel verification checks all 1,514 public exports,
exact committed bytes for 547 source/data files, licenses/notices, all 138
catalog entries and both new examples. Verification takes 16.434 seconds,
peaks at 102.08 MiB and reports zero swaps. No broad numerical suite, additional
CI workflow or dependency was introduced. Coverage remains 63 implemented,
66 partial and 9 pending; randomized Normal/survival workflows and randomized
design calibration keep BOP2-DC partial.

Local `master` is fast-forwarded to this verified code plus the present audit.
Fresh read-only GitHub checks still show both `master` and `main` at `45b6e307`;
the verified code has 123 newer local commits. The existing shell DNS failure
and connector approval rejection continue to prevent publication. No rejected
write route was retried or bypassed. The updated packages and verified all-refs
bundle preserve committed root/Luna work. Randomized Normal and survival worker
work remains outside this verified package and local `master` pending independent
reference checks and public integration.


## Randomized Normal and survival checkpoint

Randomized Normal monitoring/replay is integrated at `bcb58e4`, with aggregate
simulation and corrected conservative quadrature work accounting at `f0a915c`.
Independent arm NIG priors give Student-t mean posteriors; their difference is
computed by bounded quadrature rather than a Gaussian approximation. Shared
centering preserves signed differences under common measurement offsets.
Independent base-R density integration and analytic Cauchy tails verify 12
cases/21 looks. A further 48 fixed latent paths with 78 reached looks verify
simulated decision/stopping probabilities, enrollment and MCSEs, including
16 early graduations. All 263 numeric summaries agree; the maximum posterior
discrepancy is `6.273e-10`, within reported errors. The check takes 9.472 seconds
after imports, with a conservative combined Python/R peak bound of 196.66 MiB
and zero Python swaps. A base-R decimal parsing defect at `1e15` was isolated
and removed from the oracle by storing the common shift separately; Python
still receives actual shifted inputs. Numeric tolerances were not relaxed.

Randomized survival monitoring/replay is integrated at `3d46df9`, with total
replay work/zero-duration-event corrections at `d43eedc` and simulation at
`2add4a4`. Independent IG priors on arm mean survival yield median-time
difference probabilities. Calendar replay uses as-of censoring and absorbs
no-go or graduation; simulation supports fixed/Poisson accrual. An independent
R Gamma-density integral and calendar implementation verify 10 cases/17 looks
and 48 simulation paths/71 looks. All 763 numeric summaries agree, including
24 early graduations, per-arm events/exposure, duration and MCSEs. The maximum
posterior discrepancy is `4.922e-11`, within reported errors. The check takes
.5517 seconds after imports, with a conservative combined Python/R peak bound
of 192.80 MiB and zero Python swaps. Review repaired an omitted graduation
category and made retained-storage accounting include owned copies, Unicode
decision arrays and replay scratch.

Public APIs, guides and reference tooling are committed at
`c0ce18d9cfa1311229dedf88b5778171f6e77f76`. All 14 focused tests pass in the
integrated checkout (5.93 seconds), as do targeted Ruff/format/mypy checks and
both new guide examples. Cached wheel/source builds pass. Isolated wheel
verification checks 1,527 exports, exact committed bytes for 551 source/data
files, licenses/notices, all 138 catalog entries and both examples. Verification
takes 12.267 seconds, peaks at 125.20 MiB and reports zero swaps. No extra CI
workflow, dependency or broad numerical suite was introduced. Numerical work
remained serial. Coverage stays 63 implemented, 66 partial and 9 pending;
randomized calibration keeps BOP2-DC partial.

Local `master` is fast-forwarded to this verified code plus the present audit.
Fresh read-only GitHub checks still show `master` and `main` at `45b6e307`,
with 130 newer commits at the verified code checkpoint. Publication remains
blocked by the earlier shell DNS failure and connector approval rejection;
no rejected write route was retried or bypassed. Updated packages and a
verified all-refs bundle preserve committed work, including worker-only
randomized-binary calibration `43a6570`. That calibration and ongoing randomized
Normal/survival calibration work remain outside this verified package/master
until their independent reference checks and public integration are complete.


## Randomized single-endpoint calibration checkpoint

The three randomized calibration workflows are integrated and publicly exported
at `c44eddd3fb03593762fd429f7afc7d1f762837a1`. Luna implemented the numerical
modules; root reviewed, integrated and ran independent reference comparisons.
Binary calibration caches Beta count-state comparisons and uses exact
conditional recursion. Normal and survival calibration cache posterior tails
on shared paths and retain independent holdout feasibility without reselection.
All candidates remain inspectable when a feasible design is selected.

Independent R Beta integration/path enumeration verifies 66 binary candidate
configurations and 1,980 numeric summaries. Direct Student-t integration verifies
36 Normal candidate configurations and 1,092 summaries; distinct CGR/ESS winners
and a 3/24 calibration no-go rate versus 3/18 validation rate exercise the
calibration-pass/holdout-fail contract. Direct Gamma-density integration and
calendar replay verify 36 survival candidate configurations and 1,791 summaries,
including distinct objective winners, fixed/Poisson accrual, signed margins and
held-out arm event/exposure summaries. Combined: 4,863 numerical summaries.

Review repaired binary floating-accumulation tie selection, Normal combined-event
count arithmetic and absorbed-path decision checks, and conservative workspace
accounting across the three optimizers. Twenty-nine focused randomized tests
pass in 9.44 seconds. Targeted Ruff/format checks and mypy across six affected
source modules pass. No dependency or CI workflow was added, and the full
repository numerical suite was not rerun. Numerical workloads ran serially
with BLAS/OpenMP thread limits of one. The largest conservative combined
Python/R peak bound among these reference checks was 211.57 MiB, with no swaps.

Cached wheel/source builds pass. An isolated interpreter imports the wheel and
checks 1,540 public exports, exact committed bytes for 554 source/data files,
licenses/notices, all 138 catalog entries and four examples across two guides.
Verification takes 14.197 seconds, peaks at 120.41 MiB and records zero swaps.
Coverage remains 63 implemented, 66 partial and 9 pending. A fresh source audit
identifies randomized multiple/co-primary endpoints as the remaining substantive
BOP2-DC family; the catalog stays partial and the scope is documented in
`research/bop2-dc-remaining-methods.md`.

Local `master` is fast-forwarded to this verified code plus this audit. A fresh
read-only GitHub check still shows both `master` and `main` at `45b6e307`; the
verified code has 144 newer local commits, or 145 with this audit checkpoint.
The earlier shell DNS failure and connector rejection (`MCP tool call requires
approval, but approval policy is never`) still block publication. No rejected
write route was retried or bypassed. Refreshed wheel/source packages and a
verified all-refs bundle preserve the root and committed Luna checkpoints,
including all three now-integrated randomized single-endpoint calibrators.
