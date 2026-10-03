# Community publication checkpoint — 2026-09-28

Latest verified package checkpoint: `1da08a2` extends exponential and both
Weibull posterior workflows to right censoring, fixes long Boolean event lists,
and adds event-count inputs for survival success-criterion design. The final section records validation; earlier sections preserve
checkpoint history. Published `9f6c1eb` has passed all hosted quality and
Python 3.12/3.13/3.14 checks in
[run 37157021641](https://github.com/dx-li/mdanderson-stats/actions/runs/37157021641).
That is the latest completely passed hosted checkpoint observed before this
publication; the new revision's hosted results are recorded separately.
The local artifact manifest records full branch SHAs after each independently
verified publication to `master`, `main` and `feat/condis-svm`.

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

At that point no remote publication succeeded. No further write attempt, force
push, approval override or alternative publication transport was attempted
until the later environment change documented under Publication restored.

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

## Randomized paired endpoint checkpoint

Verified package code `e0d319d58bca84eb3a49290f9a0007d000a92c02` integrates
Luna's randomized paired monitoring/replay, exact OC/calibration and simulation
modules. Independent arm Dirichlet models support multiple efficacy and
efficacy/toxicity with raw treatment-minus-control margins, OR/AND decisions
and optional composed interim graduation. Exact recursion preserves endpoint
association and caches marginal comparisons across candidates. Serial simulation
supports larger designs without constructing the four-dimensional exact state
lattice. Twelve public exports and a runnable guide make these workflows
available to package users.

Independent base-R integration and exhaustive joint-category tapes verify 388
case-specific posterior states, 27,104 reached path/look decisions, 64 candidate
metric rows, 256 per-scenario/candidate/look OC rows, both optimization objectives,
infeasibility and 24 public replays. Maximum posterior disagreement is
`6.2507e-10`, within reported numerical error; OC and candidate metric differences
are at most `1.3323e-15`. Reference scripts and all generated tables are retained
for reproducibility. Review corrected count-axis indexing, single-scenario batch
shape, live-lattice memory accounting and the marginal simulation-cache bound.

All ten integrated focused tests pass in 1.94 seconds. Targeted Ruff/format and
mypy checks across three numerical modules pass. Three public guide examples
pass; a two-million-patient simulation request is rejected before consuming
the caller's RNG. The full repository numerical suite was not rerun, and no
CI workflow or dependency was added. Numerical processes ran serially with
thread limits of one. The R reference process peaked at 214.75 MiB; Python
reference comparison peaked at 154.70 MiB, both with zero swaps. The documented
100-trial/40-patient simulation took 0.687 seconds, peaked at 103.47 MiB and
recorded zero swaps.

Cached wheel/source builds pass. Isolated wheel verification checks 1,552 public
exports, exact committed bytes for 557 source/data files in both artifacts,
preserved licenses/notices, all 138 catalog entries and all three new guide
examples. It takes 12.285 seconds, peaks at 116.44 MiB and records zero swaps.
Coverage remains 63 implemented, 66 partial and nine pending. The randomized
two-endpoint family is now covered; more than two decision endpoints, supplement
settings and native application/optimizer/report equivalence remain outside
verified scope, so BOP2-DC retains its partial label.

Local `master` is advanced to the verified code plus this audit. A fresh
read-only GitHub check still shows both `master` and `main` at `45b6e307`.
There are 150 newer local commits at the verified code checkpoint, or 151
including this audit. Publication remains blocked by the earlier shell DNS
failure and connector approval rejection; neither write route was retried or
bypassed. Updated wheel/source artifacts and the verified all-refs bundle
preserve the committed root and Luna work for publication when access permits.


## GAO calendar and CiBolus prior checkpoint

The verified code checkpoint is `514bf2a`. Two Luna implementers added the
explicit-prior GAO calendar driver and balanced CiBolus pseudo-data calibration;
a third Luna agent authored independent references. Root reviewed the equations,
resource bounds and integration. The new APIs have usable guides and remain
within the existing partial catalog entries: counts stay 63 implemented,
66 partial and 9 pending. Native GAO prior calibration, automatic CiBolus
variance selection and the documented native workflow gaps remain open.

Four independent R GAO ledgers match nine patient rows, six interim snapshots,
four final snapshots and sixteen utility values within 6.66e-16. CiBolus reduced
pseudo-posterior means agree with independent quadrature within 0.552 reported
sampler MCSE; direct prior toxicity means agree within 0.035 MCSE. Review fixed
artificial response variance caused by summing toxicity joint cells, preserving
zero variance and infinite ESS when response parameters are fixed. Numerical
and source details are in the two method audits.

Fourteen affected checks passed together before the direct-CDF correction;
all five calibration checks passed after that correction. The independent
reference generator was rerun on the corrected code (18.560 seconds,
119.91 MiB RSS, zero swaps). A separate 400-patient SD20 smoke completed at
118.80 MiB with no swaps, but its eight-draw chains had maximum Rhat4.11;
this demonstrates workload support, not convergence or native calibration.
Targeted Ruff, formatting and source type checks pass. No new CI workflow,
dependency or broad numerical-suite run was introduced.

Cached wheel and source builds pass. An isolated wheel check verifies all
1,558 public exports, byte equality of all 559 source/data files against the
committed code, all 138 catalog entries, preserved licenses/notices, and three
examples across the two new guides. It took 10.602 seconds, peaked at
124.78 MiB and reported zero swaps. Heavy numerical work remained serial with
single-thread BLAS settings throughout this batch.

Local `master` is advanced only after these checks. The refreshed wheel,
source distribution and all-refs Git bundle under ignored `dist/` preserve
the completed work. Read-only GitHub checks still show both remote branches
at `45b6e307`; 159 local-master commits including this audit remain unpublished.
The connector approval restriction and earlier shell DNS failure have no
confirmed resolution. No write retry, permission bypass or alternate transport
was attempted. Local artifacts do not establish GitHub publication.

## Categorical survival forests and anti-split importance

The verified code checkpoint is `326d718`. Two Luna implementers added
unordered categorical splitting and anti-split OOB importance, with a third
Luna reviewer checking the pinned native contracts and extracting reference
kernels. Root integrated the shared routing, corrected source-boundary and
category-map details during review, exposed the API and updated community
examples. Catalog counts remain 63 implemented, 66 partial and nine pending;
entry 166 still has documented native application and model-family gaps.

Twenty focused forest/OOB/importance/shared-contour checks pass. Ten categorical
partition masks, four known-level routing cases and three anti-routing cases
match unchanged pinned C kernels. Existing independent numeric permutation
references continue to agree exactly for three block sizes. Mixed-category
bootstrap fits preserve predictions/OOB/both importance estimates under
order-preserving label changes, including `1e100` labels and stochastic anti
threshold 0.4. A 40-category case remains bounded; four input/work failures
preserve caller RNG state. Numerical scope and native-wrapper differences are
recorded in `research/survival-forest-categorical-audit.md`.

Targeted type, lint and formatting checks pass. Cached wheel and source builds
pass without new dependencies. An isolated wheel check verifies all 1,560
public exports, all 559 packaged source/data files against committed bytes,
all 138 catalog entries, licenses/notices and all six examples across both
forest guides. It took 31.559 seconds (including a temporary plotting font-cache
build), peaked at 153.16 MiB and reported zero swaps. Numerical processes stayed
serial with one BLAS/OpenMP thread. The broad numerical suite was not rerun and
no new CI workflow was added.

Local `master` is advanced to the verified code plus this audit. A fresh read-only
check at 2026-09-29 03:09 UTC still shows GitHub `master` and `main` at
`45b6e307`; 166 local-master commits including this audit remain unpublished.
The earlier shell DNS failure and connector approval rejection have no confirmed
resolution; neither write route was retried or bypassed. The refreshed wheel,
source distribution and verified all-refs Git bundle preserve this checkpoint
and committed Luna work. They do not establish remote publication.

## Interval-PH coefficient bootstrap and competing-risk visit preparation

The verified code checkpoint is `4873075`. Two Luna implementers added these
workflows in separate checkouts; a Luna reviewer checked the pinned R source
contracts and prepared independent fixtures. Root reviewed the resource and
numerical edge cases, integrated the public APIs and added community examples.
Entry 166 remains partial, and the catalog remains 63 implemented, 66 partial
and nine pending. Baseline survival identification bounds are not presented as
sampling confidence bands.

Twelve focused checks pass together in 2.437 seconds at 140.95 MiB peak RSS,
with zero swaps. The interval-PH references cover ordinary and weighted
resampling, singular replicates and sample covariance. Review corrected an
omitted native endpoint-preprocessing step in the reference harness before
accepting numerical agreement. All eight parent/bootstrap reference CSVs
regenerate byte-identically (4.724 seconds, 309.00 MiB peak child RSS including
the serial native builder, zero swaps). The visit generator reproduces the
pinned native outputs and records deliberate fixes for its sorting and
first-visit-event defects.

Integration checks reconstruct weighted RNG draws from an explicit tape with
fractional case weights, preserve coefficient/covariance results under
covariate units of `1e100` and `1e-100`, and recover a 120-subject two-cause
study exactly from 203 reverse-ordered visit rows before fitting it. They take
1.365 seconds at 120.12 MiB RSS, zero swaps. Review also corrected potential ID
coercion, nested-input allocation and covariance underflow problems. Targeted
Ruff, formatting and type checks pass. No new CI workflow or broad numerical
suite run was added; numerical execution remained serial and single-threaded.

Cached wheel and source builds pass without installing dependencies. An
isolated wheel check verifies all 1,564 public exports, byte equality of all
561 packaged source/data files against committed bytes, all 138 catalog
entries, retained licenses/notices and four examples across the two affected
guides. It takes 31.559 seconds including the plotting font-cache build, peaks
at 160.55 MiB and reports zero swaps.

Local `master` is advanced only after these checks, with this publication
record as the sole additional change beyond the verified code. The wheel,
source distribution and all-refs Git bundle preserve the completed work.
A read-only check at 2026-09-29 03:37:55 UTC still shows GitHub `master` and
`main` at `45b6e307`; 171 local-master commits including this audit are
unpublished. The earlier DNS failure and connector approval restriction have
no confirmed resolution. Neither write route was retried or bypassed, and
local artifacts do not establish remote publication.

## Competing-risk bootstrap and TITE-CRM prior ESS

The verified code checkpoint is `5115a87`. Two Luna implementers worked in
separate checkouts; a Luna reviewer supplied unchanged-source references and
checked the statistical contracts. Root integrated the public interfaces,
examples and notices, reviewed numerical edge cases and verified packaging.
Entries 154 and 166 retain partial status: the full catalog remains 63
implemented, 66 partial and nine pending. These counts do not estimate the
remaining engineering time or establish full native application coverage.

The competing-risk bootstrap resamples complete rows, rebuilds each sample's
spline knots and returns coefficient covariance and standard errors. Invalid
resamples and optimizer failures have explicit policies and aligned records.
Eight focused checks pass in 2.52 seconds at 139.78 MiB peak process RSS, with
zero swaps. Seeded draws reproduce an explicit tape exactly, and coefficient
and standard-error scaling remains correct under covariate units of 1e-100
and 1e100. The public example's eight refits converge. All five native R
bootstrap fixtures regenerate byte-identically in 2.224 seconds at 92.50 MiB
peak child RSS, with zero swaps. Known native optimizer defects are disclosed;
native fitted slopes are compatibility records, not certified optima.

TITE-CRM supports fixed and Poisson accrual, pending toxicity histories,
explicit assessment timing, signed expected-subset information and full-real
or legacy truncated-numerator posterior moments. Seven focused checks pass
in 1.36 seconds at 130.48 MiB RSS, zero swaps. Fixed and Poisson trial paths
match the unchanged dfcrm source, and the assessment excludes a toxicity
that occurs after the cutoff. Twelve independent posterior-moment comparisons
at default and diffuse prior scales have maximum absolute discrepancies of
1.31e-10 for full moments and 7.46e-11 for the legacy convention. The five
portable R fixtures regenerate byte-identically in 0.541 seconds at 84.30 MiB
child RSS, zero swaps. Review corrected near-one curvature cancellation,
normalizer convergence checks and overflow-prone ESS crossing detection.

Targeted lint, formatting and type checks pass. No new CI workflow or broad
numerical test run was added. Numerical processes ran serially with one
BLAS/OpenMP thread; no dependencies were installed. Cached wheel and source
builds pass. The isolated wheel check verifies all 1,568 public exports,
563 packaged source/data files against committed bytes, all 138 catalog entries,
retained licenses/notices and both new guide examples. It takes 10.629 seconds,
peaks at 125.27 MiB RSS and reports zero swaps.

Local `master` includes the verified code plus this publication record. The
refreshed wheel, source distribution and verified all-refs Git bundle preserve
the completed root and Luna commits. A read-only check at 2026-09-29 04:14:18 UTC
still shows GitHub `master` and `main` at `45b6e307`; 178 local-master commits
including this audit are unpublished. The earlier shell DNS failure and
connector rejection (approval required while approval policy is never) have no
confirmed resolution. Neither write route was retried or bypassed. Local
artifacts and local branch advancement do not establish remote publication.


## Cluster bootstrap and random-routing forest importance

The verified code checkpoint is `1abdfdb`. Two Luna implementers worked in
separate checkouts; a Luna reviewer checked the source contracts and supplied
an unchanged-C routing reference. Root integrated public exports, examples,
notices and source comparisons. Entry 166 retains partial status: the catalog
still has 63 implemented entries, 66 partial and nine pending. These counts
are not an estimate of remaining engineering time.

The interval-PH cluster bootstrap samples complete groups with replacement.
Integer frequency weights preserve repeated group observations without
allocating an expanded input matrix. Unequal group sizes, deterministic
label/tape mapping, default failure propagation and explicit optional failure
records are documented. Two fixed tapes expanded by the unchanged icenReg
helper produce 84 and 100 rows. Python slopes and sample covariance agree
with expanded-row native PH fits within 2.795e-7 and 2.905e-8 respectively.
The public example completes all eight refits. Shared-loader extraction
preserves the five original PH reference cases; all eight existing
PH/ordinary-bootstrap CSVs remain unchanged. Serial native generation takes
6.977 seconds, peaks at 309.80 MiB child RSS and reports zero swaps.

Random-routing importance retains each node's represented sample count,
including bootstrap duplicates. It implements the pinned source's coupled
uniform threshold rule and uses the existing complete-block OOB concordance
estimator, categorical encoding and undefined-block/tail reporting. All 15
unchanged native C routing/draw-count reference rows regenerate byte-for-byte
in 0.480 seconds at 36.66 MiB child RSS, zero swaps. Native RNG and full native
forest parity are not claimed.

The 31 focused affected checks pass in 3.50 seconds at 145.64 MiB peak RSS,
zero swaps. They cover expanded-row equivalence, resample failure handling,
bootstrap multiplicities, core forest/OOB behavior, categorical relabeling,
zero-threshold identity and complete-block replay. After adding the native
coefficient/covariance fixture assertions to the existing expansion test,
that test passes again in 1.81 seconds. Both public examples and the native
coefficient comparison pass in 0.937 seconds at 121.78 MiB RSS, zero swaps.
Targeted lint, formatting and worker type checks pass. No CI workflow was
added, no dependencies installed, and numerical jobs ran serially with one
BLAS/OpenMP thread.

Cached wheel/source builds pass. The isolated wheel check verifies 1,572
public exports, byte equality of all 564 packaged source/data files against
committed code, all 138 catalog entries, retained notices and both new guide
examples. It takes 9.841 seconds, peaks at 125.41 MiB RSS and reports zero
swaps. The distribution retains its mixed-license qualification.

Local `master` contains this verified code plus this publication record as the
sole additional change. The wheel, source distribution and verified all-refs
Git bundle preserve completed root and Luna work. A read-only check at
2026-09-29 04:52:00 UTC still shows GitHub `master` and `main` at `45b6e307`;
185 local-master commits including this audit are unpublished. The earlier
shell DNS failure and connector rejection (approval required while approval
policy is never) have no confirmed resolution. Neither write route was
retried or bypassed. Local branch advancement and artifacts do not establish
remote publication.


## Publication restored — September 29

The execution environment changed from restricted network access to enabled
network access. An ordinary remote read and fetch succeeded. All three remote
branches were verified to be ancestors of validated local master, then an
atomic fast-forward push published `a796f1d` to `master`, `main` and
`feat/condis-svm`. A separate `git ls-remote` check at 2026-09-29 15:32:40 UTC
returned the full SHA `a796f1d7fce7f3adf0a42cfc71c0d4703f36c387` for each branch.
This resolves the earlier 185-commit publication backlog. No force push,
connector-approval bypass or history rewrite was used. Ongoing Pinnacle and
Phase2Delay implementations remain in separate checkouts pending validation.


## Pinnacle and Phase2Delay community workflows

Verified code is `c38cca5`. Luna implemented both workflows in isolated
checkouts and a separate Luna reviewer examined the calendar contract. Root
reviewed numerical and memory boundaries, integrated public exports and guides,
and verified the combined package. No dependencies or CI workflows were added.

Pinnacle adds explicitly configured per-gel wavelet denoising and rectangular
local backgrounds. Detection uses the denoised raw average; optional per-gel
reconstructions supply peak/background measurements while image-volume
normalization uses raw pixels. The processing order is a documented Python
choice pending native executable confirmation. Combined memory accounting
includes retained pipeline images and result matrices, respects both pipeline
and per-gel budgets, and releases measurement views between gels.

Phase2Delay adds scheduled calendar replay, calibrated Weibull event delays,
Poisson accrual and serial operating-characteristic summaries. Its monitoring
gate counts full assessment windows, not early observed responses. Absolute
completion/event timestamps handle same-time enrollment and non-binary time
rounding. Stopping excludes future patients and decisions. Explicit terminal
policies distinguish decision time from completion of enrolled follow-up.
Seed lineage reproduces individual trials and their analyses; compact results
and work/storage bounds avoid retaining every posterior draw across trials.
Continuous-monitoring equivalence, native priors/calibration and reports remain
open. Both source entries remain partial; counts stay 63 implemented, 66
partial and nine pending.

All 30 affected Pinnacle and Phase2Delay checks pass in 5.896 seconds at
137.84 MiB peak process RSS with zero swaps. They include existing unchanged-C
wavelet and independent R measurement/hazard references, cropped nonzero
per-gel denoising, combined-memory rejection, calendar gate/ties/early stopping,
seed replay and independent Weibull fixtures. The regenerated base-R reference
verifies eight calibrations, 48 inverse-CDF values and sevenfold time scaling
in 0.243 seconds at 78.45 MiB peak child RSS, zero swaps. Targeted lint,
formatting and type checks pass. Numerical processes ran serially with one
BLAS/OpenMP thread; the full repository test suite was not rerun because the
change is confined to these workflows.

Cached wheel and source builds succeed. Isolated wheel verification passes
all 1,580 exports, byte equality for 565 committed package source/data files,
all 138 catalog entries, preserved licenses/notices and four examples across
the two affected guides. It takes 13.135 seconds at 125.50 MiB peak RSS with
zero swaps. The distribution retains its mixed-license qualification.

The package artifacts correspond to `c38cca5`. The publication checkpoint adds
only this audit record before advancing local master. Publication uses an
ordinary atomic fast-forward push of the same checkpoint to GitHub master,
main and the development branch, followed by independent remote-SHA verification.
The local artifact manifest records the resulting remote state and hashes;
a locally advanced branch alone is not evidence of publication.


## MDS-HOPE and Parallel Phase I/II community checkpoint

Verified package code is `28fc12faf492c6f4673c9fc2dc3131b271401788`.
Luna implemented both additions. An independent Luna review checked the MDS-HOPE
publisher equation and input semantics; root integrated the exports, guides,
source catalog and independent numerical references. No dependencies or CI
workflows were added. ARAND calendar work remains isolated and is not part of
this checkpoint.

MDS-HOPE implements Supplemental Equation S1 in original clinical units,
per-predictor contributions and relative hazards against an explicit reference
profile. Raw differences are formed before coefficient weighting. Six risk
groups require an already-standardized score or explicit caller reference
center and SD. Numeric cytogenetic-category encoding, native calibration and
baseline survival are still unavailable; no app equivalence or absolute
survival prediction is claimed. Entry 171 moves from pending to partial, so
coverage is now 63 implemented, 67 partial and eight pending.

Parallel Phase I/II adds bounded serial operating-characteristic summaries for
the validated four-arm C design, retaining reproducible per-trial seeds and
aggregates. Selection, stopping, allocation, enrollment, toxicity and response
summaries include Monte Carlo uncertainty; pooled outcome-rate errors treat
trials as independent clusters. It does not infer optimal-arm criteria or
reproduce a stale native output column. Entry 85 remains partial.

The 11 affected tests, all 43 independent base-R reference cases and three
examples across both guides pass together in 1.276 seconds at 146.98 MiB peak
process RSS with zero swaps. Reference arithmetic differs by at most
7.11e-15. Fixtures include every MDS-HOPE coefficient, TP53 profiles, explicit
reference comparisons, exact and neighboring risk cutoffs, and nonfinite
rejections. The R generator uses 17-digit CSV output to preserve distinct
floating-point neighbors. Synthetic cytogenetic values and standardization
constants are explicitly labelled. Focused Ruff, formatting and type checks
pass. The full repository numerical suite was not rerun; validation was confined
to the affected methods and package integration.

Cached wheel and source builds pass. An isolated wheel import verifies all
1,588 public exports, byte equality of 567 committed package source/data files
in both artifacts, all 138 catalog entries, preserved license notices and all
three new guide examples. It takes 10.304 seconds at 125.62 MiB peak RSS with
zero swaps. Numerical/build processes ran one at a time with thread limits;
there were no dependency installations. The distribution retains its existing
mixed-license qualification.

This audit is the only change after the verified package revision. Publication
advances master and main together by an ordinary atomic fast-forward push, also
updating the development branch. Independent remote-SHA verification and local
artifact hashes are recorded in the ignored `dist/community-checkpoint.json`;
a local commit alone does not establish GitHub publication.


## ARAND calendar and MDS-HOPE risk-group checkpoint

Verified package code is `78076d992112183f5a443436e72697f9627e7133`.
Luna implemented both additions in isolated checkouts. Root integrated public
exports, executable guides, source catalog and independent risk-group references.
No dependencies or CI workflows were added.

ARAND adds bounded binary and exponential calendar replay: potential outcomes
remain hidden until observed, suspended arms can reactivate, permanent futility
prevents allocation, and final selection includes temporarily suspended arms.
Explicit policies expose source-ambiguous ordering, allocation floors, ties and
duration/minimum-enrollment precedence. Preflight work and retained-history
limits precede large allocations. The simulator and aggregate reports remain
separate development work; native controller equivalence is not claimed.

MDS-HOPE adds the published five-group alternative while preserving the
six-group default. Explicit calibration remains required, and native numeric
cytogenetic encoding and absolute-survival calibration remain unavailable.
Coverage stays 63 implemented, 67 partial and eight pending catalog entries.

All 21 affected tests, 34 independent R risk-boundary cases and four guide
examples pass together in 1.481 seconds at 147.53 MiB peak RSS with zero swaps.
The affected posterior tests retain their 52 independent R probability checks.
Focused lint, formatting and type checks pass. The full repository suite was
not rerun because the change is confined to these methods and integration.

Cached wheel and source builds pass. Isolated wheel verification checks all
1,592 exports, byte equality for 568 committed package source/data files in
both artifacts, all 138 catalog entries, preserved license notices and all four
guide examples. It takes 11.183 seconds at 126.08 MiB peak RSS with zero swaps.
Numerical and build execution remained serial with thread limits. Existing
mixed-license qualifications remain explicit.

This audit is the only change after the verified package revision. Publication
advances master, main and the development branch by ordinary atomic
fast-forward push. Independent remote-SHA verification and local artifact
hashes are recorded in the ignored `dist/community-checkpoint.json`; a local
commit alone does not establish GitHub publication.


## ARAND simulation community checkpoint

Verified package code is `71538dd9b922da35b7f65d310d4d5a804835854d`.
Luna implemented bounded serial Poisson-accrual simulation and per-arm
operating-characteristic summaries for binary and exponential outcomes. Root
reviewed completion proofs and numerical stability, integrated public exports
and guides, and checked the assembled package. Candidate exhaustion raises
instead of silently truncating trials; duration summaries use scaled arithmetic
and each history is released before the next trial. Separate suspension,
permanent-futility and early-winner-displacement events avoid guessing the
native dropped-arm aggregation. Trial seeds reproduce individual histories.

All 15 affected tests pass. Three seeded two-arm exponential comparisons
preserve assignments, observed events and probabilities when converting means
to medians. A 512-trial single-arm simulation matches Poisson expected enrollment
and zero-enrollment probability within six analytical Monte Carlo standard
errors. Combined checks take 1.639 seconds at 147.12 MiB peak RSS, zero swaps;
the public simulation example also passes. Targeted lint, formatting and type
checks pass. No broad numerical suite or new CI pipeline was introduced.

The previous GitHub run stopped at import ordering in two pre-existing reference
scripts. This checkpoint makes the two narrow import-only corrections shown by
that job; it does not claim that a new hosted run has already passed.

Cached wheel/source builds and isolated wheel verification pass: 1,596 exports,
569 committed package source/data files byte-identical in both artifacts, 138
catalog entries, preserved notices and three examples from two guides. Package
verification takes 11.020 seconds at 125.95 MiB peak RSS, zero swaps. Coverage
remains 63 implemented, 67 partial and eight pending. Native ARAND controller,
RNG and desktop report equivalence remain open; mixed-license terms still apply.

This audit-only addition follows the verified code. Publication uses an ordinary
atomic fast-forward push to master, main and the development branch, with
independent remote verification and artifact hashes in the local ignored
manifest. Trinary EffTox calibration remains isolated development work.


## Trinary EffTox prior elicitation community checkpoint

Verified package code is `ca0f1f7c665933d3dda266567a5cc12aa286f09c`.
Luna implemented induced trinary probability moments and toxicity-first prior
elicitation; root and a separate Luna reviewer checked the model contract,
source limitations and numerical stability. The objective targets marginal
efficacy means and beta-moment ESS while integrating uncertainty from both
independent prior blocks. Stable complements and nonnegative variance terms
preserve rare probabilities and concentrated-prior variances. The sequential
objective is explicitly a Python policy because the recovered sources do not
specify native trinary calibration. Native parity is not claimed.

All seven focused checks and two public examples pass together in 29.749
seconds at 145.17 MiB peak RSS, zero swaps. Independent base-R fixtures cover
twelve dose/prior combinations, including near-fixed cases; the reported
objective is recomputed from returned moments and parameters. The ordinary
three-dose example converges with default settings at 431 toxicity and 494
efficacy evaluations. Achieved mean ESS values are 1.1000007421 and
0.8000035097, against 1.1 and 0.8, with maximum efficacy mean residual
0.00122147. Targeted lint, formatting and type checks pass.

The prior hosted run progressed past lint and then reported formatting in 33
existing guides. Mechanical formatting preserves the syntax trees of all 49
affected Python blocks and leaves their prose unchanged. This checkpoint
contains that correction without changing CI or adding new checks. Hosted
validation of this new checkpoint is tracked separately from local success.

Cached wheel/source builds and isolated wheel verification pass: 1,600 public
exports, all 570 committed package source/data files byte-identical in both
artifacts, all 138 catalog entries, preserved license notices and three examples
from the ARAND simulation and trinary calibration guides. Verification takes
22.797 seconds at 128.12 MiB peak RSS, zero swaps. Numerical/build processes
ran serially with thread limits and no installations. Coverage remains 63
implemented, 67 partial and eight pending; existing mixed-license terms apply.

Only this audit follows the verified code. Publication advances master, main
and development together by ordinary atomic fast-forward push. The local
ignored manifest records independently verified remote SHAs and artifact hashes.


## Multc duration simulation community checkpoint

Verified package code is `f901429bdf7853ea7f699f9e531af589b6a10711`.
Luna implemented serial paired-outcome generation, calendar replay and duration
summaries for Multc Lean and the supported Multc99 Phase IIa mapping. Root and
an independent Luna reviewer checked timing, aggregation and memory bounds.
The API requires a conditional-truncation or clipping policy for response
observation and an explicit toxicity delay; durations start at first enrollment.
These choices do not claim native timing or random-stream parity.

Eight focused simulator/calendar checks pass, including analytic fixed-enrollment
duration mean and Monte Carlo error, reproducible child seeds, joint outcomes,
resource preflight and extreme response-window representation. Worker validation
takes 2.13 seconds, with 131.89 MiB peak RSS and zero swaps. Targeted lint,
formatting and mypy pass. All four guide blocks execute after integration,
including the public 32-trial example, in 1.922 seconds at 123.22 MiB peak RSS.

Cached builds and isolated wheel verification pass: 1,604 public exports, all
571 package source/data files byte-identical to committed Git contents in wheel
and source distribution, four guide blocks, the complete 138-entry catalog and
preserved license notices. Verification takes 10.399 seconds at 128.39 MiB peak
RSS with zero swaps. Coverage remains 63 implemented, 67 partial and eight
pending. No broad local numerical suite or new CI checks were added.

The preceding published checkpoint `98551c1` passed GitHub's quality checks and
full Python 3.12 and 3.13 suites. Its Python 3.14 run was still active at the last
inspection; hosted results for this new checkpoint require separate verification.
Only this publication audit follows the verified package code. Publication uses
an ordinary atomic fast-forward of master, main and development, followed by
independent remote-SHA verification in the ignored artifact manifest. WFMM
variance initialization remains isolated work in progress.


## WFMM variance initialization community checkpoint

Verified package code is `8ba8e874d117ba617b2fd85d6627b68740eb17f0`.
Luna implemented per-coefficient REML initialization for the existing Gaussian
mixed model; root and a separate Luna reviewer checked the objective, scale
corrections, covariance identifiability and boundary handling. A single residual
component uses its analytical estimate. Mixed/stratified fits are serial and
bounded, retain raw estimates and convergence diagnostics, and distinguish
positive sampler starts from inference or native prior calibration.

Five focused tests match independent base-R residual-only and balanced
random-intercept references and cover degenerate/confounded designs and actual
likelihood-evaluation caps. Six public guide blocks pass. Additional root checks
confirm two-coefficient analytical variance estimates, fixed-design unit changes
in both coefficients and restricted likelihood, and use of the estimated starts
in the existing sampler. Combined checks take 2.755 seconds at 148.72 MiB peak
RSS and zero swaps. Targeted lint, formatting and type checks pass after the
final edits. Native initialization and inverse-gamma defaults remain open.

The preceding Multc hosted run stopped at formatting in its two new Python
files. Mechanical correction leaves both syntax trees unchanged. This checkpoint
includes that correction without adding CI checks or repeating the unchanged
Multc numerical suite. No full local numerical suite was run.

Cached wheel/source builds and isolated wheel verification pass: 1,606 exports,
all 572 package source/data files byte-identical to committed Git in both
artifacts, six guide blocks, the 138-entry catalog and preserved license notices.
Verification takes 11.236 seconds at 126.25 MiB peak RSS with zero swaps. Source
coverage remains 63 implemented, 67 partial and eight pending. The distribution
retains its existing mixed-license terms.

Only this audit follows the verified package code. Master, main and development
are advanced together using an ordinary atomic fast-forward push and independent
remote verification. The ignored manifest records exact remote SHAs, artifact
hashes and hosted validation separately. MTADF logistic inference and the optional
TITE-BOIN12 run-in remain isolated development work.


## TITE-BOIN12 optional run-in community checkpoint

Verified package code is `157be6d65ab21a1019d271c927e4ee39b751204f`.
The optional run-in now applies the recovered source note's exact rule: only
at toxicity limit 0.25, with exactly three or six patients at the current dose,
at least two observed DLTs trigger de-escalation. Pending-information and global
admissibility checks precede it; it precedes precision stopping and other dose
decisions. These ordering and unavailable-neighbor policies are explicitly Python
choices. Defaults preserve existing conduct. A lowest-dose safety stop retains
posterior admissibility separately rather than inventing a new elimination rule.

Luna's twenty focused conduct/reference checks pass with 130.38 MiB peak RSS
and zero swaps; targeted lint, format and mypy pass. Root and a separate reviewer
checked the small behavior change. Both guide examples pass, as do direct checks
of precision-stop precedence and lowest-dose stopping versus admissibility.
The root guide process peaks at 122.73 MiB with zero swaps. No broad local suite
or new CI checks were introduced.

Cached builds and isolated wheel verification pass: 1,606 public exports,
572 package source/data files byte-identical to committed Git contents in both
artifacts, two guide blocks, all 138 catalog entries and preserved license
notices. Verification takes 10.323 seconds at 128.16 MiB peak RSS, zero swaps.
Coverage remains 63 implemented, 67 partial and eight pending; TITE-BOIN12's
other documented method/native gaps remain open.

GitHub quality validation passed for the preceding `ba0226a` checkpoint,
confirming the Multc formatting correction. Its full Python-version test matrix
was still running at the last inspection. Hosted results for this new checkpoint
are tracked separately. This audit-only commit follows the verified code;
ordinary atomic fast-forward publication and independent remote verification
keep master, main and development aligned. MTADF logistic work remains isolated.

## MTADF logistic and TITE-BOIN12 final-selection community checkpoint

Verified package code is `319916ab24cc4ceb832d7a9bf170b70de32f02c9`.
Luna implemented the paper's global quadratic and local linear logistic efficacy
models with explicit Cauchy priors, dose coding and bounded serial MCMC.
Independent transformed-Cauchy quadrature checks both models; global 128/192
rules agree within 3.32e-14 and the seeded fit agrees within 1.82 batch MCSEs.
Root and a separate Luna reviewer checked safety-first decisions, observed
current-dose requirements, retained bounce-window diagnostics and aggregate
resource bounds. Identical lower-boundary windows reuse the current fit.
Native sampling, unspecified conduct details and logistic trial simulation
remain distinct from these documented Python methods.

TITE-BOIN12 now accepts fully resolved patient histories for final selection,
preserves joint binary counts and delegates to the existing complete-data
BOIN12 two-step rule. Pending endpoints are rejected. Cumulative follow-up may
continue past an observed event's window without changing the complete-data
selection. The unrecovered BDA joint-prior and supplementary details remain
explicit gaps; no native defaults were guessed.

Final focused MTADF checks pass 18 tests in 2.20 seconds at 134.92 MiB peak RSS;
TITE conduct/reference/final checks pass 22 tests in 1.70 seconds at 132.58 MiB.
Both runs treat warnings as errors and report zero swaps. Targeted lint,
formatting and type checks pass. No broad local suite or new CI was added.

Cached wheel/source builds and isolated wheel verification pass: 1,615 exports,
all 573 package source/data files byte-identical to committed Git contents in
both artifacts, five guide blocks across two guides, 138 catalog entries and
preserved license notices. Verification takes 10.600 seconds at 128.47 MiB peak
RSS with zero swaps. Coverage remains 63 implemented, 67 partial and eight
pending. Existing mixed-license terms remain unchanged.

The previously published `fc81de5` checkpoint has successful hosted quality,
Python 3.12 and Python 3.13 checks; Python 3.14 was still running at the last
inspection. Those results do not imply this new checkpoint's full matrix has
passed. This audit-only commit follows verified package code. Ordinary atomic
fast-forward publication and independent remote-SHA verification align master,
main and development; the ignored manifest records publication and hosted
validation separately.

## Logistic simulation and Bayesian augmentation community checkpoint

Verified package code is `f4158b47e1d368865f030c66bdbc5560e14d07fe`.
Luna implemented complete-cohort simulation for both MTADF logistic methods,
with separate outcome/sampler seeds for replay, compact operating-characteristic
summaries and bounded serial fits. The independent-marginal truth assumption
is explicit. Review verified safety-first actions, final selection, distinct
local-window accounting and release of posterior draws between looks. Root
increased the fixed storage allowance for seed/result copies and verified a
tight budget rejects before RNG allocation, without running the large request.

The TITE primary article's main text supplied all three BDA missing-outcome
formulas, superseding an earlier incomplete retrieval. A positive four-cell
Dirichlet prior remains caller-specified because the published concentration
and marginal means do not identify prior association. The sampler repeatedly
imputes from fixed observed-data masks, updates the joint probabilities, and
separately averages BOIN12 complete-data quasi-Beta and marginal-tail summaries.
Log-domain conditionals, explicit numerical failures and bounded diagnostic
temporaries preserve numerical and memory constraints.

Twenty-five MTADF and twenty-five TITE focused checks pass with warnings
treated as errors. Peak RSS is 128.06 MiB and 134.97 MiB respectively, with
zero swaps. The independent BDA reference analytically integrates all 16
compatible assignments in a small synthetic-prior example: all 13 checked
posterior summaries agree within 1.372 estimated batch MCSEs. That comparison
takes 3.321 seconds at 126.39 MiB, with zero swaps. Targeted lint, formatting
and type checks pass; no broad local suite or new CI is added.

Cached builds and isolated wheel checks pass: 1,623 exports, all 575 package
source/data files byte-identical to committed Git in wheel and source archive,
three executable blocks across two guides, all 138 catalog entries and retained
license notices. Package checks treat warnings as errors, take 11.340 seconds,
peak at 129.48 MiB and report zero swaps. Coverage remains 63 implemented,
67 partial and eight pending. BDA conduct/calendar integration and other
documented native/model gaps remain open; mixed-license terms are unchanged.

The previous guide-formatting issue was corrected in published `203d85c`,
with unchanged example syntax trees. Hosted quality, Python 3.12 and Python
3.13 checks for that revision pass; Python 3.14 was still running when last
inspected. The new hosted run is recorded separately. Only this audit follows
verified package code before ordinary atomic fast-forward publication and
independent remote-SHA verification of master, main and development.

## Bayesian dose conduct and generalized rare-disease cohorts

Verified package code is `32061b468e8f8a996c403975a8c98873954c2200`.
Luna implemented BDA neighboring-dose decisions with pending-data suspension
before sampling, persistent exclusions and completed-imputation toxicity rates.
A shared transition helper preserves the existing AL behavior. Source-defined
rules are distinguished from explicit Python rate, run-in and precision policies.
The new public API and examples are documented in the BDA guide.

The rare-disease design and simulator now accept the captured app's 15 cohort
size pairs: a=1..3 and b=1..5. Nondefault sizes require an explicit admissibility
count threshold, because the saved source does not specify generalized exclusion
settings. Review caught and repaired a fixed three-outcome simulation width;
the 1+3+5 regression verifies all nine patients' outcomes are counted. Default
1+2+3 behavior and its independent reference checks remain intact. The README
now gives direct master-branch checkout and installation instructions.

Twenty-nine focused TITE checks and nine rare-disease checks pass with warnings
treated as errors. Targeted Ruff, formatting and type checks pass. The final
worker processes peak at 133.62 and 150.42 MiB RSS with zero swaps. Root's
independent exact-integral comparison validates the BDA movement-rate estimate
within 0.201 conservative MCSE, taking 3.517 seconds at 124.19 MiB with zero
swaps. Validation runs sequentially; no broad local suite or new CI is added.

Cached wheel/source builds and isolated wheel verification pass: 1,625 public
exports, all 575 package source/data files byte-identical to committed Git in
both artifacts, eight executable examples across three affected guides, all
138 catalog entries and preserved license notices. Verification takes 14.038
seconds, peaks at 118.09 MiB and reports zero swaps. Coverage remains 63
implemented, 67 partial and eight pending. Calendar/native-policy gaps and
the existing mixed-license terms remain explicit.

At the last pre-publication inspection, the preceding `1ea1efa` checkpoint has
successful hosted quality, Python 3.12 and Python 3.13 checks; Python 3.14 is
still running. These are not results for the new checkpoint. This audit-only
commit follows the verified code. The local manifest records the subsequent
atomic fast-forward publication, independently checked remote SHAs and the new
hosted run separately.

## rBOP2 calibration and EasyCellType reference checkpoint

At `a41ade7`, binary rBOP2 calibration ranks caller-supplied cutoff curves by
power subject to a declared null error cap. Calibration and analysis priors
have separate diagnostics, and the selected design uses the analysis prior.
The API preserves strict futility and inclusive superiority, exact rational
cutoff ties, no-feasible-design reporting and deterministic work limits.
Its scope is a finite candidate set and one declared null pair; native grid
generation and composite-null guarantees are not implied.

EasyCellType now reads bounded local CSV/gzip reference tables, applies native
species/tissue filtering, preserves duplicate/order/blank-organ conventions,
and records file SHA-256 and caller-supplied provenance. The optional base-R
exporter converts the pinned author snapshot locally. Full marker datasets
are not bundled; upstream data terms and gene-ID mapping remain distinct
from this Python loading/annotation workflow.

Nineteen focused worker tests pass, with targeted Ruff, formatting and mypy.
Independent root checks verify 64 exact-rational candidate/selected-design
outputs, 24 fractional-prior/signed-margin/unequal-arm comparisons, duplicate
ties and infeasible designs. All 236,219 source association rows have the
expected species counts, and six independent R tissue-filter outputs match
exactly. A compressed oversized line is rejected before unbounded allocation.
The root numerical/source check takes 2.485 seconds, peaks at 132.53 MiB and
reports zero swaps; worker checks peak at 133.23 MiB.

Cached wheel/source builds pass without installations. The isolated wheel
check verifies 1,659 public exports, exact committed bytes for all 587 package
source/data files in both artifacts, preserved notices and both standalone
guide examples. It takes 11.969 seconds, peaks at 111.41 MiB and reports zero
swaps. Counts remain 63 implemented, 67 partial and eight pending. No broad
local test suite, new dependencies or CI workflow changes were added.

Before this publication, the preceding `266194a` hosted run had successful
quality and Python 3.12/3.13 jobs, while Python 3.14 was still running. The
earlier `d770fde` run passed all four jobs. The local manifest records new
remote branch verification and hosted state separately.

## BOIN12 two-stage and BF-BOIN accelerated titration checkpoint

At `0055027`, BOIN12 supports toxicity-only escalation followed by the
existing joint toxicity/efficacy optimization. The required S threshold is
evaluated at a completed cohort boundary; accumulated outcomes and per-dose
exclusions are retained. Public decision and simulation APIs expose stage
and transition diagnostics. Strict posterior safety, adjacent-dose movement,
stopping precedence and final selection are checked explicitly. The cached
help specifies the stage criteria; detailed cohort timing and stop precedence
remain documented Python policies.

BF-BOIN adds optional singleton titration with first-DLT and second-grade-2
triggers, immediate highest-dose top-up and lower-cap full-next-cohort
transitions. The terminal singleton and top-up count as the first ordinary
cohort. Grade-2 probabilities conditional on no DLT and assessment delay are
explicit caller inputs. Patient histories report exit time, outcomes observed
at exit and complete follow-up through later grade-2 assessments. Ordinary
backfill resumes after titration and optional expansion remains available.

Twenty-two BOIN12 and eleven BF-BOIN focused checks pass with warnings as
errors. Targeted Ruff, formatting and mypy pass. Independent 70-digit
finite-binomial references match 320 toxicity-posterior safety decisions;
efficacy invariance, adjacent safe destinations, deterministic stage ledgers
and pre-RNG work rejection pass. All existing fields reproduce 80 published
BF-BOIN trials exactly. Sixty enabled-titration ledgers cover all four exit
reasons, chronology, cohort budgets, exit counts and complete follow-up.
The root integration check peaks at 125.12 MiB; worker checks peak below
130 MiB. All report zero swaps, with numerical processes run serially.

Cached wheel/source builds pass. The isolated wheel check verifies all 1,655
public exports, exact committed bytes for 585 package source/data files in
both artifacts, preserved licenses/notices and both new guide examples.
It takes 17.267 seconds, peaks at 107.23 MiB and reports zero swaps. Catalog
totals remain 63 implemented, 67 partial and eight pending; these additions
complete documented components without claiming full native-app coverage.
No new dependencies or CI workflows were added.

The preceding published checkpoint `d770fde` passed the full hosted quality
and Python 3.12/3.13/3.14 matrix in workflow run `36636425154`. The current
publication's remote hashes and hosted status are recorded separately in the
local artifact manifest; prior hosted success does not imply that a newer
revision's full matrix has finished.


## September 29 — survival and TITE calendar workflows

Validated code checkpoint `2ccfc5a77ec7868c6e8ac72c0daac1d99761522c`
adds BayesFactorTTE event/arrival/censor replay and serial exponential simulation
with separate accrual/outcome seeds, early/final monitoring, patient-count
summaries and explicit work/storage bounds. A second workflow wraps TITE-BOIN12
AL or BDA conduct with staggered patient arrivals within fixed-dose cohorts,
pending-endpoint suspension, decision lag, persistent elimination and complete
final ascertainment. BDA requires explicit prior and sampler settings and stores
only compact interim summaries. Luna implemented both modules; root reviewed,
integrated, documented and verified them.

Validation passed 32 affected TITE checks and 10 BayesFactorTTE checks with
warnings treated as errors. The latter include a guard against a positive final
follow-up increment disappearing at a large calendar time. Independent ledger
calculations checked event/censor tie accounting and all six observation rows
from the published TITE patient table. The table's day-315 AL dose recommendation
remains inconsistent with the recovered estimator and movement rule; the guide
and source audit document the discrepancy without claiming adaptive-trial
parity. Both entries retain partial status for remaining documented scope.

The built wheel and sdist contain all 577 committed package source/data files
byte-for-byte. All 1,632 public exports resolve. Five executable examples from
three affected guides passed directly against the wheel with warnings treated
as errors. Package verification took 10.706 seconds, peaked at 114.55 MiB and
reported zero process swaps. The focused test processes peaked at 144.73 MiB
(BayesFactorTTE) and 130.80 MiB (TITE), with zero swaps. Numerical, type-checking
and build processes ran serially under the existing thread limits. Targeted
Ruff, mypy and diff checks passed; no new CI job or dependency was added.

Catalog totals remain 138 entries: 63 implemented, 67 partial and 8 pending.
The publication manifest records the independently verified remote branch
SHAs and refreshed wheel, sdist and all-refs bundle. At the last check of the
previous published checkpoint `aef4a91`, GitHub quality and Python 3.12/3.13
jobs had passed and Python 3.14 was still running; no full matrix success is
inferred from the focused local checks. Existing mixed-license terms remain
included in both distributions.


## September 29 — iBOIN and TITE-BOIN12 operating characteristics

Verified package code is `6de56f1471648e426378240166bfc87b891c4b80`.
Luna implemented both workflows. iBOIN now exposes final isotonic MTD selection
with explicit historical-borrowing, weight, candidate, tie and final-bound
policies, a seeded single-trial runner and serial operating characteristics.
The aggregate runner shares the existing conduct transitions without retaining
cohort decision histories. It reports selection/stopping probabilities,
enrollment and toxicity means, enrollment quantiles and Monte Carlo errors.

TITE-BOIN12 now runs serial AL/BDA calendar trials from joint binary truths,
with Gumbel conversion, explicit fixed/exponential arrival and conditional
uniform event-time policies, independently replayable trial streams and
selection, allocation, stopping and time summaries with Monte Carlo uncertainty.
Both runners preflight bounded work and storage. Undocumented native policies
remain explicit limitations; neither addition establishes full application parity.

The iBOIN worker reports 20 affected checks passing; the integrated final
selector/simulator passes nine focused checks after explicit array-typing
adjustments. All seven final TITE simulation checks pass. Warnings are errors.
Independent integration verifies 36 exact rational isotonic projections
(maximum absolute difference 5.56e-17), all 1,456 small replay histories against
published `61f151f`, and the aggregate summaries from 12 separately replayed
trial seeds. Robust-effective ESS and empty final-bound eligibility also pass.
The exact fixture generator uses only standard-library rational arithmetic
and exhaustive contiguous partitions. Targeted Ruff, formatting, mypy and
diff checks pass. No new dependency or CI workflow was introduced.

The cached wheel/source build and isolated package verification pass:
1,642 public exports, all 580 committed package source/data files byte-identical
in both distributions, seven executable blocks across three affected guides,
all 138 catalog entries and preserved license notices. Package verification
took 11.358 seconds, peaked at 120.73 MiB and reported zero swaps. Final
focused numerical processes peaked at 148.27 MiB; the independent comparison
peaked at 124.80 MiB. Numerical, build and type-checking processes ran serially
under the existing numerical-library thread limits.

Coverage remains 63 implemented, 67 partial and eight pending. iBOIN native
final-selection defaults/reports and TITE categorical/native details remain
open; the latter's published day-315 AL illustration discrepancy remains
documented. Existing mixed-license terms are included in both distributions.

The preceding published `61f151f` checkpoint has independently confirmed
successful GitHub quality and Python 3.12, 3.13 and 3.14 jobs (run 36622145716).
Those hosted results are for that revision. This audit-only commit follows the
verified code before atomic fast-forward publication; the local manifest records
the independently checked remote branch SHAs, refreshed artifacts and new run.


## September 29 — interim calibration and informative TPI priors

Verified package code is `2194af61788a312a8bb0b3ef6e816c0a0190fe65`.
Luna implemented both additions. MERIT searches maximum enrollment and final
integer toxicity/efficacy boundaries under an explicit interim schedule,
persistent stops and survivor-only final pooling. It reports each paper
corner's error, both powers, enrollment and Monte Carlo uncertainty. Adaptive
paths share latent draws across candidate sizes; integer event counts avoid
rounding drift in empirical probability constraints and ties. Work and array
storage are preflighted before advancing the random generator.

TPI accepts caller-specified Beta shape pairs by dose, consistently applying
them to posterior masses, safety, dose-specific tables, trial conduct, final
isotonic point selection and simulation. The simulator shares compact
ordinary/barred-move and safety tables across identical prior rows. This is
an explicit conjugate generalization of the recovered common-prior model;
undocumented native prior calibration is not inferred. Common-prior defaults
and sampled trial outputs are preserved.

The workers pass 11 affected MERIT checks and eight TPI checks. Four focused
MERIT search checks pass again after final work-accounting and real-input
guards. All numerical checks treat warnings as errors. Independent verification
checks 30 posterior references computed with 70-digit Decimal arithmetic and
finite binomial sums, with maximum absolute error 3.89e-16. Six common-prior
configurations (240 trials) reproduce published `78d319f` outputs; a 4,096-trial
heterogeneous-prior simulation agrees with exact probabilities and moments
from 21 enumerated terminal states within the stated sampling tolerance.

Independent MERIT verification reproduces the published no-interim search
and all nine nonempty-interim corner summaries through separate trial-runner
replays, with zero differences in empirical probabilities. These checks verify
implementation and event accounting; they are not guarantees of underlying
power/error after selecting a design using Monte Carlo data. Targeted Ruff,
formatting, mypy and diff checks pass. No new dependency or CI job was added.

The built wheel and source archive match all 581 committed package source/data
files byte-for-byte. All 1,644 public exports resolve, and five executable
examples across four affected guides pass directly from the wheel. License
notices remain included. Package verification takes 11.608 seconds, peaks at
127.94 MiB and reports zero swaps. Final focused numerical validation peaks
at 157.17 MiB; numerical, build and static processes ran serially under the
existing thread limits. No broad local suite was run.

Catalog totals remain 63 implemented, 67 partial and eight pending. TPI tuning,
isotonic intervals and native workflows, and MERIT native pooling/rounding
conventions and reports remain open. Mixed-license terms are unchanged.
At the latest inspection, published `78d319f` has successful hosted quality,
Python 3.12 and Python 3.13 jobs; Python 3.14 is still running. These statuses
are for the preceding revision. This audit-only commit follows verified code;
the local manifest records the new atomic fast-forward publication, independently
verified branch SHAs and hosted run separately.

## BOIN combination accelerated titration checkpoint

At `29e7ad6`, ordinary BOIN combination simulation supports the CRAN 2.7.2
single-patient staircase, first-DLT/upper-right exit and first-cohort top-up.
Prelude counts, endpoints and exit reasons are retained; the total-patient
limit includes earlier staircase visits. A conservative 128 MiB working-state
estimate is checked before allocation or RNG use. The app's separate cap,
moderate-toxicity option, 3+3 run-in and waterfall titration remain open.

Nine original R transition expressions agree exactly with Python, and 100
no-titration trials reproduce every existing result field from the published
`39a67b3` implementation. Seven focused simulation checks pass, as do targeted
Ruff and mypy. Independent comparison took 1.612 seconds, peaked at 121.55 MiB
RSS and reported zero swaps; focused tests peaked at 131.72 MiB with zero swaps.

Cached wheel/source builds pass. The isolated wheel check verifies all 1,644
exports, all 581 committed source/data files in both artifacts, retained
licenses/notices and four executable blocks across both combination guides.
It took 12.133 seconds, peaked at 127.20 MiB and reported zero swaps. Coverage
remains 63 implemented / 67 partial / 8 pending. Local numerical and build
processes were serial; no dependencies or CI workflows were added. Neural
SurvivalContour work remains in separate feature checkouts until reviewed.

Before this publication, the preceding `39a67b3` hosted workflow had passed
quality and Python 3.12/3.13; Python 3.14 was still running. This is not a
complete-matrix success claim. The local publication manifest records freshly
verified branch SHAs and subsequent hosted status separately.


## Five neural survival families checkpoint

At `ec1c213`, DeepSurv, CoxTime, DeepHitSingle, LogisticHazard and PCHazard
provide fitting, point survival prediction and covariate contours through a
bounded NumPy network. The source audit documents full-risk-set Cox objectives,
Breslow ties, source-defined discrete likelihoods and label transforms, and
explicit differences from native stochastic training. Fitted transforms and
weights are immutable; training diagnostics remain visible. There is no claim
of optimizer-stream parity, convergence for every fit or uncertainty intervals.

Independent 85-digit Decimal references cover all five losses, 56 derivatives,
six log-baseline increments and 48 Cox/CoxTime predictions. Maximum differences
are below 2.23e-16. Extreme score shifts, PCHazard logits and a 2,000-profile by
10,000-baseline-event streamed prediction check pass. An overflowing Adam
squared-gradient accumulator now fails explicitly. Ten focused checks pass
with warnings as errors; targeted Ruff, formatting and mypy pass. Focused
validation peaks at 145.84 MiB RSS and reports zero swaps. A separate read-only
review finds no material integration blocker.

Cached wheel/source builds pass. The isolated wheel check verifies all 1,649
exports and exact committed bytes for all 583 package source/data files in
both artifacts, preserved licenses/notices and two executable examples across
the neural and BOIN titration guides. It takes 12.169 seconds, peaks at
114.53 MiB RSS and reports zero swaps. Numerical and build processes remain
serial, with one BLAS thread and no new dependencies or CI workflows.

Catalog totals remain 63 implemented, 67 partial and eight pending.
SurvivalContour remains partial because stratified interval-PH bootstrapping,
remaining forest workflows, broader categorical encoding and full native app
workflows are still open. Mixed-license terms remain unchanged. At inspection,
the preceding `f009c60` hosted workflow has successful quality and Python
3.12/3.13 jobs; Python 3.14 is still running. The local manifest records the
newly published branch SHAs and hosted status separately; no complete-matrix
success is inferred from the earlier jobs.


## Stratified bootstrap and BF-BOIN expansion checkpoint

At `84a3ed9`, shared-coefficient stratified interval-PH models gain coefficient
bootstrapping with required within-stratum or pooled resampling policies.
Weighted draw sizes use accurate sums, repeated sampled rows become frequency
weights, and tapes/counts/failures remain visible. Pooled samples missing an
original group fail explicitly. Covariance and standard errors are conditional
on successful refits. This is a documented statistical extension of the
implemented interval-likelihood target, not a recovered native app default.

BF-BOIN simulation gains optional post-escalation expansion at one dose below
the last cohort actually treated. It stops at the assigned-count cap or first
toxicity closure and retains separate expansion diagnostics. The target and
stopping rules come from the cached BARD expansion guide; asynchronous timing
is an explicit Python policy. The response window and disabled-option random
stream remain unchanged. No BF-BLRM or hidden stage-two calendar extension is
inferred.

Eight affected stratified-model/bootstrap checks and 12 BF-BOIN checks pass
with warnings as errors and one BLAS thread. Targeted Ruff, formatting and
mypy pass. Independent base-R cloglog references reproduce five successful
bootstrap coefficients to 4.98e-7 and covariance to 2.42e-7, with one explicit
unsupported all-censored-group draw. Covariate units 1e-100 and 1e100 and
pre-RNG work rejection pass. Eighty published no-expansion trials reproduce
every existing result field exactly. Sixty expansion trials verify 206 new
assignments, fixed-dose/cap rules and chronological follow-up; two toxicity
closures are independently confirmed at their first qualifying assessments.
The combined root check peaks at 108.52 MiB RSS and reports zero swaps;
focused worker checks peak below 131 MiB.

Cached wheel/source builds pass. Isolated wheel validation verifies all
1,651 public exports, exact committed bytes for all 584 package source/data
files in both artifacts, preserved license notices and both executable guide
examples. It takes 12.265 seconds, peaks at 111.42 MiB RSS and reports zero
swaps. Catalog counts remain 63 implemented, 67 partial and eight pending.
No dependencies, broad local test runs or CI workflows were added. Remaining
native workflow and mixed-license limitations stay documented.

The preceding `cf802a2` hosted quality job passed. Its Python 3.13 runner
received a shutdown signal at 21:44:17 UTC on September 29; pytest had shown
progress through 12% without an assertion failure before cancellation. The
other two matrix jobs were cancelled. This is incomplete hosted validation,
not a full pass. The local manifest records the new publication and subsequent
hosted run separately.


## BCHM and aPCoA analysis-plot checkpoint

Code revision `52b12a517137f80d4db85fcb2841889fb7ceb946` adds BCHM subgroup
clusters, posterior means/HPD intervals and posterior density plots. It also
adds opt-in aPCoA group data ellipses and medoid/member connectors, plus a
reusable geometry result. Existing aPCoA plot defaults remain unchanged.
Luna implemented both components; root reviewed source contracts, integrated
public exports and fixtures, and checked the resulting package.

Twenty-four focused tests pass across the two components. Independent R
references cover 18 native HPD intervals, six density bandwidth cases, 3,072
direct Gaussian ordinates, 208 ellipse vertices and ten medoid tie cases.
The integrated check's maximum absolute differences were `5.56e-15` for direct
Gaussian densities and `9.11e-15` for ellipse vertices; all medoids matched.
Native FFT density interpolation differs by up to `0.000800597` and is an
explicitly documented evaluation difference. Source rounding, collapsed
ellipses, tiny/large units, tight translated groups and materially invalid
Gram distances are checked. An unrepresentable density peak raises clearly.
Both rendered previews were visually inspected. Targeted Ruff, formatting and
mypy checks pass. No new CI workflow, broad local numerical run or dependency
installation was added.

The final integrated numerical/plot check took 0.538 seconds, peaked at
155.83 MiB resident memory and reported zero swaps. Numerical processes were
serial and numerical-library thread counts were fixed at one. The cached
wheel/source builds pass. An isolated wheel check verifies all 1,665 public
exports, four examples across two guides, preserved license notices, and byte
identity for all 588 committed package source/data files in both archives.
It took 13.231 seconds, peaked at 146.16 MiB and reported zero swaps.

Coverage remains 63 implemented, 67 partial and 8 pending entries. BCHM's
cached scientific workflow is now covered by the fit, summaries and three
plots; its app CSV/PDF/MCMC exports have unavailable server/output contracts.
aPCoA file/formula and complete app parity remain open. Partial status is
retained, and the distribution's mixed-license limitations remain explicit.
The local artifact manifest records the independently verified master/main/
development SHAs after publication; package contents correspond to the code
revision above and this final audit-only commit changes no numerical code.


## Subgroup ESS and delayed-toxicity observation checkpoint

Verified package revision `11a91c574f5aa30ba5013fcb8df2a3664efada4c` adds
optional BaCIS per-replication/subgroup ESS, stable arithmetic means and
Monte Carlo errors. The existing simulation defaults and random streams are
preserved. The underlying ESS matches the archived software's fixed-success-
count beta variance equation, with its previously documented admissibility
correction; it is not a claim to reproduce the paper's different mean/variance
description or published simulation tables. Undefined matches fail with
replication/subgroup context. A one-replication MCSE is explicitly unavailable.

Dose Schedule Finder gains explicit onset/adjudication records and an as-of
observation helper. Qualifying events are backdated to onset once known, while
future adjudications remain masked. Actual delivered treatment is retained
separately from the administrations entering the event likelihood. Pending
earlier onsets prevent final readiness, and unresolved episodes beginning
within the risk horizon may require later ascertainment. The caller supplies
clinical adjudication; no low-grade episode generator, exact day-14 clinical
rule, or within-patient treatment policy is inferred.

Luna implemented both components. Twenty-three focused checks pass with
warnings treated as errors. Independent R polynomial calculations reproduce
20 ESS values and five subgroup mean/MCSE summaries; maximum ESS discrepancy
is `9.31e-13`. Fourteen hand-calculated observation looks pass at time scales
`1e-150`, `1` and `1e150`, covering future-information masking, resolved and
qualifying episodes, earlier-event revision, horizon crossings and treatment
history. A two-administration event likelihood matches `log(.03)-.125` to
`1e-14`. Targeted Ruff, formatting and mypy pass.

The integrated reference check took .148 seconds, peaked at 121.50 MiB RSS
and reported zero swaps. Focused measured worker runs remained below 130 MiB.
Numerical processes ran serially with numerical-library thread counts fixed
at one. The first package example check correctly rejected an inadmissible
ESS from a very short posterior sample. The guide now uses an independently
validated small example and explains the admissibility limitation; no failed
replications were discarded and no numerical safeguard was weakened.

Cached wheel/source builds pass. The final isolated wheel check executes all
three examples across the two guides, verifies all 1,669 public exports and
exact committed bytes for all 589 packaged source/data files in both
artifacts, and checks preserved license notices and excluded native binaries.
It took 11.468 seconds, peaked at 127.39 MiB RSS and reported zero swaps.
No broad local suite, dependency installation or CI workflow was added.

Coverage remains 63 implemented, 67 partial and eight pending entries. Native
workflow limitations and mixed-license terms remain explicit. The preceding
`03804d5` hosted quality job and all three Python 3.12–3.14 jobs are now
independently confirmed successful. The subsequent `ceafd12` hosted run found
only formatting violations in two executable guide examples before reaching
the numerical jobs. Those examples were reformatted, and repository-wide Ruff
lint and formatting checks now pass. No numerical code or checks were changed.
The local manifest records fresh independent master/main/development branch
verification and the replacement hosted run; its status is reported separately
from the completed preceding run.


## WFMM custom transforms and U2OET adaptive precision checkpoint

Verified package revision `83f2df9db833d64f47e3b27d253cbbc3d38257a3` adds
WFMM square analysis/synthesis inverse pairs, retaining the existing orthogonal
API. Reconstruction and data-space covariance use the supplied synthesis
matrix. Two-sided inverse checks and a scale-stable condition estimate reject
unstable pairs; variance weighting avoids premature overflow under reciprocal
uniform scales. PCA, rectangular retained-component rules and native workflows
remain separate gaps.

U2OET gains a standalone adaptive PDS/CMI/hybrid fitter. It continues complete
chain states, monitors each chain's four corner-utility batch-means MCSE/SD
ratios, and returns the achieved precision, draw-cap termination and separate
classical split-Rhat diagnostics. Work and live-array estimates are bounded
before sampling. Constant-utility precision is undefined and cannot pass.
The undocumented batch construction and extension schedule are explicit Python
choices. GAO and fixed-budget trial drivers are outside this addition.

Luna implemented both components; root reviewed and integrated them. Twenty-two
focused tests pass with warnings treated as errors (18 WFMM and four U2OET).
The condition-guard follow-up also reran its twelve affected tests. Independent
base-R comparisons cover 1,199 WFMM transform/reconstruction, posterior-summary
and covariance values, with maximum scaled discrepancy `5.49e-15`, and 252
U2OET precision/Rhat diagnostics at utility scales `1e-200`, `1` and `1e200`,
with maximum normalized absolute discrepancy `2.89e-15`. These are independent
mathematical references, not native executable comparisons.

A bounded end-to-end U2OET posterior run extended two chains from 512 to 1024
retained draws, stopped before the 4096-draw cap, and met the .05 precision
target with maximum corner ratio .047268 and maximum corner split-Rhat
1.000069. Its efficacy intercept and probability means agreed with existing
independent R quadrature references within 2.363 and 2.145 estimated MCSEs.
This does not establish every parameter's precision or trial operating
characteristics. See the component audits for settings and remaining scope.

Repository-wide Ruff lint and formatting and targeted mypy checks pass.
Numerical processes ran serially with numerical-library thread counts fixed
at one. The largest measured focused run peaked at 140.61 MiB RSS; independent
WFMM and U2OET comparisons peaked at 123.83 and 123.27 MiB, and the posterior
integration run at 121.34 MiB. All reported zero swaps. No broad local numerical
suite, dependency installation or new CI workflow was added.

Cached wheel/source builds pass. Isolated wheel validation executes both new
guide examples, verifies all 1,671 public exports and exact committed bytes
for all 590 packaged source/data files in both archives, and checks preserved
license notices and excluded native binaries. It took 11.967 seconds, peaked
at 122.55 MiB RSS and reported zero swaps. Coverage remains 63 implemented,
67 partial and eight pending entries; mixed-license limitations remain explicit.

The preceding `fcffaf8` hosted quality job and all three Python 3.12–3.14 jobs
are independently confirmed successful. The local artifact manifest records
fresh master/main/development publication verification and the new hosted run
separately. This final audit-only commit changes no packaged numerical code.


## U2OET adaptive calendar and original GAO checkpoint

Verified package revision `68f1e8d880801edba90417dbbc8244fb96634be5` connects
adaptive corner-utility precision to PDS/CMI/hybrid calendar trials. Each new
complete/toxicity-only count state, including final follow-up, must meet the
requested precision target. Cached states reuse their posterior and compact
diagnostics; an insufficient draw cap raises before that fit can determine an
assignment. Whole-trial work and per-fit storage estimates are checked before
splitting the caller's random stream. The fixed-budget route remains available.

The original 2010 GAO probability model has a separate public interface for
centered dose grids and endpoint-specific signed interactions. Its negative
interaction constraint is checked across all dose pairs and thresholds, with
bounded Decimal fallback for material cancellation. It returns ordinal joint
probabilities, utilities and complete/toxicity-only likelihoods. It is distinct
from the existing 2017 comparison model; original-prior posterior fitting,
calibration and native parameter-file mapping remain open.

Luna implemented both components. Twenty-four focused checks pass with warnings
treated as errors: 14 adaptive/fixed calendar and precision checks, and ten
new/existing GAO probability checks. Independent base-R references reproduce
680 joint cells, 60 likelihoods and 100 utilities, with maximum absolute error
`4.84e-13`. Translation and dose-unit rescaling through `1e±150` also pass.
These are mathematical references, not native executable parity claims.

The public calendar example was replayed through the standalone fitter from
its recorded posterior seed. All posterior summaries and precision diagnostics
matched exactly for three cached interim decisions and the final analysis.
The interim fit used 512 draws per chain and the final fit extended to 4096;
maximum corner MCSE/SD ratios were .047138 and .046497, below the .05 target.
This reduced-model example checks integration, not full trial operating
characteristics or precision for every model parameter.

Repository-wide Ruff lint/format and targeted mypy pass. Numerical processes
ran serially with library thread counts fixed at one. Final focused worker runs
peaked below 132 MiB RSS; the independent comparison and calendar replay peaked
at 120.02 and 123.55 MiB. All reported zero swaps. No broad local numerical
suite, dependency installation or CI workflow was added.

Cached wheel/source builds pass. Isolated wheel validation executes both new
guide examples, verifies all 1,674 public exports and exact committed bytes for
all 591 package files in both archives, and checks preserved license notices
and excluded native binaries. It took 15.838 seconds, peaked at 110.92 MiB RSS
and reported zero swaps. Coverage remains 63 implemented, 67 partial and eight
pending entries. Mixed-license terms and remaining workflow limits are explicit.

The preceding `dfcc002` hosted quality job and all Python 3.12–3.14 jobs are
independently confirmed successful. The local artifact manifest records fresh
master/main/development publication verification and the new hosted run
separately. This audit-only commit changes no packaged numerical code.


## Original GAO inference and WFMM prediction checkpoint

Verified package revision `7a4553f4350a046e1fc7d17e7fe32919455002f5` adds
posterior fitting for the original 2010 GAO model. Caller-supplied Gaussian
coordinates cover alpha, log-lambda and signed gamma, jointly restricted to
parameter values valid on every supplied dose pair and threshold. Association
has a separate uniform prior or an explicit fixed value. Endpoint-block
elliptical slice updates and uniform Metropolis proposals retain read-only
parameter/probability draws, original dose grids and diagnostics. Native
prior-center calibration and executable sampler parity remain separate work.

WFMM gains posterior prediction for explicit future fixed and random designs.
Existing random levels reuse retained conditional draws; each new level is
sampled once per posterior state and shared by all rows loading that level.
Optional residuals are independent by row in coefficient space. Existing
inverse reconstruction and summaries then propagate uncertainty into curves.
The independent-level model and prediction target are explicit; native
prediction-file formats and PCA/retention semantics are not inferred.

Luna implemented both additions. Twenty-one focused tests pass with warnings
as errors (16 new/existing GAO and five WFMM checks), along with repository-wide
Ruff lint/format and targeted mypy. Independent base-R quadrature checks the
joint restricted prior and complete/toxicity-only posteriors plus an analytic
uniform-correlation posterior. All 31 summaries agree within 1.891 estimated
Monte Carlo errors; maximum checked split-Rhat is 1.00454. These reduced fits
do not establish mixing for every full-model analysis.

Independent WFMM finite-mixture references cover 192 deterministic conditional
curve values and 168 predictive means/covariances. They include shared new
levels, distinct variance strata, varying posterior variance draws and a
nonorthogonal synthesis matrix. Stochastic discrepancies stay within 2.081
estimated Monte Carlo errors. This checks prediction conditional on supplied
posterior draws, separately from earlier fitting validation.

Numerical processes ran serially with library threads fixed at one. Focused
worker runs peaked at 132.11 MiB RSS or less; independent WFMM and GAO checks
peaked at 125.23 and 116.58 MiB and took .266 and 32.056 seconds. All reported
zero swaps. No dependency installation, broad local numerical suite or new CI
workflow was added.

Cached wheel/source builds pass. Isolated wheel verification executes all three
examples in the two new guides, resolves all 1,678 public exports, and matches
all 593 committed package files in both archives. License notices remain intact
and native binaries/raw research files are excluded. It took 16.617 seconds,
peaked at 128.02 MiB RSS and reported zero swaps. Coverage remains 63
implemented, 67 partial and eight pending; these additions deepen two partial
entries without claiming full native workflow coverage.

The local manifest separately records independent master/main/development
publication verification, the latest fully passed hosted checkpoint and the
fresh hosted run. This audit-only commit changes no packaged numerical code.

## Profile inference and waterfall titration checkpoint

Verified package revision `408766c45f4f701571e8c9de0b71c1a61c149f6d` adds the
source-defined Proportional Density likelihood-ratio test for an arbitrary
beta under explicitly asserted common censoring. The intercept is profiled;
confidence limits invert the same one-degree-of-freedom test. Stable time
scaling, both confidence tails, likelihood residual checks and bounded work
protect the numerical calculation. Unequal-censoring bootstrap calibration
and incidence inference are separate from this new interface.

BOIN waterfall replay and serial simulation now support the original R
first-subtrial titration branch. A complete staircase draw prefix is retained,
but only visited patients count as assignments. The reached cell is topped
up once; subsequent subtrials use ordinary cohorts. Actual counts govern
enrollment stopping, including the source's possible budget overshoot and
the suffix exclusion applied at the last selected dose. The Python result
retains observations that the native wrapper can drop. Native app-specific
titration caps and 3+3 run-in remain open.

Luna implemented both additions. Fourteen focused existing/new tests pass
with warnings as errors, along with targeted Ruff, formatting and mypy.
Independent base-R GLMs and profile inversion match 90 estimates/tests/limits
across five cases and 15 intervals within 2.003e-13; 30 endpoint LR residuals
are below 8.882e-15. Extreme time units, arm exchange and confidence tails
are checked. Seven original BOIN 2.7.2 workflows reproduce 43 assignments,
11 subtrial snapshots and final exclusion masks, with explicit accounting for
the native dropped-count defect. All seven final source-contour selections
also match independently.

Numerical checks ran one process at a time with library threads limited to
one. Focused worker runs peaked below 132 MiB RSS and reported zero swaps.
No dependency installation, broad local numerical suite or new CI workflow
was added. The preceding published `729abd8` quality and all Python 3.12–3.14
jobs were independently confirmed successful at 01:45 UTC on September 30.

Cached wheel/source builds pass. Isolated wheel verification executes all
four examples in the two affected guides, resolves all 1,680 public exports
and matches all 594 committed package files in both archives. License notices
are preserved and native binaries/raw sources are excluded. Verification
took 10.97 seconds, peaked at 114.8 MiB RSS and reported zero swaps. Counts
remain 63 implemented, 67 partial and eight pending. These additions improve
method coverage without claiming complete native application equivalence.

The local manifest records independent publication verification and fresh
hosted validation separately. This audit-only commit changes no packaged code.


## September 30 STPLAN matched-pairs publication

Package-code revision `59d5dbdb43e8b801dbea2ea49333e9a348e36fc1` adds
the archived Miettinen paired-binary power calculation, bounded pilot-based
effect/sample-size/significance inversion and explicit no-pilot initial-size
heuristics. It preserves the native dominant-tail convention and unrounded
pilot recommendations, while rejecting a native squared inverse that returns
107.204 pairs for requested power 0.01 but achieves approximately 0.165025.
The inactive native menu status and limitations are documented. STPLAN remains
partial for automatic native planning and session/report workflows.

Thirteen focused tests passed with warnings treated as errors, including
12 new original-Fortran/independent-R cases and existing inverse-planning
references. Checks also exercise extreme count scaling, tiny variance factors,
near-boundary no-pilot geometry and allocation preflight. The integrated test
process took 2.978 seconds, peaked at 144.0 MiB RSS and recorded zero swaps.
Targeted Ruff, format checks and mypy passed; independent final review found
no substantive blockers. No full local suite or CI expansion was added.

The wheel and source archive match all 595 committed package files. The wheel
exposes 1,683 public names, and all six Python examples in the matched-pairs
and inverse-planning guides execute from that isolated wheel. Package verification
took 11.802 seconds, peaked at 117.16 MiB RSS and recorded zero swaps.
Catalog status counts remain 63 implemented, 67 partial and 8 pending.

Before this publication, GitHub master/main/development were independently
verified at `2b53d37689ed4c274379bf18142866dad972ff86`. That checkpoint's
[hosted run](https://github.com/dx-li/mdanderson-stats/actions/runs/36657466078)
passed quality and Python 3.12, 3.13 and 3.14 jobs. The ignored local artifact
manifest records the new remote SHAs and latest hosted-run status after push.


## September 30 mTPI and Phase I/II inference publication

Package-code revision `42045e9da4dbe252c8e0ff19633dd0a60a2b0f9b` adds mTPI
posterior intervals by drawing independent beta probabilities and isotonic-
transforming each joint draw. It reports marginal bounds and optional joint
samples under explicit grid, weight and empirical-quantile conventions. The
paper's Table-3 common-prior sensitivity now applies consistently to decisions,
safety, selection, simulation and intervals, with Uniform-calibrated losses
held fixed. Default decisions and positional API arguments are preserved.
Prior-specific penalty recalibration remains unsupported by the available source.

The six-dose Parallel Phase I/II model now has a separate bounded importance
fitter. Full-mixture density weighting and the original evidence-plus-60-summary
stopping rule are retained. The implementation reports raw-integral errors,
posterior-ratio errors and whether the stopping criterion passed. It uses an
analytic Hessian and a bounded optimizer rather than the native numerical
proposal construction. It integrates with the source decision functions;
the existing calendar driver continues to use elliptical-slice fitting.

Luna implemented these methods in isolated checkouts; root and a read-only
reviewer checked the source contracts and integration. Twenty-one focused mTPI
checks pass, including exact beta identities, fractional-prior quadrature,
closed-form two-dose projected moments/CDF, default screenshot decisions and
seeded replay. Five focused importance checks pass, including direct paired-
ratio Monte Carlo errors and an independent R response-mean comparison. The
maximum response-mean difference was 0.0010111, or 1.04 combined Monte Carlo
standard errors. Existing model/decision compatibility checks also passed.
Targeted Ruff, formatting and mypy pass. The largest measured focused run used
136.16 MiB RSS and reported zero swaps. Numerical work ran serially with library
threads limited to one; no full local suite, dependency installation or new CI
workflow was added.

Cached wheel/source builds pass. Isolated wheel verification matches all 597
committed package files in both archives, resolves all 1,687 public exports and
executes all three new guide examples. Notices are retained, and ignored native
sources/binaries are excluded. Verification took 11.98 seconds, peaked at
114.39 MiB RSS and reported zero swaps. Counts remain 63 implemented, 67 partial
and eight pending; full native software coverage is not claimed.

Before publication, remote master/main/development were independently verified
at `5634ec317e756cf9c27557e14065163649ed99fb`. Its hosted quality, Python 3.12
and Python 3.13 jobs have passed; Python 3.14 is still running at this check.
The latest entirely passed hosted checkpoint remains `2b53d37`. The ignored
artifact manifest records new remote SHAs and hosted status after publication.
This audit-only commit does not change the verified package code.


## September 30 calendar importance and survival-rank splits

Package-code revision `a83936b1f4bc068730c6f501e7e3e76795f4e511` connects the
six-dose importance fitter to actual interim and final calendar decisions.
Observed tallies reuse the latest fit, while current snapshots retain endpoint
availability. Separate data/posterior streams, pre-seed whole-trial work bounds,
actual uncached work counts and per-analysis evidence/error/convergence fields
make the behavior inspectable. The existing MCMC backend remains the default.
Native cap-return behavior can use an unconverged estimate; the result exposes
that flag and its Monte Carlo errors.

Survival forests now offer Hothorn–Lausen standardized rank-score splitting,
including maximum-rank time ties and expanded bootstrap duplicates. Numeric and
categorical candidates use their existing routing and tie conventions. The
unchanged pinned `coin` R transform independently supplies 23 score references;
a concrete four-row reconstruction documents RF-SRC's alternative-branch
indexing discrepancy. This implements the documented statistical criterion,
without claiming exact behavior of that native branch. Ordinary log-rank
splitting remains the default.

Luna implemented both changes in isolated checkouts. Root and a read-only
reviewer checked the contracts and integration. Twenty-two focused checks pass
with warnings as errors: ten calendar checks include real interim/final
importance analyses and explicit unchanged-tally reuse, while twelve forest
checks include exhaustive numeric/categorical split references and input-row
permutation. Targeted Ruff, formatting and mypy pass. The final integrated
calendar run took 2.601 seconds and peaked at 145.95 MiB RSS. The forest's serial
reference/test/static/example sequence peaked at about 161.5 MiB. All measured
processes reported zero swaps; numerical work ran one process at a time with
library threads limited to one. No full local suite, installation or CI expansion
was added.

Cached wheel/source builds pass. Isolated wheel verification matches all 597
committed package files in both archives, resolves all 1,687 public exports and
runs the two new public guide examples. Notices remain included; ignored native
sources and binaries remain excluded. Verification took 10.829 seconds, peaked
at 128.56 MiB RSS and reported zero swaps. Catalog counts remain 63 implemented,
67 partial and eight pending: these extensions improve two partial entries.

Before publication, remote master/main/development were independently verified
at `54ae04834859873d38f37a5832b2e608f925b24b`. Its hosted quality and Python
3.12/3.13 jobs passed; Python 3.14 was still running. The older `5634ec3` run's
Python 3.14 job was cancelled. No overall success is claimed for either run.
The ignored artifact manifest records fresh publication verification and the
new hosted run separately. This audit-only commit changes no packaged code.


## September 30 six-dose operating characteristics and Brier splits

Package-code revision `400fe6cb30a5b382e74c51b054204f2bcef5ea22` adds bounded
serial six-dose calendar summaries for both posterior backends. Direct per-trial
seeds support replay; early/final/no-selection partitions preserve source
eligibility quirks. Generated endpoint truth and observed endpoint counts have
separate denominators. Trial-level Monte Carlo errors, enrollment/time summaries
and cached-fit convergence/work diagnostics expose uncertainty and computation.
Only one trial is retained. Allocation preflight accounts for an old fit
coexisting with its replacement and scratch arrays, plus final aggregate copies;
it is explicitly an estimate, not a total process-RSS guarantee.

Survival forests now offer scalar `bs.gradient` splitting with RF-SRC's selected
prior event-grid point, shared failure weights and strict censor-survival left
limits. The default log-rank rule is preserved. Compiled unchanged pinned C
helpers and an independent R ledger agree on 17 gradients across five cases
and the corresponding candidate scores. Numeric/categorical candidates and
bootstrap row multiplicities retain existing forest conventions. The helper
uses linear node workspace; complete native forest/RNG equivalence is not claimed.

Luna implemented both additions in separate checkouts. Root and a read-only
reviewer checked source contracts, scientific behavior and integration. Ten
focused calendar/OC checks and 25 focused forest checks pass. After the memory
estimate adjustment, the four OC checks passed again with warnings as errors:
3.339 seconds, 142.83 MiB peak RSS, zero swaps. The forest's serial validation
sequence took 3.708 seconds, peaked at 172,326,912 bytes (about 164.3 MiB), and
reported zero swaps. Targeted Ruff, formatting and mypy pass. Numerical work
ran serially with numerical-library threads limited to one; no full local
suite, new dependencies or CI expansion was added.

Cached wheel and source builds pass. An isolated interpreter matched all 598
committed package files in both archives, resolved all 1,689 public exports,
verified retained license notices and ran both new public guide examples.
It took 10.56 seconds, peaked at 129.0 MiB RSS and reported zero swaps. Ignored
native source caches and binaries remain excluded. Counts remain 63 implemented,
67 partial and eight pending; these are coverage additions within partial entries.

Before publication, remote master/main/development independently matched
`77a5fb4543706a900e709e54a626e15051f2c3a3`. Its existing hosted validation
run `36663751699` passed quality and Python 3.12, 3.13 and 3.14. The ignored
artifact manifest records the new verified remote SHAs and new hosted run
separately. This audit-only commit changes no verified package code.


## September 30 bundled EasyCellType reference data

Package-code revision `70d92b6453b892c7f7a266865df718f0665f5231` adds the
three reference tables embedded in the pinned EasyCellType 1.5.4 author package.
A public loader selects Human/Mouse and optional tissues without R or downloads.
It retains row order and duplicates, verifies the compressed file identity and
full source-row count, and returns stable provenance from ordinary or zipped
wheel resources. The source's Entrez IDs remain unchanged; versioned symbol
conversion and native plotting/app parity remain separate gaps.

Luna implemented the addition in an isolated checkout. A read-only review and
root integration checks found no remaining blocker. Nine focused bundled/local
reference and Fisher checks pass, with targeted Ruff, formatting and mypy.
The worker validation peaked at 137.30 MiB RSS with zero swaps. An independent
streaming comparison verified original R-object, plain CSV and gzip hashes,
deterministic gzip headers, and exact decompressed equality for all 236,219
cached author-export rows. It took 0.018 seconds, peaked at 29.77 MiB and
reported zero swaps. Applicable Artistic-2.0 terms and Clustermole attribution
are retained; individual upstream database releases and all contribution-level
licenses are not recorded in the source snapshot. The data are not represented
as current releases or relicensed as MIT.

Cached wheel/source builds pass. An isolated interpreter matched all 603
committed package files in both archives and resolved all 1,690 public exports.
The public annotation example executes from the wheel, and all three bundled
references load with exact expected source/species row counts and stable
filenames. Licenses/notices remain included, and ignored native caches/binaries
remain excluded. This check took 12.384 seconds, peaked at 115.50 MiB RSS and
reported zero swaps. Numerical checks ran serially with library threads limited
to one. No full local suite, dependency installation or CI expansion was added.
Counts remain 63 implemented, 67 partial and eight pending.

Before publication, remote master/main/development independently matched
`b25b5c3b1b4da643f0c5b0e606a29a21a90ed616`. Its hosted quality and Python
3.12/3.13 checks passed; Python 3.14 remained in progress. The latest completely
passed hosted checkpoint observed was `77a5fb4` (run `36663751699`). The ignored
artifact manifest records fresh remote verification and the new hosted run
separately. This audit-only commit changes no verified packaged code.


## September 30 IPDfromKM report diagnostics

Package-code revision `00e871d0697338629011ff1ac76ac421face5be3` adds the
original report's rounded precision summaries and two-sample KS diagnostic.
It preserves unrounded observed values, paired missing-row omission with
retained indices, and the native signed maximum under an explicit name beside
the true maximum absolute error. Exact tied-label probabilities use integer
counts to retain tiny tails; larger samples use the limiting KS distribution.
The nominal p-value is not calibrated inference for paired reconstructed curves.
Existing reconstruction and its unrounded error metrics remain unchanged.

Luna implemented the component, followed by root integration and read-only
review. Ten independent R reference scenarios and five focused checks pass.
A separate binary64 holdout matches R 4.4.1 rounding on all 6,000 values;
identical input bytes exclude decimal-parser differences from the comparison.
The actual 99-point separation tail agrees with `2 / choose(198, 99)` (about
`8.79e-59`) under a relative check. The final focused run took 1.795 seconds,
peaked at 148.70 MiB RSS and reported zero swaps. Targeted Ruff, guide formatting
and worker mypy pass. Applicable IPDfromKM and R GPL terms are retained.

Cached wheel/source builds pass. An isolated interpreter matched all 604
committed package files in both archives, resolved all 1,692 public exports,
verified retained license notices and executed the public diagnostic example.
Verification took 10.752 seconds, peaked at 129.28 MiB RSS and reported zero
swaps. Numerical work ran serially; no new dependencies, full local suite or
CI workflow were added. Counts remain 63 implemented, 67 partial and eight
pending. Image digitizing and native graphics remain separate IPDfromKM gaps.

The preceding `c2577d7` hosted run failed only because the new EasyCellType
guide's code block needed formatting. The correction was published at
`4c4a57362a4e170b75cda0f3a438cf14d738038b`, independently verified on
master/main/development. Its hosted quality and Python 3.12/3.13 jobs passed;
Python 3.14 was still running before this publication. The ignored artifact
manifest records the new publication and hosted status separately. This
audit-only commit changes no verified packaged code.

## October 3: SYNERGY wild-bootstrap resampling

Luna implemented the source-defined Mammen generator and serial marginal-
baseline/REML refits; a separate Luna agent produced an independent base-R
reference. Root reviewed and integrated the result at `89c1268`. Five fixed
multiplier tapes cover 13 observations, including repeated dose pairs. The
maximum absolute departure difference against R is `1.30e-10`.

Ten focused tests pass with warnings treated as errors. Response scaling by
`1e-200` and `1e200` preserves the normalized mean and sample SD within
`1.06e-10` and `6.53e-12` absolute error respectively. The integrated check
used 2.106 seconds, 149.36 MiB peak RSS and zero process swaps. Targeted Ruff,
formatting and worker mypy checks pass. Numerical processes ran serially.
No new CI workflow or full local test run was added.

The wheel and source archive match all 605 committed package files; all 1,694
public exports resolve, and the self-contained bootstrap guide runs from the
isolated wheel. Package verification used 11.614 seconds, 114.16 MiB peak RSS
and zero process swaps. Artifacts retain the existing mixed-license notices.

Bootstrap samples and descriptive sample SD are usable now. The source's exact
normal-interval centering/denominator and four older parametric surfaces remain
unverified. The entry stays partial, and catalog counts remain 63 implemented,
67 partial and eight pending. Bounded new primary-source searches did not
recover the missing fitted risk-model parameters; the K-COMPASS predictor list
was clarified without inventing coefficients or absolute survival predictions.

Before this publication, remote master/main/development were independently
verified at `b67c028`. The artifact manifest records the subsequent publication
SHA, rebuilt artifact hashes, Git bundle verification and new hosted run state
after the remote push is independently checked.

## October 3: three additional BCSTTE family fitters

Luna implemented the guide's Gamma, inverse-Gamma and log-logistic models,
using explicit proper correlated Gaussian priors on log shape and log scale.
The fitters support noninformative right censoring, including zero follow-up
and prior-only data. Complete-data draws retain paired CDF evaluations for
Johnson's diagnostic; censored fits do not invent a native diagnostic.
Root integrated public exports and documentation at `89f6082`.

Independent base-R quadrature covers six complete/censored cases. All 72
posterior moment, covariance and CDF summaries agree within 3.232 estimated
batch-means Monte Carlo errors; maximum split R-hat is 1.005445. A separate
Luna review and root review corrected density normalization, the event-only
time Jacobian, tiny-shape Gamma tails and large-shape cancellation before
publication. Nine focused tests pass with warnings treated as errors.
Twelve unit changes at factors `1e-200` and `1e200` preserve centered draws
within `5.73e-14` and expected likelihood shifts within `4.55e-13`.
Integrated checks peak at 145.12 MiB with no process swaps. Targeted mypy,
Ruff lint and formatting pass.

Cached wheel and source builds pass. An isolated wheel resolves 1,698 public
exports, matches all 606 committed package files in both artifacts, preserves
license notices and runs the new public guide example. This verification used
11.258 seconds, 132.53 MiB peak RSS and no process swaps. No full local suite,
new dependency or CI workflow was added. Catalog status remains partial;
counts remain 63 implemented, 67 partial and eight pending.

The preceding `9f6c1eb` hosted run passed all four jobs on attempt two; the first
attempt failed during a GitHub API request in setup, before tests. No CI change
was needed. The local manifest records the new publication and hosted state
separately. The independently sourced log-odds-rate core and its posterior
integration remain in their isolated development checkouts at this checkpoint.

## October 3: seventh BCSTTE distribution family

Luna implemented the generalized log-odds-rate model and posterior fitter; root
reviewed the numerical contract, integrated public exports and updated coverage
at `4509a8a`. Shen and Thall's primary model resolves the guide's missing factor
of c. The model includes log-logistic survival at c=1 and the Weibull limit as
c approaches zero. The explicit Gaussian prior on all three log parameters is
a Python contract, not a recovered native default.

Independent three-dimensional R quadrature and four serial Python chains agree
on all 34 posterior summaries within 2.619 batch-means Monte Carlo errors;
maximum split R-hat is 1.000712. Root identified and corrected a transposed R
Cholesky factor before comparison against the final targets; preliminary
comparisons against the wrong covariance were discarded. Quadrature sensitivity
is at most 1.70e-6. The final comparison used 25.443 seconds and 139.58 MiB RSS.

All 22 focused integration tests pass. Sixteen complete/censored unit changes
at factors 1e-200/1e200 preserve centered draws within 5.73e-14 and expected
likelihood shifts within 4.55e-13. These checks peak at 148.38 MiB with zero
process swaps. Targeted lint, formatting and type checks pass.

Cached wheel and source builds pass. An isolated interpreter matches all 607
committed package files in both archives, resolves 1,700 public exports, checks
license notices and executes the public log-odds-rate example. Package checking
used 14.050 seconds and 112.22 MiB RSS with zero process swaps. No full local
suite, dependency installation or new CI workflow was needed.

All seven advertised distributions now have fitting workflows. Censoring in
the earlier family fitters, native prior defaults, the censored Johnson
diagnostic, rank/trim conventions and native reports remain separate gaps.
Counts remain 63 implemented, 67 partial and eight pending.

The previous 6319a4f hosted run has passed quality and Python 3.12/3.13; its
Python 3.14 job was still running at the last observation. The local manifest
records this publication's independently verified master/main/development SHAs
and hosted state separately from the latest fully passed hosted checkpoint.

## October 3: censored exponential/Weibull and survival design inputs

Luna added noninformative right censoring to the exponential and fixed-shape
Weibull conjugate fits, and to joint unknown-shape Weibull sampling. The
conjugate update counts events in posterior shape and all follow-up exposure
in posterior rate. Zero-time censors contribute likelihood one. Proper
all-censored posteriors are explicitly a Python extension beyond the native
minimum-one-event input rule. Censored fits do not invent a Johnson diagnostic.

An independent direct R quadrature reference covers mixed and all-censored
unknown-shape Weibull posteriors: 21 summaries agree within 1.501 batch-means
Monte Carlo errors, maximum split R-hat 1.001993, and quadrature/domain changes
at most 7.20e-11. The Python comparison used 1.017 seconds and 131.77 MiB RSS.
The shared event validator now accepts Boolean lists longer than 16 observations
and rejects nested inputs before materialization. Existing public Weibull result
field ordering is preserved.

The success-criteria adapter accepts expected total events and treatment
allocation directly, deriving the source's log-hazard-ratio standard error and
using the existing normal operating-characteristic kernel. Fractional expected
event counts are accepted; no accrual or censoring generator is inferred.

Root integrated the batch at `1da08a2`. All 48 focused tests passed with warnings
treated as errors; validation used 2.609 seconds, 203.42 MiB peak RSS and zero
process swaps. Targeted lint/format and mypy over all five changed statistical
modules passed. The latter also resolves the worker-invocation discrepancy;
no unresolved type-check failure remains at this integrated checkpoint.

Cached builds and isolated wheel checks passed: all 607 committed package
files match both archives, all 1,701 public exports resolve, license notices
are retained and five relevant example blocks across four guides execute. The
success-calibration check selected the changed normal/survival example and
supplied the NumPy import shown earlier in that guide. Package checking used
11.603 seconds, 118.09 MiB RSS and zero process swaps. No full local suite,
new CI workflow or dependencies were added.

Catalog counts remain 63 implemented, 67 partial and eight pending. Lognormal
censoring is being implemented and independently referenced in isolated
checkouts; it is not included in this published checkpoint. Native priors,
censored diagnostics and remaining native workflow conventions remain explicit
gaps. Remote publication and hosted checks are recorded separately in the
local artifact manifest after an independently verified push.
