# Community publication checkpoint — 2026-09-28

Latest verified package checkpoint: `02efdac` adds PLBARPO control operating
characteristics, Dose Schedule Finder and Multc calendar trials, and U2OET GAO
probabilities/likelihoods to the preceding APIs. Local `master` contains this
validated checkpoint. Fresh
read-only checks still show GitHub `master` and `main` at `45b6e307`; their
documentation/CI commits are already merged locally. The newer statistical
additions have not been confirmed published. See the final section for the
current package and connection checks.

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
