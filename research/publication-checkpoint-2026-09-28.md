# Community publication checkpoint — 2026-09-28

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
