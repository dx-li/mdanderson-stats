# Community completion continuation — October 12, 2026

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
audit of every app. The count remains **93 implemented / 39 partial / 6 pending**;
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
