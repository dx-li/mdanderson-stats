# BOIN desktop coverage audit

Baseline main `f5a2b20`. The preceding goal turn made progress: Pinnacle's
streamed image analysis, original C/R references, documentation and package
checks were integrated. Coverage is 62 implemented, 62 partial and 14 pending.
The full catalog/publication objective remains active.

Root uses `feat/boin-desktop-coverage`; the sole Luna worker uses
`feat/boin-desktop-luna` in the existing independent checkout. Both started
clean. System memory reported 48% free. Numerical work remains serial with
one BLAS/OpenMP thread; no full-suite run or installation is planned.

## Authoritative scope

The official desktop entry describes four method families: single-agent BOIN,
TITE-BOIN, combination single-MTD and combination MTD-contour designs. It also
describes conventional-design comparisons, standardized follow-up calculations
and protocol/report generation. Its version is 1.1.0, modified December 22,
2021. The current TITE-BOIN guide separately confirms Rolling Six comparison
output and the ordinary STFT example `(30+48+75)/90 = 1.7`.

Repository inspection found executable APIs and existing independent references
for all four method families, but not every workflow. The single-agent tests
include published boundaries, exact beta/binomial safety identities and native R
isotonic selection. TITE-BOIN tests include published Table S1 and rational
imputation; Rolling Six tests enumerate the published assignment table.
Combination references cover R boundaries, movement, bivariate isotonic fits,
contour selection and the next-subtrial planner. Source versions and differences
are already recorded in the corresponding method documentation.

The desktop mapping must retain these distinctions: combination contour
selection/planning exists, but full waterfall simulation is absent; shared
Python calendar conventions are explicit and not desktop executable parity;
English/Chinese protocol text currently covers the single-agent design only.
No coverage claim is inferred merely from matching software names.

## Implementation focus

Luna is implementing a coherent TITE-BOIN/Rolling Six comparison that reuses
the existing simulators. It accepts common scenario inputs, preserves both
result objects and exposes summaries/reporting. It must preserve the distinction
between a Rolling Six MTD and a highest-planned-dose recommendation, avoid
invented sample-size matching, and preflight both branches before advancing
the caller's random generator.

The original BOIN 2.7.2 source under ignored `research/raw/BOINComb` was also
inspected to identify the next scientific gap: a complete waterfall simulator.
Its source contract and discrepancies are recorded separately before delegation.

## Comparison checkpoint

Luna committed `3e3cac7`, integrated as `6676ba3`. Two focused tests passed in
1.21 seconds, along with focused Ruff and mypy checks. Root verified both
documented usage examples, independently recomputed selection frequencies,
Monte Carlo errors and allocation/time summaries, and preserved allocations
and outcomes under a `1e100` change of time units. The mean of two `1e308`
durations remains finite. Invalid comparator settings, timing masses and an
excessive aggregate workload leave a supplied RNG unchanged. Settings retain
their precision when estimates use zero decimal places.

Root's bounded numerical check took 0.114 seconds after import, with peak RSS
113.52 MiB and zero swaps. Changed Python files pass Ruff and formatting checks;
no full suite or additional CI work was performed. Entry 99 now maps verified
shared APIs and the new workflow as partial, bringing the catalog to 62
implemented, 63 partial and 13 pending. The desktop executable and native
protocol/project formats remain unverified.

Six deterministic original-R waterfall subtrial fixtures were generated in
0.29 seconds without an R package installation. They cover safety, extra-safe
stopping, movement and the destination-dose precision rule. Full waterfall
implementation is now assigned to the same single Luna worker. No publication
is claimed: the prior automatic approval review rejected the GitHub write,
and this session does not permit the required approval.

## Subsequent integration

The complete no-titration waterfall workflow has since been integrated and
validated; see [boin-waterfall-workflow-audit.md](boin-waterfall-workflow-audit.md)
for source defects, Python contracts and resource measurements. Four new
comparison/desktop/waterfall examples passed in an isolated wheel. The retired
BOP2 desktop entry is also mapped to its documented online successor's existing
Python methods, without claiming old executable parity. EasyCellType's Fisher
branch subsequently added another partial entry; its source audit carries the
current catalog totals.
