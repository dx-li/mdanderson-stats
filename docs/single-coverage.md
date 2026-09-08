# SINGLE coverage audit

Catalog entry 55 is implemented with the numerical, solver and interface
substitutions documented in [single.md](single.md). This audit covers the exposed
capabilities of SINGLE version 2.0 (August 1997), shipped in `SINGLE_V1.tar.gz`.
It does not claim identical optimization iterates, terminal menus or output bytes.

The authoritative inputs are `single.f` and `single.doc` from the archived download.
Their SHA-256 hashes are respectively
`e28dd3bb8c51d02a667a1f12995e4f591bea7ecfb706e3160b7bcf02f565c8c2` and
`8dd621d7b58a1bc29fb055ca0f0f464ed25c70d0d673edc3632f2ca343fb69fd`.
Original source and binaries remain research inputs and are not bundled.

| Source capability | Python coverage | Evidence |
| --- | --- | --- |
| MINIT: one/two samples, logistic/log-log, linear/centered predictors | Fixed-design and optimization APIs, explicit model/form/comparison | Native CPROB/CDPDB/MIX fixtures and analytic information/contrast checks |
| CINIT/CRIT: slope or quantile SD/variance; common-slope location or common-location slope difference SD | `single_design_precision`, `single_two_sample_precision`, prior evaluators and optimizers | Native response-information fixtures, independent delta-method calculations, all model/form/criterion combinations |
| PINIT: point or independent uniform prior | Point parameters or uniform parameter nodes/weights | Native RECGS comparisons, analytic integrals, order convergence and point-prior reduction |
| PINIT/ALNTON/PARRAW: normal/log-normal means, variances and correlations | `single_prior_parameters`, `single_normal_criterion` | Marginal moment checks, latent covariance checks, normal/log-normal node moments, native RECHRM comparisons |
| CGTCOV/DSTCOV: design-based prior correlations | `single_design_correlation` | Independent covariance reconstruction at the original reference doses and group counts; no native DSTCOV execution claimed |
| PEIGEN/PRCOMP/RECHRM: correlated prior integration | Batched eigentransformed Hermite nodes; explicit covariance scaling and order | Correlated and degenerate covariance tests, 64 native quadrature cases, full manual studies |
| DINIT: dose interval, grid size, maximum dose count, subjects, improvement threshold | `single_search_design` and `SingleStudySpecification` | Bounds, limits, totals, threshold acceptance/rejection and invalid-input tests |
| DCPPAR: separate ANIMLS total in each group | `group_totals`, with experiment total equal to their sum | Analytic fixed-group precision, unequal group sizes, 32 weighted-prior combinations, search history and replay |
| DQSCAN/DIQSCN: initial pair scan and added-dose seeds | Initial grid pairs and successive dose addition | Printed point-prior optimum, broad-prior support improvements, seed counts, infeasible-seed and local-solver failure tests |
| Main optimization phases: move doses, change allocations, jointly optimize | Analytic-gradient joint solver; separate fixed-dose allocation solvers | 48 finite-difference gradient contracts, fixed-dose analytic optima, printed point-prior optima and uncertain-prior studies |
| Main-loop improvement and support stopping | Relative improvement; optional original negligible-count/overlap rules | Rejected-stage retention, 13 native support-predicate cases, precedence and minimum-dose tests |
| Main menus: change design/model/criterion/prior, restart | `SingleStudy.revise`, new specifications and JSON replay | Independent input snapshots, validation, one/two-sample replay and revision |
| Reports: settings, prior inputs, designs, counts and criterion | Full study report plus each attempted stage and selected design | Parsed full-precision values, totals, zero entries, accepted/rejected stages and file I/O tests |

## Original conventions and substitutions

For the original normal-prior numerical convention, use
`single_prior_parameters(..., conversion="legacy")`, then
`single_normal_criterion(..., legacy_scale=True, order=8)` and pass its nodes and
weights to a harmonic-aggregation study. PEIGEN sets the source Hermite order to
**eight**; the Python evaluator defaults to six and allows explicit order checks.
The separate uniform integrator uses six-point quadrature by default in both.

For the original two-sample allocation constraint, set, for example,
`total_subjects=200, group_totals=[100, 100]`. The shared-total default is an
additional feasible set. For the original support checks, select
`support_stopping="original"`; its first-group-only and entry-order conventions
are documented in the main guide.

The Python search uses a joint SLSQP solve instead of reproducing the original
sequence of fixed-count dose optimization, fixed-dose allocation optimization,
and joint optimization at every stage. Reports retain the resulting joint stage
designs, not those internal optimizer substeps. The seed scan evaluates all seven
allocation fractions and starts from the prior joint optimum; the original can
stop its seed scan early and also retains an equal-allocation intermediate.
These are explicit optimization-algorithm substitutions, not claims of exact
trajectory equivalence or global optimality.

Cholesky solves replace explicit matrix inversion. Stable response information
replaces the source's probability clipping and exponential caps. Positive
log-normal coordinates and singular prior nodes are validated explicitly.
Exact raw-moment conversion and correctly scaled normal covariance are the
Python defaults; the original approximate conversion and quarter-covariance
convention remain available. Two-sample variance objectives extend the original
SD choices. The commented-out power option is not selectable in CINIT and is
not treated as an implemented original capability.

Python calls and UTF-8 TSV/JSON files replace terminal prompts and interactive
file management. Reports record the actual integration nodes and weights,
without inventing a distribution family from them. Original raw prior inputs
can be retained separately when that provenance is needed.

## Manual and performance validation

`tests/test_single_manual.py` runs the manual's two uncertain-prior studies from
raw normal/log-normal means and variances through prior conversion, quadrature
and design search. It reproduces the quantile criterion `0.495569` with three
doses and the slope criterion `0.157458` with two doses, at the manual's printed
precision. Optimized dose/count comparisons allow for the original solver's
termination precision. Existing point-prior tests cover the manual's `0.444280`
quantile optimum and slope optimum. The full original executable is not the
oracle for the modern optimizer; these are published-output and independent
mathematical checks alongside extracted native numerical kernels.

`tools/benchmark_single.py` compares one batched call with 1,000 calls to the same
API and verifies numerical equivalence during each repetition. The recorded
three-run median on this machine is approximately 79× faster for one-sample
quantile precision and 26× for two-sample location contrast precision. See
[single-benchmark.json](single-benchmark.json) for workloads, versions and timings.
These results do not measure the full design search or compare with Fortran.
