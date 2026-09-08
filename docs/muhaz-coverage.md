# MUHAZ coverage audit

Catalog entry 49 is implemented. This audit covers the numerical, reporting and
plotting workflows in MUHAZ version 1. The Python package replaces the S-PLUS
interpreter, dynamic loader and generated installation Makefile; original code
and binaries remain local reference inputs and are not bundled. Details and
examples are in [muhaz.md](muhaz.md).

## Audited archive contents

All 11 files in the archive's S directory were reviewed: the numerical Fortran,
S entry points, seven help files, installation readme and generated Makefile.
The archive contains no additional sample dataset or executable workflow.

| File | SHA-256 |
| --- | --- |
| Makefile | 70d66f367c190bb453cefd916eae4e422fcd7688972cdb14ef791028c8bf85ef |
| all.s | 80ddfe00996429a27d15c7cc5463c8d2d3b34be12508e0c2dfacc33081417d71 |
| kphaz.fit.d | f9af57614f0f40edd3f944081cfa5b3ed6375e190b4788bb4473d3497bf36f13 |
| kphaz.plot.d | 70b9ddfe936826ec333a2c69399b86888e0d3690211553c5c53935385f225513 |
| muhaz.d | 6c714a9cc0809c71bb445f5f11a5bfdf2a87c8dafcab096cdc6d02e25eaf3191 |
| muhaz.f | 139c8e8e2080e536ab81fefd4e32f875c768bdd6396bb8821e5a26f842d28b5c |
| muhaz.object.d | ff8798911e9ca1c61f0f8979b0d115f745359ddf75f6036d4cc125454415ee0a |
| pehaz.d | 38500db140b2c41b2d25ce295d20c96b203cedc061f40832cff737c4c8fde095 |
| plot.muhaz.d | f995b2834efc8301212c0fec808974be52b8c60a5b3d62c749d22f9438ce1ada |
| readme | ca52ab43c6c021ccf912841f8eea93a184af0b17393d317d25cdc9f47fcc7b1f |
| summary.muhaz.d | 76f6470292921e51c128e57de9ddb9583b09d0f31cb73f9924682a03fee8693d |

## Capability mapping

| Archived capability/routines | Python implementation | Validation evidence |
| --- | --- | --- |
| HAZDEN, KERNEL, IBNDS: four kernels and three boundary settings | `muhaz_fixed` and prepared-sample evaluator | 36 native grids; independent kernel mass/moment integrals, endpoints, ties, clipping, scaling and chunk tests |
| FUNC, SURFCT, INTGRL, TRY, MSEMSE: pilot bias/variance/MSE | `muhaz_mse` | 24 native grids; analytical constant-pilot checks, convergence/cap reporting, varying-bandwidth matrix versus scalar evaluations |
| NEW_HAD/GLMIN: common-bandwidth minimization | `muhaz_global` | 39 native fits, grid-sum minima, tie/zero-score selection, single-candidate bypass |
| NEW_HAD/LOCLMN: pointwise bandwidth selection | `muhaz_local` | 117 native fits including overlapping boundaries and unavailable legacy diagnostics; independent pointwise minimization |
| BSMOTH: bandwidth kernel regression | Shared chunked smoother | Native local and neighbor fits; constant preservation, rectangle arithmetic mean, left-only correction and negative/undefined smoothing rejection |
| KNNCEN: kth failure-distance bandwidth | `muhaz_neighbor_bandwidths(method="failures")` | Native radius grids, independent order statistics, censor exclusion, ties and 25,001-observation buffer-limit case |
| KAPMEI, GETS, ATPOS, ONEOLF, LOCOLF, OLAFBW: survival-mass radii | `muhaz_neighbor_bandwidths(method="survival")` | Native grids, endpoint conventions, terminal-singleton correction, terminal ties, scaling and chunk boundaries |
| KNNMIN, OLAFMN, KNNHAD: neighbor selection and fitting | `muhaz_knn` | 96 native fits: 92 valid comparisons, four explicit negative-bandwidth rejections; score cutoff and bypass checks |
| SORTER and source linear/binary lookups | NumPy sorting, partitioning and searchsorted | All native numerical comparisons plus order/permutation tests |
| S `muhaz`: method selection and defaults | Separate `muhaz_global`, `muhaz_local`, `muhaz_knn` functions | Both omitted-status and supplied-status paths; default candidate grids, pilot and smoothing formulas |
| Independent min/max time arguments | `bounds=(left, right)`, either endpoint may be None | All three selectors checked against explicit intervals; ten-at-risk interpolation and invalid defaults |
| Logical subset and sorting | Shared input preparation | Joint subset applied before finite-data checks; explicit counts; unsorted/permuted samples |
| Configurable minimization/estimation grids | `n_min_grid`, `n_est_grid`; grid arrays in results | Default dimensions, custom grids, invalid counts and 1,001-point pilot-buffer regression |
| S `summary.muhaz` | `summarize_muhaz`, immutable summary, text/UTF-8 reports | 14 metadata, convergence, bypass, precision and file-output tests |
| S `pehaz` | `pehaz`, events/risk/person-time arrays | 10 native S cases; independent exposure, bin endpoints, default width, scale and large-origin checks |
| S `print.pehaz` | `PiecewiseHazard.report`/`write_report` | Significant digits, numerical columns, missing/unbounded values and file round trips |
| S `kphaz.fit`: Nelson/product-limit, strata, q | `kphaz` | 24 native S cases; independent increments, Greenwood variance, q windows, ties and terminal behavior |
| S `plot.muhaz`, `lines.muhaz` | `plot_muhaz`, optional existing axes | All kernel result types, exact data, overlay preservation and rendered inspection |
| S `plot.pehaz`, `lines.pehaz` | `plot_pehaz`, optional existing axes | Exact bin edges, no invented endpoint baseline, missing-bin gaps and rendered inspection |
| S `kphaz.plot` | `plot_kphaz` | Stratum legends, step geometry, consistent infinite-value gaps and rendered inspection |
| Help's candidate-score and bandwidth-function diagnostic plots | Retained candidate grids/scores, selected and smoothed bandwidth arrays; Matplotlib examples | Four-panel global/neighbor-score and local/neighbor-bandwidth figure rendered and visually inspected |
| Help's rerun with refined grid/smoothing | Explicit candidate/count/pilot/smoothing arguments | Candidate order/duplicates, minimization and smoothing tests; no hidden mutable session state |
| S-PLUS object/installation conventions | Typed results, Python functions, package build and optional plot extra | Wheel import/fit/report/plot checks; CI on Python 3.12–3.14 |

The eight native fixture collections contain 370 cases in total: fixed (36),
MSE (24), global (39), local (117), neighbor radii (24), neighbor fits (96),
piecewise (10), and failure-interval (24). Four of the native neighbor fits yield
negative smoothed bandwidths and are checked for explicit rejection; the other
native numerical outputs are compared within recorded tolerances. Tests also
check behavior independently of source agreement. Fortran comparisons use the
recorded compiler configuration, including non-fused floating-point operations
where required; no claim of bitwise equality across arbitrary compiler flags is
made. Source/driver hashes accompany each fixture collection.

## Explicit adaptations and limits

Python uses complete method names, explicit exceptions and typed results rather
than S partial matching/coercion. Missing strata are rejected instead of relying
on S's ambiguous NA-indexing behavior. Raw input samples are not duplicated in
result objects; callers retain their input arrays for replay. Effective settings,
counts and computational outputs are retained. S expression strings, device pixels
and installer internals are not statistical workflows.

Default tied hazards group deaths at each time; legacy mode retains sequential
risk weights and support endpoints. Legacy pilot defaults, zero-score sentinels,
boundary-smoothing quirks, terminal Kaplan–Meier omission and quadrature addition
order are explicit. The default includes zero MSE in minimization and removes the
neighbor selector's arbitrary 1e5 cutoff. Uninitialized source outputs become None
or NaN, or a clear error when no valid estimator is defined. Zero denominators,
nonpositive smoothed bandwidths and numerical overflow are not silently repaired.

The piecewise default closes/truncates the final bin; legacy mode retains equal
width overshoot and endpoint exclusion. Product-limit infinite estimates and
legacy undefined variances remain visible in data and become plot gaps. Reports
use significant digits and explicit bypass/convergence status. The source MSE
criterion, event-count survival factor, grid-sum score, radius perturbations and
left-boundary precedence are documented; grid selection is not continuous
optimization and reaching a quadrature cap is not certified convergence.

## Performance and validation

[Recorded benchmark](muhaz-benchmark.json), generated by
`tools/benchmark_muhaz.py`, compares one batched call with 501 single-point calls
to the same Python API on 2,000 observations with 25% censoring. It verifies
numerical agreement on every repetition. Median speedups were approximately
9.0× for fixed hazards, 8.6× for failure-distance radii and 61.6× for survival-mass
radii. These are machine-specific batching measurements, not speedups over
Fortran or end-to-end bandwidth-selection benchmarks. Intermediate arrays are
chunked, sample preparation is reused across query points, and arbitrary static
Fortran buffer limits are removed.

The full Python validation suite, static checks, build, installed-wheel checks
and rendered plots support this mapping. No additional numerical, reporting or
plotting workflow in the audited archive remains pending for this entry.
