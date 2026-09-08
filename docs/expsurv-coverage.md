# EXPSURV coverage audit

Catalog entry 28 is implemented. This audit covers all 20 named routines in the
version-1 source, including both platform-conditioned SAVE-DATA declarations,
and every workflow in the bundled manual. Python and optional Matplotlib replace
the XLISP-STAT environment, global-variable assignments and platform file/menu
dialogs. Original code and the reference runtime remain local research material
and are not bundled. Usage and detailed semantic differences are in
[expsurv.md](expsurv.md).

## Audited archive

The archive contains exactly three files; it has no additional dataset or binary.

| File | SHA-256 |
| --- | --- |
| expsurv.lsp | b3807d82f5e18b9d6c564642bcb6b0f785103847578e1d9d5a19581a846a2492 |
| expsurv.tex | f1629f3c27a17d2c46da07aaba62800a3410de62afd11c8959ef948d02c91d2c |
| readme | ab955d156f99980cc50b742e2cb0adab0e5b42bdeda19a94c38242c716d18b8c |

The readme allows redistribution and identifies E. Neely Atkinson's exploratory
survival work. The manual covers loading, data import/export and generation,
cut-point comparison, three linked scatterplot displays, selection/brushing,
and two model-alignment displays. Its background references add context, not
additional executable routines in this archive.

## Source-to-Python mapping

| Source routine | Python capability | Validation |
| --- | --- | --- |
| KMEST | `exploratory_survival`, grouped ties by default and sequential ties with `legacy=True` | Exact rational products on censored/tied samples; independent grouped risk counts and permutation checks |
| KM-PLOT | Fit `step_time` / `step_survival` | Exact corners, censor plateaus and initial zero-time jumps |
| SCAT-KM | `plot_survival_scatter` | Original-row selection, exact linked curves, real canvas rectangle and brush events |
| EXP-RAND | NumPy exponential draws at reciprocal rate in both generators | Analytic moments, rate scaling and seeded replay; NumPy streams explicitly replace XLISP-STAT streams |
| COSORT | `ExploratoryTable.cosort` and stable preparation in fits/generators | Tied order preservation, unsorted inputs, joint row alignment |
| ASSIGN-VARS | Named `ExploratoryTable` construction | Unique names, dimensions, copied read-only data |
| GET-DATA | `ExploratoryTable.read(path, names)` | Real whitespace files, malformed/missing values, single-row/column inputs |
| MYSET | Explicit named columns and returned objects, replacing global symbol mutation | Column access and unknown-name rejection |
| PRINT-LINE | Numeric row formatting within `ExploratoryTable.write` | Float64 round-trip precision and row boundaries |
| PRINT-LIST | Matrix row iteration in `write` | Multirow file round trips |
| SAVE-DATA (Macintosh / other) | `write(path)` | File read/write/survival workflow; dialogs replaced by explicit paths |
| GEN-DATA | `generate_exploratory_data` | Scalar sample-SD/censoring oracle, exact study-end boundary, large-scale stability, covariate moments and alignment |
| CHOOSE-CUT-PLOT | `survival_cutpoint`, `plot_cutpoint` | Equal-to-cut membership, original rows, empty groups, 50-position slider and real callback artist updates |
| SCAT-EVENT | `plot_event_scatter` | Exact duration/arrival segments and failure/censor endpoints, zero times and empty selection |
| SCAT-BOX | `censored_box`, `plot_censored_box` | All source geometry branches, unreached/plateau quartiles, tied failure survival, all-censored/empty cases |
| QUANT | `survival_quantile(method="source")`; explicit ordinary `step` alternative | Linear interpolation, exact last-plateau choice, unreachable levels, zero times and scaling |
| ACCEL-FAIL-PLOT | `plot_survival_alignment(method="accelerated-failure")` | Source multiplier ranges, rescaled-sample alignment, callbacks and zero endpoint |
| PROP-HAZ-PLOT | `plot_survival_alignment(method="proportional-hazards")` | Survival powers, cumulative-hazard scaling, callbacks and zero power |
| GEN-EXPO-DATA | `generate_exponential_samples` | Censoring probabilities 0/interior/1, independent analytic moments, reproducibility and rate scaling |
| GEN-EXPO-EXAMPLE | Same function's source defaults | 100/50 observations, rates 1/10 and censoring probability .1, file round trips |

## Manual interaction and runtime replacements

All three linked matrices support point clicks, Shift-add, rectangle selection,
continuous hover brushing, brush resizing, clear selection and explicit cleanup.
B toggles mode; +/- changes size; APIs permit programmatic control. Real canvas
mouse/key tests cover each linked controller. Repeated brush selections reuse
unchanged fits. Click tolerance and brush dimensions use explicit display pixels;
these replace XLISP-STAT's runtime defaults and menu dialogs.

The cut-point and model-alignment controls retain the source's 50-position
sequences and permit programmatic updates. The source's alignment slider value
50 lies beyond zero-based indices 0–49; Python initializes directly at the intended
maximum parameter. The AFT executable multiplies plotted time by m, corresponding
to S(t/m); the manual's S(k*t) uses the reciprocal parameter.

Matplotlib supplies tick placement, figure lifecycle and exports. The cut-point
density is an explicitly documented Gaussian guide with sample-SD*n^(-1/5)
bandwidth, replacing the unspecified XLISP-STAT kernel-dens defaults. The event
chart uses the executable's diamond censor symbol, although the manual says circle.
Named tables replace global symbols; paths replace file dialogs; source Lisp
loading/install instructions become normal Python imports and the optional plot
extra. No Lisp interpreter, arbitrary Lisp evaluator or platform menu emulator is
part of the Python package.

## Defined differences and edge cases

- Default KM ties are grouped; source sequential tie order is an explicit option.
- Inputs are stably sorted instead of requiring callers to run COSORT first.
- Source QUANT interpolation is retained explicitly; censored-box defaults use it,
  including quartile bars beyond the last failure on exact censoring plateaus.
- Initial linked displays render all selected patients consistently. Empty groups
  and selections clear their curves/geometry. All-censored boxes show an explicit
  no-failure state instead of indexing an empty source list.
- Constant cut covariates and zero-maximum AFT samples have explicit undefined-view
  errors; valid numerical cut comparisons and other views still work.
- Zero PH power is displayed as survival one, including 0**0=1.
- Missing/nonfinite data are rejected. Output uses round-trip float precision,
  rather than Lisp printer defaults; blank input lines are ignored.
- GEN-DATA uses the runtime's verified n−1 SD denominator and stable normalization.
  Both simulation families preserve the source probability mechanisms but use
  NumPy random streams. Neither source-seed identity nor cross-version bitwise
  replay is claimed.

## Evidence and performance limits

There are 138 EXPSURV tests across the numerical, data, simulation and plotting
modules. They include independent mathematical calculations and real filesystem
and canvas interactions. Rendered cut-point, endpoint, alignment, matrix, survival,
event, incomplete-box, all-censored-box and brush views were visually inspected
during implementation. Installed-wheel workflows were exercised with and without
the optional plotting extra.

No archived XLISP-STAT interpreter was executed: source inspection and independent
checks provide the numerical evidence. The reference runtime's sample standard
deviation definition was inspected at revision
f1bea6053df658ee48612bf1f63c35de99e2c649. This is not a native differential test claim.

[Benchmark results](expsurv-benchmark.json) are reproducible with
`uv run python tools/benchmark_expsurv.py`. Three repetitions compare outputs with
rtol=1e-13 and atol=1e-15 before reporting medians. A 20,000-observation sequential
KM fit was 6.52× faster than a scalar Python recurrence; 10,000 vector queries
were 216.61× faster than individual Python calls in the recorded environment.
These are Python comparison workloads, not archived-interpreter or GUI latency
measurements. Sorting, products, queries, simulation and selection masks use
NumPy; no additional JIT dependency is needed for these measured workloads.
