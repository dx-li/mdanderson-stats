# SPPCR method and source coverage

Catalog entry 26 is implemented against the pinned SPPCR 1.0 source archive
identified in [the archive research](sppcr-research.md). Completion means coverage
of its statistical methods and usable Python analysis workflows, including the
four source input modes. It does not mean identical terminal typography, a
Fortran runtime emulator, or reproduction of known numerical defects.

## Statistical coverage

| Method | Python implementation and numerical evidence |
|---|---|
| Independent binomial detections under Poisson allele means; maximum likelihood and observed information | [Mean fitting](sppcr-fit.md); native interior fits, closed-form single-level solutions, high-precision score checks and extreme-scale tests |
| Calibration, allele and mutant frequencies, delta-method variances, arcsine transformation | [Frequency summaries](sppcr-frequencies.md); independent Jacobian/covariance identities, native values, boundary and scale tests |
| Observed-fraction resampling and truth-model simulation | [Generation](sppcr-generation.md), [truth designs](sppcr-truth.md); native truth probes, stable probability identities, simulation moments and independent-cell checks |
| Bootstrap fitting and population moments | [Bootstrap](sppcr-bootstrap.md); explicit replicate composition, retained individual estimates, centered variance and undefined-frequency handling |
| Bootstrap normal and transformed intervals, inverse calibration | [Intervals](sppcr-intervals.md); native interior expressions, support boundaries, reciprocal unboundedness and missing uncertainty |
| Historical random streams and binomial draws | [RNG reconciliation](sppcr-random.md); native state/draw fixtures, all 32 streams, antithetic and block operations, phrases and source float32 probabilities |

The default is 1,000 bootstrap replicates, as in the source; callers can change
this. Explicit caller-owned random state makes repeated analyses reproducible.
Both modern NumPy sampling and source-compatible RANDLIB sampling are available.
Benchmarks linked from the method documents compare vectorized and repeated
scalar Python workloads; they do not claim speedups over native Fortran.

## Complete source-file mapping

Each of the 27 archived `.f90` files appears once below. Generic support modules
are mapped by their role in SPPCR; their private helpers are not additional
statistical methods requiring separate public APIs.

| Archived source files | Covered responsibility |
|---|---|
| `fit_mu_mod.f90`, `zero_finder.f90` | Stable bracketed likelihood solution, convergence checks and curvature in `sppcr_fit.py`; shared general zero finders also exist in the completed CDFLIB implementation |
| `fit_freq_mod.f90`, `sppcr_aux_mod.f90` | Calibration/frequency summaries, variance propagation and transforms in `sppcr_frequencies.py`; inverse-transform intervals in `sppcr_intervals.py` |
| `accumulate_mod.f90` | Retained replicate series and stable population summaries in `sppcr_bootstrap.py` |
| `generate_mod.f90` | Empirical/truth probabilities and modern/historical samples in `sppcr_generate.py`, `sppcr_legacy_generate.py` and `sppcr_truth.py` |
| `one_data_set_mod.f90` | Boundary policy, observed and bootstrap fitting in `sppcr_fit.py`, `sppcr_bootstrap.py` and `sppcr_analysis.py` |
| `ecuyer_cote_mod.f90`, `random_binomial_mod.f90`, `random_standard_uniform_mod.f90` | Existing RANDLIB generator and `sppcr_legacy_generate.py`; source-state and draw reconciliation |
| `phrase_to_seed.f90`, `set_seeds_mod.f90` | Existing phrase/clock seed support in RANDLIB/RANLIST; analysis deliberately receives explicit continuous RNG state |
| `cdf_aux_mod.f90`, `cdf_normal_mod.f90` | Shared CDFLIB normal/inverse-normal and numerical support; SPPCR intervals retain the source multiplier 1.959964 |
| `data_in_struct_mod.f90`, `structures_mod.f90`, `resize_mod.f90` | Validated immutable data/results with explicitly sized arrays, active-row allele handling and DNA units in the SPPCR data and analysis modules |
| `problem_in_mod.f90`, `qlex_mod.f90` | Batch and FileMaker parsing, observed interactive entry, truth design and truth dialogue in the corresponding `sppcr_*` modules |
| `get_numbers_mod.f90`, `print_it.f90` | Existing bounded console input plus SPPCR dialogue and reporting APIs; corrections, continuations and EOF semantics |
| `results_out_mod.f90`, `print_array_mod.f90` | Data, truth, fitted values, uncertainty and all replicate rows in `sppcr_reporting.py` and `sppcr_truth_reporting.py` |
| `file_mod.f90`, `open_file_mod.f90` | Bounded read-only file input, path handling, staged report creation/replacement/append and cancellation in `sppcr_files.py` and `sppcr_output.py` |
| `banner.f90`, `sppcr.f90` | Repeated four-mode menu, explicit seeds, per-analysis output choices, reports and file dialogue in `sppcr_console.py` and the `mdanderson_stats.sppcr` CLI |

The other 15 archive files are accounted for in the archive research: build and
acquisition instructions, exact retained legal notice, historical binaries and
misplaced SOGS documentation/examples. SOGS remains a separate pending catalog
entry. No additional SPPCR statistical method was found in those files.

## Deliberate changes and limits

Python repairs the source's partial-saturation adjustment, never-seen pseudo-fit,
unchecked root-search failure, cancellation-prone bootstrap variance and periodic
confidence-limit inversion. It preserves true zero boundaries and reports
unavailable uncertainty explicitly. It fixes stale input rows, negative-count
sign loss, missing progenitor indices, repeated allocation and uninitialized
per-analysis options. Truth generation always uses supplied truth for its initial
sample. The historical all-stream reseed defect is corrected by shared RANDLIB.

Input files are read-only. Saving waits for complete selection and stages the
output before publication. Two output files are not one atomic transaction;
append uses a bounded snapshot and does not promise concurrent-writer safety.
Reports retain numerical information, units and identities in labeled text rather
than emulating native print layouts. Native executables have not been run;
reference fixtures rebuild archived source. See the linked method and workflow
documents for parameter ranges, boundary policies and failure contracts.

## Validation record

The catalog's validation paths link the existing native-reference generators,
fixtures and focused numerical/workflow tests. They cover all four input modes,
both RNG paths, observed/truth bootstrap choices, consecutive analyses, EOF,
reports and actual file publication. The immediately preceding implementation
milestone (`9fbc25a`) passed **31,060 tests**, Ruff, mypy, package build and a fresh
installed-wheel exercise with consecutive 1,000-replicate CLI analyses and
save/append. This completion update changes documentation and catalog status;
it does not change the numerical implementation or require another full-suite
run. Archive membership, coverage-table membership and validation-file existence
were checked when completing this mapping.
