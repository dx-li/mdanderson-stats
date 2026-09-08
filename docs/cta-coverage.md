# CTA coverage audit

Catalog entry 30 is implemented. This audit accounts for all 22 named program
units in the archived source and the program's analysis-selection, repeat-study
and reporting workflows. It covers numerical functionality, not byte-for-byte
console formatting or every historical floating-point defect. Detailed API
conventions and deliberate changes are documented in [cta.md](cta.md).

## Audited archive

CTA_V1.tar.gz contains exactly three files, with no additional data, manual or
executable. The catalog labels version 1 with a March 19, 1992 modification date;
the source itself identifies February 2, 1998.

| File | SHA-256 |
| --- | --- |
| cta0298.f | 7c25440a91d37c2cb437c486965c1142debcba6e0cf589065504d989c0251834 |
| HOWTOGET | 274862a242b8ef44f10a4aa015bba83478aba4a6bbd6616eb4b291965d2df557 |
| LEGALITIES | 22eb90a4052e8fe508c68d52c4c79dd079752c26e54d75bb2668f1495d2451ef |

HOWTOGET is a distribution/contact note. The archived LEGALITIES is retained
byte-for-byte in `notices/mdanderson-cta-LEGALITIES.txt`; its redistribution,
modification and commercial-package provisions remain applicable as described
in the repository notices. Original Fortran, archives and compiled reference
executables remain local research material and are not bundled with the package.

## Program-unit mapping

| Original unit | Python implementation or disposition | Evidence |
| --- | --- | --- |
| PROGRAM cta | `CTAStudySpecification.run`, reusable/revised settings, `CTAStudy` snapshots and reports | Full unchanged program run with two tables and retained choices; study integration tests |
| CHISQT | `contingency_chi_square`: expected cells, percentages, Pearson, Yates, Cochran and small-cell diagnostics | 9 native cases; rational statistics, batching and boundary tests |
| FISHXT | `fisher_exact`, shared support selection and detailed table listings | 40 native cases; exact rational enumeration of all 625 small tables for four alternatives |
| ROTCAF | Hypergeometric probability evaluation replaces log-factorial workspace | Same Fisher comparisons and rational term checks |
| KAPPA | `cohen_kappa`: agreement, multinomial/null variances and source indexing option | 9 native cases; exact rational gradient calculations, degeneracy, scaling and rare categories |
| MCNEMAR | `mcnemar_analysis`: individual pairs, sum, pooled direction and heterogeneity | 7 native cases; rational decomposition, independent chi-square tails and zero-discordance tests |
| SENSPEC | `diagnostic_accuracy`: sensitivity, specificity, predictive values and errors | 16 native cases; independent conditional probabilities and binomial errors |
| RELRISK | `odds_ratio`: group odds, log odds ratio, log standard error and interval | 48 native cases; independent log-Wald limits, orientation and extreme-count checks |
| BINCOMP | `binomial_comparison`: event-category selection and conditional Poisson comparison | 32 native cases; rational binomial tails, category/axis choices and zero events |
| BINOP | Existing vectorized `binomial_test` engine and detailed ordered tail listings | BINCOMP comparisons and per-term rational report tests |
| BLFEW | Stable binomial-tail engine; vectorized binomial PMF for details | Same tail and report validations |
| CHI2 | Accurate incomplete-gamma survival probabilities | CHISQT/MCNEMAR native cases and independent finite gamma/erfc identities |
| ZIP | Internal log-gamma support superseded by distribution kernels | Distribution-level validations above |
| GAMMA | Internal gamma approximation superseded by distribution kernels | Distribution-level validations above |
| PHI | Accurate normal CDF/quantile and distribution kernels | RELRISK comparisons and independent normal-quantile checks |
| OVERFL | No stub needed; source always reports overflow and forces an approximation | Explicit source-approximation comparison in McNemar tests |
| CBRT | No separate public routine; source fallback replaced by accurate chi-square tails | McNemar approximation comparison and accurate-tail checks |
| B | Dormant alternative binomial-mass helper; not called by the active program | Static source/call review; no independent public API claimed |
| FACT | Called only by dormant B | Static source/call review; not executed as a supported workflow |
| COMBO | Dormant combination helper; no active caller | Static source/call review |
| BL | Dormant log-binomial helper; BINOP's possible call is commented out | Static source/call review |
| SUML | Used only by dormant COMBO/BL | Static source/call review |

Dormant helpers are accounted for rather than advertised as new functionality.
Their combinations/factorials are already subsumed by the supported probability
calculations. Internal theta-expansion traces and diagnostic arithmetic printed
by the original are not separate public APIs; the resulting estimates, variances,
probabilities, tables and correction conventions are exposed.

## Workflow audit

The original asks for table dimensions/observations, six yes/no analysis choices,
and orientation/class/alpha choices for relevant analyses. Fisher runs automatically
following a chi-square analysis for a 2x2 table with minimum expected count below
10, subject to the 50,000-case workspace limit. It writes a report, offers another
table and optionally reuses the prior yes/no choices.

The Python specification exposes each choice and parameter explicitly. Reusing
it runs another table; `dataclasses.replace` revises settings. Each result keeps
an independent input snapshot. Auto Fisher records its decision, including
unsupported fractional counts or excessive totals; an explicitly requested
invalid analysis fails clearly. The main numerical functions accept batches,
while a combined study describes one table. Console prompts and pagination are
replaced by Python calls and explicit UTF-8 files.

Reports include input cells/margins, settings, chi-square cells/percentages and
statistics, selected Fisher probabilities, agreement/variances, paired-category
statistics, diagnostic estimates/errors, odds/limits, and conditional binomial
results. Optional details provide every cell contribution and included Fisher
and binomial probability term with running sums. A configurable output-size
limit fails explicitly before large detail allocations and preserves an existing
file on failure. Tests check probability rows against fractions and verify their
sums, source traversal, cutoff inclusion and report-file behavior.

The original declares a ten-category workspace but KAPPA writes marginal totals
into an additional row and column. Python computes those separately and tests a
10x10 study against independent uniform-table identities without mutating input.
It does not reproduce the source's out-of-bounds memory writes.

## Native and independent validation

There are 161 routine-level native cases, plus two tables from a complete original
program session. Every generator records compiler flags/version and source hash.
`tools/reference_cta_workflow.py` compiles the unchanged whole source, supplies
all analysis choices, enables binomial details, and repeats with retained choices.
`tests/test_cta_workflow.py` compares all seven analyses and detailed cells against
that report. Tolerances reflect each field's printed precision; higher-precision
routine fixtures and independent mathematical tests provide tighter checks.

The 329 CTA tests include native cases, independent rational/closed-form checks,
exhaustive small-table tests, shape/orientation invariants, zero and extreme
counts, uncertainty conventions, shared studies and real report-file integration.
The fixtures do not require a Fortran compiler in normal package testing.

The audit retains these deliberate differences:

- Accurate double-precision calculations replace single-precision rounding,
  overflow stubs and cancellation-prone CDF subtraction.
- Default Yates is clipped and limited to 2x2; source formulas remain explicit.
- Default kappa variances correct asymmetric source indexing; legacy can retain
  negative source-formula variances.
- Default diagnostic errors are probability standard errors; legacy supplies
  the source's count standard deviations.
- RELRISK is correctly named an odds ratio; default intervals use a normal
  quantile, while legacy retains the erroneous CDF multiplier.
- Default Fisher is probability-ordered two-sided; source direction and cutoff
  are explicit options rather than silently inferred standard inference.
- Default conditional binomial comparison uses capped doubled-minimum tails;
  legacy retains duplicate-tail sums that can exceed one.
- Undefined statistics are explicit, malformed inputs fail, fractional exact-test
  counts are rejected, and reports preserve fractional descriptive observations.

These differences are covered by tests and documented for users; “implemented”
does not claim faithful reproduction of undefined behavior or all source bugs.

## Performance evidence

`tools/benchmark_cta.py` compares one array call with 10,000 calls to the same
Python API on deterministic positive 2x2 tables, checking equality on each of
three repetitions. Median speedups in [cta-benchmark.json](cta-benchmark.json)
are 213.30x for chi-square statistics, 531.82x for diagnostic standard errors,
and 349.62x for conditional binomial p-values. The environment and timings are
recorded. These measure Python batching, not acceleration against the Fortran
executable, report formatting or a universal workload.
