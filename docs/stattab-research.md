# STATTAB source and workflow audit

Catalog entry **23, STATTAB**, is partially implemented. Its
[discrete probability terms](stattab-probability.md) are available; the application
workflow and remaining responsibilities below are still pending. The pinned
[archive inventory](stattab-archive.json) and [104 native sessions](../tests/fixtures/stattab.json)
establish its application scope and defect evidence. They do not turn the existing
CDFLIB90 library into a completed STATTAB application.

## Archive identity

The catalog advertises `STATTAB      _V1.3.zip`, containing a `stattab13` directory.
Its manual and unchanged compiled program both identify themselves as
**Version 2.0: March, 2002**. The reference is the exact downloaded archive,
SHA-256 `0325f15818176418774216e546e0f61267cfd228cef184e63101c68e2b7e76f9`.
The PDF has 25 pages; its title and calculation notes were visually checked.
CDFLIB90's embedded STATTAB `DOC.TEX` is an additional historical manual, not a
substitute for this separate application archive.

| Material | Files | Disposition |
|---|---:|---|
| F95 source | 22 | Twelve distributions, seven shared support modules, file I/O, application descriptors and the main program |
| Build instructions | 3 | All 22 source files compile unchanged in archived order with gfortran |
| Historical build log | 1 | Interrupted TeX run producing no pages; no additional program interface |
| Windows/Mac executables | 2 | Inventoried; reference sessions use the rebuilt source, not these binaries |
| Documentation | 6 | PDF/PS/LaTeX manual, HOWTOGET, INSTALL and LEGALITIES |

Nineteen source filenames also occur in CDFLIB90. Only `biomath_mathlib_mod` and
`cdf_gamma_mod` are byte-identical. The other seventeen contain changes, including
explicit declarations, formatting and public attributes; byte differences alone
do not establish numerical differences or equivalence. The inventory retains both
hashes and source declarations, including inline PUBLIC attributes. Full cross-version
contract reconciliation remains required. The application adds `biomath_file_io_mod`,
`stattab_aux_mod` and `stattab_main`.

The INSTALL file incorrectly names `confint`; the actual source, build scripts,
executables and runtime banner identify STATTAB. Historical compiler/platform
instructions will be replaced by Python packaging. The original
[LEGALITIES](../notices/mdanderson-stattab-LEGALITIES.txt) is retained byte-for-byte;
original native source and executables are not bundled in the Python wheel.

## Application contract still to implement

| Responsibility | Required Python behavior and current evidence |
|---|---|
| Twelve distributions | Preserve the source menu's beta, binomial, negative-binomial, chi-square, noncentral chi-square, F, noncentral F, gamma, normal, Poisson, t and noncentral t workflows |
| Computed parameters | All 42 supported computed groups, with source-order parameter mapping; every group has a native session |
| F/noncentral F df | Source explicitly rejects df inversion in this application; expose any additional legacy-library df API separately and document the distinction |
| Complementary values | Accept a primary or complementary coordinate/probability; preserve the smaller tail and calculate the paired output |
| Parameter requests | Exactly one unknown (`?`), omitted complement (`.`), optional list selector (`T`), numeric formats, separators and comments |
| Reuse | `=` must refer to a defined previous value of the same distribution; preserve completed values without cross-call contamination |
| Tables | Batched results for a list at any supported input position, including complementary inputs and gamma's reordered parameters; reset table state between requests |
| List editing | All eight source actions, capacity policy, pagination, linear/logarithmic sequences, deletions, sorting and duplicate handling; existing CDFLIB list support is a foundation, not application validation |
| Extra probability columns | Two-sided normal/t probabilities; chi-square/F many-sided probabilities are their existing upper-tail columns, not extra density-ordered tests |
| Discrete terms | [Implemented](stattab-probability.md): binomial, negative-binomial and Poisson individual probabilities with consistent truncation; the manual only mentions binomial/Poisson but source also includes negative-binomial |
| Count inversions | Retain the continuous solution and the separate neighboring integer evaluations; binomial/negative-binomial have both count inversions, Poisson has its event-count inversion |
| Gamma ordering | Input A is the rate and B the shape; the source swaps them before CDF evaluation and prints shape before rate; Python names and report labels must remove ambiguity |
| Output/reporting | Structured results, source column meanings, formatted tables, per-session streams and report-file output; no silently printed invalid or uninitialized results |
| File I/O | `open_file` and `report_file_dialogue`, including read/write selection, existing-file/append/new-file policies, errors and caller ownership |
| Help and examples | Distribution/parameter help, annotated manual workflow and report examples |
| Failures | Checked invalid input, finite arithmetic, well-defined numerical failures and resource limits; no process STOP or stale answers |
| Public support | Reconcile all changed shared modules and the added descriptor/file-I/O interfaces; retain justified replacements explicitly |
| Delivery | Python implementations, behavioral/numerical tests, relevant batching benchmarks, installed-wheel checks and full catalog metadata update |

The completed CDFLIB90 kernels, lexer, console, list editor and formatting routines
provide reusable foundations. STATTAB-specific workflow mapping, result construction,
state management and remaining outputs still require implementation and tests.

## Native evidence and independent checks

`tools/reference_stattab.py` compiles all 22 unchanged sources with `-std=legacy`,
`-O0`, `-ffp-contract=off` and `-fcheck=all`. Each complete session has a three-second
limit and its own temporary directory. The fixture records source hashes, compiler
version/diagnostics, exact inputs, stdout, stderr, exit codes and report contents.
All 104 sessions finish within that limit; two terminate with checked runtime errors.
A zero process exit means the session ended, not that every calculation succeeded.

Sessions cover all 42 computed groups, upper-tail queries and tables for every
family, discrete fractional counts, negative/zero normal and t inputs, input errors,
help, comments, repeated calculations and report output. This initial set does not
cover every list-editing action, every file-dialog branch, or all extreme numerical
domains. Those remain in the requirements above.

`tests/test_stattab_reference.py` validates ordinary output using independent beta
polynomials, exact binomial sums, exponential/gamma identities, normal erfc, a
closed-form t(5) integral, a Poisson mixture for noncentral F, and conditional
quadrature for noncentral t. Quadrature is checked at two resolutions. Comparisons
respect the native six-decimal display precision; they do not assert full binary64
accuracy from a rounded table. Exact-state and error tests classify the defects below.

## Defects to repair, not preserve as reference answers

- The beta median inverse for Beta(2,2) prints X=0.75 and 1-X=0.5 without an error.
  Both the complement sum and the beta polynomial contradict this result.
- Binomial and negative-binomial chance inversions can print zero chance for a
  positive target CDF. The F(4,4) median inverse prints 1.000344 instead of 1.
- Several forward calculations have mathematically correct tails while a stale root
  status reports a lower-bound failure. For negative-binomial calculations, this
  also prevents calculation of the individual term.
  At F=2, S=3, P=0.5, the correct mass is 0.1875; the failed session prints zero.
- Failed noncentral calculations may print uninitialized values. Their raw bytes
  are retained for provenance, but these values are not stable numerical oracles.
- For fractional counts below one, binomial and Poisson use the continuous CDF as
  the individual probability rather than the probability at the truncated count.
  For example, Poisson count 0.5 and mean 3 prints about 0.111610 instead of exp(-3).
  Above one, the source does truncate counts for the term.
- Poisson term calculation overwrites saved tail values. After count 2, mean 3,
  reusing the prior CDF with `=` uses the CDF at count 1 and returns the wrong inverse.
- A prior `T` request leaves its table position active. A later scalar request
  unexpectedly opens an empty list editor and may produce no answer.
- A numeric overflow such as `1e999` is silently accepted as zero. Extra parameters
  access beyond the descriptor array before validating parameter count; parameter
  EOF invokes the Fortran runtime. First-use `=` reads values without a defined
  prior calculation. These must become explicit Python input errors.
- A HELP substring such as `SHELPER` invokes help; the intended command grammar
  needs an explicit exact-token policy.

The application also prints failed zero-mean Poisson results although the
mathematical degenerate distribution is well-defined. A Python extension to that
boundary must be deliberate and tested, using the package's existing zero-mean
legacy kernel where appropriate. Existing documented CDFLIB numerical repairs
remain relevant, but cannot replace STATTAB workflow validation.

## Reproduction

```text
uv run python tools/audit_stattab.py
uv run python tools/reference_stattab.py
uv run pytest -q tests/test_stattab_reference.py
```

The native tools require the pinned ignored archive and, for the inventory,
its exact extraction under `research/raw/STATTAB/source`; the reference tool
also requires gfortran. The committed fixture tests require neither original
source nor a native compiler. This audit makes no performance claim and leaves
the full STATTAB application unfinished in the 138-entry conversion goal.
