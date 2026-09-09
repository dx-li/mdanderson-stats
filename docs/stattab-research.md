# STATTAB source and workflow audit

Catalog entry **23, STATTAB**, is implemented. Its
[discrete probability terms](stattab-probability.md),
[structured distribution results](stattab-results.md),
[requests and sessions](stattab-sessions.md), and
[console/reporting application](stattab-console.md) are available. The
[completion audit](stattab-completion.md) reconciles the source and manual. The pinned
[archive inventory](stattab-archive.json) and [104 native sessions](../tests/fixtures/stattab.json)
establish its application scope and defect evidence. The application adds tested
workflow layers to the existing CDFLIB90 foundation.

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
hashes and source declarations, including inline PUBLIC attributes. The
[shared-source reconciliation](stattab-shared-source.md) reviews all nineteen modules
and compiles 259 public imports against the STATTAB archive. The application adds `biomath_file_io_mod`,
`stattab_aux_mod` and `stattab_main`.

The INSTALL file incorrectly names `confint`; the actual source, build scripts,
executables and runtime banner identify STATTAB. Historical compiler/platform
instructions are replaced by Python packaging. The original
[LEGALITIES](../notices/mdanderson-stattab-LEGALITIES.txt) is retained byte-for-byte;
original native source and executables are not bundled in the Python wheel.

## Application coverage

| Responsibility | Required Python behavior and current evidence |
|---|---|
| Twelve distributions | [Numerical result layer implemented](stattab-results.md) for all twelve source families; console selection and requests are implemented |
| Computed parameters | Implemented: all 42 supported computed groups with named parameters and source-order columns; native sessions and independent target tests cover every group |
| F/noncentral F df | Source explicitly rejects df inversion in this application; expose any additional legacy-library df API separately and document the distinction |
| Complementary values | Implemented in numerical results: accept either member, preserve the smaller tail and calculate the paired output; positional omission syntax is implemented |
| Parameter requests | Implemented: exactly one unknown (`?`), omitted complement (`.`), optional list selector (`T`), numeric formats, separators and comments |
| Reuse | Implemented: defined last-row reuse within one selected distribution, saved tiny complements, no cross-call contamination and transactional failure handling |
| Tables | Numerical batches implemented at every supported input position, including complements and gamma parameters; request-local T selection, bounded list snapshots and state reset implemented; interactive list-editor dialogue implemented |
| List editing | Implemented and tested through the application: all eight source actions, capacity policy, pagination, linear/logarithmic sequences, deletions, sorting and duplicate handling |
| Extra probability columns | Implemented for forward results: two-sided normal/t probabilities; chi-square/F many-sided probabilities are their existing upper-tail columns |
| Discrete terms | [Implemented](stattab-probability.md): binomial, negative-binomial and Poisson individual probabilities with consistent truncation; the manual only mentions binomial/Poisson but source also includes negative-binomial |
| Count inversions | Implemented for all five count inversions: continuous solution plus separate floor/floor+1 rows, explicit invalid masks and compact valid outputs |
| Gamma ordering | Implemented named rate/shape inputs and source-order result columns; source A is rate and B is shape |
| Output/reporting | Implemented: structured/source-order results, bounded formatted tables, caller-owned streams and optionally owned report files |
| File I/O | Implemented as stattab_open_file and stattab_report_file_dialogue: read/create/overwrite/append, cancel/retry/confirmation, errors and explicit stream ownership |
| Help and examples | Implemented distribution/parameter/formula help and annotated console/report examples; all manual sections and worked examples reconciled in the completion audit |
| Failures | Checked invalid input, finite arithmetic, well-defined numerical failures and resource limits; no process STOP or stale answers |
| Public support | [Shared-source review complete](stattab-shared-source.md): all nineteen shared modules and seven added public interfaces mapped, with 259 native imports compiled |
| Delivery | Python implementations, behavioral/numerical tests, relevant batching benchmarks, installed-wheel checks and full catalog metadata update |

The completed CDFLIB90 kernels, lexer, console, list editor and formatting routines
provide reusable foundations. The numerical result layer adds application parameter
mapping and extra outputs. The request/session layer adds checked parsing and
transactional reuse. The console layer integrates menus, list editing, help, output
and file dialogs. Source and manual reconciliation is complete; see the
[completion audit](stattab-completion.md) for evidence and explicit semantic replacements.

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
mathematical degenerate distribution is well-defined. The result layer deliberately
extends forward evaluation at that boundary through the existing zero-mean legacy
kernel, with mixed-batch tests. Parameter inversions retain their documented
CDFLIB90 domains. Existing documented CDFLIB numerical repairs remain relevant,
but cannot replace STATTAB workflow validation.

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
