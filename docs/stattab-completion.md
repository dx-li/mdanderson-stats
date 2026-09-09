# STATTAB completion audit

Catalog entry **23, STATTAB**, is implemented with documented Python semantics.
The package supplies the twelve-distribution application, its 42 supported computed
parameter groups, individual discrete probabilities, table/list workflow, reuse,
help, formatted reports, and file dialogs. Start it with
`python -m mdanderson_stats.stattab`, or use `stattab_solve`, `STATTABSession`,
and `run_stattab` from Python.

The [archive inventory](stattab-archive.json) accounts for all 34 regular files.
The [shared-source review](stattab-shared-source.md) reconciles all nineteen shared
modules and 259 compiled public imports. The [completion manifest](stattab-completion.json)
maps all seventeen procedures in the main program and every manual section to
implementation and validation evidence. These inventories establish scope;
correctness comes from the independent numerical and workflow tests they reference.

## Manual reconciliation

All 25 pages of the pinned PDF were read, including the annotated run and formulas.
Pages 21–25 were rendered and visually inspected to distinguish printed mathematical
errors from extraction errors. The accompanying LaTeX definitions were inspected;
the PostScript rendition was converted and its extracted content compared with
the PDF. Document hashes and the comparison outcome are recorded in the manifest.

| PDF pages | Requirement | Evidence and disposition |
|---|---|---|
| 1–4 | Identity, terms, references and acknowledgments | The archive labeled 1.3 contains the 2.0 March 2002 manual. Exact LEGALITIES bytes and attribution remain packaged; Python modifications are identified |
| 5–6 | Twelve distributions, extra probabilities, continuous count CDFs, truncated terms, numeric syntax, complementary inputs, unknown/list/reuse markers | Probability, result and session tests cover all supported families, groups and input positions, including small complementary tails |
| 7–12 | Menu, report, help, t calculation and inverse table | Manual tests replay help, t=1.97 with df=9, reuse and nine inverse-table rows through the console; independent angular integration checks the t probabilities |
| 12–14 | Continuous count inverses and adjacent integer rows | All five count inversions are covered; the manual Poisson mean=10 table and fourteen neighbor probabilities are replayed and checked with finite Poisson sums |
| 14–15 | Unattainable inverse and error notification | The manual binomial target 0.01 with n=5 and chance=0.5 is rejected; the previous successful result remains available |
| 15–17 | Eleven-row binomial table | Console replay checks each mass and cumulative probability against exact combinatorial sums |
| 17–20 | Normal/t two-sided and chi-square/F upper-tail p-values | Both tails remain available; extra normal/t columns use twice the smaller tail. The normal height example is checked against erfc |
| 20–25 | Parameterization and distribution definitions | Each family has Python help; corrected formulas and explicit gamma rate/shape names resolve the errata below |

The p-value discussion illustrates how a caller selects a tail for an already
calculated statistic. It does not define additional regression fitting, data
analysis or experimental-design software interfaces. Both tails are returned so
the caller can use the direction appropriate to the hypothesis.

## Manual errata and explicit choices

- Page 6's approximate IEEE magnitude discussion is obsolete for subnormal values.
  Python retains representable float64 subnormals; nonzero numeric literals that
  underflow during parsing and overflowing/nonfinite inputs are rejected explicitly.
- Page 10's commentary describes `=` although the displayed request repeats `9`.
  Both are accepted after a completed calculation. Page 11's interval commentary
  says 0.9–0.8 while its actual endpoints are 0.1 and 0.9; eight intervals give nine rows.
- Page 13 informally describes strict neighboring integers. Source behavior is
  floor and floor+1, including an integral continuous answer. Python preserves that
  rule and marks invalid or unrepresentable neighbors unavailable.
- Page 15's failed inversion prints a boundary value despite failure. Python raises
  a numerical error and the console reports it without publishing a misleading answer.
  A batch is transactional; valid preceding rows are not silently committed after failure.
- Page 21's binomial and negative-binomial expressions omit combinatorial factors.
  Those factors depend on the count, so the expressions are not complete mass functions
  up to a count-independent normalizer. Python includes them. The negative-binomial
  description's `M` is the supplied failure count, called `f` in Python.
- Page 22's noncentral chi-square definition omits the square on each shifted normal.
  The intended variable is the sum of `(N_i+d_i)^2`, with noncentrality equal to the
  sum of squared shifts. Python's Poisson-mixture implementation and help use that law.
- Page 23 labels gamma A as scale although `exp(-A*t)` makes A a rate. Its undefined
  exponent R should be shape B. Python uses `rate` and `shape`, with density proportional
  to `t^(shape-1)*exp(-rate*t)`.
- Page 24 has a stray slash before the normal exponential. Python uses the normal CDF
  and standard deviation parameterization. The omitted Poisson `exp(-mean)` factor is
  constant in the count, but Python includes it to produce normalized probabilities.
- Page 25's noncentral-t definition requires an independent standard normal numerator
  and central chi-square denominator. Python help makes that independence explicit.

The source also supplies negative-binomial individual terms even though page 5
mentions only binomial and Poisson terms. Python includes all three. The application
rejects F and noncentral-F degree-of-freedom inversion, matching its source; additional
legacy-library APIs remain separate. Noncentral-t degree-of-freedom inversion may
have multiple roots; a session bracket can select an interval. Numerical search
limits and explicit failures are documented in the underlying distribution APIs.

## Source and historical artifacts

`stattab_main`'s twelve solver procedures map to the result dispatcher. Its parser,
help, distribution loop and two status procedures map to the session, reporting,
console and exception handling layers. The added file module maps to the two Python
file-dialog helpers. Descriptor fields map to `STATTABDistribution`, the immutable
menu collection, help text and formatted columns; no Fortran memory-layout alias is
claimed. Shared modules reuse the validated CDFLIB90 foundation.

The three native build scripts are superseded by `pyproject.toml` and Python wheel
installation. The Windows and Mac binaries are inventoried; tests rebuild unchanged
source rather than running historical executables. The `.log` records an interrupted
TeX run with no output pages and introduces no runtime feature. HOWTOGET contains
historical acquisition information. INSTALL incorrectly names `confint`; the source
and executable names establish that this archive is STATTAB. PDF, PostScript and
LaTeX provide the same application manual scope. Exact license/notice bytes remain
in the wheel; archived native code and binaries are not bundled.

## Validation and performance boundaries

The tests cover native sessions, independent probability identities and high-precision
extreme cases, all computed groups and table positions, state transitions, all eight
list-editor actions, file modes/cancellation/ownership, CLI invocation, and the worked
manual examples. Native undefined outputs and known bugs are recorded as defects,
never adopted as numerical oracles. Text is deliberately reformatted; the old terminal
layout, Fortran unit numbers, process STOP behavior and mutable global state are
replaced by bounded text, streams, exceptions and explicit session objects.

The published probability, result and session benchmarks measure vectorized batches
against repeated scalar calls to the same Python implementation. They show batching
benefits; they do not claim speedups over native Fortran or faster terminal printing.
The full package's test, lint, type, build and installed-wheel checks remain the
release gates. Completion of STATTAB advances one catalog entry; it does not imply
that the other pending catalog programs are implemented.
