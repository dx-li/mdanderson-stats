# TPI informative-prior implementation audit

The cached author slides under `research/raw/TPI/ji-2009-slides.pdf` describe
independent dose probabilities with i.i.d. `Beta(alpha, alpha)` priors and
Beta posteriors after binomial observations. They do not specify a
dose-varying prior calibration or native interface. The per-dose pair
extension in `TPIDesign` therefore implements the same conjugate Beta update
with caller-supplied shapes as an explicit Python generalization, not a
recovered native TPI feature. Common `prior=(a,b)` behavior and its defaults
remain unchanged.

The 2007 original-paper URL recorded in `docs/tpi-sources.json` was not
available in the cached source directory. The neighboring cached
`paper-bioc.xml` is a 2010 mTPI paper (DOI 10.1177/1740774510382799), not the
original TPI paper, and was not used to attribute dose-specific priors. The
slides are the only local primary artifact used for the prior-family
statement.

Dose-specific prior inputs are one `(a,b)` pair per dose. Count arrays use
their final axis as the dose axis; scalar and single-dose posterior summaries
must identify a one-based dose. Action tables are dose-specific in this mode.
The complete-outcome simulator precomputes per-distinct-prior ordinary and
escalation-barred moves plus unsafe states in bounded compact arrays before
constructing or consuming its random generator. Duplicate prior rows share
lookup tables. The existing common-prior precomputation and random draw path
is left intact.

Eight affected TPI tests pass, covering common-prior behavior, dose-axis
posterior summaries, explicit-dose tables and simulation. The worker run took
3.79 seconds, peaked at 136.59 MiB and reported zero swaps. Targeted Ruff
and mypy checks pass. No native dose-specific prior calibration or application
parity is claimed.

## Independent integration checks

`tools/reference_tpi_informative.py` computes 30 integer-shape Beta posterior
references using 70-digit Decimal arithmetic and finite binomial sums, without
NumPy, SciPy or package imports. Posterior means, standard deviations, interval
bounds/masses and overdose probabilities match within 3.89e-16 absolute error.
The fixture is `tests/fixtures/tpi-informative-beta.csv`.

Six seeded common-prior comparisons (240 trials) match every result field and
readonly flag from published `78d319f`. Separately, exact enumeration of six
patient-level Bernoulli outcomes yields 21 distinct terminal trial states for
a heterogeneous three-dose prior. A 4,096-trial batched run agrees with exact
selection probabilities and enrollment/toxicity means within six Monte Carlo
standard errors plus one-trial discretization tolerance. This checks the
compact simulator against direct patient-level decisions and final selection.
The independent run takes 0.545 seconds, peaks at 124.48 MiB and reports zero swaps.
