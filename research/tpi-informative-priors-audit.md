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

Validation should cover common-prior regression, direct Beta posterior means
and interval masses under dose-specific shapes, selected-dose action tables,
dose-axis mismatch errors, and seeded simulation against a small path
enumeration. No native dose-specific prior calibration or application parity
is claimed.
