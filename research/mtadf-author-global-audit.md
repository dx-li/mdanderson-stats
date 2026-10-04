# MTADF author global logistic fit audit

## Recovered model and call

The primary author reference is
`research/raw/mtadf-author-reference/targetAgentDF.r` (SHA-256
`e27be5581fdcb7c71a7739a4f1b25026dc054896fcbf7134ff9b1a1c8960caac`). The
real-trial global function `df.logistic` is at lines 516–565. It computes
`x1s=(1:J-mean(1:J))/(2*sd(1:J))`, `x2s=x1s^2`, expands binary responses via
`collapse(yeff,n)`, and repeats both columns by dose patient counts before
calling `arm::bayesglm(yreg~x1reg+x2reg,
family=binomial(link="logit"))`. It evaluates the fitted quadratic on every
dose, chooses the rightmost maximum, moves one level toward it for interim
conduct, and clips against a newly computed admissible-dose count (lines
549–565). No prior or iteration arguments are specified at the callsite.

## `arm::bayesglm.fit` contract inspected

The source inspected is official CRAN `arm` 1.6-06.01, a pre-file-date
archive, with 1.6-07.01 used as a nearby comparator. The historical app's
exact installed version is unknown. Tarball hashes and cache paths are in
`research/mtadf-author-logistic-source-audit.md`; those ignored archives are
not included in this repository.

Relevant source in `arm/R/bayesglm.R`:

- `bayesglm` defaults to `prior.df=1`, `scaled=TRUE`, logit scalar
  `prior.scale=2.5`, and `n.iter=100`; `bayesglm.fit` delegates to the same
  fitting loop with `glm.control(maxit=n.iter)` (lines 1–10, 77–85,
  117–128).
- `.bayesglm.fit.initialize.priors` expands scalar prior means/scales/df to
  `nvars`, including the model-matrix intercept. The manual separately
  describes an intercept scale of 10, but this implementation uses that
  special scale only when a non-scalar `prior.scale` vector is supplied.
  Since the author passes no prior arguments, the observed scalar-default
  path expands scale 2.5 to the intercept as well (lines 275–325).
- `.bayesglm.fit.initialize.priorScale` divides each scale by 1 for a
  one-value column, the range for exactly two distinct values, and `2*sd(x)`
  for more than two values; scales are floored at `1e-12` (lines 327–354).
  Because `bayesglm` receives patient-expanded columns, the unique count,
  range, and sample standard deviation are from the expanded prefix, not an
  unweighted dose grid.
- `.bayesglm.fit.initialize.x` appends one prior row per coefficient, then
  replaces the first (intercept) prior row with column means of the observed
  design when `intercept & scaled` (lines 356–365). The prior row is centered
  at the mean design profile; its likelihood rows remain the patient data.
- The fit loop starts from the binomial family's `mustart` on individual
  binary observations. Under unit weights this is `.25` for a zero and `.75`
  for a one. In the first working response, each Bernoulli row has squared
  weight `.1875` and `z=±(log(3)+4/3)`. The Python implementation groups these
  two outcome categories by dose without materializing patient rows; later
  iterations use the usual grouped-binomial IRLS weights and working response
  (lines 404–423, 457–487, 525–535).
- Before each weighted QR update, prior rows have weight
  `sqrt(dispersion)/prior.sd`; after each fit, the binomial dispersion is 1
  and each finite-df prior scale updates to
  `sqrt(((centered_coefficient-prior_mean)^2 + sampling_variance +
  df*prior_scale^2)/(1+df))`. For the intercept, its centered coefficient is
  the fitted linear predictor at the expanded design-column means and its
  sampling variance is the covariance quadratic form there; slope variances
  are covariance diagonal entries (lines 529–572).
- Default `glm.control` has epsilon `1e-8`; for binomial fits the dispersion
  remains 1, so stopping requires relative deviance change below epsilon after
  the first iteration. The loop runs up to 100 steps. It warns on
  nonconvergence; it does not convert the returned final estimate into a
  Cauchy-prior MAP (lines 498–523, 744–814).

The implementation uses grouped count algebra and stable NumPy QR rather than
patient-row storage. It matches the recovered fit's first weighted step,
adaptive scale updates, centered intercept prior row and deviance stopping
criterion. It rejects a nonconverged estimate for dose decisions while
retaining the fit object for inspection. Numerical comparisons to direct
`bayesglm.fit` output for four deterministic prefixes are in
`tests/fixtures/mtadf-author-global-reference.csv`.

## Toxicity gate and dose conduct

The author source at lines 520–536 calibrates a Beta prior with total
concentration `.5` and `P(Beta(.3) lower-tail)=.22`, computes per-dose
posterior overdose probabilities, applies increasing PAVA, and sets the
admissible count to `max(1, count(adjusted<=ct))`. The Python decision reuses
the shared author-reference safety helper so the isotonic and global APIs
share this calibration and inclusive threshold.

At lines 558–563, the source selects `tail(which(est==max(est)),1)`. If the
target lies above current, it advances one dose and clips by both the grid end
and fresh cap. If below, it retreats one dose (with floor one) and clips by
the cap; if equal, it stays subject to the cap. Final selection is represented
by an explicit Python `final=True` mode that goes directly to the fitted peak
then caps it. The underlying source function accepts one current dose and
always applies its one-step rule.

## Independent reference

`tools/reference_mtadf_author_global.R` loads only function definitions from
the cached `arm/R/bayesglm.R` member and invokes the private
`bayesglm.fit` directly; it neither installs arm nor runs the author driver.
Set `MTADF_ARM_ARCHIVE` to the ignored cached 1.6-06.01 tarball before running
the script. The four cases cover one observed dose, unequal patient
replication across a prefix, all-zero outcomes and all-one outcomes. Each
records coefficients, full-grid fitted probabilities, the rightmost peak dose, source prior scales,
final adaptive scales, deviance, iterations, and convergence.

This validates the coefficient and decision contracts against a recovered
version of the dependency. It does not establish exact app-version identity,
byte-identical output, or random-stream parity for source simulations.

For one observed dose, the observed design row equals the centered intercept
prior row. The other two prior rows therefore give exactly zero slope and
quadratic coefficients in the mathematical weighted least-squares solution.
Python removes QR roundoff in those coefficients after fitting and returns an
exactly flat curve. The direct R fixture's rightmost peak is the highest grid
dose in the inspected case; an uncorrected Python QR fit selected the lowest
dose because of coefficients on the order of `1e-16`. This is an explicit
numerical convention for the one-observed-dose case, not a general tie
tolerance or a claim that all native platforms emit identical rounding.
