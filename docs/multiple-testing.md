# MULTI: implemented procedures and remaining scope

Catalog entry 50 includes a desktop program and an S library. This port currently
covers the desktop's nine adjustment/threshold procedures, two sharpened procedures,
and Schweder line fitting, plus the S library's Schweder bootstrap,
order-statistic diagnostics, clustered p-value generation, and S/desktop
nonparametric fitting. [Beta-mixture evaluation, initialization, and EM fitting](beta-mixtures.md)
are also available. It remains **partial**: the direct likelihood optimizer,
automatic component selection, simulation-based model checks, and remaining
plotting/reporting workflows still need implementations and validation.
The catalog does not count this entry as complete.

## Adjustment procedures

```python
from mdanderson_stats import multiple_testing

result = multiple_testing([0.04, 0.001, 0.2, 0.03], "holm", alpha=0.05)
print(result.adjusted_pvalues)
print(result.reject)

rom = multiple_testing([0.02, 0.021, 0.8], "rom", alpha=0.05)
print(rom.critical_values)  # Rom returns thresholds, not adjusted p-values
```

For sorted p-values p(i), m tests and i=1,...,m, these are the source's algorithms:

| Source option | Python method | Rankwise quantity and accumulation |
|---|---|---|
| One-step Bonferroni | `bonferroni` | m*p(i) |
| One-step Sidak | `sidak` | 1-(1-p(i))^m |
| Step-down Holm | `holm` | Running maximum of (m-i+1)*p(i) |
| Step-down Sidak | `holm-sidak` | Running maximum of 1-(1-p(i))^(m-i+1) |
| Step-down Finner | `finner` | Running maximum of 1-(1-p(i))^(m/i) |
| Step-up Hochberg | `hochberg` | Reverse running minimum of (m-i+1)*p(i) |
| Step-up “Hommel” | `multi-hommel` | Reverse running minimum of m*H(m)*p(i)/i, H(m)=sum(1/j) |
| Step-up Simes | `simes` | Reverse running minimum of m*p(i)/i |
| Step-up Rom | `rom` | Alpha-dependent recursive critical values, followed by step-up rejection |

Adjusted p-values are capped at one. The last axis is one testing family;
leading axes are independent batches. Results retain the original input order,
and `order` records the sorted-rank indices. Input p-values must be finite and
in [0,1]; invalid values are not discarded. Alpha is in [0,1].

The S library's function named `bonferroni` actually runs sequential Holm;
its equivalent here is `method="holm"`. The Python `bonferroni` option follows
the desktop's one-step option. The S `sidak` function maps to `method="sidak"`.

The source's “Hommel” formula is the harmonic step-up adjustment, mathematically
the BY formula. It is not the usual Hommel closed-testing method. The ambiguous
`hommel` string is therefore rejected; use `multi-hommel` to request the archived
formula. Similarly, the source's step-up Simes adjusted values use the BH formula,
not a single global Simes p-value. Holm controls familywise error without a
dependence assumption; BH and BY target false discovery rate, with BY allowing
arbitrary dependence. These are different statistical guarantees.
See the [R statistics documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)
for these distinctions. The other procedures have their own dependence conditions;
their presence here does not imply that they can be interchanged in every design.

Rom's critical values are exposed by `rom_critical_values(m, alpha)`. They are in
ascending rank order. The implementation evaluates the original recurrence using
log-domain binomial terms to avoid coefficient overflow; it does not mistake those
cutoffs for adjusted p-values. A qualifying larger rank also rejects smaller ranks,
even when a smaller rank does not meet its own cutoff. The Sidak/Finner transforms
use `log1p` and `expm1` so p-values below machine epsilon do not round to zero.

## Schweder fitting and sharpened decisions

```python
from mdanderson_stats import schweder_fit, schweder_bootstrap, sharpened_testing

p = [0.1, 0.1, 0.2, 0.4, 0.6, 0.8, 0.8]
fit = schweder_fit(p)
decisions = sharpened_testing(p, fit.null_estimate, "hochberg", alpha=0.05)
bootstrap = schweder_bootstrap([i / 41 for i in range(1, 41)], samples=100, rng=123)
```

The line uses unique p-values in descending order, x=1-p, and cumulative counts of
observations **at or above** p. Those counts include ties. The source divides them
by the number of distinct p-values while fitting an origin-constrained line, then
rescales the slope to a null-count estimate. The Python result exposes the plot
coordinates and the number of retained points. It uses the source's prediction
bound to trim the rightmost point until a line is accepted. At least four distinct
values are required; failure raises `SchwederFitError` rather than returning zero.

`prediction_variance` reproduces the S library's BETAVR value on its normalized
ordinate scale. It is not the variance of the null-count estimate. The bootstrap
reports sample variance of fitted null-count estimates instead. Like SWBOOT, it
requires the requested number of successful fits within at most twice that many
attempts. It returns original-data plot coordinates rather than the last resample's
coordinates. An explicit NumPy Generator or seed replaces the old global two-seed
generator; bitwise reproduction of the original random stream is not claimed.

The sharpened methods use the source's integer truncation and capping of the
supplied null-count estimate. A value below one causes all hypotheses to be rejected,
as in the original. A failed line fit must not be substituted with zero. These
estimate-dependent decisions do not automatically inherit a general FWER guarantee.

The original sharpened Holm routine returns after its second rejection threshold;
the subsequent denominator update described in its comments is unreachable.
`sharpened_testing(..., "holm")` preserves the executable's behavior. The Hochberg
variant follows the source's descending-rank thresholds. Neither returns fabricated
adjusted p-values.

## Order-statistic diagnostics

```python
from mdanderson_stats import order_statistic_diagnostics

diagnostics = order_statistic_diagnostics([0.04, 0.001, 0.2, 0.03])
print(diagnostics.cumulative)
print(diagnostics.legacy_transformed_cdf)
print(diagnostics.combined_score)
```

This ports the S library's `multi.os`/`OSFIT`. For sorted p-values x(i), the
cumulative diagnostic is the Beta(i, n-i+1) CDF at x(i). Equivalently, it is
the probability that at least i of n independent uniform draws are at most x(i).
It is not a posterior probability that a particular null hypothesis is true.

The second output deliberately uses the executable source's formula:
Beta(1, n-i+1) CDF at min(x(i)/(1-x(i)), 1), except the first rank uses x(1)
without transformation. The documentation and comments instead describe a
denominator involving x(i-1); that is not what the archived code computes.
`legacy_transformed_cdf` makes this discrepancy explicit and does not label the
result as an ordinary multiple-testing adjusted p-value or spacing probability.
Values at one are handled without division by zero, and small tails use
`log1p`/`expm1` arithmetic.

`combined_score` reproduces n*min(cumulative), including values above one.
For example, three input values equal to one give a score of three. It must not
be interpreted as a posterior probability. Output arrays retain input order;
`order` gives ascending-rank indices. Leading axes provide independent batches,
and `combined_score` has the shape of those leading axes.

## Clustered simulation

```python
from mdanderson_stats import clustered_pvalues

p = clustered_pvalues(100, 20, cluster_size=10, correlation=0.3, rng=123)
assert p.shape == (120, 10)
```

This implements `rcpval`/`CLUSTP`: each row is an independent cluster of
equicorrelated normal statistics converted to **one-sided upper-tail** p-values.
The first 100 rows in this example have zero normal mean; the final 20 have
`alternative_mean=1.96` by default. Flatten with `p.ravel()` to obtain the S
function's cluster-major vector. Two zero cluster counts return an empty array
with the requested cluster width. Counts must be nonnegative integers; invalid
counts are rejected rather than truncated or silently changed to zero.

`correlation` describes the latent normal statistics, not the p-values' Pearson
correlation. For cluster size k>1, the covariance must be positive definite:
-1/(k-1) < correlation < 1. For k=1, correlation is irrelevant but still must
lie in [-1,1]. The source's 100-observation workspace limit is removed.
NumPy's explicit seed/Generator replaces the global, single-precision Fortran
random stream. The distribution is preserved; identical legacy random draws
are not promised.

Mean and contrast projections apply the covariance square root in O(clusters*k)
work without constructing or factoring a k-by-k covariance matrix. The
[recorded benchmark](pvalue-models-benchmark.json), reproducible with
`uv run python tools/benchmark_pvalue_models.py`, compared 10,000 clusters of
size 200 against a precomputed dense square root with identical normal draws.
Both took about 0.042 seconds on that machine; no speed advantage was measured
at this workload. The structured method avoids the dense covariance storage and
its quadratic multiplication cost. These are Python implementation comparisons,
not original Fortran throughput measurements or CI performance thresholds.

## S nonparametric diagnostic

```python
from mdanderson_stats import nonparametric_pvalues

fit = nonparametric_pvalues([i / 21 for i in range(1, 21)])
print(fit.bandwidth)
print(fit.density)
print(fit.scores)
```

This implements the S library's `multi.np`/`NPFIT`, which differs from the
desktop's `NP1P` method described below. It accepts one vector with at least four
distinct p-values. Tied observations retain separate empirical ordinates rank/n.
Outputs retain input order, with sorted-rank indices in `order`.

The port preserves these executable conventions:

- The full bandwidth is the maximum width returned by the original nearest-point
  expansion rule. Its internal expansion counter is not a correct count of
  distinct points: some branches subtract an absolute coordinate from a distance.
  The Python implementation follows those branches, while preventing out-of-bounds
  reads. Do not interpret the bandwidth as a corrected nearest-neighbor estimate.
- The S source resets its cross-validation minimum to zero for every candidate.
  Since the squared-error objective cannot be negative, the initial width always
  wins. The port returns that width directly, omitting the ineffective search.
- Windows extend half the full width on each side. The source's binary search
  includes the first tied observation at an exact upper endpoint, with a special
  two-element bracket at the minimum. Those tie conventions are preserved.
- Normalized kernel weights are proportional to `(1-((x-center)/width)**2)**2`.
  The regression uses their **reciprocals**, as the original `LQBETA` does; it
  does not use conventional kernel-weighted regression. Normalized weights at or
  below 1e-10 are excluded.
- A local quadratic derivative d is converted to `I_(1/n)(d+1, n-d)` when d<n,
  and zero otherwise. d is not rounded to an integer. The source therefore uses
  a fractional extension through the beta function, rather than an ordinary
  discrete binomial tail. A running maximum enforces increasing scores by rank.

`scores` are historical diagnostics, not calibrated posterior null probabilities
or FDR estimates. The source's description as a probability of a true null does
not by itself establish that interpretation. Local quadratic density estimates
may be negative. Values at or below -1 make the beta shape invalid and raise
`NonparametricFitError`; failed regression windows also raise instead of inserting
the original -1 sentinel into an output described as a probability.

The regression uses centered, scaled coordinates and a least-squares solve,
avoiding inversion of uncentered normal equations. This preserves the fitted
polynomial mathematically while materially improving numerical stability. On a
grid with spacing 2^-30 near 0.75, the Python result recovers the known slope
2^30/20; the original suffers large numerical error. Undefined original results
are not a compatibility target. No corrected bandwidth-selection algorithm is
silently substituted for the archived one.

`tools/reference_nonparametric.py` records eleven original S fits in
`tests/fixtures/nonparametric.json`. The NPFIT and WDTHMX symbols are renamed only
to distinguish their desktop counterparts; CUMBIN is extracted unchanged. The
shared smoothing helper bodies were verified identical to the S versions.
The fixtures cover the published example, random families, ties, endpoints,
linear/quadratic empirical CDFs, an ill-conditioned grid, and invalid beta shapes.
Ordinary comparisons allow for precision lost by the original normal equations;
independent known-polynomial tests verify the Python derivatives to near machine
precision. Integer-density cases are also checked by explicit binomial-tail
calculations. The narrow-grid and invalid-shape cases verify the documented
stability/error differences rather than asserting equality with defective output.

## Desktop nonparametric decisions

```python
from mdanderson_stats import nonparametric_testing

result = nonparametric_testing([i / 1000 for i in range(1, 21)], null_estimate=8.7)
print(result.fitted_points)  # 20 - int(8.7) = 12
print(result.scores)
print(result.reject)
```

`nonparametric_testing` implements desktop `NP1P`, `FUNCFT`, and `SMOOTH`.
It fits the smallest `n-int(null_estimate)` p-values, using the original integer
truncation. If the estimate is omitted, it uses `schweder_fit` at its default
alpha of 0.05, independently of the decision alpha, as the original interface
does. Invalid/negative estimates are rejected; failed Schweder fits propagate.
An estimate above the family size retains all hypotheses.

The fitted subset determines the regression branch:

| Fitted points | Model |
|---|---|
| Fewer than 2 | Retain all; no density fit |
| 2–5 | One least-squares line |
| 6–10 | One least-squares quadratic |
| More than 10 | Local quadratic windows with pointwise bandwidth selection |

Empirical ordinates are rank divided by the **full family size**, including when
only a subset is fitted. Polynomial fits use equal weights. Local fits retain
the same reciprocal biquartic weighting and historical window-bound rules as the
S version. Their initial bandwidth is computed separately at each fitted point.
When it is smaller than one quarter of the fitted range, the desktop evaluates
ten equally spaced widths, including the initial width and excluding that upper
limit. Each point selects the smallest leave-one-out weighted-mean squared error,
retaining the earlier width on exact ties. Floating-point summation can choose
different numerically tied widths than the original, as on exact linear grids;
the resulting density agrees on those validation cases. Unlike S NPFIT, this
bandwidth search
is active; it is not replaced by a common global width. Both branches use stable,
centered/scaled least-squares solves rather than inverses of normal equations.

Fitted densities become reciprocal scores. Zero density gives one; reciprocal
values are clipped to [0,1] and made nondecreasing by rank. The source therefore
assigns zero to a negative initial reciprocal, which may lead to rejection; this
historical rule is preserved and is not evidence of posterior calibration.
Fitted hypotheses are rejected when their score is at most alpha. Unfitted
hypotheses are always retained, including when alpha is one.

`scores` and `reject` retain original input order. `density` contains only the
fitted points in ascending-rank order, corresponding to
`order[:fitted_points]`. For window fits, `bandwidths` has that same order;
it is `None` for polynomial or retain-all results. This avoids inventing density
estimates outside the fitted subset. The original early-return branch leaves its
arrays uninitialized; the Python API implements its documented retain-all intent
with scores of one, false decisions, and an empty density vector. Singular fits
and invalid windows raise `NonparametricFitError` rather than using undefined
regression results.

The reference fixture additionally contains 48 original desktop runs across the
branch boundaries, fractional null estimates, ties/endpoints, the published
example, and two significance levels. Ordinary density/score comparisons allow
for the original normal-equation precision loss; rejection masks agree exactly
on these cases. Fitting all 150 published values reveals larger original density
errors (up to about 0.25%) in narrow windows near the upper end. A traced copy of
SMOOTH records its selected bandwidths without changing the numerical operations.
Independent 70-digit Decimal regressions at those original windows verify the
Python densities to near double precision; those tests replace an inaccurate
original density as the numerical oracle. Independent tests also check
full-family rank normalization, exact
linear/quadratic derivatives, narrow-coordinate stability, inclusive thresholds,
input permutations, automatic Schweder wiring, and failed-fit propagation.
The original untouched-output sentinel is recorded for the early-return branch
to distinguish the Python repair from an exact reproduction of undefined data.

## Validation and provenance for other MULTI procedures

`tools/reference_multi.py` compiles the original desktop numerical routines after
removing only the interactive main program. The S SCHWED routine is compiled with
a distinct symbol name to compare its additional variance output. S OSFIT and
its CDFBET driver are also extracted unchanged. All numerical
routine bodies are retained. The source archive hash, compiler and compatibility
flags are recorded in `tests/fixtures/multi.json`.

The reference fixture contains 32 adjustment outputs, 35 Rom threshold vectors,
96 sharpened-testing outputs, and 30 desktop/S Schweder fits. It includes the
archive's 150-value example, ties, endpoints, fractional null-count estimates,
and different significance levels. An additional 50 bootstrap samples generated
by NumPy are fitted by the archived Fortran routine, allowing deterministic
comparison of bootstrap estimates and their mean/variance.

Ten additional OSFIT reference cases cover the published example, ties, endpoints,
tiny p-values, and combined scores greater than one. Independent binomial-tail
enumeration verifies the order CDF, and hand calculations verify the legacy
transformation. Clustered simulation is checked against an independently computed
dense eigendecomposition, analytic one-sided normal probabilities, and fixed-seed
simulations of null uniformity, alternative means, latent covariance, transformed
p-value correlation, and independence between clusters. No exact comparison of
the two different random number streams is claimed.

Independent checks include exhaustive closed Bonferroni testing for Holm,
hand calculations for step-up adjustments and small Rom families, Decimal
arithmetic for tiny Sidak p-values, exact Schweder line recovery, invalid/degenerate
inputs, and bounded bootstrap failure. Fixed-seed simulations check independent
global-null error rates and BH false discovery rate in a mixture of true and false
nulls. These validate the tested conditions; they do not prove all dependence
assumptions or replace a design-specific analysis.

The implementation uses NumPy broadcasting and prefix/suffix accumulations;
Rom's recurrence has quadratic work instead of repeatedly forming coefficients
by inner product loops. Original copyright and redistribution terms are retained
in [the notices](../THIRD_PARTY_NOTICES.md).

`uv run python tools/benchmark_multiplicity.py` reproduces the
[recorded benchmark](multiplicity-benchmark.json). On the recorded machine,
2,000 families of 100 p-values took about 0.0135 seconds in one batched Holm call
versus 0.0470 seconds in separate row calls, about 3.5x faster. A 2,000-test Rom
threshold vector took about 0.105 seconds and stayed finite and monotone. These
timings compare Python call patterns, not original Fortran throughput, and are
not CI performance thresholds.
