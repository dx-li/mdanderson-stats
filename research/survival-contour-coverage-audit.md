# SurvivalContour source coverage boundary

This bounded audit distinguishes the author R package's stated model routes
from capabilities of its optional `randomForestSRC` dependency. It does not
claim complete catalog-entry or Shiny-site parity.

## Cached source and model routes

The inspected package source is pinned to author revision
`d4645f69f23fc1146c07432f576b4c40f85e1bba`; cached files are under the root
repository's ignored `research/raw/survivalContour/R`. The package-level model
table is in `R/survivalContour.R` lines 3–13, and dispatch is at lines
180–255. It exposes ordinary and stratified Cox (`coxph`), ordinary and
stratified interval-censored PH (`mets::phreg`), parametric and spline
`flexsurvreg`, ordinary Fine–Gray (`FGR`), interval-censored competing risks
(`ciregic`), five PyCox-backed neural classes, and `rfsrc` forests. The table
also says 3D confidence surfaces are available for Cox, stratified Cox,
interval-PH, and parametric/spline models; Fine–Gray and neural routes have no
such option.

The current Python package has corresponding method modules and guides for
these model families: `survan_cox.py`, `interval_survival*.py`,
`parametric_survival*.py`/`survival_spline.py`, `fine_gray*.py` and
`interval_competing_risk*.py`, `survival_neural*.py`, and
`random_survival_forest*.py`. This mapping is a method-family inventory only;
it does not establish identical plotting, data-upload, report, or application
defaults.

## Recovered prediction helpers and remaining statistical contract

On October 4, a read-only check of the author's complete Git tree at the
pinned revision recovered the two stratified helpers and their predictor.
The original cache was incomplete; the files were available upstream.
Saved bytes match these Git blob hashes:

| File | Git blob |
| --- | --- |
| [R/coxIntStrataContour.R](https://github.com/YushuShi/survivalContour/blob/d4645f69f23fc1146c07432f576b4c40f85e1bba/R/coxIntStrataContour.R) | `35a2599badb0e2a5a98b2c9fdea92c5e35826b0b` |
| [R/coxIntStrataContour3D.R](https://github.com/YushuShi/survivalContour/blob/d4645f69f23fc1146c07432f576b4c40f85e1bba/R/coxIntStrataContour3D.R) | `96a24706661e3774052f3ad48887281f6d0017f4` |
| [R/predictPhreg.R](https://github.com/YushuShi/survivalContour/blob/d4645f69f23fc1146c07432f576b4c40f85e1bba/R/predictPhreg.R) | `d0c436c241bd1a1bd35ec86609b25a945dce1b49` |

The dispatcher calls the helpers at lines 207–212. They use first-seen
stratum order, a common covariate grid between the pooled 2.5th and 97.5th
percentiles, and pooled numeric means or categorical modes for adjustment
unless a profile is supplied. Time points come from the fitted object's
unique times, prepending zero when needed. The 2D route plots survival;
the 3D route also passes lower and upper confidence surfaces to the renderer.

`predictPhreg` defaults to model-based uncertainty, log-scale limits and 95%
confidence. It computes survival as `exp(-exp(x beta) H0(t))`. Its cumulative
hazard variance combines baseline variance, coefficient covariance and a
baseline/coefficient cross term derived from the fitted `mets` object's
`E`, `S0`, `II` and `se.cumhaz` fields. Log-scale survival limits use
`exp(log(S) +/- z * SE_cumulative_hazard)`, with the upper limit capped at
one. Thus confidence surfaces are an actual statistical output, not only
plot styling.

Recovering the predictor does not resolve the advertised
`mets::phreg(Surv(..., type="interval2"))` response-contract mismatch. The
current Python estimator maximizes a genuine interval-censored likelihood;
its Turnbull support-identification bounds are not sampling confidence
limits. Counting-process variance fields cannot be substituted into that
estimator without a valid derivation. Existing coefficient bootstraps do not
provide baseline-survival confidence surfaces either. The
[interval-likelihood audit](interval-survival-audit.md) and
[stratified guide](../docs/interval-survival-stratified.md) describe the
implemented target and remaining uncertainty scope. No prediction or
confidence-interval calculation was executed or added during this source review.

## Dependency contract review

A subsequent October 4 review checked the official
[`mets::phreg` documentation](https://search.r-project.org/CRAN/refmans/mets/html/phreg.html)
and [author implementation](https://github.com/kkholst/mets/blob/master/R/phreg.R).
The inspected implementation reads a two-column survival response as exit time
and event status, and a three-column response as entry, exit and status. It
does not branch on the `Surv` censoring-type attribute before passing those
values into its Cox risk-set/partial-likelihood calculation. This does not
establish support for an interval-censored likelihood. The similarly named
`IC.phreg` concerns influence-function decompositions, not interval censoring.
Context7 did not index this R package; the review used primary CRAN/author
material and did not install it. Exact historical dependency source remains
unverified, so this finding does not establish behavior of every old release
or the deployed app.

The wrapper's own table and `Surv(lower, upper, type="interval2")` example
explicitly describe interval-censored input. Its separate `CI3D` flag requests
confidence surfaces. Thus the advertised interval route cannot be resolved by
interpreting “interval” as only a confidence-interval option. A three-column
shape accepted by the dependency is not a valid conversion between these
different response contracts. This is evidence of an estimator mismatch in
the inspected source path, rather than a recovered interval-data mapping.

The remaining statistical work is a justified baseline-survival uncertainty
procedure for the genuine interval-censored model. Current support-location
bounds and coefficient bootstraps must not be relabeled as the advertised
sampling confidence surfaces. No counting-process variance formula or
unverified historical behavior has been substituted into that model.

The cache contains the package wrapper, not the deployed Shiny server. It
therefore cannot establish the live site's complete user workflow or report
contract. Options supported by `randomForestSRC` but not selected or surfaced
by this wrapper are dependency features, not additional demonstrated
SurvivalContour workflows.
