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

## Unavailable source contract

The R dispatcher calls `coxIntStrataContour` and `coxIntStrataContour3D` for
stratified interval-PH models (`survivalContour.R` lines 207–212), but neither
helper is among the cached package R files. The package table and `CI3D`
documentation claim confidence intervals for this route, yet the missing
helper means the app's exact stratified interval contour and uncertainty
calculation cannot be recovered from this cache. Do not fill that gap by
assuming the ordinary interval-PH helper generalizes unchanged.

The cache contains the package wrapper, not the deployed Shiny server. It
therefore cannot establish the live site's complete user workflow or report
contract. Options supported by `randomForestSRC` but not selected or surfaced
by this wrapper are dependency features, not additional demonstrated
SurvivalContour workflows.
