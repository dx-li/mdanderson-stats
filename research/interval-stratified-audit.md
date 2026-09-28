# Stratified interval-censored PH contract

The author package advertises stratified interval Cox in
`research/raw/survivalContour/R/survivalContour.R`, lines 198–212. Its
`mets::phreg` response interpretation and absent stratified contour helpers
prevent a claim of native numerical parity. The existing
[interval-likelihood audit](interval-survival-audit.md) records that mismatch.

The Python statistical target is the sum of genuine interval-censored PH
likelihoods, with one shared regression coefficient vector and a separate
Turnbull support and baseline distribution per stratum. Independent fits with
averaged coefficients do not implement this target. Probability inside a
non-singleton support interval is still location-unidentified; prediction
bounds must remain identification bounds, not confidence limits.

Covariate rank must be evaluated after removing stratum constants, because
each baseline can absorb those constants. Combined support, design, work and
prediction bounds apply across groups. The initial scope explicitly rejects
groups with only right-censored observations, matching the ordinary fitter's
unsupported baseline-tail case. It does not invent a fitted event curve for
such groups. Bootstrap covariance and confidence surfaces remain separate.

## Independent reference reduction

`tools/reference_interval_stratified.R` uses only base-R `stats::glm`. When
each observation is `(0,1]` or `(1,infinity)`, interval-PH likelihood is exactly
binomial complementary-log-log likelihood. Stratum intercepts are log baseline
cumulative hazards at time one; the covariate coefficient is shared. The
reference contains eight weighted observation rows with event/nonevent counts
`A: (2,8),(5,5)` and `B: (4,6),(7,3)` at covariates zero and one.

The converged R fit gives beta `0.962469326124804`, log likelihood
`-24.8084555704438` excluding binomial constants, and baseline cumulative
hazards `0.251054260277` and `0.478374591831488` at covariate zero. Ten survival
predictions span both strata and five covariate values. The generator completed
with warnings treated as errors.

## Validated implementation

Luna's `fb560081` was integrated as `0022625`. The fitter sums regression
scores/information for one coefficient vector, with separately projected
baseline updates. Review corrected a stratum-centering inconsistency before
integration, preserved baseline-only fitting, moved shape checks before large
conversions and rejected underflowed normalized weights. Per-stratum fit views
reuse the ordinary predictors and contours with their matching centers.

Four focused checks pass in 1.49 seconds. Targeted Ruff/formatting and mypy
pass. The independent R fit agrees with the shared coefficient and likelihood;
the maximum error across ten predictions is `3.2927e-8`. Time/covariate unit
factors `1e-100` and `1e100`, common weight factors `1e-200` and `1e200`, and
stratum offsets `+1e8` and `-2e8` preserve the fitted effects and predictions.

A separate 64-row mixed-censoring case uses direct survival-difference
probabilities to check likelihood `-115.6476413148439`. Its finite-difference
shared regression score is below `2.876e-9` per row; the reported combined
score/constraint diagnostic is `9.505e-8` after 172 iterations. The root audit
took 0.249 seconds after imports, peaked at 116.53 MiB and reported no swaps.
No broad suite, new dependency or CI workflow was introduced.
