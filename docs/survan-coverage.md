# SURVAN calculation coverage audit

The original `survan.doc` introduction enumerates seven calculation families.
All seven now have public Python interfaces and source-based numerical checks.
The catalog's **implemented** status refers to this advertised calculation scope.

| Advertised calculation | Python interface | Native reference and validation |
|---|---|---|
| Kaplan–Meier, survival confidence limits, quantiles and quantile intervals | `survan_km`, `.quantiles` | KMC/FSROUT/KMCI/CALQV/KMCIQT; complete 33-row life table and nine quantiles |
| Multi-group log-rank, optionally stratified | `survan_group_test(method="logrank")` | LRC and stratified summation; manual and three-group tied data |
| Multi-group Gehan–Breslow, optionally stratified | `survan_group_test(method="gehan_breslow")` | GBC and stratified summation; native scores and covariance |
| Cox coefficients and underlying survivor/hazard function | `survan_cox`, `survan_baseline` | PHLD likelihood/score/information and FHZPT/HAZF roots; worked example and multivariable data |
| Logistic regression | `survan_logistic` | LGRLL/LGRGH, manual and exact two-group fit, with/without intercept |
| Frequency distributions | `survan_frequencies` | FREQ, exact counts and native approximate grouping rule |
| Descriptive statistics | `survan_describe` | SUMMT, manual summary quantities and explicit legacy SE |

Detailed domains, numerical improvements, source provenance and validation
limits are in the linked [SURVAN documentation](survan.md). Original archives and
executables are not bundled. The legacy DMNH optimizer's iteration path is not
reproduced; the fitted likelihoods, scores and information are checked directly.

## Interface translation

The public package exposes numerical arrays and structured results. Python/NumPy
record selection, column selection and transformations replace the old menu,
variable-name, expression and file-session interface. Missing-value and signed
survival encodings are decoded explicitly before fitting; estimators reject
malformed inputs rather than silently deleting records. Group-specific KM curves
are obtained by selecting each group before calling `survan_km`; a pooled curve
uses all selected records. Logistic event lists or thresholds become explicit
zero/one indicators. Numerical result objects provide the legacy report's
quantities without duplicating printer pagination or the historical plotting UI.
This is not a binary-compatible clone of the original interactive executable.

## Time-dependent-covariate audit

Earlier progress notes listed time-dependent covariates as pending, based on a
comment in PHL/PHLD about why covariates are not normalized in those routines.
That comment is not evidence of an implemented user workflow. The manual's
advertised methods and example do not specify changing-covariate inputs, and
QGTOBS reads fixed observation records while PHL/PHLD accumulate their weighted
risk sums without reevaluating prior records at later event times. The available
Python API therefore covers the documented static-covariate Cox workflow.
Time-varying covariates, counting-process input, delayed entry and stratified Cox
baselines are not claimed as SURVAN features implemented by this port.

## Deliberate numerical differences

- Stable float64 likelihoods, moments, direct tails and log-domain roots replace
  cancellation-prone expressions, single-precision approximations and cutoffs.
- The ordinary descriptive standard error is corrected to SD/sqrt(n); the exact
  native SD/(n−1) convention remains available by name.
- Default frequencies use exact equality; the order-dependent native approximate
  grouping is an explicit option.
- Rank deficiency, separation, unresolved numerical state and unrepresentable
  results produce errors. Missing summary/quantile values use NaN instead of
  arbitrary numeric sentinels.
- The no-intercept logistic likelihood-ratio null is nested (beta=0), correcting
  the native report's absolute difference from a nonnested constant fit.

These differences are documented individually and exercised by focused checks;
they are not claims of byte-for-byte output compatibility.
