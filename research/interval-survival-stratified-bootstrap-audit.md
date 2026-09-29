# Stratified interval-PH coefficient bootstrap audit

## Source contract and explicit extension

The shared-coefficient statistical target is implemented by
`fit_stratified_interval_survival`: one regression coefficient vector is
estimated from the sum of interval-censored PH likelihoods, with an
independent Turnbull support and baseline distribution for each stratum. The
fit and its source limitation are described in
[`interval-stratified-audit.md`](interval-stratified-audit.md). The original
SurvivalContour author dispatches stratified `phreg` to
`coxIntStrataContour` helpers that were not recovered; no stratified
bootstrap contract was found. Consequently this module does not claim native
application parity.

The cached and pinned `icenReg` 2.0.16 source (commit
[`26fadac37c6b54dd0e29c91c2bf07942ae120356`](https://github.com/cran/icenReg/tree/26fadac37c6b54dd0e29c91c2bf07942ae120356))
documents the ordinary `ic_sp` coefficient bootstrap. It samples
`ceil(sum(weights))` rows globally with probabilities proportional to case
weights, collapses duplicate rows to sampled integer multiplicities, does not
redraw failed fits, and computes covariance over successful coefficients
using sample covariance. The separate cluster helper resamples subject
clusters. Neither specifies how to bootstrap a stratified interval-PH fit.

This Python extension therefore requires one of two policies. `within_stratum`
draws `ceil(sum(weights_g))` rows independently inside each first-seen group;
this holds each declared stratum's weighted sample size fixed. `pooled` applies
the ordinary global weighted-row scheme to the stratified likelihood; a
replicate missing any original group is explicitly recorded as failed rather
than fitting a changed set of baselines. In both schemes selected duplicate
rows become frequency weights and are not multiplied by the original weights.
Neither choice is presented as a recovered native default.

## Replay, failures, and bounds

An explicit global-row-index tape has one row per replicate. Within-stratum
tapes concatenate group blocks in first-seen order and validate that every
index belongs to the matching group. Pooled tapes contain unrestricted source
indices and use the global weighted draw size. Generated tapes are returned
read-only for exact replay. Failure rows and error text remain aligned; no
redraw changes the requested resample set. Covariance and standard errors are
computed only from successful rows and are undefined for fewer than two.
Baseline confidence bands are out of scope.

The preflight accounts for the retained tape and result arrays, original and
worst-case resampled support/work, all requested optimizer iterations, and
sampling work before RNG creation or fit execution. Refits run serially, with
each temporary fit released before the next replicate. The initial fit is
completed before an RNG is created or advanced.

## Independent reference

`tools/reference_interval_stratified_bootstrap.R` is an independent base-R
reference for the current-status special case, where interval-PH reduces to
binomial complementary-log-log regression with stratum-specific intercepts
and a shared slope. It fixes six within-stratum expanded-row tapes across two
groups, records five valid shared-slope fits and one requested all-censored
group failure, and writes portable fixtures under
`tests/fixtures/interval-stratified-bootstrap-*.csv`. This checks the shared
coefficient and survivor-conditional covariance for an analytically reduced
case; it does not establish native application behavior or R/Python RNG
parity.

The focused Python check (new bootstrap plus the existing stratified fitter)
passes all eight tests in 2.53 seconds with one BLAS thread and warnings as
errors; peak RSS is 135,036,928 bytes and swaps are zero. Against the six
fixed tapes, the largest shared-slope absolute difference over five successful
fits is `4.98e-7`. The original-fit slope/log-likelihood differences are
`1.08e-7` and `2.49e-14`; the successful-only variance and standard-error
differences are `2.41e-7` and `2.23e-7`. The separate reference comparison
took 0.057 seconds after imports, peaked at 111,656,960 bytes and used no
swaps. Targeted Ruff check/format and mypy pass for the new module and tests.
