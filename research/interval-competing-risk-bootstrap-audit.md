# Interval competing-risk coefficient bootstrap

## Pinned source contract

The reference is `cran/intccr` 3.0.4, commit
`252644c0d347a663ea5d7bef88fa2ab04f114b1e`. The native
`R/bssmle_se.R` blob is `c923ec3cd71842c4e4895d34905c635381ade5d6`
(SHA-256 `fa3d700006e0b6ac284f1e1ba87211ad5712faa5e85eb368991e0bc872018ba5`).

`bssmle_se` creates each resample as `data[sample(n, replace=TRUE), ]`: it
draws exactly the original number of rows, uniformly, with replacement.
Repeated rows stay repeated when `bssmle` rebuilds its model frame and
empirical event-time support/knots. The returned `beta` starts with two
cause-specific spline-baseline blocks and ends with the two cause-specific
regression-slope blocks. The wrapper extracts only those slopes. For multiple
replicates it removes rows whose first retained slope is NA and computes
`var` on the remaining rows, using denominator successful replicates minus
one. It does not redraw a failed fit. The native one-replicate branch returns
the vector itself as `Sigma`; this implementation corrects that shape bug by
returning undefined covariance and standard errors whenever fewer than two
fits succeed.

The `ciregic` caller supplies `objfun="bssmle"`. A resample with no observed
event of one cause fails in `Surv2` and aborts the native call. An optimizer
nonconvergence instead produces an NA coefficient vector, which the wrapper
omits. Python therefore raises for missing-cause/invalid resample data by
default, and records nonconverged refits without redrawing. An explicit
`on_invalid_resample="record"` option records missing-cause or resampled
input-validation failures; this is a Python extension. Unexpected errors are
not swallowed. At least one covariate is required in this API; native's
intercept-only dummy slope is not exposed as a meaningful predictor.

Native `bssmle_se` does not estimate or need the separate residualized-score
covariance stored by the existing Python point fitter. Bootstrap refits skip
that unrelated calculation; the original fit retained in the bootstrap
result exposes NaNs in its score-based covariance field. The bootstrap
covariance lives in the bootstrap result itself. No baseline or cumulative
incidence confidence bands are claimed.

## Python contract and bounds

`bootstrap_interval_competing_risk_coefficients` accepts either a NumPy
generator/seed or an explicit zero-based integer tape with shape `(B, n)`.
NumPy and R random streams are not claimed to match. Each sampled row
sequence, including duplicates, is sent to a fresh fitter call, recomputing
the empirical knots from that replicate. Fits run serially and only slope
vectors are retained. The tape, result arrays and maximum fit scratch are
preflighted together; sampling work is bounded by `B*n`; the cumulative
iteration-weighted fitter estimate has a separate hard cap. These checks run
before fitting or advancing random state.

## Validation evidence

The source-execution fixtures are in
`tests/fixtures/interval-competing-risk-bootstrap-*.csv`. They use two successful native fits with `alpha=(0,1)`,
`k=0.5`, fixed row tapes, and a third absent-cause sample. The source outputs
record sampled row order, recomputed support/knots, extracted slopes,
covariance, and the missing-cause error. The native optimizer has known
derivative defects, so its fitted slopes/covariance are compatibility records,
not certified optimization targets. The focused test checks the tape and knot
contract, then independently refits each Python tape row and verifies
convergence/KKT and the returned sample-covariance identity.

Integration validation reran the new bootstrap tests and existing competing-risk
fit tests after the final covariance-skip and failure-policy changes: 8 passed
in 2.52 seconds (139.78 MiB peak process RSS, zero swaps). The public guide's
eight resamples all converged. A separate integration check reproduced seeded
draws exactly with an explicit tape and preserved coefficients and standard
errors after covariate-unit changes by 1e-100 and 1e100 (0.591 seconds,
121.83 MiB peak process RSS, zero swaps).

The portable generator is `tools/reference_interval_competing_risk_bootstrap.R`.
From the repository root, run:

```sh
Rscript tools/reference_interval_competing_risk_bootstrap.R [INPUT.csv] [OUTPUT_DIR]
```
It uses the same pinned source hashes and optimizer dispatch as
`tools/reference_interval_competing_risk.R`, but loads only the model sources
needed for these refits. This avoids running that broader reference script or
rewriting unrelated fixtures. The default input and output locations are the
committed fixture paths; the ignored `research/raw/intccr` source cache and
the required source-only R prerequisites must be available.

Ruff, mypy, Python compilation, and `git diff --check` pass on the final source.
No full suite or large simulation was run.
