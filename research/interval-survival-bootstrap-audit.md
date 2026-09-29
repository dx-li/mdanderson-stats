# Interval-PH coefficient bootstrap scope

## Source contract

The pinned `icenReg` source is version 2.0.16, commit
`26fadac37c6b54dd0e29c91c2bf07942ae120356`; its cached source is under the
ignored `research/raw/icenReg/` tree. `R/ic_sp.R` documents the coefficient
covariance as bootstrap-estimated. With positive case weights, its
`bs_sampleData` helper draws `ceil(sum(weights))` row indices with replacement
using probabilities proportional to the original weights, then collapses
duplicate row IDs and fits with their integer multiplicities. The point fit's
weights are not normalized, so multiplying all weights can change the
bootstrap sample size even when it leaves the point estimate unchanged.

For each bootstrap sample, `getBS_coef` checks invertibility of the augmented
covariate Gram matrix `[X, 1]'[X, 1]`; singular samples yield a missing
coefficient row. The wrapper omits rows whose first coefficient is missing,
warns when at least ten percent fail, and calculates the sample covariance of
the remaining rows (`stats::cov`, divisor `B_success - 1`). It does not redraw
failed samples. The Python result therefore preserves one status and one
coefficient row per requested replicate, represents failures as NaN, and
labels covariance/SE as conditional on successful fits. With fewer than two
successes these summaries are undefined.

Python also applies its existing fit validation and convergence requirements
to every replicate; invalid or nonconvergent fits are recorded as failures.
This is stricter than the native helper's explicit singular-design check and
does not claim identical acceptance for every native optimizer result.

`ic_sp` applies `adjustIntervals(B=c(0,1))` before saving the data environment
that the bootstrap resamples. It moves the lower endpoint inward by `1e-10`
for every interval wider than `2e-10`, including right-censored rows, while
exact rows are unchanged. The independent native fixture generator
applies this preprocessing before calling the pinned bootstrap helpers. An
initial scratch comparison omitted this adjustment and disagreed; the corrected
reference and Python outputs agree, so the unadjusted scratch output is not
used as validation evidence.

The native `survCIs` wrapper explicitly accepts only parametric or Bayesian
fits; it rejects `ic_sp` because uncertainty in its nonparametric baseline is
not available. The interval model's existing survival lower/upper values are
identification bounds caused by event locations being unknown within support
intervals. This bootstrap does not reinterpret them as confidence bands or
bootstrap baseline masses. The pinned source also rejects `cluster(...)` in
`ic_sp` and refers users to a separate `ir_clustBoot` workflow; this tranche
implements neither clustered nor stratified resampling.
The existing Python interval fitter requires strictly positive case weights,
whereas native `icenReg` accepts zero weights; this bootstrap follows the
Python fitter's positive-weight input contract.

## Python replay and limits

`bootstrap_interval_survival_coefficients` accepts a zero-based
`resample_indices` matrix with `B` rows and `ceil(sum(weights))` indices per
row. This makes resample composition inspectable and reproducible without
claiming R random-stream parity. Without a tape, a NumPy generator performs
the same weighted sampling design. The original model is fit before RNG is
created or advanced. Resample matrices, retained coefficient rows, live
scratch and a conservative worst-case work estimate are checked against hard
limits before fitting or consuming randomness. Bootstrap fits are processed
serially; their baseline arrays are discarded immediately.

Coefficient covariance and standard errors are the only uncertainty
summaries. No percentile/Wald coefficient confidence interval formula is
claimed from the cached package source, and no curve confidence interval is
reported. The separate interval identification bounds remain available from
`predict_interval_survival`.

## Validation

Six focused tests pass, including fixed weighted tapes, a retained singular
replicate, RNG-preserving preflight failures, covariance range guards, and the
native fixed-resample comparison. The source-hash-guarded generator is
`tools/reference_interval_survival_bootstrap.R`; it writes the portable
fixtures under `tests/fixtures/interval-survival-bootstrap-*.csv`. Unit-weight
and non-unit-weight cases each use two successful samples and one singular
sample. Coefficients and survivor-only covariance agree with the adjusted
icenReg reference within the focused test tolerances. RNG sampling work is
bounded by `replicates * ceil(sum(weights))` before random state is consumed.

Covariance rescaling uses binary mantissa/exponent arithmetic. If a nonzero
covariance entry cannot be represented as a float64 value, the operation
raises `ArithmeticError` rather than returning a misleading zero or infinity.

Root integration runs the twelve bootstrap/visit-conversion checks together
(2.437 seconds, 140.95 MiB peak RSS, zero swaps). Four seeded weighted resamples
with a fractional total weight reproduce an independently generated explicit
row tape exactly; the sample size is 87. Changing covariate units by `1e100`
and `1e-100` preserves coefficient draws and covariance after conversion back
to the original units. The combined workflow check, including visit-to-fit
integration, takes 1.365 seconds at 120.12 MiB peak RSS, zero swaps.

The portable R generator was executed from the integrated root. All eight
parent/bootstrap CSV fixtures remained byte-identical. This run took 4.724
seconds, with 309.00 MiB peak child RSS (including the serial native reference
builder) and zero swaps. No additional dependencies were installed.
