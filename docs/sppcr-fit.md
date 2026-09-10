# SPPCR Poisson-mean fitting

`sppcr_fit_means` fits per-allele Poisson means from small-pool PCR detection counts.
It implements the likelihood/curvature core of catalog entry 26.
[Frequency summaries](sppcr-frequencies.md), [bootstrap analysis](sppcr-bootstrap.md)
and [confidence intervals](sppcr-intervals.md) build on these fits. Input formats
and application reporting remain outstanding; SPPCR is **partial**.

```python
from mdanderson_stats import sppcr_fit_means

fit = sppcr_fit_means(
    dna=[1, 2],
    seen=[[20, 50], [36, 75]],
    wells=[100, 100],
)
# fit.mu is approximately [0.22314355, 0.69314718]
# fit.variance is approximately [0.0009, 0.00428571]
```

DNA amounts are positive and share a common unit. `mu[j]` is the allele's mean
count per DNA unit, so detection probability at level i is
`1-exp(-dna[i]*mu[j])`. Several alleles may occur in one well; detection counts
across alleles do not have to sum to the number of wells.

## Input and result contract

`dna` is a nonempty one-dimensional vector. `seen` has shape
`(..., DNA levels, alleles)`, with at least one allele. `wells` is a scalar or
broadcastable to `seen.shape[:-1]`; each entry supplies the number of wells at that
DNA level. Counts are nonnegative integers below `2**53`, wells are strictly positive,
and detections cannot exceed wells. Fractional observations, nonfinite inputs and
mismatched shapes fail explicitly. Fractional half counts arise only from the
requested saturation correction.

Leading axes are independent experiments, enabling vectorized replicate fitting.
An empty leading batch is supported. DNA levels need not be sorted or distinct.
All returned arrays own immutable byte storage and remain independent of subsequent
changes to the caller's arrays.

`SPPCRMeanFit` provides:

- `dna`, `original_seen`, and broadcast `wells`: the supplied experiment.
- `seen` and `unseen`: counts actually used in the fit.
- `mu`, `variance`, `log_likelihood`: per-allele estimates, inverse observed
  information, and fitted log likelihood without binomial coefficients.
- `interior`: whether the fit has a positive interior root. Boundary variances
  are deliberately NaN rather than an invented asymptotic uncertainty estimate.
- `adjusted`: whether the requested half-count correction was applied.
- `mu_lower`, `mu_upper`: the numerical root bracket in original DNA units.
  These are solver diagnostics, not confidence limits or rigorous interval arithmetic.

Except for input/count arrays, fields have shape `batch + (alleles,)`.
With adjusted data, likelihood and variance describe that adjusted fit, not the
unmodified original observations.

## Boundaries and source repairs

An allele never detected at any level has maximum-likelihood estimate zero. Its
log likelihood is zero and both bracket endpoints are zero. `interior=False` and
`variance=NaN` identify the nonregular boundary where ordinary interior-information
uncertainty is unavailable. No pseudo-detection is inserted.

An allele detected in every well at every DNA level has no finite maximum-likelihood
mean. The default `saturation='raise'` rejects the input explicitly. With
`saturation='half'`, half a detection is moved to nondetection at the **first minimum
DNA level**, matching the location and amount of the source's stated adjustment.
The adjustment is disclosed in `adjusted`, `seen` and `unseen`. A half count that
cannot be represented exactly is rejected. With tied minimum levels, reordering
them can change which count is adjusted; the rule is positional and intentional.

Unlike the native implementation, a fully detected row at only one DNA level does
not trigger an adjustment when another level has nondetections. This avoids the
source bug that can turn a zero count at a different level into -0.5. The
[source audit](sppcr-research.md) and native fixture preserve that defect evidence.

## Numerical method and limits

For each allele the score is strictly decreasing for positive mu when detections
are present. A finite root exists when nondetections are also present. The solver
uses `tau=mu*max(dna)` and normalized DNA weights w. Multiplying the score by tau
gives a bounded expression involving `x/expm1(x)`, evaluated with negative
exponentials and `expm1` to avoid overflow and cancellation. The bound
`sum(seen)/sum(unseen*w)` supplies the upper search endpoint; no arbitrary native
`1e10` cap is copied.

All alleles and batches advance together through a NumPy bisection loop, ending
when the bracket width is at most eight machine epsilons times its upper endpoint.
`max_iterations` defaults to 2048; exhaustion raises `ArithmeticError`. Observed
information is summed in log space, avoiding intermediate DNA squares and ratios
that would overflow or underflow in ordinary arithmetic.

Inputs still use finite float64 arithmetic. An unrepresentable DNA ratio, bracket,
positive mean, likelihood or overflowing variance raises `ArithmeticError`.
Representable means do not guarantee that every intermediate bracket is representable
for extremely disparate DNA levels. A positive variance smaller than the smallest
subnormal can round to zero; this differs from a boundary's unavailable NaN variance.
These are floating-point estimates, not arbitrary-precision inference.

## Validation and performance

135 tests check single-level closed forms over DNA scales from `1e-150` to `1e150`,
unequal-level roots computed independently with 70-digit Decimal arithmetic, four
valid native reference fits, likelihood values and information, batch/allele axes,
DNA-unit and order invariance, both saturation policies, zero boundaries, ownership,
invalid inputs and explicit iteration/numerical failures.

The [benchmark](sppcr-fit-benchmark.json) compares one batch with repeated calls to
this same Python API, checking identical mean, variance, likelihood and bracket
arrays. Three-run medians measured about **21x** for 32 experiments and **66x** for
1024 experiments, each with three DNA levels and four alleles. These are batching
speedups on the recorded machine, not comparisons with native Fortran.
