# SPPCR frequency and calibration summaries

`sppcr_frequencies` converts allele means into frequencies, calibration and total
mutant frequency. It adds the original independent-mean delta-method variances,
standard errors and forward arcsine transformations to the
[likelihood fitting core](sppcr-fit.md). Bootstrap uncertainty, confidence intervals
and the complete file/report workflow remain unfinished; catalog entry 26 is partial.

```python
from mdanderson_stats import sppcr_fit_means, sppcr_frequencies

fit = sppcr_fit_means([1, 2], [[20, 50], [36, 75]], [100, 100])
summary = sppcr_frequencies(fit.mu, fit.variance, progenitor=(0, 0))

summary.frequency.value  # approximately [0.2435292, 0.7564708]
summary.frequency.standard_error
summary.calibration.value  # approximately 0.9162907
summary.mutant.value  # approximately 0.7564708
```

## Inputs and outputs

The last axis of `mu` indexes alleles; leading axes index independent experiments.
Means must be nonnegative finite numbers, with a positive total in each experiment.
All-zero means have undefined frequencies and raise `ValueError`. Empty leading
batch axes are supported. `variance` broadcasts to `mu`; values must be nonnegative
and finite, or NaN to explicitly indicate unavailable uncertainty. Omitting it
produces point estimates with unavailable uncertainty.

`progenitor` is a required pair of **zero-based** allele indices. Repeat an index
for a homozygote; use two distinct indices for a heterozygote. Mutants are all other
alleles. This differs from the native code's one-based array indexing; later file
adapters must map allele identities explicitly.

`SPPCRFrequencies` retains immutable copies of the means and mean variances, the
progenitor pair, and three summaries:

- `calibration`: `SPPCREstimate` with `value`, `variance`, and `standard_error`.
- `frequency`: `SPPCRProportion` with those fields, the directly calculated
  `complement`, and a `transformed` estimate for `2*asin(sqrt(p))`.
- `mutant`: the corresponding proportion for the combined non-progenitor group.

Frequency arrays retain the allele axis. Calibration and mutant arrays omit it.
Each estimate exposes `variance_available`, an immutable Boolean array. NaN variance
and standard error mean unavailable uncertainty, not a zero estimate. All storage
is immutable and independent of caller mutations.

## Uncertainty and boundary semantics

The variances follow the native assumption of independent estimated allele means.
They are delta-method approximations, not confidence intervals or bootstrap results.
For total mean T, calibration is T and its variance is the sum of mean variances.
For a group with mean x and variance vx, and its complement y and vy,

```text
p = x/T, q = y/T
var(p) = (q^2*vx + p^2*vy)/T^2
var(2*asin(sqrt(p))) = var(p)/(p*q)
```

All mean variances must be available for an ordinary uncertainty summary. Passing
`fit.variance` preserves NaNs from boundary mean fits; they are not replaced with
the native program's zero variance or a hidden pseudo-observation.

Structural constants are exceptions with genuinely known zero variance. With one
allele its frequency is identically one. If the progenitors cover every supplied
allele, mutant frequency is identically zero. Both raw and transformed variances
are zero in these cases even when calibration uncertainty is unavailable.

A nonstructural proportion at exactly zero or one has an unavailable transformed
variance because the transform derivative is singular. Its raw variance can still
be computed from explicitly supplied mean variances. The native arbitrary clipping
of p to `[1e-10,1-1e-10]` is not used to invent finite transformed uncertainty.

## Stability and precision

Mutant and complementary groups are summed directly; they are not reconstructed
by subtracting nearly equal totals. Prefix and suffix log sums compute each
allele's complement in linear work and storage. This retains a small group even
when the dominant frequency rounds to one.

Variance calculations use logarithms to avoid overflow in squared/fourth-power
expressions. The angle is evaluated using the square roots of both complementary
probabilities through `atan2`, avoiding the precision loss of `asin(sqrt(p))` near
one. Standard errors and transformed variances are computed from their logarithmic
expressions independently of the rounded raw variance.

Consequently a positive standard error may remain representable when its squared
variance underflows to zero. A raw probability can also underflow while its
transformed estimate remains positive. Results use float64, not arbitrary precision.
An unrepresentable total mean or overflowing reported variance raises
`ArithmeticError`; absent uncertainty remains NaN. Proportions and their complements
are floating-point values and are not promised bitwise exact summation.

## Validation and performance

51 tests cover independent Jacobian/covariance propagation, four valid native
reference cases through mean fitting and summaries, homozygous/heterozygous
progenitors, DNA-unit changes over factors `1e-150` to `1e150`, tiny mutant groups
checked with Decimal arithmetic, raw-variance underflow, absent/boundary uncertainty,
structural constants, batching, empty batches, ownership and input/error handling.

The [benchmark](sppcr-frequencies-benchmark.json) checks batched outputs against
repeated calls to this same Python summary API. Three-run medians measured about
**17x** for 32 experiments and **51x** for 1024 experiments, each with eight alleles.
These are Python batching measurements on the recorded machine, not native Fortran
speed comparisons. The summaries contain no per-experiment or per-allele Python loop.
