# SPPCR bootstrap fitting and summaries

`sppcr_bootstrap` generates independent binomial replicate data, fits each
experiment and summarizes the replicate distributions. `sppcr_bootstrap_summary`
can also summarize previously fitted or externally supplied replicate means.
[Confidence intervals](sppcr-intervals.md) consume these summaries.
[Explicit legacy sampling](sppcr-random.md) is available through a RandlibGenerator.
[Reusable analysis workflows](sppcr-analysis.md) connect input and reports.
File/application workflows remain outstanding; SPPCR is partial.

```python
import numpy as np
from mdanderson_stats import sppcr_bootstrap

result = sppcr_bootstrap(
    dna=[1, 2],
    seen=[[4, 8, 2], [10, 15, 5]],
    wells=[20, 30],
    progenitor=(0, 1),
    rng=np.random.default_rng(93),
    replicates=1000,
)
frequency_mean = result.summary.frequency.mean
frequency_sd = result.summary.frequency.standard_deviation
undefined_replicates = ~result.summary.defined
```

## Sampling and fits

The default probabilities are original observed cell fractions, matching
`set_generate(from_truth=False)` in `generate_mod.f90`. They are computed before
any half-count boundary adjustments. An explicit `probability` argument,
broadcastable to `seen`, selects a different model; for truth-based simulation
use `sppcr_detection_probabilities(dna, calibration * frequency)`.
The result retains the actual probabilities and generated counts.

`seen` has shape `(..., levels, alleles)`. The generated data add a leading
replicate axis. `wells` broadcasts to `(..., levels)`. The default is 1,000
replicates, matching the source, but any positive Python integer is accepted.
Progenitor indices are zero-based; repeat an index for a homozygote.

An explicit NumPy Generator or RandlibGenerator is required. The latter uses
historical float32 binomials and the documented legacy safety limits. Input validation and the observed-data
fit precede sampling. Once generation starts, the supplied RNG is consumed;
subsequent fitting or numerical errors propagate and do not roll back RNG state.
No clock reseeding or silent retry occurs. Memory use grows with the number of
replicates because generated counts, fits and replicate values are retained.

The `saturation` policy applies to both observed and replicate fits and defaults
to `"half"`: subtract half a detection at the first minimum-DNA level only for an
allele detected in every well at every DNA level. This implements the source's
stated intent while repairing its broader, erroneous adjustment condition.
`"raise"` rejects such data instead. Never-seen alleles retain mean zero.
Fit counts, adjusted flags, interior flags and root brackets remain available.

`SPPCRBootstrap` contains `observed_fit`, `samples`, `fit` and `summary`.
No numerical failure is caught and converted into a fabricated successful fit.

## Replicate summaries and undefined frequencies

`sppcr_bootstrap_summary(mu, *, progenitor)` accepts finite, nonnegative means
with shape `(replicates, ..., alleles)`. Replicate and allele axes must be nonempty;
empty leading experiment axes are supported.

`SPPCRBootstrapSummary` retains the validated progenitor indices and contains these `SPPCRBootstrapSeries` results:

- `mu`: individual allele means;
- `calibration`: total allele mean;
- `frequency`: normalized allele frequencies;
- `transformed_frequency`: `2 * asin(sqrt(frequency))`;
- `mutant`: combined frequency outside the progenitor indices;
- `transformed_mutant`: the same forward transform of mutant frequency.

Each series retains all replicate `values` and reports `mean`, `variance` and
`standard_deviation` with the replicate axis removed. Variance uses divisor B,
as in the original accumulation module, rather than B−1. The standard deviation
measures the spread of bootstrap estimates; it is not the Monte Carlo standard
error of their mean and is not divided by sqrt(B). One replicate has zero
population variance. These empirical variances do not average the per-fit
observed-information or delta-method variances.

A replicate with zero total mean has undefined normalized frequencies. Its
`defined` flag is false and its frequency/mutant values are NaN, including their
transforms. If any replicate is undefined for an experiment, that experiment's
frequency/mutant mean, variance and standard deviation are all NaN. Other
experiments remain available. No replicate is discarded, replaced or implicitly
conditioned away. The mean and calibration series still include the zero result.
An allele with mean zero in an otherwise positive experiment has frequency zero,
which remains a valid replicate value even if its asymptotic variance is unknown.

All output arrays have immutable storage independent of inputs.

## Numerical behavior and validation

The source uses `sum(x*x)/B - (sum(x)/B)**2`, which can lose all meaningful
variance near a large common value. The Python calculation subtracts a replicate
anchor, computes a scaled mean offset, centers residuals and scales them before
squaring. This avoids large raw squared sums and preserves small spread around a
large mean. Standard deviation is computed separately so it can survive variance
underflow. Unrepresentable finite-range totals or variances raise ArithmeticError.
Frequency and transformed values reuse the stable complementary-group formulas
of `sppcr_frequencies` and inherit their floating-point accuracy limits.

Tests compare all six distributions against independent formulas, check centered
variance against Decimal calculations, recover identical generated counts and
scalar fits, and exercise all-zero and fully detected replicates, unknown
frequencies, empty batches, one replicate, ownership and invalid-input RNG state.
`tools/benchmark_sppcr_bootstrap.py` compares batched fitting and summaries with
individual fits on identical counts; it verifies exact fit and summary equality.
Recorded three-run medians showed 16.98× speedup for 32 replicates and 92.15× for
1,024 replicates, each with two DNA levels and three alleles. Timings and
environment appear in `sppcr-bootstrap-benchmark.json`.
These are same-Python measurements, not a native Fortran speed comparison.
