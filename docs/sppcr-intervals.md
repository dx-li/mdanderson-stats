# SPPCR confidence intervals

`sppcr_bootstrap_intervals` constructs the confidence limits used by the SPPCR
report from observed estimates and bootstrap standard deviations. The lower-level
`sppcr_intervals` accepts those inputs explicitly. [Analysis reports](sppcr-reporting.md)
include these limits. The full application workflow remains outstanding.

```python
import numpy as np
from mdanderson_stats import sppcr_bootstrap, sppcr_bootstrap_intervals

bootstrap = sppcr_bootstrap(
    [1, 2],
    [[4, 8, 2], [10, 15, 5]],
    [20, 30],
    progenitor=(0, 1),
    rng=np.random.default_rng(93),
    replicates=1000,
)
limits = sppcr_bootstrap_intervals(bootstrap)
frequency_bounds = (limits.frequency.lower, limits.frequency.upper)
calibration_bounds = (limits.calibration.lower, limits.calibration.upper)
```

## Statistical contract

The original `results_out_mod.print_answers` uses the rounded normal multiplier
`1.959964` for its nominal 95% intervals. This is the Python default. A custom
positive finite `multiplier` is accepted, but the API does not infer or label a
confidence level for it. These remain normal/bootstrap approximations, not exact
finite-sample coverage guarantees or percentile bootstrap intervals.

Calibration uses observed total mean ± multiplier × bootstrap calibration SD.
Allele and mutant frequencies use the observed `2*asin(sqrt(p))` transform ±
multiplier × bootstrap SD of the transformed frequency, then invert the limits.
Bootstrap means are not the interval centers. Asymptotic delta-method SDs and
untransformed frequency SDs are not substituted for transformed bootstrap SDs.

The bootstrap summary now retains its validated zero-based progenitor indices.
The convenience function uses these indices, the observed fit's means and the
corresponding bootstrap deviations. This keeps the mutant definition consistent.
The lower-level signature is:

```python
sppcr_intervals(
    mu,
    progenitor=(0, 1),
    calibration_sd=calibration_bootstrap_sd,
    frequency_transformed_sd=allele_transformed_bootstrap_sd,
    mutant_transformed_sd=mutant_transformed_bootstrap_sd,
    multiplier=1.959964,
)
```

`mu` has shape `(..., alleles)` and must be finite and nonnegative. Calibration
and mutant SDs broadcast to `(...)`; frequency SDs broadcast to `mu`. SDs must be
nonnegative and finite, or NaN to mark unavailable uncertainty. Empty experiment
batches are supported. The result retains the multiplier and progenitor indices.

## Boundaries and unavailable limits

Each `SPPCRInterval` contains `estimate`, `lower`, `upper`, `available`,
`lower_clipped` and `upper_clipped`. Arrays have immutable storage independent of
inputs. `SPPCRIntervals` groups calibration, inverse calibration, allele frequency
and mutant frequency intervals.

Calibration has nonnegative support, so a negative lower confidence limit is
clipped to zero and marked `lower_clipped`. The inverse calibration point estimate
is `1 / calibration`; its interval reverses the reciprocal endpoints. When a
positive calibration estimate has a lower bound of zero, the inverse interval
has an infinite upper endpoint, explicitly representing an unbounded interval.
This is the only intentional infinite output. If the calibration estimate is
zero, its reciprocal estimate and bounds are NaN and unavailable. Clipping flags
record an endpoint moved from outside support, so a bound already exactly zero is
not marked as clipped.

Frequency angles are clipped to `[0, pi]` before inversion. The original source
applies `sin(angle/2)**2` without clipping, which wraps out-of-domain angles and
can put a lower bound above the estimate or an upper bound below it. Python
instead returns bounds within `[0, 1]`, includes the point estimate despite
transform roundoff, and preserves exact point intervals when SD is zero. Very
large finite SD/multiplier combinations can overflow their width; on this bounded
frequency support that correctly produces the full `[0, 1]` interval.

All-zero mean vectors have undefined frequencies. Their frequency/mutant bounds
are NaN and unavailable, while calibration can remain defined. A NaN SD likewise
makes only the associated interval unavailable; known point estimates are still
retained. Bootstrap summaries containing an undefined replicate already expose
NaN frequency deviations, which flow through unchanged. No interval fabricates
uncertainty or silently excludes a replicate. Unavailable clipping flags are false.

Unrepresentable calibration bounds or nonzero reciprocal endpoints raise
ArithmeticError. A genuine reciprocal upper limit at zero is distinguished from
floating-point overflow for a small positive endpoint.

## Validation

`tools/reference_sppcr_intervals.py` pins the source archive, compiles its unchanged
transform module and evaluates the expressions transcribed from `print_answers`.
The fixture retains the driver, hashes, compiler and raw output. It establishes
interior numerical agreement and reproduces boundary defects; it does not claim
to execute or validate the complete report workflow.

Tests cover those native comparisons, zero/one and rare frequencies, exact point
intervals, interval ordering and inclusion, reciprocal endpoint reversal and
unboundedness, unavailable uncertainty, empty batches, immutability, invalid
inputs and the observed-center bootstrap integration.
`tools/benchmark_sppcr_intervals.py` checks exact equality of every bound and
diagnostic against individual calls while measuring batched execution. Local
three-run medians showed 24.34× speedup for 32 experiments and 184.82× for
1,024 experiments with four alleles each. Environment and timings are in
`sppcr-intervals-benchmark.json`; this compares the same Python API, not native
Fortran performance.
