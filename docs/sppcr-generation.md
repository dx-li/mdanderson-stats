# SPPCR probabilities and reproducible generation

SPPCR's independent binomial sampling layer is available through three functions.
[Bootstrap fitting and summaries](sppcr-bootstrap.md) build on this layer.
[Confidence intervals](sppcr-intervals.md) build on the bootstrap results.
The full application workflow remains pending.

```python
import numpy as np
from mdanderson_stats import (
    sppcr_detection_probabilities,
    sppcr_generate,
    sppcr_fit_means,
    sppcr_observed_probabilities,
)

# Truth-based simulation: mu = calibration * allele frequencies.
dna = [0.5, 1, 2]
mu = np.array([0.2, 0.5, 0.3])
p = sppcr_detection_probabilities(dna, mu)
samples = sppcr_generate(p, [40, 80, 120], rng=np.random.default_rng(123), replicates=100)
fits = sppcr_fit_means(dna, samples.seen, samples.wells, saturation="half")

# The source's empirical bootstrap model uses original observed cell fractions.
p_observed = sppcr_observed_probabilities([[5, 10], [12, 16]], [20, 30])
resampled_counts = sppcr_generate(
    p_observed, [20, 30], rng=np.random.default_rng(456), replicates=100
)
```

## Models and shapes

`sppcr_detection_probabilities(dna, mu)` computes `-expm1(-dna * mu)`.
`dna` is a positive one-dimensional vector; `mu` has shape `(..., alleles)`
and contains finite nonnegative means. The output shape is
`(..., levels, alleles)`. To specify calibration and frequencies, construct
`mu = calibration * frequency` explicitly; the function does not silently
normalize weights. Large positive products correctly saturate at probability
one, including products that overflow float64. Tiny probabilities can underflow
to zero; representable rare probabilities survive the subtraction cancellation
in the original `1 - exp(-x)` formula.

`sppcr_observed_probabilities(seen, wells)` returns `seen / wells` for the
original, unadjusted counts. `seen` has shape `(..., levels, alleles)` and
`wells` is scalar or broadcasts to `(..., levels)`. Counts are nonnegative
integers below `2**53`; wells must be positive and seen cannot exceed wells.
This is the `set_generate(from_truth=False)` model in `generate_mod.f90`.
It generally differs from probabilities computed from fitted Poisson means.
Use the original counts, rather than a fit's half-count-adjusted `seen` array.

`sppcr_generate(probability, wells, *, rng, replicates=1)` accepts probabilities
of shape `(..., levels, alleles)` in `[0, 1]`, with the same well-count contract.
It draws independent binomials across replicates, experiments, levels and alleles
in a single NumPy call. `replicates` is a nonnegative Python integer.
Empty leading batch axes and zero replicates produce empty arrays without
consuming random state; level and allele axes must be nonempty.

The immutable `SPPCRSamples` result contains:

- `probability`: the unreplicated probability array;
- `wells`: the broadcast unreplicated design, omitting the allele axis;
- `seen`, `unseen`: shape `(replicates, ..., levels, alleles)`.

All arrays own immutable byte storage independent of inputs. Counts use float64,
are exact integers within the accepted range, and feed directly into the fitting
API. Each cell satisfies `seen + unseen == wells`. Zero/one detection probabilities
produce deterministic counts. No per-allele categorical constraint is imposed:
SPPCR models detections as independent binomials, not a multinomial partition.

## Random state and source differences

An explicit `numpy.random.Generator` is required. The function neither uses the
global NumPy RNG nor resets seeds from the clock. Input validation precedes the
sampling call. Reproducibility requires the same NumPy environment, bit generator,
initial state, parameters and call ordering; cross-version sequence stability is
not promised. Repeated calls advance the supplied generator.

The original source narrows probabilities to single precision and calls its
Fortran binomial generator in replicate/level/allele order. The Python path retains
float64 probabilities and uses NumPy's generator. It implements the same binomial
model but does not reproduce historical random sequences. Reconciliation of
SPPCR's bundled RNG with the package's RANDLIB implementation remains pending.
The original bootstrap also forcibly seeds from the clock; that behavior is not
part of this explicit-state API.

Generating counts does not guarantee a regular fitted experiment. A replicate
can have a never-seen allele, a fully detected allele, or all-zero counts. The
existing fitter retains its explicit boundary policy; all-zero total means still
make frequency summaries undefined. This layer neither drops replicates nor
invents replacement observations. The bootstrap layer retains all-zero replicates
with explicit undefined-frequency diagnostics.

## Validation and performance

`tests/test_sppcr_generate.py` checks detection probabilities against 230-digit
Decimal calculations, model distinctions, extreme means, independent binomial
moments and cross-cell covariance over 100,000 draws, and recovery of known means
from simulated experiments. It also checks RNG continuation, batch ordering,
endpoints, large exact counts, ownership, empty batches and invalid inputs.
NumPy's `Generator.binomial` signature and broadcasting contract were verified
against the installed NumPy 2.5.3 documentation after a Context7 lookup.

`tools/benchmark_sppcr_generate.py` compares one batched call with repeated calls
to the same Python API, asserting identical samples and final RNG states. In the
recorded three-run medians, batches of 32 and 1,024 replicates were respectively
8.47 and 19.35 times faster. Each experiment had three DNA levels and four alleles.
These are local same-Python measurements, not comparisons with native Fortran;
see `sppcr-generation-benchmark.json` for environment and timings.
