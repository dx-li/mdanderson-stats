# SPPCR truth-generation designs

`sppcr_truth(dna, wells, weights, calibration, *, progenitor)` prepares one
simulation design. It returns an immutable `SPPCRTruth` with DNA, wells,
normalized allele frequencies, calibration, zero-based progenitor indices,
allele means, and detection probabilities. Homozygous progenitors repeat an index.
Zero-weight alleles remain present.

The source `problem_in_mod.generate_parameters_in` normalizes supplied weights;
`generate_mod.set_generate(.true.)` uses
`p[i,j] = 1 - exp(-dna[i] * calibration * frequency[j])`. The Python factory
uses the same model and the existing stable `expm1` probability implementation.
Its DNA argument is used **directly**, just as in native truth entry. Do not apply
the genome-to-allele factor of two used by observed-data input readers a second
time. The returned design is ready for numerical generation and fitting APIs.

```python
import numpy as np
from mdanderson_stats import sppcr_truth, sppcr_generate, sppcr_bootstrap

truth = sppcr_truth(
    dna=[0.25, 1, 2],
    wells=100,
    weights=[4, 5, 1],
    calibration=1.5,
    progenitor=(0, 1),
)
rng = np.random.default_rng(42)
first = sppcr_generate(truth.probability, truth.wells, rng=rng)
result = sppcr_bootstrap(
    truth.dna,
    first.seen[0],
    truth.wells,
    progenitor=truth.progenitor,
    probability=truth.probability,
    rng=rng,
    replicates=1000,
)
```

The first experiment is always drawn from the supplied truth in this example.
Passing `probability=truth.probability` bootstraps from truth; omitting that keyword
bootstraps from the first experiment's observed fractions. This makes the two
choices explicit. The native dialogue instead passes its bootstrap-choice flag
to `set_generate` before its first experiment, potentially reading uninitialized
observed counts when that choice is false. Python callers should generate the
first experiment from truth before choosing their bootstrap model.

For historical sampling, pass the design's probabilities and wells to
`sppcr_generate_legacy` with an explicit `RandlibGenerator`; see
[the RNG reconciliation](sppcr-random.md). The high-level bootstrap function still
uses a NumPy generator. Its replicates can instead be composed explicitly from
legacy sampling, `sppcr_fit_means`, and `sppcr_bootstrap_summary`.

## Validation and numerical choices

DNA is a finite positive vector; weights are a finite nonnegative vector with a
positive total. Calibration is a finite positive scalar. Wells are positive
integer counts below `2**53`, broadcast over DNA levels. Per-level wells are
supported, extending the native dialogue's common-well restriction. This numerical
API does not impose the dialogue's narrower input ranges.

Weights are divided by their maximum before summation, so finite inputs such as
`[1e308, 1e308, 1e308]` normalize successfully. If a positive frequency or mean
would round to zero, construction raises `ArithmeticError`. Detection probabilities
may underflow to zero or saturate at one as documented by the probability API.
Arrays own immutable storage; changing input arrays cannot change a design.
Construction consumes no RNG state.

`tools/reference_sppcr_truth.py` compiles the pinned original sources unchanged
and calls their actual `set_generate(.true.)` and `generate` routines. Its driver
uses the normalization expression from native truth entry. Three recorded designs
cover ordinary weights, zero weights, a single allele, multiple DNA levels, and
unequal wells. Python tests match their normalized frequencies within floating-point
roundoff and their three seeded experiments exactly through the legacy sampler.
This is finite reference evidence, not a promise of matching all native rounding
or platform behavior. Decimal checks independently validate normalization and
probabilities; integration tests exercise both bootstrap choices.

SPPCR remains partial. Truth-parameter dialogue/reporting, full application and
file workflows, and the final source-interface completion audit remain.
