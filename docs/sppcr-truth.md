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
[the RNG reconciliation](sppcr-random.md). The high-level bootstrap and [analysis workflows](sppcr-analysis.md) also accept
a RandlibGenerator. Replicates can also be composed explicitly from legacy sampling,
`sppcr_fit_means`, and `sppcr_bootstrap_summary`.

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

SPPCR remains partial. Full application and file workflows and the final
source-interface completion audit remain.

## Interactive parameter entry and reporting

`read_sppcr_truth(input_stream=None, output_stream=None, *, max_attempts=3,
max_records=10000, max_line_length=10000)` returns an immutable
`SPPCRSimulationRequest` containing `truth`, `bootstrap_from_truth`, and
`write_simulations`. It uses caller-owned streams (standard input/output by default).
No random draws or file operations occur. These flags describe the requested
subsequent analysis and output; callers still perform generation and route reports.

Input order follows `generate_parameters_in`:

1. DNA-level and allele counts together, each bounded to 1..50.
2. Common wells, 1..1000.
3. Nonnegative allele weights, normalized by the truth factory.
4. Model DNA amounts, .001..10000, without doubling.
5. Calibration, .001..1000.
6. Two **one-based** progenitor indices, converted to zero-based in the result.
7. Whether to bootstrap from truth (`y`) or observed fractions (`n`).
8. Whether to request replicate estimates for output (`y`/`n`).

The source does not bound its first dimension input or validate weights. Python
bounds dimensions consistently with other SPPCR entry APIs, rejects negative
weights and permits correction of all-zero lists. Numeric corrections restart
the complete requested vector. Numeric and all-zero-list retries have separate
`max_attempts` bounds; `max_records` applies per input request, and line length is
bounded. Continuations, commas, repeat syntax, and excess-final-record handling
reuse the CDFLIB console contract. EOF returns no partial request; stream failures,
exhausted retries, and numerical underflow propagate explicitly.

`format_sppcr_truth(request, *, precision=10, max_characters=1000000)` returns a
bounded tab-separated report. It includes native parameter-report quantities
(dimensions, normalized frequencies, calibration, progenitors and wells), plus
explicit simulation choices, model DNA, allele means and detection probabilities.
It prints one-based indices and lists every run's wells, supporting unequal-well
programmatic designs. It validates consistency of public truth records and never
silently renormalizes reported values. Precision is 1..17 significant digits.

```python
from io import StringIO
from mdanderson_stats import read_sppcr_truth, format_sppcr_truth

request = read_sppcr_truth(StringIO("2 3\n100\n2 5 3\n.25 1\n1.7\n1 2\ny\nn\n"), StringIO())
text = format_sppcr_truth(request)
```

`tools/reference_sppcr_truth_console.py` calls the unchanged native dialogue,
including its parameter report, for ordinary, continued/homozygous, and corrected
well-count transcripts. It supplies an open scratch report unit: the source
`asterisks` routine writes to any supplied unit even when its caller considers
reporting disabled. Clock-seeded counts from the native dialogue are excluded from
comparison. Tests compare all entered parameters and choices, report quantities,
corrections, EOF, stream ownership, repeated entry and output bounds.

Full application orchestration, CLI/file routing and final completion audit remain.
