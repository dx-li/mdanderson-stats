# SPPCR historical random streams

The SPPCR archive's L'Ecuyer–Côté generator, phrase seeding, standard uniforms and
binomial sampler have been reconciled against the package's existing RANDLIB
implementation. `sppcr_generate_legacy` provides an explicit historical sampling
path. The default `sppcr_generate` remains the fast NumPy implementation.

```python
from mdanderson_stats import (
    RandlibGenerator,
    sppcr_generate_legacy,
    sppcr_fit_means,
    sppcr_bootstrap_summary,
)

rng = RandlibGenerator()
seeds = rng.set_phrase("SPPCR seed", source="fortran")
samples = sppcr_generate_legacy(
    [[0.1, 0.4, 0.8], [0.2, 0.5, 0.9]],
    [20, 40],
    rng=rng,
    replicates=1000,
)
fits = sppcr_fit_means([1, 2], samples.seen, samples.wells, saturation="half")
summary = sppcr_bootstrap_summary(fits.mu, progenitor=(0, 1))
final_seeds = rng.get_seeds()
```

## Source interface mapping

| SPPCR source routine | Existing Python interface |
|---|---|
| `random_large_integer` | `RandlibGenerator.integers` |
| `get_current_generator`, `set_current_generator` | `stream`, `select` |
| `get_seeds`, `set_current_seed` | `get_seeds`, `set_seeds` |
| `set_all_seeds` | `set_all_seeds`, with the defect repair below |
| `set_antithetic` | `set_antithetic` |
| `reinitialize_current_generator` | `reinitialize(-1/0/1)` |
| `advance_state` | `advance_state`, using efficient modular arithmetic |
| `random_standard_uniform` | `uniform(legacy=True, source="fortran")` |
| `random_binomial` | `binomial(legacy=True, source="fortran")` |
| `phrase_to_seed` | `ranlist_seeds`; `set_phrase` also initializes the bank |
| clock branch of `set_seeds` | `set_time`, accepting an explicit datetime for replay |

Stream numbers are 1–32. A bank is caller-owned mutable state. Reinitialization
uses the original state, current block start or next block start according to mode.
The next block advances by 2**30 draws. `advance_state(k)` adopts the state 2**k
draws ahead as the selected stream's new initial/block state. Antithetic mode
complements raw integers before their float32 conversion.

The Fortran uniform path rounds the raw integer to float32 before multiplying by
the float32 source constant. The C RANDLIB scaling path differs and is not the
SPPCR compatibility path. Likewise, the nonlegacy RANDLIB binomial inverse-CDF
method does not reproduce the source BTPE/inversion draw sequence.

## Explicit legacy SPPCR generation

`sppcr_generate_legacy(probability, wells, *, rng, replicates=1)` shares the modern
sampler's shapes: probabilities are `(..., levels, alleles)`, wells broadcast to
`(..., levels)`, and counts add a leading replicate axis. It consumes the selected
RANDLIB stream in replicate, experiment-batch, DNA-level, allele order. Within a
single experiment, this matches the original `generate` loop. Arrays are immutable;
returned probabilities are the actual float32 values represented in float64.

All design validation precedes random draws. Wells must be positive integers no
larger than 2,147,483,646, following RANDLIB's checked legacy domain. Exact p=0 and
p=1 work and preserve native random consumption. Positive probabilities rounding
to zero, or subunit probabilities rounding to one in float32, raise rather than
silently changing the model. Zero replicates and empty experiment batches consume
no randomness. The existing binomial rejection/overflow safeguards still apply.
If a later cell fails numerically, earlier cells' RNG consumption is not rolled
back; callers requiring transactional orchestration must account for this.

Historical rejection sampling is sequential and this explicit path prioritizes
sequence compatibility. Use NumPy-based `sppcr_generate` for fast modern batches.
The high-level `sppcr_bootstrap` and [analysis workflows](sppcr-analysis.md) now
also accept a RandlibGenerator; the example remains an explicit composition of
legacy sampling with the common fitting/summary primitives.
Fitting retains the documented Python boundary repairs, so matching RNG draws
does not claim reproduction of the native program's defective boundary estimates.

The supplied probabilities also matter: the stable Python `expm1` model formula
can preserve rare probabilities that the source's `1-exp(-x)` loses. Matching the
native sampler for supplied probabilities is distinct from claiming that every
modern probability calculation reproduces native rounding.

## Verified reseeding defect and safety differences

Native `set_all_seeds` writes stream 1's initial seeds, then reinitializes the
currently selected stream before looping over streams 2–32. If another stream
is selected, stream 1's current state can remain stale. The native fixture
reproduces this after selecting stream 7. RANDLIB correctly resets all streams
while preserving selection and antithetic flags. An independent modular-recurrence
check verifies the repaired stream-1 draws from the requested new seeds.

Original routines use global state and can terminate the process on bad inputs.
Python uses private generator banks, validates seeds/stream indices/bounds and
raises exceptions. The original bootstrap forcibly reseeds from the clock; the
Python sampling APIs require explicit RNG state and never do that implicitly.
The source clock phrase is HHMMSS.mmm; recording an explicit datetime/base seed
pair through the existing seeding API permits reproducible use.

## Native evidence

`tools/reference_sppcr_random.py` pins the archive and hashes the four compiled
source files. The fixture retains the complete driver, compiler identification,
commands and raw output for 101 native scenarios. Comparisons cover all 32 streams,
antithetic draws, selected-stream seeding, block resets, jumps through exponent
64, phrase variants, binomial inversion and BTPE cases including endpoints, cached
parameter changes, and a complete replicate/DNA/allele draw grid. Tests compare
every recorded draw and component state, not just distributional moments.

The observed compatible cases match exactly on the recorded compiler/platform.
This is finite reference evidence, not a universal guarantee across every seed,
parameter or floating-point compiler. The fixture explicitly isolates the native
reseed defect rather than treating it as a desired Python result. Additional tests
verify input rejection without state consumption and the legacy SPPCR sampler's
shape, actual probabilities, immutable results, endpoints and final state.
