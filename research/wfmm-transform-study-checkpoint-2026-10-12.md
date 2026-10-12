# WFMM PCA, pass filtering and saved Python workflow

This checkpoint follows accepted `c512e2abb0ceef6252f0eefb27ea75941109d944`.
WFMM70 remains partial; all 138 application counts remain 103/31/4.

## Verified handoff and reference correction

The replacement executor read the complete 84,949-byte JSON handoff from Library
`libfile_77a1f77d74c0819183791f7992068870`, restoring the single terminal newline
omitted by the text read. Each embedded file was UTF-8 encoded and verified
against its authoritative byte count and SHA-256 before inspection:

| Original path | Bytes | SHA-256 |
| --- | ---: | --- |
| tools/reference_wfmm_pca.py | 4,007 | e024a2bd88c44ff582400a7aba71b7598792ce6f92e4260f91c0a3eec8ac492c |
| tests/fixtures/wfmm-native-pca.json | 69,940 | 237d06b1ca4c6eba14ab946dbdd02b4686f5b9a5e238ab434f9461e239aba554 |
| tests/test_wfmm_native_pca_reference.py | 2,008 | d7dbf2148f3571ba2428f8fb5a17996b311bf1fbf2136587f4e63a57d9f94fb5 |

The original copies and Library identity remain outside the repository at
`/workspace/recovered-wfmm/pca-text`. This was authenticated text reconstruction,
not a successful ZIP download. The earlier two ZIP download failures remain
historical evidence, superseded by this successful text transfer.

The predecessor explicitly marked the PCA reference test provisional: seven
failures, sixteen passes, excluded from the accepted 76,423-test baseline.
Its independent one-level Haar matrix used negative-first detail differences,
while the actual source and existing production Haar transform use positive-first
differences in the same packing. Correcting only those matrix signs makes all
23 source tests pass. No production Haar taps or signs were changed. All
23 native cases regenerate byte-identically from the checksum-pinned original
WFMM1 executable, with 19 successes and four explicit failures (alphaPC=1 for
PC/PCw/wPC, and a zero-energy input). Original executable/archive hashes and
download provenance remain in the earlier native-prior audit and the generator.

## PCA scientific and source contracts

`wfmm_pca_basis` implements uncentered right-singular-vector PC projections,
PCw concatenation with wavelet-transformed residuals, and wPC after wavelet
analysis. The returned analysis/synthesis pair operates on original time axes.
Native fixtures validate retained subspaces, scores after sign alignment,
projection/full reconstruction and manual/equal/automatic partition counts.
Independent diagonal-matrix spectra validate cutoff, energy and row-space
reconstruction; extreme scales are handled through a single global rescaling.
Native SVD signs are not imposed on the Python decomposition.

The native retention rule crosses cumulative squared singular energy strictly.
Python preserves that rule below alpha=1, but at one retains the numerical row
space rather than reproducing the native zero-column failure. Zero-energy input
and tied cutoff directions are rejected. Automatic partitions use half-unit
bins of log-base singular values (source addresses 0x458ede–0x45902b), followed
by forward singleton merging and final backward merging. This corrects the
guide's eigenvalue wording. Native endpoint/allocation defects are not copied;
floating-point bin-boundary parity is not claimed.

`WFMMBasis.coefficient_count` now differs from time_count where appropriate.
Selection, compression, inverse reconstruction, posterior summaries and
covariance propagation support these rectangular pairs. Existing square custom
constructor validation remains unchanged. Independent explicit matrix products
verify covariance and draw summaries with both fewer and more coefficients than
time points. Dense covariance memory bounds still account for T-by-T buffers
even for rank-one projections.

PCw is redundant: independent coefficient priors/variances are an explicit
working model, not an exact independent-coordinate likelihood transformation
of the original curves. Coefficient/RNG/native whole-chain equivalence and
clinical validity are not consequences of subspace agreement.

## Pass-filter and compression source order

108 original runs cover one through four Haar levels, all highpass/lowpass
combinations in each tested range, and both uncompressed and compressed output.
Positive highpass h removes the approximation and the h coarsest detail bands;
lowpass l removes the l finest bands. These behaviors reverse the guide's
descriptions. Invalid and empty source cases become bounded Python errors.
The native program filters first, then ranks energy within the retained space.
The new eligible_indices compression control implements that composition while
preserving original coefficient IDs and zero-filled vote counts outside the
eligible subset. Reported compression energy is relative to eligible energy.

## Saved workflows and reporting

The version1 Python study schema stores observations, designs/strata, transform,
selection, initialization limits, sampling controls, seed and population
prediction inputs. Identity, periodic wavelet, square custom matrix-pair and
PCA/hybrid workflows save/reopen and compose with the existing initializer,
sampler, posterior summaries, covariance functions and predictive coefficients.
Initializer holds return a held report without MCMC. Reports use explicit nulls
for unavailable diagnostics and never equate successful sampling with certified
posterior precision or clinical validation. Separate PCG64 child streams keep
prediction requests from altering fitting draws.

Tests verify replay, prediction/fit stream separation, native pass selection,
independent frequency decomposition, custom pairs, random effects, held
initialization, schema/duplicate/nonfinite rejection and report output.
JSON input, summary JSON and zero-based curve CSV are Python formats; the
native MAT and binary report contracts have not been relabeled as complete.

Acceptance passed 76,651 full tests in 371.41s, with all 776 production Python
files and 1,599 test/fixture files unchanged during the run. Focused WFMM checks
passed 519 tests; a fresh installed wheel passed 547 tests including 28 plot/TIFF
checks, all three examples and dependency checks. Lint, formatting, types and
build passed. Both artifacts retain 29 identical notices and 780 matching package
files, excluding original binaries. Artifacts precede final evidence documentation.
See [usage](../docs/wfmm-studies.md) and
[validation manifest](wfmm-transform-study-validation-2026-10-12.json). Native boundary extensions, broader wavelet-family
parity, multidimensional transforms, original file/report interoperability,
hyperparameter updates and full native sampling compatibility remain open.
No status promotion, license/notices change, packaging split, PR bypass, merge,
release, deployment, permissions change or outreach is included.
