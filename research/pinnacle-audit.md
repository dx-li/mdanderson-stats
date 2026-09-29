# Pinnacle coverage audit

Baseline main `d44c2e2`. The previous goal turn made progress: CiBolus's core
workflow, independent references and packaged examples were integrated.
Coverage is 62 implemented, 61 partial and 15 pending entries. The complete
catalog/publication objective remains active.

Root uses `feat/pinnacle-core`; one Luna agent uses `feat/pinnacle-luna` in the
existing independent checkout. Initial system memory reported 47% free.
Numerical work stays serial with one BLAS/OpenMP thread. No large gel stack,
dependency installation or full-suite run is planned.

The publisher-hosted 2008 primary paper and official detailed manual establish
average-gel detection, undecimated Daubechies wavelet denoising, orthogonal local
maxima, square proximity suppression and per-gel maximum intensity quantification
with background subtraction and normalization. Live PMC reads and PDF equation
screenshots were unavailable. Native Pinnacle archive/executable behavior is
not yet verified; ties, image-edge handling and region indexing require explicit
Python conventions.

The original RWT 2.4 source archive was retrieved through a read-only GitHub
file fetch from Rice's repository, fixed revision
`ba7587cef713cd23f17de8c7c788b63a5d2c9454`, path `dist/2.4/rwt.tar.gz`.
The 15,447-byte archive SHA-256 is
`cf242020f598718768f4fc6f7dfdcffee997caae1cf55b051af837afbf51b207`.
Its source algorithms were inspected and its complete license is preserved.
Reference-only source stays under ignored `research/raw`; it is not bundled.

The original forward/inverse redundant transform uses periodic boundaries,
strided filter operations, three detail images per level and no decimation.
Native `denoise.m` selects `floor(log2(min(shape)))` levels and requires each
dimension to be divisible by the corresponding power of two. The low-pass
component is not thresholded. Native noise estimation uses the median absolute
finest diagonal coefficients divided by .67, with hard-threshold equality
discarded. The article instead specifies .6745 and equality retained. The
Python interface must distinguish these conventions. The paper's default
filter length is eight (four vanishing moments) and multiplier two; the later
GUI specifies length six and multiplier 3.6. The paper's MAD wording is
interpreted as median-centered absolute deviation; native RWT instead computes
absolute coefficients about zero. This distinction is exposed and documented,
not claimed to reproduce an uninspected Pinnacle executable.

Base-R filter generation completed for even lengths two through twenty.
The original C forward/inverse algorithms were compiled with an allocation-only
shim: MATLAB's implicit per-call allocation lifetime becomes explicit tracked
allocation/free. Transform arithmetic is unmodified. Three tiny square/rectangular
image cases reconstructed successfully; six denoising references compare
thresholded inverse outputs. No MATLAB runtime or additional package was
installed.

## Implemented checkpoint

Luna's image primitives (`3a63b08`) and wavelet/workflow implementation
(`af1ce26`) were integrated as `15734bf` and `532f2fd`. Reference fixtures and
their source/license records were committed separately as `dccc45c`.
`run_pinnacle` averages raw gels, denoises the single average, detects peaks,
and quantifies the original individual gels in a second pass. It checks
ordered pixel identity with a streaming digest, preserves original-image
coordinates for cropped regions, and never caches an image stack.

The NumPy implementation covers minimum-phase Daubechies lengths 2–20,
periodic redundant forward/inverse transforms, explicit paper/RWT denoising
conventions, four background modes and four normalization modes. Review
corrected the direct-filter high-pass sign, separated native transform and
denoiser level defaults, and included immutable copies/temporaries in a
conservative `(3*levels+16)` image-buffer budget. Nonrepresentable wavelet
coefficients raise an explicit arithmetic error.

The detailed GUI manual (§3.4, pp. 9–11) permits optional denoising of each gel
during quantification (default off; threshold multiplier 3.6) and independently
sized horizontal and vertical background windows. The Python API represents
these as opt-in `PinnacleDenoiseSettings` and a `(row_radius, column_radius)`
pair while retaining scalar square radii. It denoises each raw gel sequentially,
allows signed reconstruction only inside measurement calculations, and keeps
raw image volume normalization and the replay digest tied to original inputs.
Because the executable's operation order is undocumented, this implementation
detects on the denoised raw average first, then applies optional per-gel
denoising before peak and background measurement. Native TIFF/project formats,
interactive editing and report/executable equivalence remain open. Catalog
entry 95 is **partial**.
The catalog now has **62 implemented, 62 partial and 14 pending** entries.

## Validation and resource use

Luna ran the following from its independent checkout, using the root Python
environment and one BLAS/OpenMP thread:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  '/Users/dxli2/math stats/mdanderson-stats/.venv/bin/pytest' -W error \
  tests/test_pinnacle.py tests/test_pinnacle_wavelet.py \
  '/Users/dxli2/math stats/mdanderson-stats/tests/test_pinnacle_reference.py' -q
```

All **11 tests passed in 2.08 seconds**. These include ten independent R filter
coefficient sets, three original-C forward/inverse cases, six original-C
denoising cases, all sixteen R background/normalization combinations, cropped
coordinates and replay identity. The original C comparison checks coefficients
as well as reconstructed images, so a reversed high-pass sign cannot hide
behind a successful round trip. Luna's final Ruff and three-module mypy checks
passed after guard/typing cleanup. Root's integrated Ruff and format checks
passed on eight affected Python files.

Root built a wheel and source archive with the cached Hatch backend, without
installation or network access. An isolated, warnings-as-errors Python process
imported the wheel, verified all thirteen public names, compared packaged source,
catalog and notice bytes, checked source-archive documentation/license contents,
and verified that `research/raw` was excluded. The documented replayable example
found two peaks and passed its coordinate, shape and normalization assertions.
A patched decomposition entry point confirmed that an insufficient memory budget
is rejected before the coefficient pyramid is allocated. An extreme finite input
confirmed explicit overflow failure.

System memory was 48% free immediately before the representative check. A single
1024-by-1024 synthetic gel, default length-eight filter and ten levels denoised
in **1.005 seconds**. Output was finite, immutable and preserved the image mean.
The entire isolated verification process took **3.175 seconds**, peaked at
**415.48 MiB RSS**, and reported **zero process swaps**. This is one local
measurement, not a guarantee for every image or machine. Numerical jobs stayed
serial with one thread; no large gel stack, full repository suite or dependency
installation was run.

This goal turn made progress. The complete catalog/publication objective remains
active. The existing GitHub write approval block is unchanged; no alternate
publication transport was attempted. Next reconcile BOIN desktop entry 99 with
the already implemented method families and identify its actual remaining gaps.
