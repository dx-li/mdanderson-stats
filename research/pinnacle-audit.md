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
