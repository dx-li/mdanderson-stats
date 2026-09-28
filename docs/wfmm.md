# Wavelet functional mixed models

The [MD Anderson WFMM software](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/70)
fits functional fixed and random effects in a transformed coefficient space.
The Python transform layer is available; the fitting and posterior-summary
layers are under implementation. Catalog entry 70 remains pending until a
validated statistical workflow is integrated.

## Transform and reconstruct curves

```python
import numpy as np
from mdanderson_stats import wfmm_basis, wfmm_transform, wfmm_inverse

curves = np.arange(24.0).reshape(2, 12)
basis = wfmm_basis(12, filter_length=8)
transformed = wfmm_transform(curves, basis)
reconstructed = wfmm_inverse(transformed.coefficients, basis)
```

Each row is a curve on an equally spaced grid. The default is a periodic
Daubechies db4 transform; even filter lengths 2–20 select db1–db10.
The default decomposition level is the largest `J` for which `2**J` divides
the curve length. Supply `levels` to use fewer levels. Odd lengths require
an identity or custom transform.

The coefficient order is `[a_J,d_J,...,d_1]`: coarsest approximation followed
by details from coarse to fine. Filtering samples even indices with forward
tap offsets and periodic wrapping. `coefficient_partition` separates the
approximation and each detail band; `coefficient_scale` records its level.
These are explicit Python conventions, not a claim to reproduce native WFMM
binary ordering or boundary-extension modes.

Use `transform="identity"` to retain the original columns. For
`transform="custom"`, provide a square `custom_matrix` with orthonormal
columns: the forward calculation is `curves @ custom_matrix`, and inversion
uses its transpose. Custom transforms do not center the input. PCA, energy
compression, alternate boundary modes and multidimensional transforms remain
open.

Transforms return readonly arrays and reject nonfinite values. They support
at most 4,096 time points and 2,000,000 input cells. Work estimates are checked
before filtering or custom matrix operations; `max_work` can lower the default
200,000,000-operation estimate. Wavelet transforms do not allocate a dense
time-by-time basis matrix.

Validation includes hand-computed Haar coefficients, db4 scalar calculations,
energy conservation and reconstruction, including a non-power-of-two length.
The [source and implementation audit](../research/wfmm-audit.md) tracks the
remaining statistical and native-software gaps.
