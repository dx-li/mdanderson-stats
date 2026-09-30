# WFMM custom transform matrix pairs

WFMM's guide documents a caller-supplied custom transform using separate
analysis and synthesis matrices. For row-wise curves `Y`, the coefficient
calculation is `D = Y @ phi_inv`; reconstruction is `Y = D @ phi`. Use both
matrices directly:

```python
import numpy as np
from mdanderson_stats import wfmm_basis, wfmm_transform, wfmm_inverse

analysis = np.array([[2.0, 1.0], [0.0, 1.0]])  # phi_inv
synthesis = np.array([[0.5, -0.5], [0.0, 1.0]])  # phi
basis = wfmm_basis(
    2,
    transform="custom",
    analysis_matrix=analysis,
    synthesis_matrix=synthesis,
)
curves = np.array([[1.0, 3.0], [-2.0, 0.5]])
coefficients = wfmm_transform(curves, basis).coefficients
reconstructed = wfmm_inverse(coefficients, basis)
np.testing.assert_allclose(reconstructed, curves)
```

The paired matrices must both be finite square `(time_count,time_count)` arrays
and numerical two-sided inverses within absolute tolerance `2e-10`. The code
also requires the infinity-norm condition estimate
`||analysis_matrix||inf * ||synthesis_matrix||inf` to be at most `1e8`. The
estimate is evaluated in the log domain, so uniform reciprocal scaling does
not by itself cause rejection. This guards against round-trip information loss
in pairs that multiply to identity in floating point but are too ill-conditioned
for reliable use. The code does not calculate a missing inverse or silently
regularize a pair. The old `custom_matrix` interface remains available for
orthogonal matrices and keeps its previous transpose reconstruction behavior.
Supplying both interfaces, or only one member of the pair, is an error.

This accepts the guide's square custom-matrix case. It does not implement
rectangular PCA, retained-component projection or the named `PCw`/`wPC`
transforms. A non-orthogonal pair changes coefficient-space variance meanings;
WFMM covariance propagation therefore uses the supplied synthesis matrix as
`phi.T @ diag(omega) @ phi`. It is not an orthogonal-model invariance claim.

Matrix construction, transformation and covariance operations use the existing
WFMM dimension, operation and memory limits. The direct inverse transform is
used by selection/restoration and posterior curve reconstruction as well.
