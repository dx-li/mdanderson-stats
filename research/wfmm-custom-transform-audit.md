# WFMM paired custom-transform audit

The WFMM user guide (pages 2–4; cited in the existing
[WFMM audit](wfmm-audit.md)) names a `custom` transform and specifies distinct
forward and reverse matrices: `D = Y phi_inv` and `Y = D phi`. This contract
does not require the matrices to be orthogonal. For a row-oriented Python curve,
the direct products are `Y @ phi_inv` and `D @ phi`.

This change supports the source's square full-dimension case: both matrices are
`T x T` and are checked as finite two-sided inverses before they are retained.
The check uses absolute tolerance `2e-10`; the supplied pair's infinity-norm
condition estimate must also be at most `1e8`. That product is calculated
from log norms so reciprocal uniform scales through the tested extreme range
remain allowed. The limit rejects pairs whose product is identity but whose
round trip can erase meaningful components. No inverse is guessed, and no
jitter or rank repair is applied. Rectangular transforms are not included
because the guide's custom-matrix equations do not establish a retained-
component projection rule. The guide's separately named PCA transforms remain
distinct.

The previous `custom_matrix` API remains the orthogonal special case, using the
matrix for analysis and its transpose for synthesis. The paired path retains
separate readonly analysis and synthesis matrices. Transform, inverse,
selection/restoration and posterior reconstruction flow through the same basis
object. Covariance propagation uses `phi.T @ diag(omega) @ phi`, with
square-root variance weighting to avoid premature overflow when a supplied
matrix has a large or small uniform scale. Variance-function diagonals are
computed from the same synthesis rows.

The public helper is a transform-basis input only: using a non-orthogonal pair
changes the coefficient-space priors and variances and is not asserted to leave
the orthogonal WFMM model invariant. Native file handling, PCA calibration,
rectangular retained-PC rules and the `PCw`/`wPC` workflows remain outside this
implementation.

Focused validation in this branch covers inverse round trips, transform and
covariance orientation, variance-function diagonals, invalid matrix pairs,
legacy orthogonal-matrix behavior, rejection of an exactly represented
ill-conditioned shear, and finite covariance under uniform matrix rescaling
through `1e-200` and `1e200`. Independent base-R mathematical references are
prepared for integration comparison; they do not exercise the native WFMM
application. This branch's checks do not claim native runtime or file-workflow
parity.
