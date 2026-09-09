"""Legacy incomplete-gamma inverse with checked optional starting values."""

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._dcdflib import _probability_pair
from ._validation import FloatArray, finite
from .cdflib_gamma_factor import rcomp
from .cdflib_gamma_support import gamln
from .dcdflib_gamma import _scaled_quantile, _tails, cdfgam
from .dcdflib_quantile_helpers import dinvnr


def _small_shape_upper(a: FloatArray, q: FloatArray, x: FloatArray) -> FloatArray:
    """Refine subnormal upper tails at 1e-12<=a<1 in the log domain."""
    target, log_gamma = np.log(q), gamln(a)
    # Here x>600. Integration by parts has an alternating remainder bounded
    # by the first omitted term; after 32 terms it is below 1e-50 relatively.
    if np.any(~np.isfinite(x) | (x <= 600)):
        raise ArithmeticError("gamma log-tail refinement is outside its asymptotic regime")
    for iteration in range(4):
        term = total = np.ones(x.shape)
        for k in range(1, 33):
            term = term * (a - k) / x
            total = total + term
        log_tail = (a - 1) * np.log(x) - x - log_gamma + np.log(total)
        if iteration == 3:
            if np.any(~np.isfinite(log_tail) | (np.abs(log_tail - target) > 2e-12)):
                raise ArithmeticError("gamma log-tail refinement failed verification")
            break
        # d(log Q)/dx = -1/total for this integrated density expansion.
        x = x + (log_tail - target) * total
    return x


def gaminv(
    a: ArrayLike, p: ArrayLike | None, q: ArrayLike | None = None, x0: ArrayLike = 0.0
) -> FloatArray:
    """Invert unit-rate gamma tails, optionally checking a positive initial guess.

    Requires positive finite a and complementary probabilities. Nonpositive x0
    selects automatic inversion. Accurate positive guesses are retained; other
    guesses fall back to the validated gamma inverse. Native iteration counts
    are not exposed. Invalid/unrepresentable inputs and numerical failures raise
    exceptions. Native endpoints remain x=0 for p=0 and max-finite for q=0;
    the latter is a sentinel, not a finite solution to Q(a,x)=0.
    """
    shape, initial = finite(a, "a"), finite(x0, "x0")
    if np.any(shape <= 0):
        raise ValueError("a must be positive")
    pp, qq = _probability_pair(p, q)
    shape, pp, qq, initial = np.broadcast_arrays(shape, pp, qq, initial)
    result = np.zeros(shape.shape)
    result[qq == 0] = np.finfo(float).max
    interior = (pp > 0) & (qq > 0)
    target = np.minimum(pp, qq)
    # Avoid accepting a guess merely because two subnormal tails round to the
    # same bin. Those cases go through the automatic inverse and its repair path.
    hinted = interior & (initial > 0) & (target >= 1e-280)
    accepted = np.zeros(shape.shape, dtype=bool)
    with np.errstate(under="ignore"):
        if np.any(hinted):
            aa, xx = shape[hinted], initial[hinted]
            lower, upper = _tails(xx, aa, np.ones(aa.shape))
            actual = np.where(pp[hinted] <= qq[hinted], lower, upper)
            residual = np.abs(actual - target[hinted])
            factor = rcomp(aa, xx)
            # rcomp=x*dP/dx, so residual/rcomp estimates relative x error.
            # Require both tail and coordinate agreement before keeping a hint.
            accepted[hinted] = (
                (factor > 0) & (residual <= 1e-12 * target[hinted]) & (residual <= 1e-12 * factor)
            )
        result[accepted] = initial[accepted]
        solve = interior & ~accepted
        large = solve & (shape >= 1e20)
        if np.any(large):
            z = dinvnr(pp[large], qq[large])
            # Cornish-Fisher: the next correction is bounded by 2e-7 at
            # a=1e20 for every positive float64 tail pair, far below one x ULP.
            # At enormous a, all finite-probability quantiles round to a.
            result[large] = shape[large] + np.sqrt(shape[large]) * z + (z * z - 1) / 3
        solve = solve & ~large
        refine = solve & (shape >= 1e-12) & (shape < 1) & (qq < np.finfo(float).tiny)
        if np.any(refine):
            aa = shape[refine]
            initial_x = _scaled_quantile(pp[refine], qq[refine], aa, np.ones(aa.shape))
            result[refine] = _small_shape_upper(aa, qq[refine], initial_x)
        regular = solve & ~refine
        if np.any(regular):
            result[regular] = cdfgam(2, shape=shape[regular], p=pp[regular], q=qq[regular]).x
    return _freeze(result)
