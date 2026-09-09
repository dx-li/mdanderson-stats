"""Stable gamma scaling factor used by incomplete-gamma support routines."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln, gammasgn

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .cdflib_elementary import _log_remainder
from .cdflib_gamma_ratios import _delta
from .cdflib_gamma_support import _local_log_gamma, _positive_log_gamma


def _large_log_factor(aa: FloatArray, xx: FloatArray) -> FloatArray:
    """Log gamma factor for positive x and a>=8; retain near-center displacement."""
    near = (xx >= 0.5 * aa) & (xx <= 1.5 * aa)
    cost = np.empty(aa.shape)
    # Subtract original coordinates, not their rounded quotient.
    d = (xx[near] - aa[near]) / aa[near]
    cost[near] = aa[near] * _log_remainder(d)
    # The log difference remains finite even if x/a underflows to zero.
    cost[~near] = (xx[~near] - aa[~near]) - aa[~near] * (np.log(xx[~near]) - np.log(aa[~near]))
    return (0.5 * np.log(aa) - 0.9189385332046727) - _delta(1 / aa) - cost


def rcomp(a: ArrayLike, x: ArrayLike) -> FloatArray:
    """Compute exp(-x)*x**a/Gamma(a) for finite real broadcast inputs.

    As in F95, x<=0 returns zero. At x>0, nonpositive integer a uses
    reciprocal-gamma continuation (zero); negative nonintegers retain their
    sign. Output overflow raises ArithmeticError; signed underflow is allowed.
    """
    shape, coordinate = np.broadcast_arrays(finite(a, "a"), finite(x, "x"))
    result = np.zeros(shape.shape)
    active = (coordinate > 0) & ~((shape <= 0) & (shape == np.floor(shape)))
    local = active & (shape >= -0.2) & (shape <= 1.25)
    large = active & (shape >= 8)
    other = active & ~(local | large)
    with np.errstate(over="ignore", under="ignore"):
        aa, xx = shape[local], coordinate[local]
        result[local] = aa * np.exp(aa * np.log(xx) - xx - _local_log_gamma(aa))

        aa, xx = shape[large], coordinate[large]
        result[large] = np.exp(_large_log_factor(aa, xx))

        aa, xx = shape[other], coordinate[other]
        positive = aa > 0
        log_gamma = np.empty(aa.shape)
        log_gamma[positive] = _positive_log_gamma(aa[positive])
        log_gamma[~positive] = gammaln(aa[~positive])
        result[other] = gammasgn(aa) * np.exp(aa * np.log(xx) - xx - log_gamma)
    if np.any(~np.isfinite(result)):
        raise ArithmeticError("gamma scaling factor exceeds the finite output range")
    return _freeze(result)
