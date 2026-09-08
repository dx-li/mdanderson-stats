"""MULTI's uniform-plus-beta model, posterior calculations and fit diagnostics.

Adapted model formulas; original copyright/terms are in the MULTI notice.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betainc, betaln, logsumexp, xlog1py, xlogy

from ._validation import FloatArray, finite, scalar
from .multiplicity import _pvalues


def _values(value: ArrayLike) -> FloatArray:
    x = finite(value, "pvalues")
    if np.any((x < 0) | (x > 1)):
        raise ValueError("pvalues must lie in [0,1]")
    return x


def _logs(x: FloatArray, legacy: bool) -> tuple[FloatArray, FloatArray]:
    with np.errstate(divide="ignore"):
        lx, ly = np.log(x), np.log1p(-x)
    if legacy:
        tiny = np.finfo(np.float64).tiny
        # INITLN uses +tiny (not -tiny) for the log near one.
        lx = np.where(x <= tiny, np.log(tiny), np.where(1 - x <= tiny, tiny, lx))
        ly = np.where(1 - x <= tiny, np.log(tiny), np.where(x <= tiny, tiny, ly))
    return lx, ly


@dataclass(frozen=True, init=False)
class BetaMixture:
    """A uniform null component plus k beta components with normalized weights.

    Arrays are copied and made read-only. Weights must sum to one within 1e-12;
    the accepted rounding residual is normalized. Zero-weight components are
    excluded from density calculations, including at singular endpoints.
    """

    null_weight: float
    weights: FloatArray
    a: FloatArray
    b: FloatArray

    def __init__(
        self, null_weight: float, weights: ArrayLike = (), a: ArrayLike = (), b: ArrayLike = ()
    ) -> None:
        p0 = scalar(null_weight, "null_weight")
        p, aa, bb = (
            finite(v, name).copy() for v, name in ((weights, "weights"), (a, "a"), (b, "b"))
        )
        if any(v.ndim != 1 for v in (p, aa, bb)) or not p.shape == aa.shape == bb.shape:
            raise ValueError("weights, a and b must be equal-length vectors")
        if p0 < 0 or p0 > 1 or np.any((p < 0) | (p > 1)):
            raise ValueError("Mixture weights must lie in [0,1]")
        total = float(p.sum()) + p0
        if not np.isclose(total, 1, rtol=0, atol=1e-12):
            raise ValueError("Mixture weights must sum to one")
        if np.any((aa <= 0) | (bb <= 0)):
            raise ValueError("Beta shape parameters must be positive")
        with np.errstate(over="ignore"):
            if not np.all(np.isfinite(aa + bb)):
                raise ValueError("Combined beta shape parameters exceed floating-point range")
        p /= total
        for name, array in (("weights", p), ("a", aa), ("b", bb)):
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "null_weight", p0 / total)

    def _log_terms(self, x: FloatArray, legacy: bool) -> FloatArray:
        result = np.full(x.shape + (self.weights.size + 1,), -np.inf)
        if self.null_weight > 0:
            result[..., 0] = np.log(self.null_weight)
        active = np.flatnonzero(self.weights > 0)
        a, b = self.a[active], self.b[active]
        if legacy:
            lx, ly = _logs(x, True)
            logs = (a - 1) * lx[..., None] + (b - 1) * ly[..., None]
        else:
            logs = xlogy(a - 1, x[..., None]) + xlog1py(b - 1, -x[..., None])
        result[..., active + 1] = logs - betaln(a, b) + np.log(self.weights[active])
        if np.any(np.isnan(result)):
            raise ValueError("Beta log-density could not be evaluated")
        return result

    def logpdf(self, pvalues: ArrayLike, *, legacy_endpoints: bool = False) -> FloatArray:
        """Log mixture density; optionally reproduce INITLN's endpoint convention."""
        return np.asarray(logsumexp(self._log_terms(_values(pvalues), legacy_endpoints), axis=-1))

    def cdf(self, pvalues: ArrayLike) -> FloatArray:
        """Mixture CDF (MIXPRB); endpoints always use their exact values."""
        x = _values(pvalues)
        active = self.weights > 0
        result = np.asarray(
            self.null_weight * x
            + betainc(self.a[active], self.b[active], x[..., None]) @ self.weights[active]
        )
        if not np.all(np.isfinite(result)):
            raise ValueError("Beta mixture CDF could not be evaluated")
        return result

    def null_posterior(self, pvalues: ArrayLike, *, legacy_endpoints: bool = False) -> FloatArray:
        """BPVAL's p0 / mixture density, conditional on this specified model.

        This is not a multiple-testing adjusted p-value or a CDF ratio.
        A zero-density observation has an undefined posterior and raises.
        """
        log_density = self.logpdf(pvalues, legacy_endpoints=legacy_endpoints)
        if np.any(np.isneginf(log_density)):
            raise ValueError("Null posterior is undefined at zero mixture density")
        if self.null_weight == 0:
            return np.zeros_like(log_density)
        return np.exp(np.log(self.null_weight) - log_density)

    def log_likelihood(self, pvalues: ArrayLike, *, legacy_endpoints: bool = False) -> FloatArray:
        """Sum log densities along the last axis; leading axes are batches."""
        return np.asarray(
            self.logpdf(_pvalues(pvalues), legacy_endpoints=legacy_endpoints).sum(axis=-1)
        )

    def cramer_von_mises(self, pvalues: ArrayLike) -> FloatArray:
        """CALCVM's W-squared/n scaling, with families along the last axis."""
        x = np.sort(_pvalues(pvalues), axis=-1)
        n = x.shape[-1]
        target = (np.arange(1, n + 1) - 0.5) / n
        return np.asarray(np.mean((self.cdf(x) - target) ** 2, axis=-1) + 1 / (12 * n * n))
