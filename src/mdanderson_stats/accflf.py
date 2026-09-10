"""ACCFLF log-F probabilities, likelihood derivatives and shape conversion."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betainc, expit

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar
from .cdflib_beta_factors import _positive_parts
from .cdflib_beta_support import betaln


@dataclass(frozen=True)
class AccflfShape:
    numerator_df: float
    denominator_df: float
    tau: float
    lower_clipped: bool
    upper_clipped: bool


def accflf_shape(p: float, q: float) -> AccflfShape:
    """Prentice p,q to source-bounded degrees of freedom [0.001, 1e10].

    p=2/(a+b), q=(1/a-1/b)/sqrt(1/a+1/b), where a=dfn/2, b=dfd/2.
    Limiting infinite degrees of freedom are represented by 1e10 as in ACCFLF.
    Uses rationalization instead of the native truncated small-ratio polynomial.
    """
    p, q = scalar(p, "p"), scalar(q, "q")
    if not 0 <= p <= 1e150 or abs(q) > 1e150:
        raise ValueError("p must be nonnegative; p and abs(q) must be <=1e150")
    u = q * q + 2 * p
    if u < 4e-10:
        dfn = dfd = float("inf")
    else:
        ratio = abs(q) / np.sqrt(u)
        small = 4 / (u * (1 + ratio))
        large = float("inf") if p == 0 else 2 * (1 + ratio) / p
        dfn, dfd = (small, large) if q >= 0 else (large, small)
    lower, upper = bool(min(dfn, dfd) < 0.001), bool(max(dfn, dfd) > 1e10)
    dfn, dfd = float(np.clip(dfn, 0.001, 1e10)), float(np.clip(dfd, 0.001, 1e10))
    return AccflfShape(dfn, dfd, float(np.sqrt(1 / (2 / dfn + 2 / dfd))), lower, upper)


@dataclass(frozen=True)
class AccflfLogF:
    log_density: FloatArray
    log_cdf: FloatArray
    log_survival: FloatArray
    density_score: FloatArray
    density_curvature: FloatArray
    cdf_score: FloatArray
    cdf_curvature: FloatArray
    survival_score: FloatArray
    survival_curvature: FloatArray


def _log_fraction(a: float, b: float, x: float) -> float:
    """Log incomplete-beta continued fraction, used only for underflowed tails."""
    tiny = 1e-290
    c = 1.0
    d = 1 - (a + b) * x / (a + 1)
    d = 1 / (d if abs(d) >= tiny else np.copysign(tiny, d))
    h = d
    for m in range(1, 10001):
        for coefficient in (
            m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),
        ):
            d = 1 + coefficient * d
            c = 1 + coefficient / c
            d = 1 / (d if abs(d) >= tiny else np.copysign(tiny, d))
            c = c if abs(c) >= tiny else np.copysign(tiny, c)
            change = c * d
            h *= change
        if abs(change - 1) < 2e-14:
            if h <= 0 or not np.isfinite(h):
                raise ArithmeticError("beta tail fraction is invalid")
            return float(np.log(h))
    raise ArithmeticError("beta tail fraction failed to converge")


def accflf_logf(w: ArrayLike, numerator_df: float, denominator_df: float) -> AccflfLogF:
    """Log density/CDF/survival of log(F), and first/second w derivatives.

    Unlike native LLDRLF(case=1), log_density includes log(tau), so it is a
    normalized density in w. Tail derivatives may lose relative accuracy when
    their true curvature is nearly zero; these are float64 calculations.
    """
    if np.iscomplexobj(w):
        raise ValueError("w must be real")
    values = finite(w, "w")
    shape = values.shape
    z = values.reshape(-1)
    n, d = scalar(numerator_df, "numerator_df"), scalar(denominator_df, "denominator_df")
    if not 0.001 <= n <= 1e10 or not 0.001 <= d <= 1e10 or z.size > 2_000_000:
        raise ValueError("degrees of freedom must be in [0.001,1e10]; at most 2000000 values")
    a, b = n / 2, d / 2
    eta = z + np.log(a) - np.log(b)
    lx, ly = -np.logaddexp(0, -eta), -np.logaddexp(0, eta)
    x, y = expit(eta), expit(-eta)
    lp = np.empty(z.shape)
    interior = (x > 0) & (y > 0)
    aa, bb = np.full(z.shape, a), np.full(z.shape, b)
    with np.errstate(divide="ignore"):
        prefactor, exponent, divisor = _positive_parts(
            aa[interior], bb[interior], x[interior], y[interior], np.zeros(interior.sum())
        )
    lp[interior] = np.log(prefactor) + exponent - np.log(divisor)
    lp[~interior] = a * lx[~interior] + b * ly[~interior] - float(betaln(a, b))
    # Work on whichever incomplete-beta side is smaller, retaining its log even
    # when the probability is below the smallest representable float.
    left = x <= (a + 1) / (a + b + 2)
    ta, tb, tx = np.where(left, a, b), np.where(left, b, a), np.where(left, x, y)
    probability = betainc(ta, tb, tx)
    with np.errstate(divide="ignore"):
        small = np.log(probability)
    underflow = probability < 1e-280
    tail_score = np.zeros(z.shape)
    for i in np.flatnonzero(underflow):
        fraction = _log_fraction(ta[i], tb[i], tx[i])
        small[i] = lp[i] - np.log(ta[i]) + fraction
        tail_score[i] = ta[i] * np.exp(-fraction)
    with np.errstate(divide="ignore", invalid="ignore"):
        other = np.where(small < -np.log(2), np.log1p(-np.exp(small)), np.log(-np.expm1(small)))
    lc, ls = np.where(left, small, other), np.where(left, other, small)
    if a == 1:
        ls = b * ly
        with np.errstate(divide="ignore"):
            lc = np.where(ls == 0, lc, np.log(-np.expm1(ls)))
    if b == 1:
        lc = a * lx
        with np.errstate(divide="ignore"):
            ls = np.where(lc == 0, ls, np.log(-np.expm1(lc)))
    # Stable centered density derivatives avoid subtracting a*y and b*x.
    swapped = b > a
    v = -z if swapped else z
    k, low = min(a, b) / max(a, b), min(a, b)
    e = np.exp(-np.abs(v))
    denom = np.where(v > 0, 1 + k * e, e + k)
    score = np.where(v > 0, low * np.expm1(-np.abs(v)) / denom, -low * np.expm1(-np.abs(v)) / denom)
    if swapped:
        score = -score
    curvature = -low * (1 + k) * e / denom**2
    cs, hazard = np.exp(lp - lc), np.exp(lp - ls)
    cs[underflow & left] = tail_score[underflow & left]
    hazard[underflow & ~left] = tail_score[underflow & ~left]
    cc, sc = cs * (score - cs), -hazard * (score + hazard)
    # Exact unit-shape expressions avoid cancellation of limiting constant hazards.
    if a == 1:
        hazard, sc = b * x, -b * x * y
    if b == 1:
        cs, cc = a * y, -a * x * y
    arrays = (lp, lc, ls, score, curvature, cs, cc, -hazard, sc)
    if any(np.any(~np.isfinite(v)) for v in arrays):
        raise ArithmeticError("log-F values exceed finite floating-point range")
    return AccflfLogF(*(_freeze(v.reshape(shape)) for v in arrays))
