"""Displacement-preserving large-shape incomplete beta expansion."""

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .cdflib_beta_factors import _scaled
from .cdflib_error_exponential import erfc1
from .cdflib_gamma_ratios import bcorr
from .dcdflib_beta import cumbet


def _weighted_remainder(shape: FloatArray, lam: FloatArray, sign: int) -> FloatArray:
    u = sign * (lam / shape)
    local = np.abs(u) <= 0.5
    result = np.empty(shape.shape)
    z = u[local]
    series = np.full(z.shape, 1 / 64)
    for k in range(63, 1, -1):
        series = (-1.0 if k % 2 else 1.0) / k + z * series
    # Keep the shape scale before the square could underflow.
    result[local] = (lam[local] * (lam[local] / shape[local])) * series
    s, displacement = shape[~local], lam[~local]
    if sign < 0:
        result[~local] = -displacement - s * np.log((s - displacement) / s)
    else:
        result[~local] = displacement - s * np.log1p(displacement / s)
    return result


def _positive_integral(
    a: FloatArray, b: FloatArray, lam: FloatArray, f: FloatArray, eps: FloatArray
) -> FloatArray:
    """Positive beta-integral expansion, preserving the displacement scale."""
    scale = np.maximum(a, b)
    h = np.minimum(a, b) / scale
    inverse_total = (1 / scale) / (1 + h)
    delta = a - lam
    term, total = np.ones(a.shape), np.ones(a.shape)
    active = np.ones(a.shape, dtype=bool)
    for n in range(1, 4097):
        term[active] *= (delta[active] / (a[active] + n)) * (1 + (n - 1) * inverse_total[active])
        total[active] += term[active]
        # Ratios decrease for b>1, giving a geometric remainder bound.
        ratio = (delta[active] / (a[active] + n + 1)) * (1 + n * inverse_total[active])
        bound = np.where(ratio < 1, term[active] * ratio / (1 - ratio), np.inf)
        active[active] = bound > eps[active] * total[active]
        if not np.any(active):
            break
    else:
        raise ArithmeticError("positive beta integral did not meet its remainder bound")
    prefactor = (np.sqrt(np.minimum(a, b)) / a) / (np.sqrt(1 + h) * np.sqrt(2 * np.pi))
    return _scaled(prefactor, -f - bcorr(a, b) + np.log(total), np.ones(a.shape))


def _expansion(a: FloatArray, b: FloatArray, lam: FloatArray, eps: FloatArray) -> FloatArray:
    f = _weighted_remainder(a, lam, -1) + _weighted_remainder(b, lam, 1)
    result = np.full(a.shape, np.nan)
    # Exponential Markov optimization for two independent gamma variables
    # gives I_x(a,b)<=exp(-f) when lambda>=0.
    result[f > 800] = 0
    selected = f <= 800
    if not np.any(selected):
        return result
    aa, bb, ll, ff, tolerance = (v[selected] for v in (a, b, lam, f, eps))
    h = np.minimum(aa, bb) / np.maximum(aa, bb)
    r0, r1 = 1 / (1 + h), (bb - aa) / np.maximum(aa, bb)
    w0 = (1 / np.sqrt(np.minimum(aa, bb))) / np.sqrt(1 + h)
    e0, e1 = 2 / np.sqrt(np.pi), 1 / np.sqrt(8.0)
    z0, z2 = np.sqrt(ff), 2 * ff
    z = np.sqrt(2.0) * z0
    # One-based coefficient indices follow the source recurrence.
    a0, b0, c, d = (np.zeros((22, aa.size)) for _ in range(4))
    a0[1] = 2 * r1 / 3
    c[1] = -0.5 * a0[1]
    d[1] = -c[1]
    j0 = (0.5 / e0) * erfc1(1, z0)
    j1 = np.full(aa.shape, e1)
    total = j0 + d[1] * w0 * j1
    s, hn = np.ones(aa.shape), np.ones(aa.shape)
    h2 = h * h
    w = w0.copy()
    znm1, zn = z.copy(), z2.copy()
    values = np.full(aa.shape, np.nan)
    correction = bcorr(aa, bb)
    for n in range(2, 21, 2):
        hn = h2 * hn
        a0[n] = 2 * r0 * (1 + h * hn) / (n + 2)
        s = s + hn
        a0[n + 1] = 2 * r1 * s / (n + 3)
        for i in (n, n + 1):
            r = -0.5 * (i + 1)
            b0[1] = r * a0[1]
            for m in range(2, i + 1):
                bsum = np.zeros(aa.shape)
                for j in range(1, m):
                    bsum += (j * r - (m - j)) * a0[j] * b0[m - j]
                b0[m] = r * a0[m] + bsum / m
            c[i] = b0[i] / (i + 1)
            dsum = np.zeros(aa.shape)
            for j in range(1, i):
                dsum += d[i - j] * c[j]
            d[i] = -(dsum + c[i])
        j0 = e1 * znm1 + (n - 1) * j0
        j1 = e1 * zn + n * j1
        znm1 = z2 * znm1
        zn = z2 * zn
        w = w0 * w
        t0 = d[n] * w * j0
        w = w0 * w
        t1 = d[n + 1] * w * j1
        total = total + (t0 + t1)
        done = np.isnan(values) & (total > 0) & (np.abs(t0) + np.abs(t1) <= tolerance * total)
        if np.any(done):
            values[done] = _scaled(
                e0 * total[done], -ff[done] - correction[done], np.ones(np.count_nonzero(done))
            )
        if np.all(~np.isnan(values)):
            break
    remaining = np.isnan(values)
    if np.any(remaining):
        ax, bx, lx = aa[remaining], bb[remaining], ll[remaining]
        fallback = np.empty(ax.shape)
        scale = np.maximum(ax, bx)
        denominator = ax / scale + bx / scale
        x = ((ax - lx) / scale) / denominator
        y = (bx / scale + lx / scale) / denominator
        left = x <= y
        if np.any(left):
            fallback[left] = _positive_integral(
                ax[left], bx[left], lx[left], ff[remaining][left], tolerance[remaining][left]
            )
        if np.any(~left):
            fallback[~left] = cumbet(x[~left], ax[~left], bx[~left], cx=y[~left])[0]
        values[remaining] = fallback
    result[selected] = values
    return result


def basym(a: ArrayLike, b: ArrayLike, lambda_: ArrayLike, eps: ArrayLike = 5e-15) -> FloatArray:
    """Compute I_x(a,b), x=(a-lambda_)/(a+b), for a,b>=15 and 0<=lambda_<=a.

    The displacement is retained without rounding x. The bounded source
    expansion uses a complete beta evaluation when its convergence check fails.
    Inputs broadcast to owned, immutable float64 outputs.
    """
    aa, bb, ll, ee = np.broadcast_arrays(
        finite(a, "a"), finite(b, "b"), finite(lambda_, "lambda_"), finite(eps, "eps")
    )
    shape = aa.shape
    aa, bb, ll, ee = (v.ravel() for v in (aa, bb, ll, ee))
    if np.any((aa < 15) | (bb < 15) | (ll < 0) | (ll > aa) | (ee <= 0)):
        raise ValueError("basym requires a,b>=15, 0<=lambda_<=a and positive eps")
    result = np.zeros(aa.shape)
    midpoint = (aa == bb) & (ll == 0)
    result[midpoint] = 0.5
    active = np.flatnonzero((ll < aa) & ~midpoint)
    tolerance = np.maximum(4 * np.finfo(float).eps, np.minimum(ee, 5e-15))
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        # Limit the four coefficient matrices to about 1.4 MiB per batch.
        for start in range(0, active.size, 2048):
            index = active[start : start + 2048]
            result[index] = _expansion(aa[index], bb[index], ll[index], tolerance[index])
    if np.any(~np.isfinite(result) | (result < 0) | (result > 1)):
        raise ArithmeticError("beta asymptotic evaluation failed")
    return _freeze(result.reshape(shape))
