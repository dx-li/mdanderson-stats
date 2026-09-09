"""Paired incomplete-beta tails with the CDFLIB zero-shape contract."""

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze, _pair
from ._validation import FloatArray, finite
from .cdflib_beta_factors import _deviation
from .cdflib_beta_fraction import bfrac
from .cdflib_beta_series import _product_bound, apser, bpser
from .dcdflib_beta import cumbet


def _small_tails(
    a: FloatArray, b: FloatArray, x: FloatArray, y: FloatArray
) -> tuple[FloatArray, FloatArray]:
    swap = x > y
    a, b, x, y = (
        np.where(swap, b, a),
        np.where(swap, a, b),
        np.where(swap, y, x),
        np.where(swap, x, y),
    )
    p, q = np.empty(a.shape), np.empty(a.shape)
    eps = np.full(a.shape, 5e-15)
    tiny = (a <= eps) & _product_bound(a, eps, b, inclusive=True)
    tiny &= ~_product_bound(np.ones(a.shape), b, x)
    if np.any(tiny):
        q[tiny] = apser(a[tiny], b[tiny], x[tiny])
        p[tiny] = 1 - q[tiny]
    power = ~tiny & ((b <= 1) | ~_product_bound(np.full(a.shape, 0.7), b, x))
    if np.any(power):
        p[power] = bpser(a[power], b[power], x[power])
        q[power] = 1 - p[power]
    # Preserve a small upper tail through its own evaluation, not 1-P.
    remaining = ~(tiny | power)
    remaining[power] = p[power] > 0.5
    if np.any(remaining):
        p[remaining], q[remaining] = cumbet(
            x[remaining], a[remaining], b[remaining], cx=y[remaining]
        )
    return np.where(swap, q, p), np.where(swap, p, q)


def bratio(
    a: ArrayLike, b: ArrayLike, x: ArrayLike | None, y: ArrayLike | None = None
) -> tuple[FloatArray, FloatArray]:
    """Return (I_x(a,b), I_y(b,a)) for nonnegative finite shapes.

    One shape may be zero except at its singular coordinate endpoint. Either
    coordinate can be supplied. Explicit pairs must sum to one within three
    machine epsilons. Invalid input raises ValueError instead of source IERR.
    Both broadcast outputs are independently owned and immutable.
    """
    aa, bb = np.broadcast_arrays(finite(a, "a"), finite(b, "b"))
    if np.any((aa < 0) | (bb < 0)):
        raise ValueError("bratio requires nonnegative shapes (IERR=1)")
    if np.any((aa == 0) & (bb == 0)):
        raise ValueError("bratio requires at least one positive shape (IERR=2)")
    raw_x = None if x is None else finite(x, "x")
    raw_y = None if y is None else finite(y, "y")
    if raw_x is not None and np.any((raw_x < 0) | (raw_x > 1)):
        raise ValueError("bratio requires x in [0,1] (IERR=3)")
    if raw_y is not None and np.any((raw_y < 0) | (raw_y > 1)):
        raise ValueError("bratio requires y in [0,1] (IERR=4)")
    if raw_x is not None and raw_y is not None:
        if np.any(np.abs((raw_x + raw_y) - 1) > 3 * np.finfo(float).eps):
            raise ValueError("bratio requires x+y=1 within three machine epsilons (IERR=5)")
    xx, yy = _pair(raw_x, raw_y, "x/y")
    aa, bb, xx, yy = np.broadcast_arrays(aa, bb, xx, yy)
    shape = aa.shape
    aa, bb, xx, yy = (v.ravel() for v in (aa, bb, xx, yy))
    if np.any((aa == 0) & (xx == 0)):
        raise ValueError("bratio is undefined at a=x=0 (IERR=6)")
    if np.any((bb == 0) & (yy == 0)):
        raise ValueError("bratio is undefined at b=y=0 (IERR=7)")
    p = np.where((yy == 0) | (aa == 0), 1.0, 0.0)
    q = 1 - p
    active = (aa > 0) & (bb > 0) & (xx > 0) & (yy > 0)
    small = active & ((aa <= 1) | (bb <= 1))
    if np.any(small):
        p[small], q[small] = _small_tails(aa[small], bb[small], xx[small], yy[small])
    large = active & ~small
    if np.any(large):
        a0, b0, x0, y0 = (v[large] for v in (aa, bb, xx, yy))
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            swap = _deviation(a0, b0, x0, y0) < 0
        a0, b0, x0, y0 = (
            np.where(swap, b0, a0),
            np.where(swap, a0, b0),
            np.where(swap, y0, x0),
            np.where(swap, x0, y0),
        )
        lower = bfrac(a0, b0, x0, y0)
        p[large], q[large] = np.where(swap, 1 - lower, lower), np.where(swap, lower, 1 - lower)
    return _freeze(p.reshape(shape)), _freeze(q.reshape(shape))
