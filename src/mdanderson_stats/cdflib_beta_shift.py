"""Finite incomplete-beta shape shifts with positive, peak-scaled sums."""

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze, _pair
from ._validation import FloatArray, finite
from .cdflib_beta import _tails
from .cdflib_beta_factors import _deviation, _positive_parts, _scaled
from .cdflib_error_exponential import _integer


def _ratio(
    a: FloatArray, b: FloatArray, x: FloatArray, y: FloatArray, offset: FloatArray, k: FloatArray
) -> FloatArray:
    denominator = a + 1 + k
    correction = (-offset - 1 - k * y) / denominator
    direct = (a / denominator + b / denominator + k / denominator) * x
    return np.where(np.abs(correction) <= 0.5, 1 + correction, direct)


def _side(
    a: FloatArray,
    b: FloatArray,
    x: FloatArray,
    y: FloatArray,
    offset: FloatArray,
    peak: FloatArray,
    count: FloatArray,
    direction: int,
    tolerance: FloatArray,
) -> FloatArray:
    total, weight, done = np.zeros(a.shape), np.ones(a.shape), np.zeros(a.shape)
    active = count > 0
    steps = np.arange(1, 129, dtype=float)[None, :]
    for _ in range(4096):
        if not np.any(active):
            return total
        index = np.flatnonzero(active)
        distance = done[index, None] + steps
        valid = distance <= count[index, None]
        k = peak[index, None] + direction * distance - (direction == 1)
        k = np.maximum(k, 0)
        ratio = _ratio(
            a[index, None], b[index, None], x[index, None], y[index, None], offset[index, None], k
        )
        if direction == -1:
            ratio = 1 / ratio
        ratio = np.where(valid, ratio, 1)
        if np.any(ratio > 1 + 1e-12):
            raise ArithmeticError("beta-shift recurrence peak is inconsistent")
        terms = weight[index, None] * np.cumprod(np.minimum(ratio, 1), axis=1)
        total[index] += np.sum(np.where(valid, terms, 0), axis=1)
        weight[index] = terms[:, -1]
        done[index] = np.minimum(done[index] + 128, count[index])
        remaining = count[index] - done[index]
        current = peak[index] + direction * done[index]
        next_ratio = _ratio(
            a[index],
            b[index],
            x[index],
            y[index],
            offset[index],
            np.maximum(current - (direction == -1), 0),
        )
        if direction == -1:
            next_ratio = 1 / next_ratio
        else:
            next_ratio = np.where(b[index] < 1, x[index], next_ratio)
        # Ratios on either side of the peak bound the uncomputed tail.
        # For b<1 the forward ratios increase toward x, so use that limit.
        bound = remaining * weight[index]
        geometric = np.full(index.shape, np.inf)
        decreasing = next_ratio < 1
        geometric[decreasing] = (
            weight[index][decreasing] * next_ratio[decreasing] / (1 - next_ratio[decreasing])
        )
        bound = np.minimum(bound, geometric)
        active[index] = (remaining > 0) & (bound > tolerance[index] * (1 + total[index]) / 2)
    if np.any(active):
        raise ArithmeticError("beta-shift recurrence did not meet its remainder bound")
    return total


def _normalized(
    prefactor: FloatArray,
    exponent: FloatArray,
    divisor: FloatArray,
    anchor: FloatArray,
    log_weight: FloatArray,
) -> FloatArray:
    ratio = prefactor / anchor
    tiny = ratio < np.finfo(float).tiny
    result = np.empty(ratio.shape)
    result[~tiny] = _scaled(ratio[~tiny], (exponent + log_weight)[~tiny], divisor[~tiny])
    result[tiny] = np.exp(
        np.log(prefactor[tiny])
        - np.log(anchor[tiny])
        + exponent[tiny]
        + log_weight[tiny]
        - np.log(divisor[tiny])
    )
    return result


def _geometric_log_sum(logarithm: FloatArray, n: FloatArray) -> FloatArray:
    z = np.abs(logarithm)
    result = np.log(n.astype(float))
    nonzero = z > 0
    result[nonzero] = np.log(-np.expm1(-n[nonzero] * z[nonzero])) - np.log(-np.expm1(-z[nonzero]))
    return result + np.where(logarithm > 0, (n - 1) * logarithm, 0)


def _blocked(a: float, b: float, x: float, y: float, offset: float, n: int, block: int) -> float:
    # Bound log-ratio curvature separately on blocks. The bound was chosen
    # using k=0, so it remains conservative on every later block.
    logarithm = -np.inf
    for start in range(0, n, block * 8192):
        indices = np.arange(start, min(n, start + block * 8192), block, dtype=float)
        count = np.minimum(block, n - indices)
        mid = indices + (count - 2) / 2
        correction = (-offset - 1 - mid * y) / (a + 1 + mid)
        direct = (a / (a + 1 + mid) + b / (a + 1 + mid) + mid / (a + 1 + mid)) * x
        lr = np.where(np.abs(correction) <= 0.5, np.log1p(correction), np.log(direct))
        anchor = a + indices
        prefactor, exponent, divisor = _positive_parts(
            anchor,
            np.full(anchor.shape, b),
            np.full(anchor.shape, x),
            np.full(anchor.shape, y),
            np.zeros(anchor.shape),
        )
        terms = (
            np.log(prefactor)
            - np.log(anchor)
            + exponent
            - np.log(divisor)
            + _geometric_log_sum(lr, count)
        )
        maximum = np.max(terms)
        if np.isfinite(maximum):
            logarithm = np.logaddexp(logarithm, maximum + np.log(np.sum(np.exp(terms - maximum))))
        elif maximum != -np.inf:
            raise ArithmeticError("beta-shift block evaluation is nonfinite")
    return float(np.exp(logarithm))


def bup(
    a: ArrayLike,
    b: ArrayLike,
    x: ArrayLike | None,
    y: ArrayLike | None = None,
    n: ArrayLike = 1,
    eps: ArrayLike = 5e-15,
) -> FloatArray:
    """Return I_x(a,b)-I_x(a+n,b) for positive shapes and a positive int32 shift.

    Coordinates are complementary; the smaller supplied coordinate is retained.
    eps is a positive recurrence tolerance, capped at 5e-15 for loose requests
    and floored at four machine epsilons. Numerical nonconvergence is explicit.
    """
    xx, yy = _pair(x, y, "x,y")
    aa, bb, xx, yy, nn, tolerance = np.broadcast_arrays(
        finite(a, "a"), finite(b, "b"), xx, yy, _integer(n, "n"), finite(eps, "eps")
    )
    if np.any((aa <= 0) | (bb <= 0) | (nn < 1) | (tolerance <= 0)):
        raise ValueError("a, b, n and eps must be positive")
    shape = aa.shape
    aa, bb, xx, yy, nn, tolerance = [v.ravel() for v in (aa, bb, xx, yy, nn, tolerance)]
    tolerance = np.maximum(4 * np.finfo(float).eps, np.minimum(tolerance, 5e-15))
    result = np.zeros(aa.shape)
    active = (xx > 0) & (yy > 0)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        unit = active & (bb == 1)
        lx = np.where(xx[unit] <= yy[unit], np.log(xx[unit]), np.log1p(-yy[unit]))
        # Combine the power with the finite geometric difference before rounding.
        result[unit] = np.exp(aa[unit] * lx) * -np.expm1(nn[unit] * lx)
        active &= ~unit
        # Use existing compiled tails only when their difference is well
        # conditioned and the shapes are within the independently checked tail range.
        candidate = active & (nn > 1024) & (aa + nn <= 1e14) & (bb <= 1e10)
        if np.any(candidate):
            idx = np.flatnonzero(candidate)
            p0, q0 = _tails(xx[idx], yy[idx], aa[idx], bb[idx])
            p1, q1 = _tails(xx[idx], yy[idx], aa[idx] + nn[idx], bb[idx])
            use_p = p0 <= q1
            difference = np.where(use_p, p0 - p1, q1 - q0)
            scale = np.where(use_p, p0, q1)
            accepted = (difference > 0) & (difference >= 0.05 * scale)
            result[idx[accepted]] = difference[accepted]
            active[idx[accepted]] = False
        if np.any(active):
            idx = np.flatnonzero(active)
            a0, b0, x0, y0, n0, tol = (v[idx] for v in (aa, bb, xx, yy, nn, tolerance))
            offset = _deviation(a0, b0, x0, y0)
            peak = np.where(b0 > 1, np.clip(np.floor((-offset - x0) / y0), 0, n0 - 1), 0)
            anchor = a0 + peak
            prefactor, exponent, divisor = _positive_parts(anchor, b0, x0, y0, np.zeros(a0.shape))
            upper = _normalized(prefactor, exponent, divisor, anchor, np.log(n0))
            underflow = upper == 0
            # The derivative of log(t[k+1]/t[k]) has absolute value
            # |b-1|/((a+1+k)*(a+b+k)). Its maximum occurs at k=0.
            # A midpoint geometric series therefore bounds the relative
            # variation error by exp(n**2*curvature)-1.
            large, small = np.maximum(a0, b0), np.minimum(a0, b0)
            inverse_sum = (1 / large) / (1 + small / large)
            curvature = (np.abs(b0 - 1) / (a0 + 1)) * inverse_sum
            geometric = n0.astype(float) ** 2 * curvature <= tol / 4
            weight = n0.astype(float).copy()
            use = geometric & ~underflow & (n0 > 1)
            if np.any(use):
                mid = (n0[use] - 2) / 2
                correction = (-offset[use] - 1 - mid * y0[use]) / (a0[use] + 1 + mid)
                ratio = _ratio(a0[use], b0[use], x0[use], y0[use], offset[use], mid)
                logarithm = np.where(np.abs(correction) <= 0.5, np.log1p(correction), np.log(ratio))
                log_sum = _geometric_log_sum(logarithm, n0[use]) - peak[use] * logarithm
                weight[use] = np.exp(log_sum)
            varied = ~geometric & ~underflow & (n0 > 1)
            block_size = np.maximum(1, np.floor(np.sqrt(tol / (4 * curvature))))
            blocked = varied & (n0 > 524288) & (np.ceil(n0 / block_size) <= 1_000_000)
            for j in np.flatnonzero(blocked):
                result[idx[j]] = _blocked(
                    float(a0[j]),
                    float(b0[j]),
                    float(x0[j]),
                    float(y0[j]),
                    float(offset[j]),
                    int(n0[j]),
                    int(block_size[j]),
                )
            varied &= ~blocked
            if np.any(varied):
                av, bv, xv, yv, ov, pv, nv, tv = (
                    v[varied] for v in (a0, b0, x0, y0, offset, peak, n0, tol)
                )
                weight[varied] = (
                    1
                    + _side(av, bv, xv, yv, ov, pv, pv, -1, tv)
                    + _side(av, bv, xv, yv, ov, pv, nv - 1 - pv, 1, tv)
                )
            ordinary = ~blocked
            result[idx[ordinary]] = _normalized(
                prefactor[ordinary],
                exponent[ordinary],
                divisor[ordinary],
                anchor[ordinary],
                np.log(weight)[ordinary],
            )
    if np.any(~np.isfinite(result) | (result < 0) | (result > 1 + 1e-12)):
        raise ArithmeticError("beta-shift evaluation produced an invalid probability")
    return _freeze(result.reshape(shape))
