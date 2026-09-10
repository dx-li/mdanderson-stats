"""Fixed-width estimators from MD Anderson WINDOWS."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite, scalar


@dataclass(frozen=True)
class WindowSmoothing:
    centers: FloatArray
    estimates: FloatArray
    counts: FloatArray
    spans: FloatArray
    status: tuple[str, ...]


def _readonly(x: ArrayLike) -> FloatArray:
    result = np.array(x, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _data(x: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
    x, y = finite(x, "x"), finite(y, "y")
    if x.ndim != 1 or y.shape != x.shape or not 2 <= x.size <= 1_000_000:
        raise ValueError("x and y must be matching vectors with 2..1000000 observations")
    order = np.argsort(x, kind="stable")
    return x[order], y[order]


def window_smooth(
    x: ArrayLike,
    y: ArrayLike,
    *,
    width: float,
    centers: ArrayLike | None = None,
    estimator: str = "mean",
    weighting: str = "boxcar",
    quantile: float = 0.5,
    derivative_degree: int = 1,
    polynomial_weights: str = "kernel",
    leave_one_out: bool = False,
) -> WindowSmoothing:
    """Full window width; endpoints included. Missing estimates have explicit status.

    Estimators: mean, linear, quadratic, maximum, minimum, quantile, std,
    derivative, span, count. Polynomial weights are kernel or native_inverse.
    Leave-one-out evaluates sorted observations and excludes just the target row.
    """
    x, y = _data(x, y)
    width = scalar(width, "width")
    quantile = scalar(quantile, "quantile")
    if width <= 0 or not 0 <= quantile <= 1:
        raise ValueError("width must be positive and quantile in [0,1]")
    if estimator not in (
        "mean",
        "linear",
        "quadratic",
        "maximum",
        "minimum",
        "quantile",
        "std",
        "derivative",
        "span",
        "count",
    ):
        raise ValueError("unknown WINDOWS estimator")
    if weighting not in ("boxcar", "biquadratic") or polynomial_weights not in (
        "kernel",
        "native_inverse",
    ):
        raise ValueError("unknown weighting convention")
    if isinstance(derivative_degree, (bool, np.bool_)) or derivative_degree not in (1, 2):
        raise ValueError("derivative_degree must be 1 or 2")
    if not isinstance(leave_one_out, (bool, np.bool_)):
        raise ValueError("leave_one_out must be boolean")
    if leave_one_out and (centers is not None or estimator not in ("mean", "linear", "quadratic")):
        raise ValueError("leave-one-out requires observation centers and a regression estimator")
    points = (
        x.copy()
        if leave_one_out
        else (np.linspace(x[0], x[-1], 20) if centers is None else finite(centers, "centers"))
    )
    if points.ndim == 0:
        points = points[None]
    if points.ndim != 1 or not 1 <= points.size <= 100_000:
        raise ValueError("centers must have 1..100000 entries")
    with np.errstate(over="ignore"):
        left = np.searchsorted(x, points - width / 2, side="left")
        right = np.searchsorted(x, points + width / 2, side="right")
    if np.sum(right - left) > 20_000_000:
        raise ValueError("total window membership exceeds twenty million observations")
    values = np.full(points.size, np.nan)
    counts = np.zeros(points.size)
    spans = np.full(points.size, np.nan)
    status: list[str] = []
    degree = (
        derivative_degree if estimator == "derivative" else (2 if estimator == "quadratic" else 1)
    )
    for i, center in enumerate(points):
        indices = np.arange(left[i], right[i])
        if leave_one_out:
            indices = indices[indices != i]
        xx, yy = x[indices], y[indices]
        counts[i] = yy.size
        if not yy.size:
            if estimator == "count":
                values[i] = 0
                status.append("ok")
            else:
                status.append("empty")
            continue
        spans[i] = xx[-1] - xx[0]
        if estimator == "count":
            values[i] = yy.size
        elif estimator == "span":
            values[i] = spans[i]
        elif estimator == "maximum":
            values[i] = yy.max()
        elif estimator == "minimum":
            values[i] = yy.min()
        else:
            # Native fixed-width kernel uses full width in its denominator.
            delta = xx - center
            z = delta / width
            weight = np.ones(yy.size) if weighting == "boxcar" else (1 - z * z) ** 2
            weight /= weight.sum()
            yscale = float(np.max(np.abs(yy)))
            scaled = yy / yscale if yscale else yy
            if estimator == "mean":
                values[i] = float(weight @ scaled) * yscale
            elif estimator == "std":
                residual = scaled - weight @ scaled
                values[i] = np.sqrt(weight @ (residual * residual)) * yscale
            elif estimator == "quantile":
                if yy.size < 2:
                    status.append("insufficient_points")
                    continue
                order = np.argsort(yy, kind="stable")
                sy, sw = scaled[order], weight[order]
                positions = np.cumsum(sw) - sw / 2
                j = int(np.clip(np.searchsorted(positions, quantile), 1, yy.size - 1))
                values[i] = (
                    sy[j - 1]
                    + (sy[j] - sy[j - 1])
                    * (quantile - positions[j - 1])
                    / (positions[j] - positions[j - 1])
                ) * yscale
            else:
                if yy.size <= degree:
                    status.append("insufficient_points")
                    continue
                # Fit in centered local coordinates, scaled to the observed span.
                scale = float(np.max(np.abs(delta)))
                if scale == 0:
                    status.append("rank_deficient")
                    continue
                design = np.vander(delta / scale, degree + 1, increasing=True)
                root_weight = np.sqrt(weight if polynomial_weights == "kernel" else 1 / weight)
                u, singular, vt = np.linalg.svd(design * root_weight[:, None], full_matrices=False)
                if singular[-1] <= np.finfo(float).eps * max(design.shape) * singular[0]:
                    status.append("rank_deficient")
                    continue
                b = vt.T @ ((u.T @ (scaled * root_weight)) / singular)
                values[i] = b[1] * yscale / scale if estimator == "derivative" else b[0] * yscale
        if not np.isfinite(values[i]):
            raise ArithmeticError(f"window estimate at center index {i} is not representable")
        status.append("ok")
    return WindowSmoothing(
        _readonly(points), _readonly(values), _readonly(counts), _readonly(spans), tuple(status)
    )


@dataclass(frozen=True)
class WindowCrossValidation:
    widths: FloatArray
    sum_squared_errors: FloatArray
    valid_folds: FloatArray
    best_width: float | None


def window_cross_validation(
    x: ArrayLike,
    y: ArrayLike,
    widths: ArrayLike,
    *,
    estimator: str = "mean",
    weighting: str = "boxcar",
    polynomial_weights: str = "kernel",
) -> WindowCrossValidation:
    """Leave-one-observation-out SSE; invalid candidates are NaN, never partial SSE."""
    x, y = _data(x, y)
    widths = finite(widths, "widths")
    if widths.ndim != 1 or not 1 <= widths.size <= 1000 or np.any(widths <= 0):
        raise ValueError("widths must be a vector of 1..1000 positive candidates")
    if widths.size * x.size * x.size > 50_000_000:
        raise ValueError("cross-validation workload exceeds supported limit")
    scores = np.full(widths.size, np.nan)
    folds = np.zeros(widths.size)
    for i, width in enumerate(widths):
        result = window_smooth(
            x,
            y,
            width=float(width),
            estimator=estimator,
            weighting=weighting,
            polynomial_weights=polynomial_weights,
            leave_one_out=True,
        )
        folds[i] = sum(s == "ok" for s in result.status)
        if folds[i] == x.size:
            difference = y - result.estimates
            scale = float(np.max(np.abs(difference)))
            scores[i] = float(np.sum((difference / scale) ** 2) * scale * scale) if scale else 0
            if not np.isfinite(scores[i]):
                raise ArithmeticError("cross-validation loss is not representable; rescale y")
    valid = np.flatnonzero(np.isfinite(scores))
    best = float(widths[valid[np.argmin(scores[valid])]]) if valid.size else None
    return WindowCrossValidation(_readonly(widths), _readonly(scores), _readonly(folds), best)
