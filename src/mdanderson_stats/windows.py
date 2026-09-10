"""Fixed-width and nearest-neighbor estimators from MD Anderson WINDOWS."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite, scalar
from .windows_neighbors import _neighbor_bounds, _neighbor_weights


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
    width: float | None = None,
    neighbors: int | None = None,
    centers: ArrayLike | None = None,
    estimator: str = "mean",
    weighting: str = "boxcar",
    quantile: float = 0.5,
    derivative_degree: int = 1,
    polynomial_weights: str = "kernel",
    leave_one_out: bool = False,
) -> WindowSmoothing:
    """Choose a full window width or neighbor count; missing estimates have explicit status.

    Estimators: mean, linear, quadratic, maximum, minimum, quantile, std,
    derivative, span, count. Polynomial weights are kernel or native_inverse.
    Leave-one-out evaluates sorted observations and excludes just the target row.
    """
    x, y = _data(x, y)
    if (width is None) == (neighbors is None):
        raise ValueError("supply exactly one of width or neighbors")
    if width is not None:
        width = scalar(width, "width")
        if width <= 0:
            raise ValueError("width must be positive")
    if neighbors is not None and (
        isinstance(neighbors, (bool, np.bool_))
        or not isinstance(neighbors, (int, np.integer))
        or not 2 <= neighbors <= x.size
    ):
        raise ValueError("neighbors must be an integer in 2..len(x)")
    quantile = scalar(quantile, "quantile")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0,1]")
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
    if width is not None:
        with np.errstate(over="ignore"):
            left = np.searchsorted(x, points - width / 2, side="left")
            right = np.searchsorted(x, points + width / 2, side="right")
    else:
        assert neighbors is not None
        bounds = [_neighbor_bounds(x, float(center), neighbors) for center in points]
        left, right = np.array(bounds, dtype=np.intp).T
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
        neighbor_weight = None
        if neighbors is not None:
            neighbor_weight = _neighbor_weights(x[indices], float(center), neighbors, weighting)
        if leave_one_out:
            if neighbor_weight is not None:
                neighbor_weight = neighbor_weight[indices != i]
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
        with np.errstate(over="ignore"):
            spans[i] = xx[-1] - xx[0]
        if not np.isfinite(spans[i]):
            raise ArithmeticError("window span is not representable; rescale x")
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
            if neighbor_weight is None:
                assert width is not None
                z = delta / width
                weight = np.ones(yy.size) if weighting == "boxcar" else (1 - z * z) ** 2
            else:
                weight = neighbor_weight
            if weight.sum() == 0:
                status.append("zero_weight")
                continue
            weight = weight / weight.sum()
            positive = weight > (
                1e-10
                if polynomial_weights == "native_inverse"
                and estimator in ("linear", "quadratic", "derivative")
                else 0
            )
            yy, delta, weight = yy[positive], delta[positive], weight[positive]
            if not yy.size:
                status.append("zero_weight")
                continue
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
    scores, folds, best = _cross_validation(
        x, y, widths, estimator, weighting, polynomial_weights, False
    )
    return WindowCrossValidation(_readonly(widths), _readonly(scores), _readonly(folds), best)


@dataclass(frozen=True)
class WindowNeighborCrossValidation:
    neighbors: FloatArray
    sum_squared_errors: FloatArray
    valid_folds: FloatArray
    best_neighbors: int | None


def window_neighbor_cross_validation(
    x: ArrayLike,
    y: ArrayLike,
    neighbors: ArrayLike,
    *,
    estimator: str = "mean",
    weighting: str = "boxcar",
    polynomial_weights: str = "kernel",
) -> WindowNeighborCrossValidation:
    """Tune neighbor counts; membership and weights are determined before removing each target."""
    x, y = _data(x, y)
    candidates = finite(neighbors, "neighbors")
    if (
        candidates.ndim != 1
        or not 1 <= candidates.size <= 1000
        or np.any(candidates != np.floor(candidates))
        or np.any((candidates < 2) | (candidates > x.size))
    ):
        raise ValueError("neighbors must contain 1..1000 integer candidates in 2..len(x)")
    scores, folds, best = _cross_validation(
        x, y, candidates, estimator, weighting, polynomial_weights, True
    )
    return WindowNeighborCrossValidation(
        _readonly(candidates),
        _readonly(scores),
        _readonly(folds),
        None if best is None else int(best),
    )


def _cross_validation(
    x: FloatArray,
    y: FloatArray,
    candidates: FloatArray,
    estimator: str,
    weighting: str,
    polynomial_weights: str,
    nearest: bool,
) -> tuple[FloatArray, FloatArray, float | None]:
    if candidates.size * x.size * x.size > 50_000_000:
        raise ValueError("cross-validation workload exceeds supported limit")
    scores = np.full(candidates.size, np.nan)
    folds = np.zeros(candidates.size)
    for i, parameter in enumerate(candidates):
        result = window_smooth(
            x,
            y,
            width=None if nearest else float(parameter),
            neighbors=int(parameter) if nearest else None,
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
    best = float(candidates[valid[np.argmin(scores[valid])]]) if valid.size else None
    return scores, folds, best
