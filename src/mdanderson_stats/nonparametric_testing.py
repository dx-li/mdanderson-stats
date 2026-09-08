"""MULTI desktop NP1P, FUNCFT and SMOOTH; original terms in the MULTI notice."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, scalar
from .multiplicity import _alpha, _pvalues
from .nonparametric import (
    NonparametricFitError,
    _legacy_widths,
    _local_derivatives,
    _window_bounds,
)
from .schweder import schweder_fit


@dataclass(frozen=True)
class NonparametricTestingResult:
    """Scores/reject retain input order; density/bandwidths use fitted rank order.

    Only the first fitted_points indices of order have fitted densities.
    bandwidths is None for polynomial fits and the retain-all branch.
    Scores are historical diagnostics, not posterior null probabilities.
    """

    scores: FloatArray
    reject: NDArray[np.bool_]
    density: FloatArray
    bandwidths: FloatArray | None
    fitted_points: int
    null_estimate: float
    order: NDArray[np.intp]


def _desktop_bandwidths(x: FloatArray, y: FloatArray) -> FloatArray:
    widths = _legacy_widths(x)
    if np.any(widths <= 0):
        raise NonparametricFitError("Legacy bandwidth search gave a zero-width window")
    step = np.maximum(((x[-1] - x[0]) / 4 - widths) / 10, 0)
    active = np.flatnonzero(step > 0)
    if active.size == 0:
        return widths
    current = widths.copy()
    minimum = np.full(x.size, np.inf)
    for _ in range(10):
        starts, ends = _window_bounds(x, current)
        for i in active:
            start, end = starts[i], ends[i]
            weights = (1 - ((x[start:end] - x[i]) / current[i]) ** 2) ** 2
            weights /= weights.sum()
            weights[i - start] = 0
            denominator = weights.sum()
            if denominator <= 0:
                raise NonparametricFitError("Cross-validation window has no other observations")
            weights /= denominator
            error = (y[i] - weights @ y[start:end]) ** 2
            if error < minimum[i]:
                minimum[i] = error
                widths[i] = current[i]
        current += step
    return widths


def nonparametric_testing(
    pvalues: ArrayLike,
    null_estimate: float | None = None,
    *,
    alpha: float = 0.05,
) -> NonparametricTestingResult:
    """Run desktop NP1P using a supplied or Schweder-estimated null count.

    Fits n-int(null_estimate) smallest p-values. Fewer than two means retain all.
    Fits 2--5 points by a line, 6--10 by a quadratic, and more by local quadratic
    smoothing with the source's ten-candidate, pointwise cross-validation rule.
    Empirical ranks are divided by the full family size, not the fitted size.

    When omitted, null_estimate comes from schweder_fit at its default alpha .05,
    independently of the decision alpha, as in the desktop interface. Failed fits
    propagate rather than becoming zero estimates or undefined decisions.
    """
    values = _pvalues(pvalues)
    if values.ndim != 1:
        raise ValueError("Nonparametric testing accepts one p-value vector")
    alpha = _alpha(alpha)
    estimate = (
        schweder_fit(values).null_estimate
        if null_estimate is None
        else scalar(null_estimate, "null_estimate")
    )
    if estimate < 0:
        raise ValueError("null_estimate must be nonnegative")
    n = values.size
    fitted = max(0, n - int(estimate))
    order = np.argsort(values, kind="stable")
    scores = np.ones(n)
    reject = np.zeros(n, dtype=bool)
    if fitted < 2:
        return NonparametricTestingResult(scores, reject, np.empty(0), None, 0, estimate, order)
    x = values[order[:fitted]]
    y = np.arange(1, fitted + 1, dtype=np.float64) / n
    widths = None
    if fitted <= 10:
        degree = 1 if fitted <= 5 else 2
        span = x[-1] - x[0]
        if span <= 0:
            raise NonparametricFitError("Polynomial fit has no variation in p-values")
        t = (x - x[0]) / span
        design = np.column_stack([t**j for j in range(degree + 1)])
        coefficients, _, rank, _ = np.linalg.lstsq(design, y - y[0], rcond=None)
        if rank != degree + 1:
            raise NonparametricFitError("Polynomial fit is rank deficient")
        density = np.full(fitted, coefficients[1])
        if degree == 2:
            density += 2 * coefficients[2] * t
        density /= span
    else:
        if np.unique(x).size < 4:
            raise NonparametricFitError("Window smoothing requires four distinct fitted p-values")
        widths = _desktop_bandwidths(x, y)
        density = _local_derivatives(x, y, widths)
    if not np.all(np.isfinite(density)):
        raise NonparametricFitError("Regression produced nonfinite density estimates")
    reciprocal = np.ones(fitted)
    with np.errstate(over="ignore"):
        np.divide(1, density, out=reciprocal, where=density != 0)
    scores[:fitted] = np.maximum.accumulate(np.clip(reciprocal, 0, 1))
    reject[:fitted] = scores[:fitted] <= alpha
    inverse = np.argsort(order)
    return NonparametricTestingResult(
        scores[inverse], reject[inverse], density, widths, fitted, estimate, order
    )
