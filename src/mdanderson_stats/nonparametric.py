"""The MULTI S library's local-quadratic p-value diagnostic.

Modified NPFIT/WDTHMX/WNFWD/WTBQWD/LQBETA algorithms; see the MULTI notice.
The desktop's iterative NP1P is a separate method, not implemented here.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betainc

from ._validation import FloatArray
from .multiplicity import _pvalues


class NonparametricFitError(ValueError):
    """The archived diagnostic cannot be evaluated on these data."""


@dataclass(frozen=True)
class NonparametricFit:
    """Density and diagnostic scores retain input order; width is full width.

    Scores preserve the source's fractional beta-tail calculation and increasing
    rank constraint. They are not posterior null probabilities or FDR estimates.
    """

    density: FloatArray
    scores: FloatArray
    bandwidth: float
    order: NDArray[np.intp]


def _legacy_width(x: FloatArray) -> float:
    """Reproduce S WDTHMX, including its historical expansion counter."""
    width = 0.0
    n = x.size
    for i, center in enumerate(x):
        lo = hi = i
        changes = 0
        while changes < 3:
            if lo == 0 and hi == n - 1:
                raise NonparametricFitError("Legacy bandwidth search exhausted the data")
            if lo != 0 and hi != n - 1:
                left, right = center - x[lo - 1], x[hi + 1] - center
                if left < right:
                    lo -= 1
                    difference = left - x[lo + 1]
                else:
                    hi += 1
                    difference = right - x[hi - 1]
            elif lo == 0:
                hi += 1
                difference = x[hi] - x[hi - 1]
            else:
                lo -= 1
                difference = x[lo] - x[lo + 1]
            changes += int(difference != 0)
        width = max(width, 2 * max(center - x[lo], x[hi] - center))
    return float(width)


def nonparametric_pvalues(pvalues: ArrayLike) -> NonparametricFit:
    """Fit the S multi.np diagnostic with the archived bandwidth/weight rules.

    Requires one vector with at least four distinct p-values. The empirical
    ordinates are sorted rank/n, including tied ranks individually. A centered,
    scaled least-squares solve replaces the original inverse of normal equations.
    A rank-deficient window or invalid beta-tail parameter raises an error rather
    than returning the source's -1 sentinel or an undefined value.

    The source resets its cross-validation minimum for every candidate and hence
    always retains its initial width. That executable behavior is preserved.
    See docs/multiple-testing.md for the other historical numerical conventions.
    """
    values = _pvalues(pvalues)
    if values.ndim != 1:
        raise ValueError("Nonparametric fitting accepts one p-value vector")
    order = np.argsort(values, kind="stable")
    x = values[order]
    n = x.size
    if np.unique(x).size < 4:
        raise NonparametricFitError("At least four distinct p-values are required")
    width = _legacy_width(x)
    if width <= 0:
        raise NonparametricFitError("Bandwidth must be positive")
    y = np.arange(1, n + 1, dtype=np.float64) / n
    low = np.maximum(x - width / 2, x[0])
    high = np.minimum(x + width / 2, x[-1])
    starts = np.searchsorted(x, low, side="left")
    # IBSDB's upper bound uses the first exact match, except at the minimum,
    # where its two-element terminating bracket can include the second tie.
    upper = np.searchsorted(x, high, side="left")
    ends = np.where(x[upper] == high, np.maximum(upper, 1), upper - 1) + 1
    density = np.empty(n)
    for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
        offsets = (x[start:end] - x[i]) / width
        weights = (1 - offsets**2) ** 2
        weights /= weights.sum()
        keep = weights > 1e-10
        t = offsets[keep]
        scale = 1 / np.sqrt(weights[keep])
        design = np.column_stack((np.ones_like(t), t, t**2)) * scale[:, None]
        response = (y[start:end][keep] - y[i]) * scale
        coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
        if rank != 3:
            raise NonparametricFitError(f"Quadratic window at sorted rank {i + 1} is singular")
        density[i] = coefficients[1] / width
    if np.any(~np.isfinite(density) | (density <= -1)):
        raise NonparametricFitError("Density gives invalid fractional beta-tail parameters")
    scores = np.zeros(n)
    mask = density < n
    scores[mask] = betainc(density[mask] + 1, n - density[mask], 1 / n)
    if not np.all(np.isfinite(scores)):
        raise NonparametricFitError("Fractional beta-tail calculation failed")
    scores = np.maximum.accumulate(scores)
    inverse = np.argsort(order)
    return NonparametricFit(density[inverse], scores[inverse], width, order)
