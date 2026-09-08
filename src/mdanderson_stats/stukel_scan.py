"""Fixed-shape likelihood grids corresponding to scan.stukel.S/minim.S."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, count, finite, scalar
from .stukel_fit import StukelFitError, _optimize_stukel
from .stukel_objective import stukel_objective


def scan_stukel(
    x: ArrayLike,
    successes: ArrayLike,
    trials: ArrayLike,
    alpha1: ArrayLike,
    alpha2: ArrayLike,
    *,
    intercept: bool = True,
    beta_bound: float = 1000,
    tolerance: float = 1e-12,
    gradient_tolerance: float = 1e-8,
    max_iterations: int = 2000,
) -> FloatArray:
    """Profile negative log likelihood over a rectangular fixed-shape grid.

    Output[i, j] refits beta from zero at (alpha1[i], alpha2[j]), with the
    original scan's +/-1000 beta bounds. Binomial constants are omitted.
    This is constrained profiling, including separated or saturated data;
    it does not claim an interior MLE or provide covariance estimates.
    Failed cells raise with their grid coordinates; no partial grid is returned.
    """
    x = finite(x, "x")
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or min(x.shape) < 1 or not isinstance(intercept, bool):
        raise ValueError("x must be a nonempty matrix and intercept must be boolean")
    design = np.column_stack((np.ones(x.shape[0]), x)) if intercept else x.copy()
    y, n = count(successes, "successes"), count(trials, "trials")
    a1, a2 = finite(alpha1, "alpha1"), finite(alpha2, "alpha2")
    if a1.ndim != 1 or a2.ndim != 1 or not a1.size or not a2.size:
        raise ValueError("alpha1 and alpha2 must be nonempty one-dimensional grids")
    beta_bound = scalar(beta_bound, "beta_bound")
    if beta_bound <= 0:
        raise ValueError("beta_bound must be positive")
    coef = np.zeros(design.shape[1])
    bound = np.full(coef.size, beta_bound)
    # Validate common data before entering the grid, so malformed counts are
    # distinguished from failures of an individual constrained optimization.
    stukel_objective(design, y, n, coef, fixed_alpha=(a1[0], a2[0]))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise StukelFitError("Scan requires a full-rank design")
    result = np.empty((a1.size, a2.size))
    for i, first in enumerate(a1):
        for j, second in enumerate(a2):
            try:
                _, evaluated, _ = _optimize_stukel(
                    design,
                    y,
                    n,
                    coef,
                    bound,
                    0,
                    np.array([first, second]),
                    tolerance,
                    gradient_tolerance,
                    max_iterations,
                )
            except ValueError as error:
                raise StukelFitError(
                    f"Scan cell ({i}, {j}), alpha=({first}, {second}): {error}"
                ) from error
            result[i, j] = evaluated.negative_log_likelihood
    return result
