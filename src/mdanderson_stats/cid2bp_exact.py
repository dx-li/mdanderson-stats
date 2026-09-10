"""Brown's CID2BP exact binomial-tail inversion with nuisance maximization."""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import binom


def _exact(n1: int, x1: int, n2: int, x2: int, tail: float) -> tuple[float, float]:
    if max(n1, n2) > 100:
        raise ValueError("exact CID2BP requires at most 100 trials per sample")
    counts = np.arange(n1 + 1)[:, None]
    numerator = counts * n2 - (x1 * n2 - x2 * n1)
    cutoff = numerator // n1
    opposite_cutoff = cutoff - (numerator % n1 == 0)
    grid = np.linspace(0, 1, max(33, n1 + n2 + 1))

    def objective(delta: float, nuisance: np.ndarray) -> np.ndarray:
        width = 1 - abs(delta)
        p = max(delta, 0) + width * nuisance
        q = max(-delta, 0) + width * nuisance
        mass = binom.pmf(counts, n1, p)
        upper = (mass * binom.cdf(cutoff, n2, q)).sum(axis=0)
        lower = (mass * binom.sf(opposite_cutoff, n2, q)).sum(axis=0)
        return np.minimum(lower, upper)

    def crossing(delta: float) -> float:
        values = objective(delta, grid)
        best = float(values.max())
        if best == 0 or best == 1:
            return best - tail
        # Include endpoints and refine every grid-local maximum, rather than
        # assuming that the native objective has only one interior maximum.
        peaks = (
            np.flatnonzero(
                (values[1:-1] >= values[:-2])
                & (values[1:-1] >= values[2:])
                & ((values[1:-1] > values[:-2]) | (values[1:-1] > values[2:]))
            )
            + 1
        )
        for index in peaks:
            result = minimize_scalar(
                lambda u: -float(objective(delta, np.array([u]))[0]),
                bounds=(grid[index - 1], grid[index + 1]),
                method="bounded",
                options={"xatol": 1e-12},
            )
            if not result.success:
                raise ArithmeticError("CID2BP nuisance maximization did not converge")
            best = max(best, -float(result.fun))
        return best - tail

    difference = x1 / n1 - x2 / n2
    lower = -1.0 if difference == -1 else brentq(crossing, -1, difference, xtol=1e-10)
    upper = 1.0 if difference == 1 else brentq(crossing, difference, 1, xtol=1e-10)
    return float(lower), float(upper)
