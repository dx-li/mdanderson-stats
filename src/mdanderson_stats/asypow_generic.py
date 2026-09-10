"""Generic expected-log-likelihood SMO with bounded nuisance optimization."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from ._validation import FloatArray, finite, scalar
from .asypow_constraints import _components
from .asypow_smo import SMOPower
from .boin import _owned


def asypow_smo_generic(
    parameters: ArrayLike,
    expected_log_likelihood: Callable[[FloatArray, FloatArray], float],
    *,
    lower: ArrayLike,
    upper: ArrayLike,
    constraints: ArrayLike,
    gradient: Callable[[FloatArray, FloatArray], ArrayLike] | None = None,
    initial: ArrayLike | None = None,
    subtract_df: bool = True,
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
) -> SMOPower:
    """Fit a generic SMO null using per-observation E_alt[log f(candidate)].

    Fixed/equality constraints use original one-based indices. gradient, when
    supplied, differentiates expected_log_likelihood in its SECOND argument.
    Bounds must be finite, with alternatives and fixed values strictly inside.
    This is a local fit; a nonconcave callback needs independent global checks.
    """
    values = finite(parameters, "parameters")
    if values.ndim not in (1, 2) or not 1 <= values.size <= 500:
        raise ValueError("parameters must be a vector/matrix with 1..500 entries")
    p = _owned(values.ravel())
    lo = np.broadcast_to(finite(lower, "lower").ravel(), p.shape)
    hi = np.broadcast_to(finite(upper, "upper").ravel(), p.shape)
    if np.any((lo >= hi) | (p <= lo) | (p >= hi)):
        raise ValueError("parameters must be strictly inside ordered bounds")
    tol = scalar(tolerance, "tolerance")
    iterations = scalar(max_iterations, "max_iterations")
    if not 0 < tol <= 1e-3 or not 1 <= iterations <= 100_000 or iterations != int(iterations):
        raise ValueError("tolerance must be in (0,1e-3]; max_iterations an integer in 1..100000")
    if not isinstance(subtract_df, (bool, np.bool_)):
        raise ValueError("subtract_df must be boolean")
    components = _components(constraints, p.size)
    base = np.zeros_like(p)
    free: list[list[int]] = []
    lows: list[float] = []
    highs: list[float] = []
    already_null = True
    for indices, value in components:
        if value is None:
            left, right = float(lo[indices].max()), float(hi[indices].min())
            if left >= right:
                raise ValueError("equality component has no interior within its bounds")
            free.append(indices)
            lows.append(left)
            highs.append(right)
            already_null &= bool(np.all(p[indices] == p[indices[0]]))
        else:
            if np.any((value <= lo[indices]) | (value >= hi[indices])):
                raise ValueError("fixed values must lie strictly inside parameter bounds")
            base[indices] = value
            already_null &= bool(np.all(p[indices] == value))
    df = p.size - len(free)
    if df == 0:
        raise ValueError("constraints must impose at least one independent restriction")
    baseline = scalar(expected_log_likelihood(p, p), "expected log likelihood")
    starts = p if initial is None else finite(initial, "initial").ravel()
    if starts.shape != p.shape or np.any((starts <= lo) | (starts >= hi)):
        raise ValueError("initial must match parameters and lie strictly inside bounds")
    if initial is not None:
        for indices, value in components:
            required = starts[indices[0]] if value is None else value
            if np.any(starts[indices] != required):
                raise ValueError("initial must satisfy all fixed/equality constraints")
    if already_null:
        return SMOPower(0, df, _owned(p), bool(subtract_df))
    lower_free = np.array(lows)
    with np.errstate(over="ignore"):
        width = np.array(highs) - lower_free
    if np.any(~np.isfinite(width)):
        raise ArithmeticError("bound widths overflow; rescale parameters")

    def unpack(z: FloatArray) -> FloatArray:
        result = base.copy()
        for j, indices in enumerate(free):
            result[indices] = lower_free[j] + width[j] * z[j]
        return _owned(result)

    def objective(z: FloatArray) -> float:
        return baseline - scalar(expected_log_likelihood(p, unpack(z)), "expected log likelihood")

    def derivative(z: FloatArray) -> FloatArray:
        assert gradient is not None
        grad = finite(gradient(p, unpack(z)), "gradient")
        if grad.shape != p.shape:
            raise ValueError("gradient must match the flattened parameter vector")
        result = -width * np.array([grad[indices].sum() for indices in free])
        return finite(result, "scaled gradient")

    if free:
        z0 = np.array(
            [
                np.mean((np.clip(starts[indices], lows[j], highs[j]) - lows[j]) / width[j])
                for j, indices in enumerate(free)
            ]
        )
        z0 = np.where((z0 > 0) & (z0 < 1), z0, 0.5)
        fit = minimize(
            objective,
            z0,
            jac=derivative if gradient is not None else "3-point",
            bounds=[(0, 1)] * len(free),
            method="L-BFGS-B",
            options={"gtol": tol, "ftol": 0, "maxiter": int(iterations), "maxls": 40},
        )
        if not fit.success or np.any(~np.isfinite(fit.jac)) or np.max(np.abs(fit.jac)) > tol:
            raise ArithmeticError(f"generic SMO fit failed its gradient criterion: {fit.message}")
        if np.any((fit.x <= 0) | (fit.x >= 1)):
            raise ArithmeticError("generic SMO optimum reached a search bound; review bounds/model")
        q = unpack(fit.x)
    else:
        q = _owned(base)
    fitted = scalar(expected_log_likelihood(p, q), "expected log likelihood")
    difference = baseline - fitted
    resolution = 16 * np.finfo(float).eps * max(abs(baseline), abs(fitted))
    if difference <= resolution:
        raise ArithmeticError(
            "likelihood difference is nonpositive or unresolved; use a centered callback"
        )
    w = 2 * difference
    if not np.isfinite(w):
        raise ArithmeticError("generic SMO divergence is not representable")
    return SMOPower(w, df, q, bool(subtract_df))
