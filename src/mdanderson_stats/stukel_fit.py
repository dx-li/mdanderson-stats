"""Bounded STUKEL regression and observed-information uncertainty estimates."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog, minimize
from scipy.special import expit, xlogy

from ._validation import FloatArray, count, finite, scalar
from .stukel import predict_stukel
from .stukel_objective import StukelObjective, _family_shapes, stukel_objective


class StukelFitError(ValueError):
    """The regression could not produce a valid converged finite fit."""


@dataclass(frozen=True)
class StukelFit:
    coefficients: FloatArray
    alpha: FloatArray
    objective: StukelObjective
    deviance: float
    dispersion: float
    null_deviance: float
    null_dispersion: float
    residual_df: int
    covariance: FloatArray | None
    inference_message: str
    active_bounds: NDArray[np.bool_]
    iterations: int
    family: int
    intercept: bool

    @property
    def standard_errors(self) -> FloatArray | None:
        return None if self.covariance is None else np.sqrt(np.diag(self.covariance))

    def predict(self, x: ArrayLike) -> FloatArray:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        extra = 0 if self.family == 0 else 2 if self.family == 5 else 1
        beta = self.coefficients[: self.coefficients.size - extra]
        return predict_stukel(x, beta, *self.alpha, intercept=self.intercept)


def _check_separation(x: FloatArray, y: FloatArray, n: FloatArray) -> None:
    pure = (y == 0) | (y == n)
    if not np.any(pure):
        return
    # Normalize rows without changing the separating half-spaces.
    norm = np.max(np.abs(x), axis=1)
    scaled = x / np.where(norm == 0, 1, norm)[:, None]
    signed = scaled[pure] * np.where(y[pure] == n[pure], 1, -1)[:, None]
    mixed = scaled[~pure]
    result = linprog(
        -signed.sum(axis=0),
        A_ub=-signed,
        b_ub=np.zeros(signed.shape[0]),
        A_eq=mixed if mixed.size else None,
        b_eq=np.zeros(mixed.shape[0]) if mixed.size else None,
        bounds=[(-1, 1)] * x.shape[1],
        method="highs",
    )
    if not result.success:
        raise StukelFitError(f"Separation check failed: {result.message}")
    if result.fun < -1e-7:
        raise StukelFitError("Complete or quasi-complete separation: no finite interior maximum")


def fit_stukel(
    x: ArrayLike,
    successes: ArrayLike,
    trials: ArrayLike,
    *,
    family: int = 0,
    intercept: bool = True,
    fixed_alpha: ArrayLike = (0, 0),
    start_alpha: ArrayLike = (0, 0),
    initial: ArrayLike | None = None,
    shape_bound: float = 10,
    scale: str = "pearson",
    tolerance: float = 1e-12,
    gradient_tolerance: float = 1e-8,
    max_iterations: int = 2000,
) -> StukelFit:
    """Fit any of STUKEL's six families with analytic-gradient L-BFGS-B.

    x is observations by covariates; an intercept is added by default. Free shape
    bounds default to +/-10, beta bounds to +/-1e20, matching minimize.S.
    Beta starts at zero unless initial is supplied. Solver paths differ from
    David Gay's routines, and the opposite-shape derivative error is corrected.
    Nonconvergence, deficient design, separation and numerical failure raise.
    Covariance is omitted when boundary/nonregular curvature prevents Wald inference.
    """
    x = finite(x, "x")
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or min(x.shape) < 1 or not isinstance(intercept, bool):
        raise ValueError("x must be a nonempty matrix and intercept must be boolean")
    design = np.column_stack((np.ones(x.shape[0]), x)) if intercept else x.copy()
    y, n = count(successes, "successes"), count(trials, "trials")
    fixed, start = finite(fixed_alpha, "fixed_alpha"), finite(start_alpha, "start_alpha")
    if fixed.shape != (2,) or start.shape != (2,):
        raise ValueError("fixed_alpha and start_alpha must have two elements")
    if isinstance(family, bool) or not isinstance(family, int) or not 0 <= family <= 5:
        raise ValueError("family must be an integer from zero to five")
    extra = 0 if family == 0 else 2 if family == 5 else 1
    p = design.shape[1]
    starts = [] if not extra else start.tolist() if extra == 2 else [start[int(family == 2)]]
    coef = np.r_[np.zeros(p), starts] if initial is None else finite(initial, "initial")
    shape_bound = scalar(shape_bound, "shape_bound")
    tolerance, gradient_tolerance = (
        scalar(tolerance, "tolerance"),
        scalar(gradient_tolerance, "gradient_tolerance"),
    )
    if shape_bound <= 0 or not 0 < tolerance < 1 or not 0 < gradient_tolerance < 1:
        raise ValueError("shape_bound must be positive; tolerances must lie in (0,1)")
    if scale not in ("pearson", "fixed"):
        raise ValueError("scale must be 'pearson' or 'fixed'")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    bound = np.r_[np.full(p, 1e20), np.full(extra, shape_bound)]
    if coef.shape != bound.shape or np.any(np.abs(coef) > bound):
        raise ValueError("Initial coefficients must have the required size and lie within bounds")
    initial_result = stukel_objective(design, y, n, coef, family=family, fixed_alpha=fixed)
    df = design.shape[0] - coef.size
    if df <= 0 or np.linalg.matrix_rank(design) < p:
        raise StukelFitError(
            "Fit requires positive residual degrees of freedom and full-rank design"
        )
    _check_separation(design, y, n)
    # Column scaling improves optimization units while preserving exact beta bounds.
    units = np.r_[np.maximum(np.max(np.abs(design), axis=0), 1), np.ones(extra)]
    total = float(n.sum())

    def objective(theta: FloatArray) -> tuple[float, FloatArray]:
        result = stukel_objective(design, y, n, theta / units, family=family, fixed_alpha=fixed)
        return result.negative_log_likelihood / total, result.gradient / units / total

    result = minimize(
        objective,
        coef * units,
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(-bound * units, bound * units, strict=True)),
        options={
            "ftol": tolerance,
            "gtol": gradient_tolerance,
            "maxiter": max_iterations,
            "maxls": 40,
        },
    )
    if not result.success:
        raise StukelFitError(f"STUKEL optimizer failed: {result.message}")
    coef = result.x / units
    evaluated = stukel_objective(design, y, n, coef, family=family, fixed_alpha=fixed)
    if evaluated.negative_log_likelihood > initial_result.negative_log_likelihood + 1e-8:
        raise StukelFitError("STUKEL fit worsened the initial likelihood")
    shapes, _ = _family_shapes(coef, p, family, fixed)
    h = evaluated.log_odds
    entropy = float(np.sum(xlogy(y, y / n) + xlogy(n - y, (n - y) / n)))
    deviance = max(0.0, 2 * (evaluated.negative_log_likelihood + entropy))
    prob = evaluated.probabilities
    complement = expit(-h)
    residual = y * complement - (n - y) * prob
    variance = n * prob * complement
    terms = np.zeros_like(n)
    np.divide(residual**2, variance, out=terms, where=variance > 0)
    if np.any((variance == 0) & (residual != 0)):
        raise StukelFitError("Pearson dispersion exceeds numerical range")
    dispersion = float(terms.sum() / df) if scale == "pearson" else 1.0
    if not np.isfinite(dispersion):
        raise StukelFitError("Dispersion exceeds numerical range")
    null_p = float(y.sum() / n.sum())
    null_nll = -float(np.sum(xlogy(y, null_p) + xlogy(n - y, 1 - null_p)))
    null_deviance = max(0.0, 2 * (null_nll + entropy))
    if scale == "fixed":
        null_dispersion = 1.0
    elif null_p in (0, 1):
        null_dispersion = 0.0
    else:
        null_dispersion = float(
            np.sum((y - n * null_p) ** 2 / (n * null_p * (1 - null_p))) / (len(y) - 1)
        )
    active = np.asarray(np.isclose(np.abs(coef), bound, rtol=1e-7, atol=0))
    covariance = None
    message = "Observed-information covariance"
    if np.any(active):
        message = "Covariance unavailable: parameter bound active"
    elif extra and np.any(coef[p:] == 0):
        message = "Covariance unavailable: free shape at nonregular zero boundary"
    elif shapes.sum() != 0 and np.any(((design @ coef[:p]) == 0) & (y != n / 2)):
        message = "Covariance unavailable: predictor at nonregular zero boundary"
    else:
        try:
            np.linalg.cholesky(evaluated.hessian)
            covariance = dispersion * np.linalg.solve(evaluated.hessian, np.eye(coef.size))
            if not np.all(np.isfinite(covariance)):
                covariance = None
                message = "Covariance unavailable: numerical range exceeded"
        except np.linalg.LinAlgError:
            message = "Covariance unavailable: observed Hessian is not positive definite"
    return StukelFit(
        coef,
        shapes,
        evaluated,
        deviance,
        dispersion,
        null_deviance,
        null_dispersion,
        df,
        covariance,
        message,
        active,
        int(result.nit),
        family,
        intercept,
    )
