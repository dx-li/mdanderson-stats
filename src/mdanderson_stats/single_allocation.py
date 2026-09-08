"""Optimize SINGLE point-prior allocations on a supplied finite dose set."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from ._validation import FloatArray, finite, scalar
from .single import _response_information, single_design_precision


@dataclass(frozen=True)
class SingleAllocation:
    doses: FloatArray
    subjects: FloatArray
    variance: float
    initial_variance: float
    optimality_gap: float
    iterations: int

    @property
    def sd(self) -> float:
        return float(np.sqrt(self.variance))


def single_optimize_allocations(
    doses: ArrayLike,
    parameters: ArrayLike,
    *,
    total_subjects: float = 100,
    criterion: str = "quantile",
    model: str = "logistic",
    form: str = "linear",
    quantile: float = 0.05,
    initial_subjects: ArrayLike | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-10,
) -> SingleAllocation:
    """Minimize local slope/quantile variance over nonnegative allocations.

    Dose locations and model parameters are fixed. Allocations are continuous,
    not rounded integer counts. Initial allocations, if supplied, must sum to
    total_subjects. An analytic gradient is used. A result is returned only if
    SLSQP succeeds and the finite-dose sensitivity optimality check passes.
    Full-rank information is required, including at the returned design.
    """
    total, tol = scalar(total_subjects, "total_subjects"), scalar(tolerance, "tolerance")
    if total <= 0 or not 0 < tol < 1:
        raise ValueError("Require positive total_subjects and tolerance in (0,1)")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(
        max_iterations, (int, np.integer)
    ):
        raise ValueError("max_iterations must be a positive integer")
    if max_iterations < 1 or criterion not in ("slope", "quantile"):
        raise ValueError("Require positive max_iterations and criterion slope/quantile")
    x, b = finite(doses, "doses"), finite(parameters, "parameters")
    if x.ndim != 1 or x.size < 2 or b.shape != (2,):
        raise ValueError("Require a dose vector and one two-parameter point prior")
    q = scalar(quantile, "quantile")
    n = (
        np.full(x.size, total / x.size)
        if initial_subjects is None
        else finite(initial_subjects, "initial_subjects")
    )
    if n.shape != x.shape or np.any(n < 0) or not np.isclose(np.sum(n), total, rtol=1e-12, atol=0):
        raise ValueError(
            "initial_subjects must be nonnegative, match doses and sum to total_subjects"
        )
    initial = single_design_precision(
        x, n, b, model=model, form=form, quantile=q if criterion == "quantile" else None
    )
    if form == "linear":
        u = b[0] + b[1] * x
        gradient = np.stack([np.ones_like(x), x], axis=-1)
        slope_target = np.array([0.0, 1.0])
    else:
        u = b[0] * (x - b[1])
        gradient = np.stack([x - b[1], np.full_like(x, -b[0])], axis=-1)
        slope_target = np.array([1.0, 0.0])
    _, weight = _response_information(u, model)
    if criterion == "slope":
        target = slope_target
    else:
        link = np.log(q) - np.log1p(-q) if model == "logistic" else -np.log(-np.log(q))
        target = (
            np.array([-1 / b[1], -(link - b[0]) / b[1] ** 2])
            if form == "linear"
            else np.array([-link / b[0] ** 2, 1.0])
        )

    def evaluate(fractions: FloatArray) -> tuple[float, FloatArray]:
        effective = fractions * weight
        informative = x[effective > 0]
        if informative.size < 2 or np.ptp(informative) == 0:
            return np.inf, np.zeros_like(x)
        information = gradient.T @ (effective[:, None] * gradient)
        try:
            factor = np.linalg.cholesky(information)
            y = np.linalg.solve(factor, target)
            solution = np.linalg.solve(factor.T, y)
        except np.linalg.LinAlgError:
            return np.inf, np.zeros_like(x)
        value = float(y @ y)
        derivative = -weight * (gradient @ solution) ** 2
        return value, derivative

    fractions = n / total
    baseline, _ = evaluate(fractions)
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError("Initial design precision is invalid")

    def objective(f: FloatArray) -> tuple[float, FloatArray]:
        value, derivative = evaluate(f)
        return value / baseline, derivative / baseline

    result = minimize(
        objective,
        fractions,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * x.size,
        constraints={
            "type": "eq",
            "fun": lambda f: np.sum(f) - 1,
            "jac": lambda f: np.ones_like(f),
        },
        options={"maxiter": int(max_iterations), "ftol": tol},
    )
    if not result.success:
        raise RuntimeError(f"Allocation optimization failed: {result.message}")
    f = result.x
    if np.any(f < 0) or abs(np.sum(f) - 1) > 1e-8:
        raise RuntimeError("Allocation optimizer returned an infeasible design")
    f = f / np.sum(f)
    value, derivative = evaluate(f)
    if not np.isfinite(value) or value <= 0 or not np.all(np.isfinite(derivative)):
        raise RuntimeError("Allocation optimizer returned invalid precision or sensitivity")
    gap = float(max(0.0, np.max(-derivative) / value - 1))
    if not np.isfinite(value) or not np.isfinite(gap) or gap > max(1e-5, 10 * np.sqrt(tol)):
        raise RuntimeError("Allocation optimizer failed the finite-dose optimality check")
    initial_variance = initial.slope_variance if criterion == "slope" else initial.quantile_variance
    assert initial_variance is not None
    return SingleAllocation(
        x.copy(), total * f, value / total, float(initial_variance), gap, int(result.nit)
    )
