"""Optimize SINGLE prior-averaged allocations on a supplied finite dose set."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from ._validation import FloatArray, finite, scalar
from .single import _response_information, single_design_precision


@dataclass(frozen=True)
class SinglePriorAllocation:
    doses: FloatArray
    subjects: FloatArray
    value: float
    initial_value: float
    optimality_gap: float
    iterations: int


def single_optimize_prior_allocations(
    doses: ArrayLike,
    parameters: ArrayLike,
    prior_weights: ArrayLike,
    *,
    total_subjects: float = 100,
    criterion: str = "quantile",
    measure: str = "variance",
    aggregation: str = "arithmetic",
    model: str = "logistic",
    form: str = "linear",
    quantile: float = 0.05,
    initial_subjects: ArrayLike | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-10,
) -> SinglePriorAllocation:
    """Optimize allocations for an explicit weighted set of prior parameter nodes.

    Parameters have shape (nodes,2); positive prior_weights sum to one. Use nodes
    and weights from single_uniform_criterion or single_normal_criterion. Measure
    is variance/sd; aggregation is arithmetic/harmonic. The returned value is the
    chosen aggregate, not automatically a variance. This is local constrained
    optimization with an analytic gradient and a first-order stationarity check.
    Dose locations are fixed and subject counts remain continuous.
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
    if x.ndim != 1 or x.size < 2 or b.ndim != 2 or b.shape[1] != 2 or b.shape[0] == 0:
        raise ValueError("Require a dose vector and parameters of shape (nodes,2)")
    prior = finite(prior_weights, "prior_weights")
    if (
        prior.shape != (b.shape[0],)
        or np.any(prior <= 0)
        or not np.isclose(prior.sum(), 1, rtol=1e-12, atol=0)
    ):
        raise ValueError("prior_weights must be positive, match nodes and sum to one")
    prior = prior / prior.sum()
    if measure not in ("variance", "sd") or aggregation not in ("arithmetic", "harmonic"):
        raise ValueError("Require measure variance/sd and aggregation arithmetic/harmonic")
    power = 1.0 if measure == "variance" else 0.5
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
    single_design_precision(
        x, n, b, model=model, form=form, quantile=q if criterion == "quantile" else None
    )
    first, second = b[:, 0], b[:, 1]
    if form == "linear":
        u = first[:, None] + second[:, None] * x
        gradient = np.stack(np.broadcast_arrays(np.ones_like(u), x), axis=-1)
        slope_target = np.broadcast_to([0.0, 1.0], b.shape)
    else:
        u = first[:, None] * (x - second[:, None])
        gradient = np.stack(np.broadcast_arrays(x - second[:, None], -first[:, None]), axis=-1)
        slope_target = np.broadcast_to([1.0, 0.0], b.shape)
    _, weight = _response_information(u, model)
    if criterion == "slope":
        target = slope_target
    else:
        link = np.log(q) - np.log1p(-q) if model == "logistic" else -np.log(-np.log(q))
        target = (
            np.stack([-1 / second, -(link - first) / second**2], axis=-1)
            if form == "linear"
            else np.stack([-link / first**2, np.ones_like(first)], axis=-1)
        )

    def evaluate(fractions: FloatArray) -> tuple[float, FloatArray]:
        effective = fractions * weight
        low = np.min(np.where(effective > 0, x, np.inf), axis=-1)
        high = np.max(np.where(effective > 0, x, -np.inf), axis=-1)
        if np.any(high <= low):
            return np.inf, np.zeros_like(x)
        information = np.swapaxes(gradient, -1, -2) @ (effective[..., None] * gradient)
        try:
            factor = np.linalg.cholesky(information)
            y = np.linalg.solve(factor, target[..., None])
            solution = np.linalg.solve(np.swapaxes(factor, -1, -2), y)
        except np.linalg.LinAlgError:
            return np.inf, np.zeros_like(x)
        variance = np.sum(y[..., 0] ** 2, axis=-1)
        dv = -weight * (gradient @ solution)[..., 0] ** 2
        local = variance**power
        derivative = (power * variance ** (power - 1))[:, None] * dv
        if aggregation == "arithmetic":
            return float(prior @ local), prior @ derivative
        minimum = np.min(local)
        value = float(minimum / (prior @ (minimum / local)))
        return value, (prior * (value / local) ** 2) @ derivative

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
    gap = float(max(0.0, np.max(-derivative) / (power * value) - 1))
    if not np.isfinite(value) or not np.isfinite(gap) or gap > max(1e-5, 10 * np.sqrt(tol)):
        raise RuntimeError("Allocation optimizer failed the finite-dose stationarity check")
    return SinglePriorAllocation(
        x.copy(), total * f, value / total**power, baseline / total**power, gap, int(result.nit)
    )
