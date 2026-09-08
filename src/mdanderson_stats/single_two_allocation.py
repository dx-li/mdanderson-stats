"""Two-sample allocation optimization with SINGLE's shared-parameter models."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._single_constraints import allocation_constraints
from ._validation import FloatArray, finite, scalar
from .single import _response_information
from .single_prior_allocation import _optimize_prior_information
from .single_two_sample import single_two_sample_precision


@dataclass(frozen=True)
class SingleTwoSampleAllocation:
    doses: tuple[FloatArray, FloatArray]
    subjects: tuple[FloatArray, FloatArray]
    value: float
    initial_value: float
    optimality_gap: float
    iterations: int


def single_optimize_two_sample_allocations(
    doses: tuple[ArrayLike, ArrayLike],
    parameters: ArrayLike,
    *,
    prior_weights: ArrayLike | None = None,
    total_subjects: float = 100,
    group_totals: ArrayLike | None = None,
    comparison: str = "location",
    model: str = "logistic",
    form: str = "linear",
    measure: str = "sd",
    aggregation: str = "arithmetic",
    initial_subjects: tuple[ArrayLike, ArrayLike] | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-10,
) -> SingleTwoSampleAllocation:
    """Allocate subjects across fixed doses, optionally fixing each group total.

    Supply a three-entry point prior, or (nodes,3) parameters with positive
    prior_weights summing to one. Parameter order follows single_two_sample_precision.
    group_totals optionally fixes two positive group sizes summing to total_subjects.
    Counts are continuous. The result passes local first-order stationarity checks,
    without a general claim of global optimality for aggregated prior objectives.
    """
    if len(doses) != 2:
        raise ValueError("Require two dose vectors")
    x1, x2 = (finite(v, "doses") for v in doses)
    if x1.ndim != 1 or x2.ndim != 1 or x1.size == 0 or x2.size == 0:
        raise ValueError("Require two nonempty dose vectors")
    x = np.concatenate([x1, x2])
    b = finite(parameters, "parameters")
    if b.shape == (3,):
        b = b[None, :]
    if b.ndim != 2 or b.shape[1] != 3 or b.shape[0] == 0:
        raise ValueError("parameters must contain three entries or have shape (nodes,3)")
    if prior_weights is None:
        if b.shape[0] != 1:
            raise ValueError("prior_weights are required for multiple nodes")
        prior = np.ones(1)
    else:
        prior = finite(prior_weights, "prior_weights")
    if (
        prior.shape != (b.shape[0],)
        or np.any(prior <= 0)
        or not np.isclose(prior.sum(), 1, rtol=1e-12, atol=0)
    ):
        raise ValueError("prior_weights must be positive, match nodes and sum to one")
    prior = prior / prior.sum()
    total, tol = scalar(total_subjects, "total_subjects"), scalar(tolerance, "tolerance")
    if total <= 0 or not 0 < tol < 1:
        raise ValueError("Require positive total_subjects and tolerance in (0,1)")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    if measure not in ("sd", "variance") or aggregation not in ("arithmetic", "harmonic"):
        raise ValueError("Require measure sd/variance and aggregation arithmetic/harmonic")
    matrix, shares = allocation_constraints([x1.size, x2.size], group_totals, total)
    if initial_subjects is None:
        initial = total * (matrix.T @ (shares / matrix.sum(axis=1)))
        n1, n2 = initial[: x1.size], initial[x1.size :]
    else:
        if len(initial_subjects) != 2:
            raise ValueError("Require two initial subject vectors")
        n1, n2 = (finite(v, "initial_subjects") for v in initial_subjects)
    single_two_sample_precision(
        (x1, x2), (n1, n2), b, comparison=comparison, model=model, form=form
    )
    n = np.concatenate([n1, n2])
    if not np.isclose(n.sum(), total, rtol=1e-12, atol=0):
        raise ValueError("Initial allocations must sum to total_subjects across both groups")
    if not np.allclose(matrix @ (n / total), shares, rtol=1e-12, atol=0):
        raise ValueError("Initial allocations must match group_totals")
    compared = 0 if (form == "linear") == (comparison == "location") else 1
    gradients, weights = [], []
    for group, points in enumerate((x1, x2)):
        indices = [0, 1]
        if group == 1:
            indices[compared] = 2
        first, second = b[:, indices[0], None], b[:, indices[1], None]
        if form == "linear":
            u = first + second * points
            derivatives = np.broadcast_arrays(np.ones_like(u), points)
        else:
            u = first * (points - second)
            derivatives = np.broadcast_arrays(points - second, -first)
        gradient = np.zeros(u.shape + (3,))
        for index, derivative in zip(indices, derivatives, strict=True):
            gradient[..., index] = derivative
        gradients.append(gradient)
        weights.append(_response_information(u, model)[1])
    target = np.zeros_like(b)
    target[:, compared], target[:, 2] = 1, -1
    result = _optimize_prior_information(
        x,
        n,
        np.concatenate(gradients, axis=1),
        np.concatenate(weights, axis=1),
        target,
        prior,
        total,
        0.5 if measure == "sd" else 1.0,
        aggregation,
        tol,
        int(max_iterations),
        constraints=(matrix, shares),
    )
    return SingleTwoSampleAllocation(
        (x1.copy(), x2.copy()),
        (result.subjects[: x1.size], result.subjects[x1.size :]),
        result.value,
        result.initial_value,
        result.optimality_gap,
        result.iterations,
    )
