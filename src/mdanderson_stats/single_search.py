"""Incremental dose-point search for SINGLE response designs."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite, scalar
from .single_optimize import DesignVectors, SingleOptimizedDesign, single_optimize_design
from .single_uniform import _local_criterion


@dataclass(frozen=True)
class SingleSearchStep:
    design: SingleOptimizedDesign
    relative_improvement: float | None
    accepted: bool


@dataclass(frozen=True)
class SingleDesignSearch:
    best: SingleOptimizedDesign
    steps: tuple[SingleSearchStep, ...]
    stop_reason: str
    scan_evaluations: int
    infeasible_seeds: int
    failed_local_starts: int


def single_search_design(
    parameters: ArrayLike,
    dose_bounds: ArrayLike,
    *,
    prior_weights: ArrayLike | None = None,
    total_subjects: float = 100,
    criterion: str = "quantile",
    comparison: str | None = None,
    model: str = "logistic",
    form: str = "linear",
    measure: str = "sd",
    aggregation: str = "arithmetic",
    quantile: float = 0.05,
    max_doses: int = 10,
    scan_points: int = 10,
    relative_improvement: float = 0.01,
    tolerance: float = 1e-10,
    max_iterations: int = 1000,
) -> SingleDesignSearch:
    """Scan seeds, jointly optimize, and add doses until improvement is too small.

    Starts with two dose entries per group. Each extension adds one entry per
    group at a common grid dose, then joint optimization may move them separately.
    max_doses is per group. The original 1% stopping rule is the default; a
    rejected larger design remains in steps. This is a local heuristic search,
    not a proof of the globally best design or support size.
    """
    for limit_value, name, maximum in [
        (max_doses, "max_doses", 10),
        (scan_points, "scan_points", 20),
    ]:
        if (
            isinstance(limit_value, (bool, np.bool_))
            or not isinstance(limit_value, (int, np.integer))
            or not 2 <= limit_value <= maximum
        ):
            raise ValueError(f"{name} must be an integer from 2 to {maximum}")
    threshold = scalar(relative_improvement, "relative_improvement")
    total, tol, q = (
        scalar(total_subjects, "total_subjects"),
        scalar(tolerance, "tolerance"),
        scalar(quantile, "quantile"),
    )
    if not 0 <= threshold < 1 or total <= 0 or not 0 < tol < 1 or not 0 < q < 1:
        raise ValueError("Invalid improvement threshold, subject total, tolerance or quantile")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    if (
        model not in ("logistic", "loglog")
        or form not in ("linear", "centered")
        or comparison not in (None, "location", "slope")
    ):
        raise ValueError("Invalid model, form or comparison")
    if (
        criterion not in ("slope", "quantile")
        or measure not in ("sd", "variance")
        or aggregation not in ("arithmetic", "harmonic")
    ):
        raise ValueError("Invalid criterion, measure or aggregation")
    bounds = finite(dose_bounds, "dose_bounds")
    if bounds.shape != (2,) or bounds[0] >= bounds[1] or not np.isfinite(bounds[1] - bounds[0]):
        raise ValueError("Require finite increasing dose bounds with finite width")
    dimension = 2 if comparison is None else 3
    b = finite(parameters, "parameters")
    if b.shape == (dimension,):
        b = b[None, :]
    if b.ndim != 2 or b.shape[1] != dimension or b.shape[0] == 0:
        raise ValueError("Parameter dimensions do not match the number of samples")
    if prior_weights is None:
        if b.shape[0] != 1:
            raise ValueError("prior_weights are required for multiple parameter nodes")
        prior = np.ones(1)
    else:
        prior = finite(prior_weights, "prior_weights")
    if (
        prior.shape != (b.shape[0],)
        or np.any(prior <= 0)
        or not np.isclose(prior.sum(), 1, rtol=1e-12, atol=0)
    ):
        raise ValueError("Positive prior weights must match nodes and sum to one")
    prior = prior / prior.sum()
    grid = np.linspace(bounds[0], bounds[1], scan_points)
    groups = 1 if comparison is None else 2
    evaluations = infeasible = failures = 0
    steps: list[SingleSearchStep] = []

    def packed(values: tuple[FloatArray, ...]) -> DesignVectors:
        return values[0] if groups == 1 else (values[0], values[1])

    def score(x: DesignVectors, n: DesignVectors) -> float:
        nonlocal evaluations, infeasible
        evaluations += 1
        try:
            local = _local_criterion(
                x,
                n,
                b,
                f"{criterion}_{measure}" if groups == 1 else measure,
                model,
                form,
                comparison,
                q,
            )
        except ValueError:
            # All static options were validated above; a grid seed can still
            # have singular information, zero quantile slope or overflow.
            infeasible += 1
            return np.inf
        if aggregation == "arithmetic":
            value = float(prior @ local)
        else:
            minimum = np.min(local)
            value = float(minimum / (prior @ (minimum / local)))
        if not np.isfinite(value) or value <= 0:
            infeasible += 1
            return np.inf
        return value

    def optimize(
        candidates: list[tuple[float, DesignVectors, DesignVectors]],
    ) -> SingleOptimizedDesign:
        nonlocal failures
        if not candidates:
            raise ValueError("No feasible grid seeds for this design and prior")
        for _, x, n in sorted(candidates, key=lambda item: item[0]):
            try:
                return single_optimize_design(
                    x,
                    b,
                    bounds,
                    prior_weights=prior,
                    total_subjects=total,
                    initial_subjects=n,
                    criterion=criterion,
                    comparison=comparison,
                    model=model,
                    form=form,
                    measure=measure,
                    aggregation=aggregation,
                    quantile=q,
                    tolerance=tol,
                    max_iterations=int(max_iterations),
                )
            except RuntimeError:
                failures += 1
        raise RuntimeError("All feasible grid starts failed joint optimization")

    seeds: list[tuple[float, DesignVectors, DesignVectors]] = []
    for pair in combinations(grid, 2):
        x = packed(tuple(np.array(pair) for _ in range(groups)))
        n = packed(tuple(np.full(2, total / (2 * groups)) for _ in range(groups)))
        value = score(x, n)
        if np.isfinite(value):
            seeds.append((value, x, n))
    best = optimize(seeds)
    steps.append(SingleSearchStep(best, None, True))
    for _ in range(3, max_doses + 1):
        xs = best.doses if isinstance(best.doses, tuple) else (best.doses,)
        ns = best.subjects if isinstance(best.subjects, tuple) else (best.subjects,)
        seeds = []
        for denominator in (1, 2, 4, 8, 16, 32, 64):
            # Preserve each group's existing total during seed construction;
            # joint optimization can subsequently redistribute across groups.
            counts = []
            for n in ns:
                added = np.min(n[n > 0]) / denominator
                counts.append(np.append(n * (1 - added / n.sum()), added))
            proposed_n = packed(tuple(counts))
            for point in grid:
                proposed_x = packed(tuple(np.append(x, point) for x in xs))
                value = score(proposed_x, proposed_n)
                if np.isfinite(value):
                    seeds.append((value, proposed_x, proposed_n))
        candidate = optimize(seeds)
        improvement = (best.value - candidate.value) / best.value
        accepted = improvement >= threshold
        steps.append(SingleSearchStep(candidate, improvement, accepted))
        if not accepted:
            return SingleDesignSearch(
                best, tuple(steps), "relative_improvement", evaluations, infeasible, failures
            )
        best = candidate
    return SingleDesignSearch(best, tuple(steps), "max_doses", evaluations, infeasible, failures)
