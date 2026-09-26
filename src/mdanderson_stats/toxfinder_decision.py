"""Dose selection and target-contour helpers for ToxFinder."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from ._validation import FloatArray, count, finite, scalar
from .toxfinder_model import toxfinder_probabilities

_MAX_LEVELS = 50
_MAX_POSTERIOR_DOSES = 200_000


def _readonly(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _readonly_bool(value: ArrayLike) -> np.ndarray:
    result = np.array(value, dtype=bool, copy=True)
    result.flags.writeable = False
    return result


def _levels(value: ArrayLike) -> FloatArray:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != 2 or not 2 <= raw.shape[0] <= _MAX_LEVELS:
        raise ValueError("dose_levels must have shape (2..50, 2)")
    levels = finite(raw, "dose_levels")
    if np.any(levels < 0) or np.any(np.diff(levels, axis=0) <= 0):
        raise ValueError("dose_levels must be nonnegative and strictly increase in both agents")
    origin = levels[0]
    direction = levels[-1] - origin
    coordinates = (levels - origin) @ direction / (direction @ direction)
    if np.any(
        np.linalg.norm(levels - (origin + coordinates[:, None] * direction), axis=1)
        > 1e-8 * max(1.0, np.linalg.norm(direction))
    ):
        raise ValueError("dose_levels must lie on one L1 dose segment")
    return levels


def _draws(parameters: ArrayLike, log_parameters: bool) -> FloatArray:
    raw = np.asarray(parameters)
    if raw.ndim < 1 or raw.shape[-1] != 6 or raw.size > 200_000:
        raise ValueError("parameters must end in six values and contain at least one draw")
    value = np.asarray(raw, dtype=np.float64).reshape((-1, 6))
    if value.shape[0] == 0:
        raise ValueError("parameters must contain at least one draw")
    # Delegate alpha zero and log-coordinate validation to the core model.
    return value


def _line_coordinate(points: FloatArray, levels: FloatArray) -> FloatArray:
    origin = levels[0]
    direction = levels[-1] - origin
    norm = float(direction @ direction)
    coordinate = (points - origin) @ direction / norm
    residual = points - (origin + coordinate[:, None] * direction)
    if np.any(np.linalg.norm(residual, axis=1) > 1e-8 * max(1.0, np.linalg.norm(direction))):
        raise ValueError("treated_doses must lie on the dose-level L1 segment")
    if np.any(coordinate < -1e-10) or np.any(coordinate > 1 + 1e-10):
        raise ValueError("treated_doses must lie within the dose-level L1 segment")
    return np.clip(coordinate, 0, 1)


@dataclass(frozen=True)
class ToxFinderStage1Result:
    dose: FloatArray
    estimated_toxicity: float
    candidate_doses: FloatArray
    mean_toxicity: FloatArray
    eligible: np.ndarray
    first_toxic_dose: FloatArray | None
    reason: str


def toxfinder_stage1(
    dose_levels: ArrayLike,
    parameters: ArrayLike,
    treated_doses: ArrayLike,
    toxicities: ArrayLike,
    *,
    target: float,
    log_parameters: bool = False,
    start_index: int = 0,
) -> ToxFinderStage1Result:
    """Select the candidate whose interpolated mean toxicity is nearest target.

    History is chronological, with one nonnegative toxicity count per treated-dose
    row. Once toxicity first occurs at an original level, midpoint candidates are
    added in intervals at and above that level. This implements the paper's L1
    mean-interpolation rule; native software may add other workflow behavior.
    """
    levels = _levels(dose_levels)
    draws = _draws(parameters, log_parameters)
    goal = scalar(target, "target")
    if not 0 < goal < 1:
        raise ValueError("target must be in (0, 1)")
    if isinstance(start_index, (bool, np.bool_)) or not isinstance(start_index, (int, np.integer)):
        raise ValueError("start_index must be an integer")
    if not 0 <= start_index < len(levels):
        raise ValueError("start_index must identify a dose_levels row")
    raw_history = np.asarray(treated_doses)
    if raw_history.ndim == 1 and raw_history.size == 0:
        raw_history = raw_history.reshape((0, 2))
    if raw_history.ndim != 2 or raw_history.shape[1] != 2 or raw_history.shape[0] > 100:
        raise ValueError("treated_doses must have shape (0..100, 2)")
    history = finite(raw_history, "treated_doses")
    raw_y = np.asarray(toxicities)
    if raw_y.size > 100:
        raise ValueError("toxicities may contain at most 100 rows")
    y = count(raw_y, "toxicities")
    if y.ndim != 1 or y.size != history.shape[0]:
        raise ValueError(
            "toxicities must contain one nonnegative integer count per treated-dose row"
        )
    coordinate = _line_coordinate(history, levels) if history.size else np.empty(0)

    if draws.shape[0] * len(levels) > _MAX_POSTERIOR_DOSES:
        raise ValueError("posterior draws times dose levels exceeds 200,000")
    node_prob = toxfinder_probabilities(levels, draws, log_parameters=log_parameters)[..., 1]
    node_mean = node_prob.mean(axis=0)
    origin = levels[0]
    direction = levels[-1] - origin
    norm2 = float(direction @ direction)
    scalars = ((levels - origin) @ direction) / norm2
    first_toxic_index: int | None = None
    first_toxic: FloatArray | None = None
    toxic_rows = np.flatnonzero(y > 0)
    first_toxic_row = int(toxic_rows[0]) if toxic_rows.size else len(history)
    for row, point in enumerate(history):
        matches = np.flatnonzero(np.all(np.isclose(levels, point, rtol=0, atol=1e-10), axis=1))
        if row <= first_toxic_row and not matches.size:
            raise ValueError("history through first toxicity must use original dose levels")
        if toxic_rows.size and row == first_toxic_row:
            if not matches.size:
                raise ValueError("first toxicity must occur at an original dose level")
            first_toxic_index = int(matches[0])
            first_toxic = _readonly(point)

    candidate_scalars = list(scalars)
    if first_toxic_index is not None:
        r = first_toxic_index
        candidate_scalars.extend(
            (scalars[i] + scalars[i + 1]) / 2 for i in range(r, len(scalars) - 1)
        )
        # Below the first toxic level, the method uses linear interpolation of
        # node means. Add every target crossing in that continuous lower segment.
        for i in range(r):
            left, right = node_mean[i], node_mean[i + 1]
            if (left - goal) * (right - goal) < 0:
                candidate_scalars.append(
                    scalars[i] + (goal - left) * (scalars[i + 1] - scalars[i]) / (right - left)
                )
    candidate_scalars_array = np.unique(np.asarray(candidate_scalars))
    if candidate_scalars_array.size > 99:
        raise ValueError("candidate dose set exceeds 99 levels")
    candidates = origin + candidate_scalars_array[:, None] * direction
    posterior_pair_count = draws.shape[0] * candidates.shape[0]
    if posterior_pair_count > _MAX_POSTERIOR_DOSES:
        raise ValueError("posterior draws times candidate doses exceeds 200,000")
    candidate_mean = toxfinder_probabilities(candidates, draws, log_parameters=log_parameters)[
        ..., 1
    ].mean(axis=0)
    if first_toxic_index is not None:
        lower = candidate_scalars_array <= scalars[first_toxic_index] + 1e-12
        candidate_mean[lower] = np.interp(
            candidate_scalars_array[lower],
            scalars[: first_toxic_index + 1],
            node_mean[: first_toxic_index + 1],
        )
    if history.size == 0:
        eligible = np.zeros(candidate_scalars_array.shape, dtype=bool)
        selected = int(np.flatnonzero(np.isclose(candidate_scalars_array, scalars[start_index]))[0])
        eligible[selected] = True
    else:
        last = coordinate[-1]
        eligible = candidate_scalars_array <= last + 1e-10
        tried = np.array([((point - origin) @ direction) / norm2 for point in history])
        upward = np.flatnonzero(candidate_scalars_array > last + 1e-10)
        for index in upward:
            if np.any(np.isclose(tried, candidate_scalars_array[index], rtol=0, atol=1e-10)):
                eligible[index] = True
            else:
                eligible[index] = True
                break
        selected = -1
    if history.size:
        selected = int(np.flatnonzero(eligible)[np.argmin(np.abs(candidate_mean[eligible] - goal))])
    return ToxFinderStage1Result(
        _readonly(candidates[selected]),
        float(candidate_mean[selected]),
        _readonly(candidates),
        _readonly(candidate_mean),
        _readonly_bool(eligible),
        first_toxic,
        "initial"
        if history.size == 0
        else ("after-first-toxicity" if first_toxic is not None else "no-skip"),
    )


@dataclass(frozen=True)
class ToxFinderContour:
    doses: FloatArray
    mean_toxicity: FloatArray
    attainable: np.ndarray


def toxfinder_contour(
    parameters: ArrayLike,
    x1: ArrayLike,
    *,
    target: float,
    x2_bounds: tuple[float, float] = (0.0, 1.0),
    log_parameters: bool = False,
) -> ToxFinderContour:
    """Find mean-toxicity target crossings at requested x1 values by bisection."""
    draws = _draws(parameters, log_parameters)
    raw_x1 = np.asarray(x1)
    if raw_x1.size > 200 or raw_x1.ndim > 1:
        raise ValueError("x1 must contain at most 200 scalar values")
    first = finite(raw_x1, "x1").reshape(-1)
    goal = scalar(target, "target")
    raw_bounds = np.asarray(x2_bounds)
    if raw_bounds.shape != (2,):
        raise ValueError("x2_bounds must contain two values")
    bounds = finite(raw_bounds, "x2_bounds")
    if (
        first.size == 0
        or np.any(first < 0)
        or not 0 < goal < 1
        or bounds.shape != (2,)
        or bounds[0] < 0
        or bounds[1] <= bounds[0]
    ):
        raise ValueError("invalid target contour inputs or bounds")
    if draws.shape[0] * first.size * 105 > 2_000_000:
        raise ValueError("contour root evaluation budget exceeds 2,000,000 draw-point evaluations")

    def mean_at(a: float, b: float) -> float:
        p = toxfinder_probabilities([[a, b]], draws, log_parameters=log_parameters)[..., 1]
        return float(p.mean())

    roots = np.full(first.shape, np.nan)
    values = np.full(first.shape, np.nan)
    attainable = np.zeros(first.shape, dtype=bool)
    for i, a in enumerate(first):
        low, high = float(bounds[0]), float(bounds[1])
        f_low, f_high = mean_at(float(a), low) - goal, mean_at(float(a), high) - goal
        if f_low == 0:
            root = low
        elif f_high == 0:
            root = high
        elif f_low * f_high > 0:
            continue
        else:
            root = float(
                brentq(lambda b: mean_at(float(a), b) - goal, low, high, xtol=1e-10, maxiter=100)
            )
        roots[i] = root
        values[i] = mean_at(float(a), root)
        if abs(values[i] - goal) > 1e-8:
            raise ArithmeticError("contour root did not reach the requested mean-toxicity target")
        attainable[i] = True
    dose_pairs = np.column_stack((first, roots))
    return ToxFinderContour(_readonly(dose_pairs), _readonly(values), _readonly_bool(attainable))
