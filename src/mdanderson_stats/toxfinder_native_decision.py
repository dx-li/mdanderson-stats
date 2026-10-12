"""Declared ToxFinder 1.1.1 stage-one and fixed-grid allocation workflows."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite, scalar
from .toxfinder_model import ToxFinderPrior, toxfinder_probabilities
from .toxfinder_native_information import _frozen, _history, toxfinder_native_information


def _levels(value: ArrayLike) -> FloatArray:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != 2 or not 2 <= len(raw) <= 50:
        raise ValueError("dose_levels must have shape (2..50,2)")
    levels = finite(raw, "dose_levels")
    if np.any(levels < 0) or np.any(levels > 1) or np.any(np.diff(levels, axis=0) <= 0):
        raise ValueError("native levels must strictly increase in both agents inside [0,1]")
    return levels


def _goal(target: float) -> float:
    value = scalar(target, "target")
    if not 0 < value < 1:
        raise ValueError("target must be in (0,1)")
    return value


def _predictor(parameters: ArrayLike, log_parameters: bool) -> Callable[[float, float], float]:
    draws = np.asarray(parameters)
    if draws.ndim < 1 or draws.shape[-1] != 6 or not 6 <= draws.size <= 200_000:
        raise ValueError("parameters must contain 1..33,333 six-parameter draws")
    draws = draws.reshape((-1, 6))
    # Validate even if a controller branch does not request a prediction.
    toxfinder_probabilities([[0.5, 0.5]], draws, log_parameters=log_parameters)
    cache: dict[tuple[float, float], float] = {}

    def predict(x: float, y: float) -> float:
        key = (float(x), float(y))
        if key not in cache:
            cache[key] = float(
                toxfinder_probabilities([key], draws, log_parameters=log_parameters)[..., 1].mean()
            )
        return cache[key]

    return predict


def _stage1(
    levels: FloatArray,
    history: FloatArray,
    outcomes: FloatArray,
    target: float,
    predict: Callable,
    start_index: int,
) -> FloatArray:
    if not len(history):
        return levels[start_index]
    toxic = np.flatnonzero(outcomes)
    first = (
        len(levels) if not len(toxic) else int(np.argmin(abs(levels[:, 0] - history[toxic[0], 0])))
    )
    candidates = []
    for i, dose in enumerate(levels):
        candidates.append(dose)
        if first <= i < len(levels) - 1:
            candidates.append((dose + levels[i + 1]) / 2)
    points = np.asarray(candidates)
    upper = min(int(np.argmin(abs(points[:, 0] - history[:, 0].max()))) + 1, len(points) - 1)
    previous = 0.0
    for i in range(upper):
        probability = predict(*points[i])
        if probability > target:
            if i == 0:
                # Explicit safe boundary; native source falls through to index -1.
                return points[0]
            if i < first:
                weight = (target - previous) / (probability - previous)
                return (1 - weight) * points[i - 1] + weight * points[i]
            return points[i] if target - previous > probability - target else points[i - 1]
        previous = probability
    return points[upper]


def toxfinder_native_stage1(
    dose_levels: ArrayLike,
    parameters: ArrayLike,
    treated_doses: ArrayLike,
    toxicities: ArrayLike,
    *,
    target: float,
    start_index: int = 0,
    log_parameters: bool = False,
) -> FloatArray:
    """Chronological individual-patient native rule, with a safe lower boundary.

    History outcomes are binary and observed. Pending patients must be excluded;
    use the observed-history workflow to retain pending records separately.
    """
    levels = _levels(dose_levels)
    raw = np.asarray(toxicities)
    history, y, _ = _history(treated_doses, raw, np.ones(raw.shape))
    if isinstance(start_index, (bool, np.bool_)) or not isinstance(start_index, (int, np.integer)):
        raise ValueError("start_index must be an integer")
    if not 0 <= start_index < len(levels):
        raise ValueError("start_index must identify a dose level")
    predict = _predictor(parameters, log_parameters)
    return _frozen(_stage1(levels, history, y, _goal(target), predict, int(start_index)))


def _allocation(
    levels: FloatArray,
    target: float,
    potency: float,
    mode: str,
    predict: Callable,
    information: Callable,
    allocation: bool,
) -> FloatArray:
    grid = 30
    nodes = levels
    if levels[0, 0] > 0:
        nodes = np.vstack(([0.0, 0.0], nodes))
    if levels[-1, 0] < 1:
        nodes = np.vstack((nodes, [1.0, 1.0]))
    first_axis = np.arange(grid) / grid
    second_axis = np.interp(first_axis, nodes[:, 0], nodes[:, 1])

    def prob(i: int, j: int) -> float:
        return predict(first_axis[i], second_axis[j])

    k = 1
    while k < grid and prob(k, k) <= target:
        k += 1
    k -= 1

    def crossing(fixed: int, previous: int, swap: bool) -> int:
        def p(j: int) -> float:
            return prob(j, fixed) if swap else prob(fixed, j)

        if p(previous) > target:
            j = previous - 1
            while j >= 0 and p(j) >= target:
                j -= 1
        else:
            j = previous + 1
            while j < grid and p(j) <= target:
                j += 1
            j = j - 1 if j < grid else -1
        return j

    def sweep(direction: int, swap: bool) -> tuple[FloatArray, float]:
        best = -float("inf")
        best_dose = np.zeros(2)
        best_ckp = 0.0
        ckp_dose = np.zeros(2)
        seen = False
        previous = -1
        for fixed in range(k, grid if direction > 0 else -1, direction):
            if previous < 0:
                previous = k
            previous = crossing(fixed, previous, swap)
            if previous >= 0 and (
                (previous < k and direction < 0) or (previous > k and direction > 0)
            ):
                previous = k
            if previous <= 0:
                continue
            x, y = (
                (first_axis[previous], second_axis[fixed])
                if swap
                else (first_axis[fixed], second_axis[previous])
            )
            if allocation:
                value = information(x, y)
                if value > best:
                    best, best_dose = value, np.array([x, y])
            value = x + potency * y
            if value > best_ckp:
                best_ckp, ckp_dose = value, np.array([x, y])
            seen = True
        if not seen:
            return np.array([first_axis[k], second_axis[k]]), -1e10
        if not allocation:
            return ckp_dose, best_ckp
        if mode == "fi":
            return best_dose, best
        return (best_dose + ckp_dose) / 2, best

    result = [np.array([first_axis[k], second_axis[k]])]
    for direction in (1, -1):
        a, sa = sweep(direction, False)
        b, sb = sweep(-direction, True)
        result.append(b if sb > sa else a)
    return np.asarray(result)


@dataclass(frozen=True)
class ToxFinderNativeAllocation:
    """Rows are middle, first branch, second branch; first stage-two uses row 2."""

    doses: FloatArray
    estimated_toxicity: FloatArray
    mode: str
    allocation: bool


def toxfinder_native_allocation(
    dose_levels: ArrayLike,
    parameters: ArrayLike,
    *,
    target: float,
    potency: float = 1.0,
    mode: Literal["fi_ckp", "fi"] = "fi_ckp",
    doses: ArrayLike = (),
    toxicities: ArrayLike = (),
    subjects: ArrayLike = (),
    prior: ToxFinderPrior | None = None,
    log_parameters: bool = False,
    allocation: bool = True,
) -> ToxFinderNativeAllocation:
    """Native 30-point traversal; allocation=False selects the final CKP contour.

    FI uses a MAP fitted to the supplied observed history. Final-contour selection
    needs only posterior draws and does not fit an information matrix.
    """
    levels, goal = _levels(dose_levels), _goal(target)
    strength = scalar(potency, "potency")
    if strength <= 0 or mode not in ("fi_ckp", "fi"):
        raise ValueError("potency must be positive; mode must be 'fi_ckp' or 'fi'")
    if not isinstance(allocation, bool):
        raise ValueError("allocation must be boolean")
    predict = _predictor(parameters, log_parameters)
    if allocation:
        if prior is None:
            raise ValueError("allocation requires a gamma prior for observed information")
        info = toxfinder_native_information(doses, toxicities, subjects, prior=prior)

        def information(x: float, y: float) -> float:
            return info.score([x, y], predict(x, y))
    else:

        def information(x: float, y: float) -> float:
            return 0.0

    pairs = _allocation(levels, goal, strength, mode, predict, information, allocation)
    return ToxFinderNativeAllocation(
        _frozen(pairs), _frozen([predict(*p) for p in pairs]), mode, allocation
    )
