"""Bounded allocation and accrual planning for historical-control survival."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq, minimize_scalar

from ._validation import scalar
from .stplan_continuous import _alpha
from .stplan_survival import stplan_historical_survival_power

_MAX_EVALUATIONS = 20_000
_MAX_GRID_POINTS = 129


@dataclass(frozen=True)
class STPLANHistoricalAllocationPlan:
    """Power-verified accrual plan with bounded optimal control allocation."""

    accrual_duration: float
    control_allocation: float
    target_power: float
    achieved_power: float
    expected_enrollment: float
    accrual_bounds: tuple[float, float]
    allocation_bounds: tuple[float, float]
    at_accrual_lower_bound: bool
    evaluations: int
    inputs: Mapping[str, object]


def _number(value: ArrayLike, name: str) -> float:
    try:
        raw = np.asarray(value)
        if raw.ndim != 0 or raw.dtype.kind not in "iuf":
            raise ValueError
        return scalar(float(raw), name)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc


def _bounds(value: tuple[float, float], name: str, *, lower_open: bool) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a pair of finite bounds")
    low, high = (_number(value[0], f"{name}[0]"), _number(value[1], f"{name}[1]"))
    if (low <= 0 if lower_open else low < 0) or not low < high:
        qualifier = "positive" if lower_open else "nonnegative"
        raise ValueError(f"{name} must be ascending with a {qualifier} lower bound")
    return low, high


def stplan_historical_allocation_plan(
    experimental_hazard: ArrayLike,
    control_hazard: ArrayLike,
    accrual_rate: ArrayLike,
    followup_duration: ArrayLike,
    historical_deaths: ArrayLike,
    historical_alive: ArrayLike,
    *,
    target_power: float,
    accrual_bounds: tuple[float, float],
    allocation_bounds: tuple[float, float] = (0.0, 0.99),
    continued_followup: bool = True,
    alpha: float = 0.05,
    sides: int = 1,
    grid_points: int = 17,
    power_tolerance: float = 1e-8,
    max_evaluations: int = 20_000,
) -> STPLANHistoricalAllocationPlan:
    """Find the least accrual duration meeting target power over allocation bounds.

    At each duration, power is maximized over the bounded control allocation
    using a coarse grid and local bounded refinements. The duration root is
    searched in log space over ``accrual_bounds``. The result is numerical over
    the supplied bounds, not a proof of global optimality or automatic bound
    discovery. When continued follow-up is disabled, allocation cannot add
    control events, so the lower allocation bound is used directly.
    """
    alpha_value = _number(alpha, "alpha")
    significance = _alpha(alpha_value, sides)
    hre = _number(experimental_hazard, "experimental_hazard")
    hrc = _number(control_hazard, "control_hazard")
    rate = _number(accrual_rate, "accrual_rate")
    followup = _number(followup_duration, "followup_duration")
    deaths = _number(historical_deaths, "historical_deaths")
    alive = _number(historical_alive, "historical_alive")
    target = _number(target_power, "target_power")
    tolerance = _number(power_tolerance, "power_tolerance")
    accrual_range = _bounds(accrual_bounds, "accrual_bounds", lower_open=True)
    allocation_range = _bounds(allocation_bounds, "allocation_bounds", lower_open=False)
    if not (0 < hre < hrc and rate > 0 and followup >= 0 and deaths > 0 and alive >= 0):
        raise ValueError(
            "require 0 < experimental_hazard < control_hazard, positive accrual_rate "
            "and historical_deaths, and nonnegative follow-up/alive count"
        )
    if allocation_range[1] >= 1:
        raise ValueError("allocation_bounds must remain below 1")
    if not significance < target < 1:
        raise ValueError("target_power must be greater than achieved significance and less than 1")
    if not 0 < tolerance < 0.01:
        raise ValueError("power_tolerance must lie between 0 and 0.01")
    if (
        isinstance(grid_points, bool)
        or not isinstance(grid_points, int)
        or not 5 <= grid_points <= _MAX_GRID_POINTS
    ):
        raise ValueError(f"grid_points must be an integer from 5 to {_MAX_GRID_POINTS}")
    if (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, int)
        or not 20 <= max_evaluations <= _MAX_EVALUATIONS
    ):
        raise ValueError(f"max_evaluations must be an integer from 20 to {_MAX_EVALUATIONS}")
    if not isinstance(continued_followup, (bool, np.bool_)):
        raise ValueError("continued_followup must be boolean")
    continued = bool(continued_followup)

    evaluations = 0
    cache: dict[float, tuple[float, float]] = {}

    def evaluate(duration: float, allocation: float) -> float:
        nonlocal evaluations
        if evaluations >= max_evaluations:
            raise RuntimeError("historical allocation search exceeded max_evaluations")
        evaluations += 1
        value = stplan_historical_survival_power(
            hre,
            hrc,
            rate,
            duration,
            followup,
            deaths,
            alive,
            control_allocation=allocation,
            continued_followup=continued,
            alpha=alpha_value,
            sides=int(sides),
        )
        power = float(np.asarray(value))
        if not isfinite(power) or not 0 <= power <= 1:
            raise ArithmeticError("historical survival power is not finite in [0,1]")
        return power

    def profile(duration: float) -> tuple[float, float]:
        key = float(duration)
        if key in cache:
            return cache[key]
        if not continued:
            outcome = (evaluate(key, allocation_range[0]), allocation_range[0])
            cache[key] = outcome
            return outcome

        grid = np.linspace(allocation_range[0], allocation_range[1], grid_points)
        powers = np.array([evaluate(key, float(w)) for w in grid])
        best_idx = int(np.argmax(powers))
        best_power, best_allocation = float(powers[best_idx]), float(grid[best_idx])
        brackets: list[tuple[int, int]] = [(0, 1), (grid_points - 2, grid_points - 1)]
        for idx in range(1, grid_points - 1):
            if (
                powers[idx] >= powers[idx - 1]
                and powers[idx] >= powers[idx + 1]
                and (powers[idx] > powers[idx - 1] or powers[idx] > powers[idx + 1])
            ):
                brackets.append((idx - 1, idx + 1))
        for left_idx, right_idx in brackets:
            left, right = float(grid[left_idx]), float(grid[right_idx])
            optimized = minimize_scalar(
                lambda w: -evaluate(key, float(w)),
                bounds=(left, right),
                method="bounded",
                options={"xatol": 1e-10},
            )
            if not optimized.success:
                raise ArithmeticError(f"allocation maximization failed: {optimized.message}")
            candidate_power = -float(optimized.fun)
            if candidate_power > best_power:
                best_power, best_allocation = candidate_power, float(optimized.x)
        outcome = (best_power, best_allocation)
        cache[key] = outcome
        return outcome

    lower, upper = accrual_range
    lower_power, lower_allocation = profile(lower)
    at_lower = lower_power >= target
    if at_lower:
        duration, allocation = lower, lower_allocation
        achieved = evaluate(duration, allocation)
    else:
        upper_power, _ = profile(upper)
        if upper_power < target:
            raise ValueError("target power is not attained within accrual_bounds")
        log_lower, log_upper = np.log(lower), np.log(upper)

        def duration_at(log_duration: float) -> float:
            if log_duration <= log_lower:
                return lower
            if log_duration >= log_upper:
                return upper
            return float(np.exp(log_duration))

        def residual(log_duration: float) -> float:
            return profile(duration_at(log_duration))[0] - target

        root = brentq(residual, log_lower, log_upper, xtol=1e-12, rtol=1e-12)
        duration = duration_at(root)
        _, allocation = profile(duration)
        achieved = evaluate(duration, allocation)
        if abs(achieved - target) > tolerance:
            raise ArithmeticError(
                f"optimized design misses target power by {abs(achieved - target):.3g}"
            )

    enrollment = rate * duration
    if not isfinite(enrollment) or enrollment <= 0:
        raise ArithmeticError("expected enrollment is not representable")
    inputs: Mapping[str, object] = MappingProxyType(
        {
            "experimental_hazard": hre,
            "control_hazard": hrc,
            "accrual_rate": rate,
            "accrual_duration": duration,
            "followup_duration": followup,
            "historical_deaths": deaths,
            "historical_alive": alive,
            "control_allocation": allocation,
            "continued_followup": continued,
            "alpha": alpha_value,
            "sides": int(sides),
        }
    )
    return STPLANHistoricalAllocationPlan(
        duration,
        allocation,
        target,
        achieved,
        enrollment,
        accrual_range,
        allocation_range,
        at_lower,
        evaluations,
        inputs,
    )
