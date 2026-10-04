"""Continuous cutoff calibration for normal and survival success OCs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import finite, scalar
from .success_calibration import (
    SuccessOperatingCharacteristics,
    normal_success_oc,
    survival_success_oc,
)


@dataclass(frozen=True)
class ContinuousSuccessCalibration:
    """A feasible cutoff and its final infeasible/feasible bisection bracket."""

    cutoff: float
    target: float
    operating_characteristics: SuccessOperatingCharacteristics
    candidates_evaluated: int
    bracket: tuple[float, float]
    cutoff_tolerance: float


def _calibrate(
    evaluate: Callable[[float], SuccessOperatingCharacteristics],
    target: float,
    cutoff_range: ArrayLike,
    cutoff_tolerance: float,
    max_evaluations: int,
) -> ContinuousSuccessCalibration:
    target_value = scalar(target, "target")
    if not 0 < target_value < 1:
        raise ValueError("target must be in (0,1)")
    interval = finite(cutoff_range, "cutoff_range")
    if interval.shape != (2,):
        raise ValueError("cutoff_range must contain exactly two endpoints")
    lower, upper = map(float, interval)
    if not 0 <= lower <= upper < 1:
        raise ValueError("cutoff_range must satisfy 0 <= lower <= upper < 1")
    tolerance = scalar(cutoff_tolerance, "cutoff_tolerance")
    if tolerance <= 0:
        raise ValueError("cutoff_tolerance must be positive")
    if (
        isinstance(max_evaluations, (bool, np.bool_))
        or not isinstance(max_evaluations, (int, np.integer))
        or not 1 <= max_evaluations <= 100
    ):
        raise ValueError("max_evaluations must be an integer in [1,100]")
    max_evaluations = int(max_evaluations)

    lower_result = evaluate(lower)
    evaluations = 1
    lower_pid = lower_result.incorrect_decision_probability
    if lower_pid is None or not np.isfinite(lower_pid):
        raise ArithmeticError("PID is undefined at the lower cutoff; success mass is unresolved")
    if lower_pid <= target_value:
        return ContinuousSuccessCalibration(
            lower, target_value, lower_result, evaluations, (lower, lower), tolerance
        )

    if upper == lower:
        raise ValueError("cutoff range contains no cutoff meeting the PID target")
    if evaluations >= max_evaluations:
        raise ArithmeticError("cutoff calibration exhausted max_evaluations before bracketing")
    upper_result = evaluate(upper)
    evaluations += 1
    upper_pid = upper_result.incorrect_decision_probability
    if upper_pid is None or not np.isfinite(upper_pid):
        raise ArithmeticError("PID is undefined at the upper cutoff; success mass is unresolved")
    if upper_pid > target_value:
        raise ValueError("upper cutoff does not meet the PID target")

    # For the normal model, the truth and posterior decision statistic have
    # positive correlation. Conditioning on increasingly favorable posterior
    # evidence therefore decreases the probability of an ineffective truth.
    # The survival adapter is this same normal model with the favorable tail
    # reversed, so PID is nonincreasing in cutoff for both wrappers.
    best = upper_result
    while upper - lower > tolerance:
        if evaluations >= max_evaluations:
            raise ArithmeticError("cutoff bisection exhausted max_evaluations before tolerance")
        midpoint = lower + (upper - lower) / 2
        if midpoint <= lower or midpoint >= upper:
            raise ArithmeticError("cutoff bracket cannot be refined at float64 precision")
        result = evaluate(midpoint)
        evaluations += 1
        pid = result.incorrect_decision_probability
        if pid is None or not np.isfinite(pid):
            raise ArithmeticError("PID is undefined during cutoff bisection")
        if pid <= target_value:
            upper = midpoint
            best = result
        else:
            lower = midpoint

    return ContinuousSuccessCalibration(
        upper, target_value, best, evaluations, (lower, upper), tolerance
    )


def calibrate_normal_success_cutoff(
    target: float,
    *,
    cutoff_range: tuple[float, float] = (0.6, 0.999),
    cutoff_tolerance: float = 1e-10,
    max_evaluations: int = 80,
    standard_error: ArrayLike,
    design_mean: ArrayLike = 0.0,
    design_sd: ArrayLike = 1.0,
    analysis_mean: ArrayLike = 0.0,
    analysis_sd: ArrayLike = 1.0,
    margin: float = 0.0,
    direction: str = "greater",
    null_mean: ArrayLike | None = None,
) -> ContinuousSuccessCalibration:
    """Bisect to the smallest feasible normal-model cutoff in a bounded range.

    All model arguments, other than the cutoff, match :func:`normal_success_oc`.
    The returned cutoff is the feasible (upper) endpoint of a bracket no wider
    than ``cutoff_tolerance``. This is an independent Python search convention,
    not a claim about the application's optimization algorithm.
    """

    def evaluate(cutoff: float) -> SuccessOperatingCharacteristics:
        return normal_success_oc(
            cutoff,
            standard_error=standard_error,
            design_mean=design_mean,
            design_sd=design_sd,
            analysis_mean=analysis_mean,
            analysis_sd=analysis_sd,
            margin=margin,
            direction=direction,
            null_mean=null_mean,
        )

    return _calibrate(evaluate, target, cutoff_range, cutoff_tolerance, max_evaluations)


def calibrate_survival_success_cutoff(
    target: float,
    *,
    cutoff_range: tuple[float, float] = (0.6, 0.999),
    cutoff_tolerance: float = 1e-10,
    max_evaluations: int = 80,
    events: float,
    treatment_allocation: float,
    design_mean: float = 0.0,
    design_sd: float = 1.0,
    analysis_mean: float = 0.0,
    analysis_sd: float = 1.0,
    margin: float = 0.0,
    null_mean: float | None = None,
) -> ContinuousSuccessCalibration:
    """Bisect the survival log-hazard-ratio model's normal-approximation PID."""

    def evaluate(cutoff: float) -> SuccessOperatingCharacteristics:
        return survival_success_oc(
            cutoff,
            events=events,
            treatment_allocation=treatment_allocation,
            design_mean=design_mean,
            design_sd=design_sd,
            analysis_mean=analysis_mean,
            analysis_sd=analysis_sd,
            margin=margin,
            null_mean=null_mean,
        )

    return _calibrate(evaluate, target, cutoff_range, cutoff_tolerance, max_evaluations)
