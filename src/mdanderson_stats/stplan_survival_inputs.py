"""Conversions from survival summaries to STPLAN exponential model inputs."""

from dataclasses import dataclass
from math import prod

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray
from .stplan_continuous import _inputs

_MAX_CASES = 200_000
_EPS = np.finfo(float).eps


def _log_drop(larger_survival: FloatArray, smaller_survival: FloatArray) -> FloatArray:
    """Compute log(larger/smaller), preserving close survival probabilities."""
    difference = larger_survival - smaller_survival
    close = difference <= 0.5 * smaller_survival
    relative = np.divide(
        difference,
        smaller_survival,
        out=np.zeros_like(difference),
        where=close,
    )
    close_value = np.log1p(relative)
    far_value = np.log(larger_survival) - np.log(smaller_survival)
    return np.where(close, close_value, far_value)


def _log_survival(survival: FloatArray) -> FloatArray:
    near_one = survival > 0.5
    near_argument = np.where(near_one, survival, 1.0)
    far_argument = np.where(near_one, 1.0, survival)
    return np.where(near_one, np.log1p(near_argument - 1.0), np.log(far_argument))


def _nonnegative_rate(value: FloatArray, scale: FloatArray, name: str) -> FloatArray:
    tolerance = 64 * _EPS * np.maximum(np.abs(scale), np.finfo(float).tiny)
    if np.any(~np.isfinite(value) | (value < -tolerance)):
        raise ValueError(f"{name} must be finite and nonnegative")
    return np.where(value < 0, 0.0, value)


def stplan_exponential_hazard(
    value: ArrayLike,
    *,
    parameter: str = "median",
    time: ArrayLike | None = None,
) -> FloatArray:
    """Convert an exponential median, mean, or survival probability to hazard.

    ``parameter`` is ``"median"`` (hazard ``log(2)/value``), ``"mean"``
    (``1/value``), or ``"survival"`` (``-log(value)/time``). The time input is
    required only for survival probabilities; those probabilities lie in
    ``(0, 1]`` and time must be positive. Inputs broadcast to at most 200,000
    values.
    """
    if parameter not in ("median", "mean", "survival"):
        raise ValueError("parameter must be 'median', 'mean', or 'survival'")
    if parameter == "survival":
        if time is None:
            raise ValueError("time is required for a survival probability")
        survival, at_time = _inputs((value, "survival"), (time, "time"))
        if np.any((survival <= 0) | (survival > 1)) or np.any(at_time <= 0):
            raise ValueError("survival must lie in (0,1] and time must be positive")
        hazard = -_log_survival(survival) / at_time
        if np.any(~np.isfinite(hazard)) or np.any((survival < 1) & (hazard == 0)):
            raise ArithmeticError("hazard is not representable")
        return hazard
    if time is not None:
        raise ValueError("time is only valid with parameter='survival'")
    (positive_value,) = _inputs((value, "value"))
    if np.any(positive_value <= 0):
        raise ValueError(f"{parameter} must be positive")
    hazard = np.log(2.0) / positive_value if parameter == "median" else 1 / positive_value
    if np.any(~np.isfinite(hazard) | (hazard <= 0)):
        raise ArithmeticError("positive hazard is not representable")
    return hazard


def stplan_historical_control_hazard(deaths: ArrayLike, total_time: ArrayLike) -> FloatArray:
    """Convert observed historical-control deaths and person-time to a rate."""
    event_count, exposure = _inputs((deaths, "deaths"), (total_time, "total_time"))
    if np.any(event_count < 0) or np.any(exposure <= 0):
        raise ValueError("deaths must be nonnegative and total_time positive")
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        hazard = event_count / exposure
    if np.any(~np.isfinite(hazard)) or np.any((event_count > 0) & (hazard == 0)):
        raise ArithmeticError("historical-control hazard is not representable")
    return hazard


@dataclass(frozen=True)
class STPLANPiecewiseModel:
    """Immutable piecewise-exponential hazards and change-point status."""

    hazard_before: FloatArray
    hazard_after: FloatArray
    change_time: FloatArray
    change_time_identified: NDArray[np.bool_]


def _readonly_bool(value: NDArray[np.bool_]) -> NDArray[np.bool_]:
    return np.frombuffer(value.tobytes(), dtype=np.bool_).reshape(value.shape)


def stplan_piecewise_from_survival(
    times: ArrayLike,
    survival: ArrayLike,
    *,
    change_time: ArrayLike | None = None,
) -> STPLANPiecewiseModel:
    """Infer a two-hazard model from two points and a known break or three points.

    The final axis contains 2 or 3 points. Without ``change_time``, three
    points identify the break if the fitted hazards differ; a consistent
    constant-hazard curve returns the midpoint between the first two times and
    ``change_time_identified=False``. Near-equal hazards within 64 machine
    epsilons relative to their scale use the same unidentified-break result.
    The recovered hazards must be nonnegative and must reconstruct all supplied
    survivals to floating-point relative tolerance.
    """
    raw_times, raw_survival = np.asarray(times), np.asarray(survival)
    if raw_times.size > _MAX_CASES or raw_survival.size > _MAX_CASES:
        raise ValueError("each input is limited to 200000 values")
    if raw_times.ndim == 0 or raw_survival.ndim == 0:
        raise ValueError("times and survival need a final point axis")
    if raw_times.shape[-1] not in (2, 3) or raw_survival.shape[-1] != raw_times.shape[-1]:
        raise ValueError("times and survival must share a final axis of length 2 or 3")
    point_count = raw_times.shape[-1]
    if (point_count == 2) != (change_time is not None):
        raise ValueError("two points require change_time; three points infer it")
    shapes = [raw_times.shape[:-1], raw_survival.shape[:-1]]
    raw_change: NDArray[np.generic] | None = None
    if change_time is not None:
        raw_change = np.asarray(change_time)
        if raw_change.size > _MAX_CASES:
            raise ValueError("change_time is limited to 200000 values")
        shapes.append(raw_change.shape)
    try:
        batch_shape = np.broadcast_shapes(*shapes)
    except ValueError as exc:
        raise ValueError("time, survival, and change-time batches cannot be broadcast") from exc
    if prod(batch_shape) * point_count > _MAX_CASES:
        raise ValueError("broadcast result exceeds 200000 values")

    raw_times = np.broadcast_to(raw_times, batch_shape + (point_count,))
    raw_survival = np.broadcast_to(raw_survival, batch_shape + (point_count,))
    time_values, survival_values = _inputs((raw_times, "times"), (raw_survival, "survival"))
    if np.any(time_values <= 0) or np.any(np.diff(time_values, axis=-1) <= 0):
        raise ValueError("times must be positive and strictly increasing")
    if np.any((survival_values <= 0) | (survival_values > 1)):
        raise ValueError("survival probabilities must lie in (0,1]")
    if np.any(np.diff(survival_values, axis=-1) > 0):
        raise ValueError("survival probabilities must be nonincreasing")
    cumulative = -_log_survival(survival_values)

    if point_count == 2:
        assert raw_change is not None
        t1, t2 = time_values[..., 0], time_values[..., 1]
        s1, s2 = survival_values[..., 0], survival_values[..., 1]
        h1cum, h2cum = cumulative[..., 0], cumulative[..., 1]
        drop = _log_drop(s1, s2)
        (break_value,) = _inputs((raw_change, "change_time"))
        break_value = np.broadcast_to(break_value, batch_shape)
        if np.any((break_value <= 0) | (break_value >= t2)):
            raise ValueError("change_time must be positive and earlier than the second time")
        before_break = t1 < break_value
        h1_early = h1cum / t1
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            h2_early = (drop + h1cum * ((t1 - break_value) / t1)) / (t2 - break_value)
            h2_late = drop / (t2 - t1)
            h1_late = (h1cum - h2_late * (t1 - break_value)) / break_value
        h1 = np.where(before_break, h1_early, h1_late)
        h2 = np.where(before_break, h2_early, h2_late)
        scale = np.maximum(np.abs(h1), np.abs(h2))
        near_equal = np.abs(h1 - h2) <= 64 * _EPS * scale
        average_hazard = h1 / 2 + h2 / 2
        h1 = np.where(near_equal, average_hazard, h1)
        h2 = np.where(near_equal, average_hazard, h2)
        h1 = _nonnegative_rate(h1, scale, "hazard_before")
        h2 = _nonnegative_rate(h2, scale, "hazard_after")
        identified = ~near_equal
    else:
        t1, t2, t3 = (time_values[..., i] for i in range(3))
        h1cum, h2cum, h3cum = (cumulative[..., i] for i in range(3))
        s1, s2, s3 = (survival_values[..., i] for i in range(3))
        h1 = h1cum / t1
        h2 = _log_drop(s2, s3) / (t3 - t2)
        scale = np.maximum(np.abs(h1), np.abs(h2))
        near_equal = np.abs(h1 - h2) <= 64 * _EPS * scale
        average_hazard = h1 / 2 + h2 / 2
        h1 = np.where(near_equal, average_hazard, h1)
        h2 = np.where(near_equal, average_hazard, h2)
        denominator = h1 - h2
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            inferred = (h2cum - h2 * t2) / denominator
        midpoint = t1 + (t2 - t1) / 2
        break_value = np.where(near_equal, midpoint, inferred)
        identified = ~near_equal
        if np.any(~np.isfinite(break_value) | (break_value < t1) | (break_value > t2)):
            raise ValueError("survival points do not define a feasible two-hazard model")
        scale = np.maximum(np.abs(h1), np.abs(h2))
        h1 = _nonnegative_rate(h1, scale, "hazard_before")
        h2 = _nonnegative_rate(h2, scale, "hazard_after")

    predicted = h1[..., None] * np.minimum(time_values, break_value[..., None]) + h2[
        ..., None
    ] * np.maximum(time_values - break_value[..., None], 0)
    residual_scale = np.maximum(
        np.maximum(np.abs(cumulative), np.abs(predicted)), np.finfo(float).tiny
    )
    if np.any(
        ~np.isfinite(predicted) | (np.abs(predicted - cumulative) > 256 * _EPS * residual_scale)
    ):
        raise ValueError("inferred hazards do not reconstruct the supplied survival points")
    return STPLANPiecewiseModel(
        _freeze(h1),
        _freeze(h2),
        _freeze(break_value),
        _readonly_bool(np.asarray(identified, dtype=bool)),
    )
