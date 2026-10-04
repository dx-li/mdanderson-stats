"""Source-defined CiBolus scenario truths for simulation and calibration."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray
from .cibolus import _prediction_inputs

_MAX_RETAINED_CELLS = 2_000_000
_CURVES = ("linear", "below_linear", "above_linear", "s_shaped")


def _freeze_dtype(value: ArrayLike, dtype: np.dtype) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _joint_grid(value: ArrayLike, expected: tuple[int, ...]) -> FloatArray:
    """Validate, normalize within roundoff, and freeze a joint probability grid."""
    if int(np.prod(expected, dtype=np.int64)) > _MAX_RETAINED_CELLS:
        raise ValueError("joint probability grid exceeds the retained-cell limit")
    if isinstance(value, (list, tuple)):

        def check_nested(item: object, axis: int) -> None:
            if isinstance(item, np.ndarray):
                if item.shape != expected[axis:]:
                    raise ValueError(
                        f"joint probabilities must have shape {expected} and real values"
                    )
                return
            if axis == len(expected):
                if isinstance(item, (list, tuple)):
                    raise ValueError(
                        f"joint probabilities must have shape {expected} and real values"
                    )
                return
            if not isinstance(item, (list, tuple)) or len(item) != expected[axis]:
                raise ValueError(f"joint probabilities must have shape {expected} and real values")
            for child in item:
                check_nested(child, axis + 1)

        check_nested(value, 0)
        raw = np.asarray(value)
    elif isinstance(value, np.ndarray):
        raw = value
    else:
        size = getattr(value, "size", None)
        if size is not None and int(size) > _MAX_RETAINED_CELLS:
            raise ValueError("joint probability grid exceeds the retained-cell limit")
        try:
            length = len(value)  # type: ignore[arg-type]
        except TypeError:
            length = None
        if length is not None and length > _MAX_RETAINED_CELLS:
            raise ValueError("joint probability grid exceeds the retained-cell limit")
        raw = np.asarray(value)
    if raw.size > _MAX_RETAINED_CELLS:
        raise ValueError("joint probability grid exceeds the retained-cell limit")
    if raw.dtype.kind not in "iuf" or raw.shape != expected:
        raise ValueError(f"joint probabilities must have shape {expected} and real values")
    result = np.array(raw, dtype=float, copy=True)
    if np.any(~np.isfinite(result)) or np.any(result < 0):
        raise ValueError("joint probabilities must be finite and nonnegative")
    totals = result.sum(axis=(-2, -1))
    if np.any(np.abs(totals - 1.0) > 1e-12):
        raise ValueError("joint probabilities must sum to one for every regimen")
    result /= totals[..., None, None]
    return _freeze_dtype(result, np.dtype(float))


def _regimen_probability(value: ArrayLike, name: str, shape: tuple[int, int]) -> FloatArray:
    if isinstance(value, np.ndarray):
        if value.ndim > 2 or value.size > 400:
            raise ValueError(f"{name} must be bounded real probability data")
        raw = value
    elif isinstance(value, (list, tuple)):

        def inspect(item: object, depth: int) -> tuple[tuple[int, ...], int]:
            if depth > 2:
                raise ValueError(f"{name} must be scalar or at most two-dimensional")
            if isinstance(item, np.ndarray):
                if item.ndim + depth > 2 or item.size > 400:
                    raise ValueError(f"{name} must be bounded real probability data")
                return item.shape, int(item.size)
            if not isinstance(item, (list, tuple)):
                return (), 1
            if len(item) > 400:
                raise ValueError(f"{name} must be bounded real probability data")
            child_shapes = [inspect(child, depth + 1) for child in item]
            if child_shapes and any(part[0] != child_shapes[0][0] for part in child_shapes):
                raise ValueError(f"{name} must be a rectangular broadcastable array")
            cells = sum(part[1] for part in child_shapes)
            if cells > 400:
                raise ValueError(f"{name} must be bounded real probability data")
            return (len(item),) + (child_shapes[0][0] if child_shapes else ()), cells

        inspect(value, 0)
        raw = np.asarray(value)
    elif np.isscalar(value):
        raw = np.asarray(value)
    else:
        raise ValueError(f"{name} must be scalar, ndarray, list, or tuple data")
    if raw.dtype.kind not in "iuf" or raw.size > 400:
        raise ValueError(f"{name} must be bounded real probability data")
    try:
        result = np.broadcast_to(np.asarray(raw, dtype=float), shape)
    except ValueError as exc:
        raise ValueError(f"{name} must broadcast to regimen shape {shape}") from exc
    if np.any(~np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return np.asarray(result, dtype=float)


def _curve_increment(lower: float, upper: float, curve: str) -> float:
    """Stable difference g(upper)-g(lower) for the paper's four profiles."""
    width = upper - lower
    if curve == "linear":
        return width
    if curve == "below_linear":
        return width * (upper + lower)
    if curve == "above_linear":
        return width / (np.sqrt(upper) + np.sqrt(lower)) if width else 0.0
    if upper <= 0.5:
        return 2.0 * width * (upper + lower)
    if lower >= 0.5:
        return width / (np.sqrt(2 * upper - 1) + np.sqrt(2 * lower - 1)) if width else 0.0
    left = 2.0 * (0.5 - lower) * (0.5 + lower)
    right = 0.5 * np.sqrt(2 * upper - 1)
    return left + right


def _validate_curve(curve: str, name: str) -> str:
    if not isinstance(curve, str) or curve not in _CURVES:
        raise ValueError(f"{name} must be one of {_CURVES}")
    return curve


def cibolus_interpolated_truth(
    concentrations: ArrayLike,
    bolus_fractions: ArrayLike,
    endpoints: ArrayLike,
    *,
    response_zero: ArrayLike,
    response_one: ArrayLike,
    toxicity_zero: ArrayLike,
    toxicity_one: ArrayLike,
    toxicity_failure: ArrayLike,
    response_curve: str = "linear",
    toxicity_curve: str = "linear",
) -> FloatArray:
    """Build regimen-specific joint interval-response/toxicity probabilities.

    The response probabilities at standardized times zero and one are
    cumulative probabilities, including the bolus atom at zero. Each regimen
    value may be scalar or broadcastable to ``(len(concentrations),
    len(bolus_fractions))``. Toxicity probabilities are conditional on the
    corresponding response category; failure toxicity is supplied separately.
    The returned axes are concentration, bolus, response category, toxicity
    (0 then 1). Categories are bolus, endpoint intervals, and failure.
    """
    curve_e = _validate_curve(response_curve, "response_curve")
    curve_t = _validate_curve(toxicity_curve, "toxicity_curve")
    endpoint_raw = np.asarray(endpoints)
    if endpoint_raw.size > 20:
        raise ValueError("endpoints must contain at most 20 values")
    endpoint_count = endpoint_raw.size
    dummy_utility = np.zeros((endpoint_count + 2, 2))
    concentrations_value, bolus_value, endpoint_value, _ = _prediction_inputs(
        concentrations, bolus_fractions, endpoints, dummy_utility
    )
    regimen_shape = (concentrations_value.size, bolus_value.size)
    p0 = _regimen_probability(response_zero, "response_zero", regimen_shape)
    p1 = _regimen_probability(response_one, "response_one", regimen_shape)
    t0 = _regimen_probability(toxicity_zero, "toxicity_zero", regimen_shape)
    t1 = _regimen_probability(toxicity_one, "toxicity_one", regimen_shape)
    tf = _regimen_probability(toxicity_failure, "toxicity_failure", regimen_shape)
    if np.any(p1 < p0):
        raise ValueError("response_one must be at least response_zero for every regimen")

    categories = endpoint_value.size + 2
    joint = np.empty((*regimen_shape, categories, 2), dtype=float)
    joint[..., 0, 1] = p0 * t0
    joint[..., 0, 0] = p0 * (1.0 - t0)

    previous = 0.0
    response_width = p1 - p0
    for index, endpoint in enumerate(endpoint_value):
        upper = float(endpoint)
        mass = response_width * _curve_increment(previous, upper, curve_e)
        toxicity_fraction = _curve_increment(0.0, upper, curve_t)
        toxicity = (1.0 - toxicity_fraction) * t0 + toxicity_fraction * t1
        nontoxicity = (1.0 - toxicity_fraction) * (1.0 - t0) + toxicity_fraction * (1.0 - t1)
        cell = index + 1
        joint[..., cell, 1] = mass * toxicity
        joint[..., cell, 0] = mass * nontoxicity
        previous = upper

    failure_mass = 1.0 - p1
    joint[..., -1, 1] = failure_mass * tf
    joint[..., -1, 0] = failure_mass * (1.0 - tf)
    return _joint_grid(joint, (*regimen_shape, categories, 2))
