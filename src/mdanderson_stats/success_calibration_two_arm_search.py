"""Finite-state PID calibration for two-arm binary success decisions."""

from __future__ import annotations

from bisect import bisect_right

import numpy as np
from numpy.typing import ArrayLike

from .success_calibration import (
    BinarySuccessTable,
    SuccessCalibration,
    prepare_binary_two_arm_success,
)


def _scalar(value: object, name: str) -> float:
    if isinstance(value, np.ndarray) and value.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    if isinstance(value, (list, tuple, complex, np.complexfloating)):
        raise ValueError(f"{name} must be a real scalar")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _pair(value: object, name: str) -> tuple[float, float]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.size != 2:
            raise ValueError(f"{name} must contain exactly two values")
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must contain real values")
        first: object = value[0]
        second: object = value[1]
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        first, second = value
    else:
        raise ValueError(f"{name} must contain exactly two values")
    return (_scalar(first, name), _scalar(second, name))


def _compensated_suffix(values: np.ndarray) -> np.ndarray:
    """Suffix sums of nonnegative values with Kahan compensation."""
    result = np.empty(values.size, dtype=np.float64)
    total = 0.0
    compensation = 0.0
    for index in range(values.size - 1, -1, -1):
        adjusted = float(values[index]) - compensation
        updated = total + adjusted
        compensation = (updated - total) - adjusted
        total = updated
        result[index] = total
    return result


def _table_arrays(table: BinarySuccessTable) -> tuple[np.ndarray, ...]:
    if not isinstance(table, BinarySuccessTable):
        raise TypeError("table must be a BinarySuccessTable")
    raw_arrays = (
        table.posterior_probability,
        table.posterior_error,
        table.effective_mass,
        table.ineffective_mass,
        table.null_mass,
    )
    if any(not isinstance(value, np.ndarray) for value in raw_arrays):
        raise TypeError("binary success table fields must be NumPy arrays")
    if any(np.iscomplexobj(value) for value in raw_arrays):
        raise ValueError("binary success table arrays must be real-valued")
    shape = raw_arrays[0].shape
    if not shape or any(value.shape != shape for value in raw_arrays):
        raise ValueError("binary success table arrays must have matching non-scalar shapes")
    if raw_arrays[0].size == 0 or raw_arrays[0].size > 40_000:
        raise ValueError("binary success table must contain between 1 and 40000 states")
    if any(value.dtype.kind not in "iuf" for value in raw_arrays):
        raise ValueError("binary success table arrays must have real numeric dtypes")
    arrays = tuple(value.astype(np.float64, copy=False) for value in raw_arrays)
    pa, error, effective, ineffective, null = (value.ravel() for value in arrays)
    if (
        not all(np.all(np.isfinite(value)) for value in arrays)
        or np.any((pa < 0) | (pa > 1))
        or np.any(error < 0)
        or any(np.any(value < 0) for value in (effective, ineffective, null))
    ):
        raise ValueError("binary success table contains invalid probabilities or masses")
    return pa, error, effective, ineffective, null


def _merge_error_intervals(pa: np.ndarray, error: np.ndarray) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for probability, error_bound in zip(pa, error, strict=True):
        if error_bound <= 0:
            continue
        widened = np.nextafter(min(float(error_bound), 1.0), np.inf)
        lower = max(0.0, float(np.nextafter(probability - widened, -np.inf)))
        upper = min(1.0, float(np.nextafter(probability + widened, np.inf)))
        intervals.append((lower, upper))
    intervals.sort()
    merged: list[tuple[float, float]] = []
    for lower, upper in intervals:
        if merged and lower <= np.nextafter(merged[-1][1], np.inf):
            merged[-1] = (merged[-1][0], max(merged[-1][1], upper))
        else:
            merged.append((lower, upper))
    return merged


def _inside_intervals(value: float, intervals: list[tuple[float, float]]) -> bool:
    index = bisect_right(intervals, (value, 1.0)) - 1
    return index >= 0 and value <= intervals[index][1]


def _search_table(
    table: BinarySuccessTable,
    target: float,
    cutoff_range: ArrayLike,
) -> SuccessCalibration:
    target_value = _scalar(target, "target")
    if not 0 < target_value < 1:
        raise ValueError("target must be in (0,1)")
    lower, upper = _pair(cutoff_range, "cutoff_range")
    if not 0 <= lower <= 1 or not 0 <= upper <= 1:
        raise ValueError("cutoff_range must contain two values in [0,1]")
    if lower > upper:
        raise ValueError("cutoff_range lower bound must not exceed upper bound")

    pa, error, effective, ineffective, _null = _table_arrays(table)

    # The existing table deliberately defines c=0 separately: a positive
    # integration-error estimate makes the state count as potentially positive.
    # Preserve that established behavior at the requested lower endpoint.
    candidates: set[float] = set()
    if lower == 0.0:
        at_zero = table.evaluate(0.0)
        if at_zero.bayesian_power > 0:
            pid = at_zero.incorrect_decision_probability
            if pid is not None and pid <= target_value:
                return SuccessCalibration(0.0, target_value, at_zero, 1)
        if upper == 0.0:
            raise ValueError(
                "no cutoff in cutoff_range achieves the PID target with "
                "positive success probability"
            )
        first_count = 1
    else:
        first_count = 0

    if lower == upper == 1.0:
        # The evaluator treats c=1 as a defined endpoint with no strict successes.
        table.evaluate(1.0)
        raise ValueError(
            "no cutoff in cutoff_range achieves the PID target with positive success probability"
        )

    intervals = _merge_error_intervals(pa, error)
    unresolved = any(right >= lower and left <= upper for left, right in intervals)

    if first_count == 0 and not _inside_intervals(lower, intervals):
        candidates.add(lower)

    # Exact zero-error probabilities are safe strict breakpoints. At equality,
    # the state is excluded because the decision is posterior probability > c.
    exact_breakpoints = np.unique(pa[error == 0])
    for point in exact_breakpoints:
        candidate = float(point)
        if lower < candidate <= upper and not _inside_intervals(candidate, intervals):
            candidates.add(candidate)

    # Within a positive-error interval the existing table refuses evaluation.
    # The first representable float above its right edge gives a conservative
    # candidate for the state on the right.
    for _, right in intervals:
        if right >= lower:
            candidate = float(np.nextafter(right, np.inf))
            if lower <= candidate <= upper and not _inside_intervals(candidate, intervals):
                candidates.add(candidate)

    if not candidates:
        if unresolved:
            raise ArithmeticError(
                "no numerically distinguishable cutoff is available; posterior error intervals "
                "leave mathematical feasibility unresolved"
            )
        raise ValueError(
            "no cutoff in cutoff_range achieves the PID target with positive success probability"
        )

    order = np.argsort(pa, kind="mergesort")
    sorted_pa = pa[order]
    tp_suffix = _compensated_suffix(effective[order])
    fp_suffix = _compensated_suffix(ineffective[order])

    target_boundary_disagreement = False
    for evaluated, candidate in enumerate(sorted(candidates), start=first_count + 1):
        # This search needs PID and positive success mass only. The selected
        # cutoff's full metrics are recomputed by the authoritative table below.
        start = int(np.searchsorted(sorted_pa, candidate, side="right"))
        tp = float(tp_suffix[start]) if start < sorted_pa.size else 0.0
        fp = float(fp_suffix[start]) if start < sorted_pa.size else 0.0
        success_mass = tp + fp
        if success_mass <= 0:
            continue
        pid = fp / success_mass
        if pid <= target_value:
            result = table.evaluate(candidate)
            final_pid = result.incorrect_decision_probability
            if result.bayesian_power <= 0 or final_pid is None:
                raise ArithmeticError("selected cutoff has unresolved success probability")
            if final_pid > target_value:
                # Compensated state sums and the table's ordinary reduction may
                # differ at a target boundary; never return a contradictory result.
                target_boundary_disagreement = True
                continue
            return SuccessCalibration(candidate, target_value, result, evaluated)

    if target_boundary_disagreement:
        raise ArithmeticError(
            "compensated search sums and final table evaluation disagree at the PID target"
        )
    if unresolved:
        raise ArithmeticError(
            "no numerically distinguishable cutoff met the target; posterior error intervals "
            "leave some decision states unresolved, so mathematical infeasibility is unknown"
        )
    raise ValueError(
        "no cutoff in cutoff_range achieves the PID target with positive success probability"
    )


def calibrate_binary_two_arm_success_cutoff(
    n_treatment: int,
    n_control: int,
    target: float,
    *,
    cutoff_range: tuple[float, float] = (0.6, 0.999),
    design_treatment: tuple[float, float] = (1.0, 1.0),
    design_control: tuple[float, float] = (1.0, 1.0),
    analysis_treatment: tuple[float, float] = (1.0, 1.0),
    analysis_control: tuple[float, float] = (1.0, 1.0),
    null_rate: float = 0.5,
    margin: float = 0.0,
    null_treatment_rate: float | None = None,
    direction: str = "greater",
    absolute_tolerance: float = 1e-10,
) -> SuccessCalibration:
    """Calibrate two-arm binary posterior cutoffs over distinguishable states.

    Success uses the strict comparison ``posterior_probability > cutoff``.
    The search checks every conservative error-separated decision state and
    selects its smallest feasible cutoff; it does not assume PID is monotone.
    This is an explicit Python search convention, not native application
    optimizer parity.
    """
    if isinstance(n_treatment, (bool, np.bool_)) or isinstance(n_control, (bool, np.bool_)):
        raise ValueError("arm sizes must be integers in [1,1000]")
    nt, nc = _scalar(n_treatment, "n_treatment"), _scalar(n_control, "n_control")
    if any(
        isinstance(value, (bool, np.bool_)) or value != np.floor(value) or not 1 <= value <= 1000
        for value in (nt, nc)
    ):
        raise ValueError("arm sizes must be integers in [1,1000]")
    nt, nc = int(nt), int(nc)
    if (nt + 1) * (nc + 1) > 40_000:
        raise ValueError("two-arm enumeration is limited to 40000 response-count pairs")

    target_value = _scalar(target, "target")
    if not 0 < target_value < 1:
        raise ValueError("target must be in (0,1)")
    bounds = _pair(cutoff_range, "cutoff_range")
    if any(not 0 <= value <= 1 for value in bounds):
        raise ValueError("cutoff_range must contain two values in [0,1]")
    if bounds[0] > bounds[1]:
        raise ValueError("cutoff_range lower bound must not exceed upper bound")
    delta = _scalar(margin, "margin")
    if not -1 <= delta <= 1:
        raise ValueError("margin must be in [-1,1]")
    null = _scalar(null_rate, "null_rate")
    null_t = (
        null if null_treatment_rate is None else _scalar(null_treatment_rate, "null_treatment_rate")
    )
    if not 0 <= null <= 1 or not 0 <= null_t <= 1:
        raise ValueError("null rates must be in [0,1]")
    if direction not in ("greater", "less"):
        raise ValueError("direction must be greater or less")
    tolerance = _scalar(absolute_tolerance, "absolute_tolerance")
    if not 1e-12 <= tolerance <= 1e-3:
        raise ValueError("absolute_tolerance must be in [1e-12,1e-3]")

    normalized_priors: list[tuple[float, float]] = []
    for prior, name in (
        (design_treatment, "design_treatment"),
        (design_control, "design_control"),
        (analysis_treatment, "analysis_treatment"),
        (analysis_control, "analysis_control"),
    ):
        shapes = _pair(prior, name)
        if any(value <= 0 for value in shapes):
            raise ValueError(f"{name} must contain two positive beta shape parameters")
        normalized_priors.append(shapes)

    table = prepare_binary_two_arm_success(
        nt,
        nc,
        design_treatment=normalized_priors[0],
        design_control=normalized_priors[1],
        analysis_treatment=normalized_priors[2],
        analysis_control=normalized_priors[3],
        null_rate=null,
        margin=delta,
        null_treatment_rate=null_t,
        direction=direction,
        absolute_tolerance=tolerance,
    )
    result = _search_table(table, target_value, (float(bounds[0]), float(bounds[1])))
    return result
