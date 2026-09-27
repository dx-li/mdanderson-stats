"""Triangular single-dose hazards and variable-dose schedule likelihoods."""

from dataclasses import dataclass
from math import isfinite, log

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from ._cdflib import _freeze
from ._validation import FloatArray

_MAX_DOSES = 20
_MAX_ADMINISTRATIONS = 10_000


def _numeric(value: ArrayLike, name: str, *, max_size: int = _MAX_ADMINISTRATIONS) -> FloatArray:
    raw = np.asarray(value)
    if raw.size > max_size or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be bounded real numeric data")
    result = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _broadcast(values: tuple[ArrayLike, ...], names: tuple[str, ...]) -> tuple[FloatArray, ...]:
    arrays = []
    for value, name in zip(values, names, strict=True):
        raw = np.asarray(value)
        if raw.size > _MAX_ADMINISTRATIONS or raw.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be bounded real numeric data")
        arrays.append(raw)
    try:
        shape = np.broadcast_shapes(*(array.shape for array in arrays))
    except ValueError as exc:
        raise ValueError("hazard inputs must be broadcast-compatible") from exc
    if int(np.prod(shape, dtype=object)) > _MAX_ADMINISTRATIONS:
        raise ValueError("broadcast hazard result exceeds the bounded size")
    converted = []
    for array, name in zip(arrays, names, strict=True):
        value = np.asarray(array, dtype=float)
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite")
        converted.append(np.broadcast_to(value, shape))
    return tuple(converted)


def _positive_parameters(
    area: FloatArray, peak_time: FloatArray, tail_time: FloatArray
) -> FloatArray:
    if np.any(area <= 0) or np.any(peak_time <= 0) or np.any(tail_time <= 0):
        raise ValueError("area, peak_time and tail_time must be positive")
    with np.errstate(over="ignore"):
        total = peak_time + tail_time
    if not np.all(np.isfinite(total)):
        raise ArithmeticError("peak_time + tail_time is not representable")
    return total


def _exp_log(log_value: FloatArray, name: str) -> FloatArray:
    if np.any(log_value > np.log(np.finfo(float).max)):
        raise ArithmeticError(f"{name} exceeds floating-point range")
    with np.errstate(under="ignore"):
        value = np.exp(log_value)
    if not np.all(np.isfinite(value)):
        raise ArithmeticError(f"{name} is not representable")
    return value


def _log_hazard(
    elapsed: FloatArray, area: FloatArray, peak_time: FloatArray, tail_time: FloatArray
) -> FloatArray:
    total = _positive_parameters(area, peak_time, tail_time)
    answer = np.full(elapsed.shape, -np.inf, dtype=float)
    rising = (elapsed > 0) & (elapsed <= peak_time)
    falling = (elapsed > peak_time) & (elapsed < total)
    if np.any(rising):
        answer[rising] = (
            log(2)
            + np.log(area[rising])
            - np.log(total[rising])
            + np.log(elapsed[rising])
            - np.log(peak_time[rising])
        )
    if np.any(falling):
        remaining = total[falling] - elapsed[falling]
        answer[falling] = (
            log(2)
            + np.log(area[falling])
            - np.log(total[falling])
            + np.log(remaining)
            - np.log(tail_time[falling])
        )
    if np.any(answer > np.log(np.finfo(float).max)):
        raise ArithmeticError("single-dose hazard exceeds floating-point range")
    return answer


def dose_schedule_hazard(
    elapsed: ArrayLike, area: ArrayLike, peak_time: ArrayLike, tail_time: ArrayLike
) -> FloatArray:
    """Evaluate the paper's triangular single-administration hazard.

    Hazard is zero before dosing, rises linearly to ``peak_time`` and falls
    linearly to zero at ``peak_time + tail_time``. Inputs broadcast to at most
    10,000 output elements.
    """
    t, a, b, c = _broadcast(
        (elapsed, area, peak_time, tail_time),
        ("elapsed", "area", "peak_time", "tail_time"),
    )
    log_hazard = _log_hazard(t, a, b, c)
    with np.errstate(under="ignore"):
        answer = np.exp(log_hazard)
    return _freeze(answer)


def dose_schedule_cumulative_hazard(
    elapsed: ArrayLike, area: ArrayLike, peak_time: ArrayLike, tail_time: ArrayLike
) -> FloatArray:
    """Evaluate integrated triangular hazard from an administration to elapsed time."""
    t, a, b, c = _broadcast(
        (elapsed, area, peak_time, tail_time),
        ("elapsed", "area", "peak_time", "tail_time"),
    )
    total = _positive_parameters(a, b, c)
    answer = np.zeros(t.shape, dtype=float)
    rising = (t > 0) & (t <= b)
    falling = (t > b) & (t < total)
    saturated = t >= total
    if np.any(rising):
        log_h = (
            np.log(a[rising]) + 2 * np.log(t[rising]) - np.log(b[rising]) - np.log(total[rising])
        )
        answer[rising] = _exp_log(log_h, "cumulative hazard")
    if np.any(falling):
        elapsed_after_peak = t[falling] - b[falling]
        ratio = elapsed_after_peak / c[falling]
        if np.any(ratio > 1 + 8 * np.finfo(float).eps):
            raise ArithmeticError("falling-segment time ratio exceeds its physical range")
        ratio = np.minimum(ratio, 1.0)
        log_ratio = np.log(elapsed_after_peak) - np.log(c[falling])
        log_base = np.log(a[falling]) + np.log(b[falling]) - np.log(total[falling])
        log_increment = (
            np.log(a[falling])
            + np.log(c[falling])
            - np.log(total[falling])
            + log_ratio
            + np.log(2 - ratio)
        )
        log_h = np.logaddexp(log_base, log_increment)
        answer[falling] = _exp_log(log_h, "cumulative hazard")
    answer[saturated] = a[saturated]
    if not np.all(np.isfinite(answer)) or np.any(answer < 0):
        raise ArithmeticError("cumulative hazard is not representable")
    return _freeze(answer)


@dataclass(frozen=True)
class DoseSchedulePatient:
    """One observed event/censoring time and the actual past administrations."""

    time: float
    event: bool
    administration_times: FloatArray
    dose_indices: NDArray[np.int64]

    def __init__(
        self,
        time: float,
        event: bool,
        administration_times: ArrayLike,
        dose_indices: ArrayLike,
    ) -> None:
        raw_time = np.asarray(time)
        if raw_time.ndim != 0 or raw_time.dtype.kind not in "iuf":
            raise ValueError("time must be a finite nonnegative scalar")
        observed_time = float(raw_time)
        if not isfinite(observed_time) or observed_time < 0:
            raise ValueError("time must be a finite nonnegative scalar")
        if not isinstance(event, (bool, np.bool_)):
            raise ValueError("event must be boolean")
        times = _numeric(administration_times, "administration_times")
        raw_doses = np.asarray(dose_indices)
        if (
            times.ndim != 1
            or raw_doses.ndim != 1
            or times.size != raw_doses.size
            or raw_doses.dtype.kind not in "iu"
            or raw_doses.dtype.kind == "b"
        ):
            raise ValueError(
                "administration_times and integer dose_indices must be matching vectors"
            )
        doses = raw_doses.astype(np.int64, copy=False)
        if np.any(times < 0) or np.any(times > observed_time):
            raise ValueError("administrations must occur between time zero and observation time")
        if np.any(doses < 0) or np.any(doses >= _MAX_DOSES):
            raise ValueError("dose_indices must lie in [0,19]")
        object.__setattr__(self, "time", observed_time)
        object.__setattr__(self, "event", bool(event))
        object.__setattr__(self, "administration_times", _freeze(times))
        object.__setattr__(
            self,
            "dose_indices",
            np.frombuffer(np.ascontiguousarray(doses).tobytes(), dtype=np.int64),
        )


def dose_schedule_patient_loglikelihood(
    patient: DoseSchedulePatient,
    areas: ArrayLike,
    peak_times: ArrayLike,
    tail_times: ArrayLike,
) -> float:
    """Return one patient's survival/event log likelihood under actual dose history."""
    if not isinstance(patient, DoseSchedulePatient):
        raise ValueError("patient must be a DoseSchedulePatient")
    a = _numeric(areas, "areas", max_size=_MAX_DOSES)
    b = _numeric(peak_times, "peak_times", max_size=_MAX_DOSES)
    c = _numeric(tail_times, "tail_times", max_size=_MAX_DOSES)
    if a.ndim != 1 or b.shape != a.shape or c.shape != a.shape or not 1 <= a.size <= _MAX_DOSES:
        raise ValueError("areas, peak_times and tail_times must be matching dose vectors")
    if np.any(patient.dose_indices >= a.size):
        raise ValueError("patient contains a dose index outside the parameter grid")
    _positive_parameters(a, b, c)
    elapsed = patient.time - patient.administration_times
    cumulative = dose_schedule_cumulative_hazard(
        elapsed, a[patient.dose_indices], b[patient.dose_indices], c[patient.dose_indices]
    )
    total_hazard = float(np.sum(cumulative, dtype=float))
    if not isfinite(total_hazard):
        raise ArithmeticError("patient cumulative or event hazard exceeds floating-point range")
    if not patient.event:
        return -total_hazard
    log_hazards = _log_hazard(
        elapsed, a[patient.dose_indices], b[patient.dose_indices], c[patient.dose_indices]
    )
    log_event_hazard = float(logsumexp(log_hazards))
    if not isfinite(log_event_hazard):
        return -np.inf
    return float(log_event_hazard - total_hazard)


def _subsequence(earlier: NDArray, later: NDArray) -> bool:
    index = 0
    for value in later:
        if index < earlier.size and value == earlier[index]:
            index += 1
    return index == earlier.size


def dose_schedule_parameter_names(
    dose_count: int, *, ordered_areas: bool = True
) -> tuple[str, ...]:
    """Parameter order is per dose: log area, log peak time, log tail time."""
    raw = np.asarray(dose_count)
    if raw.ndim != 0 or raw.dtype.kind not in "iu" or isinstance(dose_count, (bool, np.bool_)):
        raise ValueError("dose_count must be an integer in [1,20]")
    count = int(raw)
    if not 1 <= count <= _MAX_DOSES:
        raise ValueError("dose_count must be an integer in [1,20]")
    if not isinstance(ordered_areas, (bool, np.bool_)):
        raise ValueError("ordered_areas must be boolean")
    result: list[str] = []
    for dose in range(count):
        area_name = "log_area_increment" if ordered_areas else "log_area"
        result.extend((f"{area_name}.{dose + 1}", f"log_peak.{dose + 1}", f"log_tail.{dose + 1}"))
    return tuple(result)


def validate_schedules(schedules: object, horizon: float) -> tuple[FloatArray, ...]:
    """Validate a nested family of increasing administration-time sequences."""
    if isinstance(schedules, np.ndarray) or not isinstance(schedules, (tuple, list)):
        raise ValueError("schedules must be a sequence of administration-time vectors")
    if not 1 <= len(schedules) <= 20:
        raise ValueError("provide 1..20 nested candidate schedules")
    rows: list[FloatArray] = []
    total = 0
    for index, schedule in enumerate(schedules):
        row = _numeric(schedule, f"schedules[{index}]", max_size=20)
        if row.ndim != 1 or not 1 <= row.size <= 20:
            raise ValueError("each candidate schedule must contain 1..20 administration times")
        if np.any(row < 0) or np.any(np.diff(row) <= 0) or np.any(row >= horizon):
            raise ValueError("schedule times must be strictly increasing in [0,horizon)")
        if rows and (row.size <= rows[-1].size or not _subsequence(rows[-1], row)):
            raise ValueError(
                "schedule lengths must increase and each schedule must contain its predecessor"
            )
        rows.append(row)
        total += row.size
    if total > 200:
        raise ValueError("total candidate schedule administrations exceed 200")
    return tuple(_freeze(row) for row in rows)
