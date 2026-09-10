"""Calendar-time trial conduct from explicit arrival gaps and potential DLT delays."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite, scalar
from .boin import BOINDesign, _owned
from .keyboard import KeyboardDesign


class InterimDecision(Protocol):
    @property
    def action(self) -> str: ...
    @property
    def next_dose(self) -> int | None: ...
    @property
    def eliminated(self) -> NDArray[np.bool_]: ...


@dataclass(frozen=True)
class CalendarStep[D]:
    time: float
    current_dose: int
    decision: D


@dataclass(frozen=True)
class CalendarTrial[S]:
    enrollment_times: FloatArray
    assigned_doses: NDArray[np.int64]
    dlt_times: FloatArray
    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    selected_dose: int | None
    eliminated: NDArray[np.bool_]
    stop_reason: str
    final_time: float
    suspension_time: float
    steps: tuple[S, ...]


def _advance(time: float, gap: float) -> float:
    result = time + gap
    if not np.isfinite(result) or (gap > 0 and result <= time):
        raise ArithmeticError("calendar time is not representable; rescale the time units")
    return result


def run_calendar_trial[D: InterimDecision, S](
    design: BOINDesign | KeyboardDesign,
    interarrival: ArrayLike,
    dlt_delays: ArrayLike,
    window: float,
    cohort_size: int,
    start_dose: int,
    decide: Callable[[ArrayLike, ArrayLike, list[FloatArray], int, ArrayLike], D],
    make_step: Callable[[float, int, D], S],
    minimum_pending_followup: float = 0,
) -> CalendarTrial[S]:
    """Fixed-dose staggered cohorts with outcome and follow-up event scheduling."""
    gaps = finite(interarrival, "interarrival")
    potential = np.asarray(dlt_delays, dtype=float)
    duration = scalar(window, "window")
    if gaps.ndim != 1 or not 1 <= gaps.size <= 200 or np.any(gaps < 0):
        raise ValueError("interarrival must be a vector of 1..200 nonnegative gaps")
    if (
        potential.ndim != 2
        or potential.shape[0] != gaps.size
        or not 2 <= potential.shape[1] <= 100
        or duration <= 0
        or np.any(
            ~(
                np.isposinf(potential)
                | (np.isfinite(potential) & (potential >= 0) & (potential <= duration))
            )
        )
    ):
        raise ValueError(
            "require potential delays shaped (patients,2..100 doses), in [0,window] or +inf"
        )
    settings = count([cohort_size, start_dose], "cohort settings")
    if settings.shape != (2,) or np.any(settings < 1) or settings[1] > potential.shape[1]:
        raise ValueError("require positive cohort size and a valid start dose")
    size, start = map(int, settings)
    if gaps.size % size:
        raise ValueError("planned enrollment must contain complete cohorts")
    levels = potential.shape[1]
    enrolled = np.zeros(gaps.size)
    assigned = np.zeros(gaps.size, dtype=np.int64)
    events = np.full(gaps.size, np.inf)
    known_at = np.zeros(gaps.size)
    n = np.zeros(levels, dtype=np.int64)
    excluded = np.zeros(levels, dtype=bool)
    steps = []
    total = 0
    current = start
    clock = 0.0
    paused = 0.0
    stop = "max_patients"
    while total < len(gaps):
        ready = _advance(clock, float(gaps[total]))
        clock = ready
        if total:
            while True:
                observed = events[:total] <= clock
                pending = known_at[:total] > clock
                y = np.bincount(assigned[:total][observed] - 1, minlength=levels)
                times = [
                    clock - enrolled[:total][pending & (assigned[:total] == dose)]
                    for dose in range(1, levels + 1)
                ]
                decision = decide(n, y, times, current, excluded)
                steps.append(make_step(clock, current, decision))
                excluded = decision.eliminated
                if decision.action.startswith("suspend_"):
                    future = known_at[:total][pending]
                    if not future.size:
                        raise RuntimeError("suspension cannot resolve without a pending outcome")
                    next_time = float(future.min())
                    if decision.action == "suspend_followup":
                        current_pending = pending & (assigned[:total] == current)
                        latest = float(enrolled[:total][current_pending].max())
                        ready_followup = _advance(latest, minimum_pending_followup * duration)
                        # Round toward a time satisfying the actual conduct comparison.
                        while (ready_followup - latest) / duration < minimum_pending_followup:
                            ready_followup = float(np.nextafter(ready_followup, np.inf))
                        if ready_followup <= clock:
                            raise RuntimeError("follow-up suspension did not advance the calendar")
                        next_time = min(next_time, ready_followup)
                    clock = next_time
                    continue
                break
            paused += clock - ready
            if decision.next_dose is None:
                stop = decision.action
                break
            current = decision.next_dose
        for offset in range(size):
            if offset:
                clock = _advance(clock, float(gaps[total]))
            enrolled[total] = clock
            assigned[total] = current
            delay = potential[total, current - 1]
            end = _advance(clock, duration)
            if np.isfinite(delay):
                events[total] = _advance(clock, float(delay))
            known_at[total] = min(end, events[total])
            n[current - 1] += 1
            total += 1
    final_time = max(clock, float(known_at[:total].max()))
    final_y = np.bincount(assigned[:total][np.isfinite(events[:total])] - 1, minlength=levels)
    selected = design.select_mtd(n, final_y, eliminated=excluded)
    return CalendarTrial(
        _owned(enrolled[:total]),
        _owned(assigned[:total]),
        _owned(events[:total]),
        _owned(n),
        _owned(final_y),
        selected.dose,
        selected.eliminated,
        stop,
        final_time,
        paused,
        tuple(steps),
    )
