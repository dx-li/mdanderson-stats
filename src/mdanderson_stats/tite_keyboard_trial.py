"""Calendar-time trial conduct from explicit arrival gaps and potential DLT delays."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite, scalar
from .boin import _owned
from .keyboard import KeyboardDesign
from .tite_keyboard import TITEKeyboardDecision, tite_keyboard_decision, toxicity_followup_weights


@dataclass(frozen=True)
class TITEKeyboardStep:
    time: float
    current_dose: int
    decision: TITEKeyboardDecision


@dataclass(frozen=True)
class TITEKeyboardTrial:
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
    steps: tuple[TITEKeyboardStep, ...]


def _advance(time: float, gap: float) -> float:
    result = time + gap
    if not np.isfinite(result) or (gap > 0 and result <= time):
        raise ArithmeticError("calendar time is not representable; rescale the time units")
    return result


def run_tite_keyboard_trial(
    design: KeyboardDesign,
    interarrival: ArrayLike,
    dlt_delays: ArrayLike,
    window: float,
    *,
    cohort_size: int = 3,
    start_dose: int = 1,
    trimester_probabilities: ArrayLike | None = None,
    pending_fraction_limit: float | None = 0.5,
) -> TITEKeyboardTrial:
    """Conduct one trial from potential outcomes without exposing future events.

    dlt_delays[patient,dose] is time since enrollment to DLT, or +inf for no DLT
    within the window. Arrival gaps restart after suspensions; no queue builds up.
    Cohorts have a fixed dose and staggered enrollment. Decisions occur before
    each new cohort, and at outcome ascertainments while that cohort is waiting.
    """
    if not isinstance(design, KeyboardDesign):
        raise ValueError("design must be a KeyboardDesign")
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
    toxicity_followup_weights([], duration, trimester_probabilities=trimester_probabilities)
    if pending_fraction_limit is not None:
        limit = scalar(pending_fraction_limit, "pending_fraction_limit")
        if not 0 < limit <= 0.65:
            raise ValueError("pending_fraction_limit must be in (0,.65] or None")
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
                decision = tite_keyboard_decision(
                    design,
                    n,
                    y,
                    times,
                    current,
                    duration,
                    trimester_probabilities=trimester_probabilities,
                    pending_fraction_limit=pending_fraction_limit,
                    eliminated=excluded,
                )
                steps.append(TITEKeyboardStep(clock, current, decision))
                excluded = decision.eliminated
                if decision.action.startswith("suspend_"):
                    future = known_at[:total][pending]
                    if not future.size:
                        raise RuntimeError("suspension cannot resolve without a pending outcome")
                    clock = float(future.min())
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
    return TITEKeyboardTrial(
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
