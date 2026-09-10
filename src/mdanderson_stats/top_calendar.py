"""Batched TOP calendar replay with suspended accrual and observed-data decisions."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite, scalar
from .boin import _owned
from .top_binary import TOPBinaryDesign


@dataclass(frozen=True)
class TOPCalendarStep:
    time: float
    patients: int
    responses: int
    pending: int
    effective_sample_size: float
    decision: str


@dataclass(frozen=True)
class TOPBinaryTrial:
    enrollment_times: FloatArray
    observed_response_times: FloatArray
    pending: NDArray[np.bool_]
    decision: str
    final_time: float
    suspension_time: float
    final_followup_time: float
    steps: tuple[TOPCalendarStep, ...]


@dataclass(frozen=True)
class TOPBinarySimulation:
    patients: NDArray[np.int64]
    responses: NDArray[np.int64]
    pending: NDArray[np.int64]
    decision: NDArray[np.str_]
    duration: FloatArray
    suspension_time: FloatArray
    final_followup_time: FloatArray
    success_probability: float
    success_mcse: float


def _add_time(time: FloatArray, delta: ArrayLike) -> FloatArray:
    delta_array = np.asarray(delta, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        result = time + delta_array
    if np.any(~np.isfinite(result) | ((delta_array > 0) & (result <= time))):
        raise ArithmeticError("calendar time is not representable; rescale time units")
    return result


def _run_top(
    design: TOPBinaryDesign,
    gaps: FloatArray,
    delays: FloatArray,
    window: float,
    *,
    record: bool = False,
) -> tuple[TOPBinarySimulation, FloatArray, FloatArray, tuple[TOPCalendarStep, ...]]:
    trials, maximum = gaps.shape
    clock = np.zeros(trials)
    enrolled = np.full(gaps.shape, np.nan)
    responses_at = np.full(gaps.shape, np.inf)
    known_at = np.full(gaps.shape, np.inf)
    patients, responses, pending = (np.zeros(trials, dtype=np.int64) for _ in range(3))
    alive = np.ones(trials, dtype=bool)
    actions = np.full(trials, "", dtype="<U13")
    pauses, final_wait = np.zeros(trials), np.zeros(trials)
    steps = []
    for j in range(maximum):
        rows = np.flatnonzero(alive)
        if not rows.size:
            break
        clock[rows] = _add_time(clock[rows], gaps[rows, j])
        enrolled[rows, j] = clock[rows]
        event_rows = rows[np.isfinite(delays[rows, j])]
        responses_at[event_rows, j] = _add_time(clock[event_rows], delays[event_rows, j])
        known_at[rows, j] = _add_time(clock[rows], np.minimum(delays[rows, j], window))
        n = j + 1
        patients[rows] = n
        if n not in np.asarray(design.looks):
            continue
        waiting = rows
        while waiting.size:
            event = responses_at[waiting, :n] <= clock[waiting, None]
            missing = known_at[waiting, :n] > clock[waiting, None]
            followup = np.clip((clock[waiting, None] - enrolled[waiting, :n]) / window, 0, 1)
            r, m = event.sum(1), missing.sum(1)
            weight = np.where(missing, followup, 0).sum(1)
            decision = design.evaluate(n, r, m, weight)
            responses[waiting], pending[waiting] = r, m
            if record:
                steps.append(
                    TOPCalendarStep(
                        float(clock[0]),
                        n,
                        int(r[0]),
                        int(m[0]),
                        float(decision.effective_sample_size[0]),
                        str(decision.decision[0]),
                    )
                )
            terminal = np.isin(decision.decision, ["stop_futility", "success"])
            done = waiting[terminal]
            actions[done] = decision.decision[terminal]
            alive[done] = False
            suspended = decision.decision == "suspend"
            waiting = waiting[suspended]
            if waiting.size:
                future = np.where(missing[suspended], known_at[waiting, :n], np.inf).min(1)
                if np.any(~np.isfinite(future) | (future <= clock[waiting])):
                    raise ArithmeticError("suspended TOP look has no later ascertainment event")
                elapsed = future - clock[waiting]
                if n == maximum:
                    final_wait[waiting] += elapsed
                else:
                    pauses[waiting] += elapsed
                clock[waiting] = future
    if alive.any() or np.any(actions == ""):
        raise RuntimeError("TOP trial did not reach a terminal decision")
    success = float(np.mean(actions == "success"))
    summary = TOPBinarySimulation(
        _owned(patients),
        _owned(responses),
        _owned(pending),
        _owned(actions),
        _owned(clock),
        _owned(pauses),
        _owned(final_wait),
        success,
        float(np.sqrt(success * (1 - success) / trials)),
    )
    observed_events = np.where(responses_at <= clock[:, None], responses_at, np.inf)
    return summary, enrolled, observed_events, tuple(steps)


def run_top_binary_trial(
    design: TOPBinaryDesign,
    interarrival: ArrayLike,
    response_delays: ArrayLike,
    window: float,
) -> TOPBinaryTrial:
    """Replay planned arrival gaps and potential response delays, without look-ahead.

    Vectors have max_subjects entries. A delay in [0,window] denotes response;
    +inf denotes no response, ascertained at window end. The first gap is measured
    from calendar time zero.
    After a suspended look resolves, the next arrival gap starts afresh (no queue).
    Duration ends at the terminal decision; pending follow-up after futility is
    not included. Final-window waiting is separate from interim suspension time.
    """
    gaps = finite(interarrival, "interarrival")
    delays = np.asarray(response_delays, dtype=float)
    duration = scalar(window, "window")
    if gaps.shape != (design.max_subjects,) or np.any(gaps < 0):
        raise ValueError("interarrival requires one nonnegative gap per planned patient")
    if (
        duration <= 0
        or delays.shape != gaps.shape
        or np.any(
            ~(np.isposinf(delays) | (np.isfinite(delays) & (delays >= 0) & (delays <= duration)))
        )
    ):
        raise ValueError("require positive window and response delays in [0,window] or +inf")
    result, enrollment, events, steps = _run_top(
        design, gaps[None, :], delays[None, :], duration, record=True
    )
    n = int(result.patients[0])
    final = float(result.duration[0])
    # Pending means neither a response nor a completed response-free window.
    ascertainment = _add_time(enrollment[0, :n], np.minimum(delays[:n], duration))
    pending = ascertainment > final
    return TOPBinaryTrial(
        _owned(enrollment[0, :n]),
        _owned(events[0, :n]),
        _owned(pending),
        str(result.decision[0]),
        final,
        float(result.suspension_time[0]),
        float(result.final_followup_time[0]),
        steps,
    )
