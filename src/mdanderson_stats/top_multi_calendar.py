"""Observed-data calendar replay for TOP two-endpoint designs."""

from dataclasses import dataclass
from math import prod

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite
from .boin import _owned
from .top_calendar import _add_time
from .top_endpoints import TOPMultiEndpointDesign


@dataclass(frozen=True)
class TOPMultiEndpointStep:
    """One observed analysis, including re-evaluations during suspension."""

    time: float
    patients: int
    events: tuple[int, int]
    pending: tuple[int, int]
    effective_sample_size: tuple[float, float]
    endpoint_status: tuple[str, str]
    decision: str


@dataclass(frozen=True)
class TOPMultiEndpointTrial:
    """Observed trial path; unobserved endpoint outcomes remain NaN/Inf."""

    enrollment_times: FloatArray
    observed_event_times: FloatArray
    observed_outcomes: FloatArray
    observed_events: NDArray[np.int64]
    observed_non_events: NDArray[np.int64]
    pending: NDArray[np.int64]
    decision: str
    final_time: float
    interim_suspension_time: float
    final_followup_time: float
    steps: tuple[TOPMultiEndpointStep, ...]


@dataclass(frozen=True)
class _TopMultiBatchResult:
    patients: NDArray[np.int64]
    enrollment_times: FloatArray | None
    events: NDArray[np.int64]
    non_events: NDArray[np.int64]
    pending: NDArray[np.int64]
    decisions: NDArray[np.str_]
    final_times: FloatArray
    interim_pauses: FloatArray
    final_waits: FloatArray
    observed_event_times: FloatArray | None
    observed_outcomes: FloatArray | None
    steps: tuple[TOPMultiEndpointStep, ...]


def _run_top_multiendpoint_batch(
    design: TOPMultiEndpointDesign,
    interarrival: FloatArray,
    delays: FloatArray,
    *,
    record_single: bool = False,
    scan_budget: list[int] | None = None,
) -> _TopMultiBatchResult:
    """Vectorized calendar kernel shared by the aggregate simulator."""
    n_trials, maximum = interarrival.shape
    if delays.shape != (n_trials, maximum, 2):
        raise ValueError("batch delays must have shape (trials, patients, 2)")
    clock = np.zeros(n_trials)
    enrolled = np.full((n_trials, maximum), np.nan)
    event_at = np.full((n_trials, maximum, 2), np.inf)
    known_at = np.full((n_trials, maximum, 2), np.inf)
    alive = np.ones(n_trials, dtype=bool)
    decisions = np.full(n_trials, "", dtype="<U24")
    pauses = np.zeros(n_trials)
    final_waits = np.zeros(n_trials)
    steps: list[TOPMultiEndpointStep] = []
    windows = np.asarray(design.windows)
    looks = np.asarray(design.looks)

    for patient in range(maximum):
        rows = np.flatnonzero(alive)
        if not rows.size:
            break
        clock[rows] = _add_time(clock[rows], interarrival[rows, patient])
        enrolled[rows, patient] = clock[rows]
        row_delays = delays[rows, patient]
        finite_event = np.isfinite(row_delays)
        for endpoint in range(2):
            active = finite_event[:, endpoint]
            if np.any(active):
                active_rows = rows[active]
                event_at[active_rows, patient, endpoint] = _add_time(
                    clock[active_rows], row_delays[active, endpoint]
                )
        known_delay = np.where(finite_event, row_delays, windows)
        known_at[rows, patient] = _add_time(clock[rows, None], known_delay)
        n_enrolled = patient + 1
        if n_enrolled not in looks:
            continue

        waiting = rows
        while waiting.size:
            if scan_budget is not None:
                charge = 2 * n_enrolled * waiting.size
                if charge > scan_budget[0]:
                    raise ValueError("TOP calendar scan work exceeds max_work")
                scan_budget[0] -= charge
            current_clock = clock[waiting]
            in_trial = np.isfinite(enrolled[waiting, :n_enrolled])
            event_time = event_at[waiting, :n_enrolled]
            ascertainment = known_at[waiting, :n_enrolled]
            event = np.isfinite(event_time) & (event_time <= current_clock[:, None, None])
            no_event = np.isposinf(event_time) & (ascertainment <= current_clock[:, None, None])
            observed = in_trial[..., None]
            pending_mask = observed & ~(event | no_event)
            event_count = event.sum(axis=1).astype(np.int64)
            pending_count = pending_mask.sum(axis=1).astype(np.int64)
            elapsed = np.maximum(
                current_clock[:, None, None] - enrolled[waiting, :n_enrolled, None], 0
            )
            followup = np.minimum(elapsed, windows)
            weights = design.timing_weight(np.where(pending_mask, followup, 0.0))
            weight_sum = np.where(pending_mask, weights, 0.0).sum(axis=1)
            result = design.evaluate(n_enrolled, event_count, pending_count, weight_sum)
            action = np.asarray(result.decision)
            if record_single:
                if n_trials != 1:
                    raise ValueError("step recording is available only for one trial")
                if waiting.size != 1 or waiting[0] != 0:
                    raise RuntimeError("single-trial recording lost its row")
                steps.append(
                    TOPMultiEndpointStep(
                        float(current_clock[0]),
                        n_enrolled,
                        (int(event_count[0, 0]), int(event_count[0, 1])),
                        (int(pending_count[0, 0]), int(pending_count[0, 1])),
                        (
                            float(result.effective_sample_size[0, 0]),
                            float(result.effective_sample_size[0, 1]),
                        ),
                        (str(result.endpoint_status[0, 0]), str(result.endpoint_status[0, 1])),
                        str(action[0]),
                    )
                )
            terminal = ~np.isin(action, ("suspend", "continue"))
            if np.any(terminal):
                ended = waiting[terminal]
                decisions[ended] = action[terminal]
                alive[ended] = False
            suspended = action == "suspend"
            waiting = waiting[suspended]
            if not waiting.size:
                continue
            missing_suspended = pending_mask[suspended]
            pending_event_at = event_at[waiting, :n_enrolled]
            pending_known_at = known_at[waiting, :n_enrolled]
            next_times = np.where(
                missing_suspended & np.isfinite(pending_event_at),
                pending_event_at,
                pending_known_at,
            )
            next_times = np.where(missing_suspended, next_times, np.inf)
            future = np.min(next_times, axis=(1, 2))
            if np.any(~np.isfinite(future) | (future <= clock[waiting])):
                raise ArithmeticError("suspended TOP look has no later endpoint ascertainment")
            elapsed_wait = future - clock[waiting]
            is_final = n_enrolled == maximum
            if is_final:
                final_waits[waiting] += elapsed_wait
            else:
                pauses[waiting] += elapsed_wait
            clock[waiting] = future

    if np.any(alive) or np.any(decisions == ""):
        raise RuntimeError("TOP endpoint trial did not reach a terminal decision")
    included = np.isfinite(enrolled)
    event = included[..., None] & np.isfinite(event_at) & (event_at <= clock[:, None, None])
    no_event = included[..., None] & np.isposinf(event_at) & (known_at <= clock[:, None, None])
    pending_mask = included[..., None] & ~(event | no_event)
    observed_event_times = None
    observed_outcomes = None
    if record_single:
        observed_event_times = np.where(event, event_at, np.inf)
        observed_outcomes = np.full(event.shape, np.nan)
        observed_outcomes[event] = 1.0
        observed_outcomes[no_event] = 0.0
    return _TopMultiBatchResult(
        _owned(included.sum(axis=1).astype(np.int64)),
        _owned(enrolled) if record_single else None,
        _owned(event.sum(axis=1).astype(np.int64)),
        _owned(no_event.sum(axis=1).astype(np.int64)),
        _owned(pending_mask.sum(axis=1).astype(np.int64)),
        _owned(decisions),
        _owned(clock),
        _owned(pauses),
        _owned(final_waits),
        None if observed_event_times is None else _owned(observed_event_times),
        None if observed_outcomes is None else _owned(observed_outcomes),
        tuple(steps),
    )


def run_top_multiendpoint_trial(
    design: TOPMultiEndpointDesign,
    interarrival: ArrayLike,
    event_delays: ArrayLike,
) -> TOPMultiEndpointTrial:
    """Replay one trial using only endpoint outcomes observed at each analysis.

    A finite delay in ``[0, endpoint_window]`` is an event; positive infinity is
    no event, known at the end of that endpoint's window. The first interarrival
    gap starts at time zero. After an interim suspension, accrual pauses until the
    next event or window completion; the next arrival gap then starts afresh.
    These are explicit Python calendar conventions, not native RNG/scheduling
    parity. Endpoint outcomes share patient arrivals but may resolve separately.
    """
    if np.shape(interarrival) != (design.max_subjects,):
        raise ValueError("interarrival must contain one gap per planned patient")
    if np.shape(event_delays) != (design.max_subjects, 2):
        raise ValueError("event_delays must have shape (max_subjects, 2)")
    gaps = finite(interarrival, "interarrival")
    raw_delays = np.asarray(event_delays, dtype=float)
    windows = np.asarray(design.windows)
    if np.any(gaps < 0):
        raise ValueError("interarrival must contain one nonnegative gap per planned patient")
    if np.any(
        ~(
            np.isposinf(raw_delays)
            | (np.isfinite(raw_delays) & (raw_delays >= 0) & (raw_delays <= windows))
        )
    ):
        raise ValueError("event delays must be in each endpoint window or positive infinity")
    if prod(raw_delays.shape) > 2_000_000:
        raise ValueError("trial input exceeds the 2,000,000 endpoint-patient-cell limit")
    batch = _run_top_multiendpoint_batch(
        design,
        gaps[None, :],
        raw_delays[None, :, :],
        record_single=True,
    )
    patients = int(batch.patients[0])
    assert batch.enrollment_times is not None
    assert batch.observed_event_times is not None
    assert batch.observed_outcomes is not None
    return TOPMultiEndpointTrial(
        _owned(batch.enrollment_times[0, :patients]),
        _owned(batch.observed_event_times[0, :patients]),
        _owned(batch.observed_outcomes[0, :patients]),
        _owned(batch.events[0]),
        _owned(batch.non_events[0]),
        _owned(batch.pending[0]),
        str(batch.decisions[0]),
        float(batch.final_times[0]),
        float(batch.interim_pauses[0]),
        float(batch.final_waits[0]),
        batch.steps,
    )
