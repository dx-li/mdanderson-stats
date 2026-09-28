"""Replay Multc Lean accrual with pending paired outcomes and look-ahead."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import count, finite
from .multc_core import MultcLeanDesign

_MAX_CALENDAR_WORK = 12_000_000


def _readonly(value: ArrayLike, dtype: type = np.float64) -> NDArray:
    result: NDArray = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class MultcCalendarLook:
    """One scheduled-look evaluation, including any pending-data waits.

    A ``stop_response`` or ``stop_toxicity`` action names a cause guaranteed
    under all pending completions; the other endpoint may also cross its stop
    boundary for some completions. ``stop_both`` means both causes are certain.
    """

    sample_size: int
    time: float
    response_observed: int
    toxicity_observed: int
    response_pending: int
    toxicity_pending: int
    response_stop_max: int
    toxicity_stop_min: int
    action: str


@dataclass(frozen=True)
class MultcCalendarTrial:
    """Calendar replay of a single paired-outcome trial.

    The first patient arrives at time zero. Inter-arrival inputs are elapsed
    accrual-open times; accrual pauses freeze that clock, so no arrivals are
    queued or discarded during a suspension. Endpoint delays run in calendar
    time. Both endpoints are observed at their separately supplied times.
    """

    outcomes: NDArray[np.int8]
    arrival_times: NDArray[np.float64]
    response_available_times: NDArray[np.float64]
    toxicity_available_times: NDArray[np.float64]
    response_known_at_decision: NDArray[np.bool_]
    toxicity_known_at_decision: NDArray[np.bool_]
    looks: tuple[MultcCalendarLook, ...]
    enrolled: int
    responses: int
    toxicities: int
    observed_responses_at_decision: int
    observed_toxicities_at_decision: int
    decision: str
    decision_time: float
    accrual_end_time: float
    last_followup_time: float
    duration: float
    paused_duration: float
    work_units: int


def _decision_with_pending(
    design: MultcLeanDesign,
    sample_size: int,
    response_observed: int,
    toxicity_observed: int,
    response_pending: int,
    toxicity_pending: int,
) -> str:
    """Resolve a stopping action only when every pending completion agrees."""
    j = int(np.searchsorted(design.looks, sample_size))
    response_bound = int(design._bounds.response_stop_max[j])
    toxicity_bound = int(design._bounds.toxicity_stop_min[j])
    response_stop_certain = response_observed + response_pending <= response_bound
    toxicity_stop_certain = toxicity_observed >= toxicity_bound
    if response_stop_certain and toxicity_stop_certain:
        return "stop_both"
    if response_stop_certain:
        return "stop_response"
    if toxicity_stop_certain:
        return "stop_toxicity"
    can_stop = (
        response_observed <= response_bound
        or toxicity_observed + toxicity_pending >= toxicity_bound
    )
    can_continue = (
        response_observed + response_pending > response_bound and toxicity_observed < toxicity_bound
    )
    if not can_stop:
        return "continue"
    if can_continue:
        return "wait"
    # The two endpoint count ranges are independently attainable because the
    # paired binary categories include all four response/toxicity combinations.
    raise ArithmeticError("pending-outcome look-ahead reached an inconsistent boundary state")


def run_multc_calendar_trial(
    design: MultcLeanDesign,
    outcomes: ArrayLike,
    interarrival_intervals: ArrayLike,
    response_delays: ArrayLike,
    toxicity_delays: ArrayLike,
) -> MultcCalendarTrial:
    """Replay a bounded trial with explicit paired outcomes and endpoint delays.

    ``outcomes`` has shape ``(design.max_subjects, 2)`` in response/toxicity
    order. ``interarrival_intervals`` has one nonnegative elapsed accrual-open
    time between each possible successive patient; patient one starts at zero.
    The two delay vectors are separate elapsed times from enrollment until each
    endpoint is observed. This intentionally does not impose an unverified
    toxicity-time model or response/toxicity independence.

    At each scheduled cohort look, possible completions of the still-pending
    endpoint indicators are assessed against the design's precomputed stopping
    bounds, without reading their supplied future outcomes. Accrual continues
    if all completions continue, stops if all completions stop, and pauses only
    when pending outcomes can change stop versus continue. If every pending
    completion stops, the endpoint stop bounds identify the cause without
    consulting unresolved outcome values. At maximum enrollment,
    ``cap_complete`` takes precedence and does not wait
    for outcomes. Trial duration is the later of the stopping/cap decision and
    last follow-up among enrolled patients.
    """
    if not isinstance(design, MultcLeanDesign):
        raise TypeError("design must be a MultcLeanDesign")
    maximum = design.max_subjects
    work_bound = 10 * maximum * maximum + 4 * maximum + len(design.looks)
    if work_bound > _MAX_CALENDAR_WORK:
        raise ValueError("Multc calendar replay exceeds the 12-million-work-unit limit")

    raw_outcomes = np.asarray(outcomes)
    if raw_outcomes.shape != (maximum, 2):
        raise ValueError(f"outcomes must have shape ({maximum}, 2)")
    outcome_counts = count(raw_outcomes, "outcomes")
    if np.any(outcome_counts > 1):
        raise ValueError("outcomes must contain paired binary response/toxicity values")
    paired = np.asarray(outcome_counts, dtype=np.int8)

    raw_intervals = np.asarray(interarrival_intervals)
    if raw_intervals.shape != (maximum - 1,):
        raise ValueError(f"interarrival_intervals must have length {maximum - 1}")
    intervals = finite(raw_intervals, "interarrival_intervals")
    if np.any(intervals < 0):
        raise ValueError("interarrival_intervals must be nonnegative")

    raw_response_delays = np.asarray(response_delays)
    raw_toxicity_delays = np.asarray(toxicity_delays)
    if raw_response_delays.shape != (maximum,) or raw_toxicity_delays.shape != (maximum,):
        raise ValueError(f"response_delays and toxicity_delays must have length {maximum}")
    response_wait = finite(raw_response_delays, "response_delays")
    toxicity_wait = finite(raw_toxicity_delays, "toxicity_delays")
    if np.any(response_wait < 0) or np.any(toxicity_wait < 0):
        raise ValueError("endpoint availability delays must be nonnegative")

    arrivals = np.zeros(maximum, dtype=np.float64)
    response_available = np.zeros(maximum, dtype=np.float64)
    toxicity_available = np.zeros(maximum, dtype=np.float64)
    outcomes_known_r = np.zeros(maximum, dtype=bool)
    outcomes_known_t = np.zeros(maximum, dtype=bool)
    history: list[MultcCalendarLook] = []
    n = 0
    now = 0.0
    paused = 0.0
    response_observed = 0
    toxicity_observed = 0

    if design.pretrial_check:
        pretrial = str(design.monitor_counts(0, 0, 0).decision.item())
        if pretrial != "continue":
            decision_time = 0.0
            last_followup = 0.0
            return MultcCalendarTrial(
                _readonly(paired[:0], np.int8),
                _readonly(arrivals[:0]),
                _readonly(response_available[:0]),
                _readonly(toxicity_available[:0]),
                _readonly(outcomes_known_r[:0], bool),
                _readonly(outcomes_known_t[:0], bool),
                (),
                0,
                0,
                0,
                0,
                0,
                f"pretrial_{pretrial}",
                decision_time,
                decision_time,
                last_followup,
                max(decision_time, last_followup),
                paused,
                1,
            )

    decision: str | None = None
    decision_time = 0.0
    work = 1
    for patient in range(maximum):
        if patient > 0:
            increment = float(intervals[patient - 1])
            next_arrival = now + increment
            if not np.isfinite(next_arrival):
                raise ArithmeticError("calendar arrival time overflowed")
            if increment > 0 and next_arrival <= now:
                raise ArithmeticError(
                    "positive inter-arrival interval does not advance calendar time"
                )
            now = next_arrival
        arrivals[patient] = now
        response_increment = float(response_wait[patient])
        toxicity_increment = float(toxicity_wait[patient])
        response_available[patient] = now + response_increment
        toxicity_available[patient] = now + toxicity_increment
        if not np.isfinite(response_available[patient]) or not np.isfinite(
            toxicity_available[patient]
        ):
            raise ArithmeticError("endpoint availability time overflowed")
        if (response_increment > 0 and response_available[patient] <= now) or (
            toxicity_increment > 0 and toxicity_available[patient] <= now
        ):
            raise ArithmeticError("positive endpoint delay does not advance calendar time")
        n += 1

        while True:
            work += 3 * n + 2
            if work > _MAX_CALENDAR_WORK:
                raise RuntimeError("Multc calendar replay exceeded its work limit")
            new_response = (~outcomes_known_r[:n]) & (response_available[:n] <= now)
            new_toxicity = (~outcomes_known_t[:n]) & (toxicity_available[:n] <= now)
            response_observed += int(np.sum(paired[:n, 0][new_response]))
            toxicity_observed += int(np.sum(paired[:n, 1][new_toxicity]))
            outcomes_known_r[:n] |= new_response
            outcomes_known_t[:n] |= new_toxicity
            r_known = response_observed
            t_known = toxicity_observed
            r_pending = n - int(np.sum(outcomes_known_r[:n]))
            t_pending = n - int(np.sum(outcomes_known_t[:n]))

            is_look = n in design.looks
            action = "continue"
            r_bound = t_bound = -1
            if n == maximum:
                action = "cap_complete"
            elif is_look:
                action = _decision_with_pending(design, n, r_known, t_known, r_pending, t_pending)
                j = int(np.searchsorted(design.looks, n))
                r_bound = int(design._bounds.response_stop_max[j])
                t_bound = int(design._bounds.toxicity_stop_min[j])
            if is_look or n == maximum:
                history.append(
                    MultcCalendarLook(
                        n,
                        now,
                        r_known,
                        t_known,
                        r_pending,
                        t_pending,
                        r_bound,
                        t_bound,
                        action,
                    )
                )
            if action == "wait":
                pending_times = np.concatenate(
                    (
                        response_available[:n][~outcomes_known_r[:n]],
                        toxicity_available[:n][~outcomes_known_t[:n]],
                    )
                )
                if pending_times.size == 0:
                    raise ArithmeticError("look-ahead requested a wait with no pending endpoint")
                next_time = float(np.min(pending_times))
                if next_time < now:
                    raise ArithmeticError("pending endpoint availability precedes current time")
                paused += next_time - now
                now = next_time
                continue
            if action in ("stop_response", "stop_toxicity", "stop_both", "cap_complete"):
                decision = action
                decision_time = now
                break
            # Stable continuation resumes the frozen accrual clock from now.
            break
        if decision is not None:
            break
    if decision is None:
        raise ArithmeticError("calendar replay ended without a decision")
    accrual_end = decision_time
    last_followup = float(
        max(response_available[:n].max(initial=0.0), toxicity_available[:n].max(initial=0.0))
    )
    duration = max(decision_time, last_followup)
    known_r = outcomes_known_r[:n]
    known_t = outcomes_known_t[:n]
    return MultcCalendarTrial(
        _readonly(paired[:n], np.int8),
        _readonly(arrivals[:n]),
        _readonly(response_available[:n]),
        _readonly(toxicity_available[:n]),
        _readonly(known_r, bool),
        _readonly(known_t, bool),
        tuple(history),
        n,
        int(np.sum(paired[:n, 0])),
        int(np.sum(paired[:n, 1])),
        response_observed,
        toxicity_observed,
        decision,
        decision_time,
        accrual_end,
        last_followup,
        duration,
        paused,
        work,
    )
