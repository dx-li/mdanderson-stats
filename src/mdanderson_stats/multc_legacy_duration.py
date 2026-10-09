"""Deterministic compatibility replay of the recovered Multc Lean duration kernel.

This legacy simulation uses latent outcome counts and balked arrivals. It is
separate from the observation-aware Python calendar trial engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray

_LOG_TWENTY = 2.99573227355399  # Value stored by Multc Lean 2.1's native DLL.
_MAX_DRAWS = 1_000_000


@dataclass(frozen=True)
class MultcLegacyDurationPatient:
    """One treated patient's generated outcome and shared follow-up endpoint."""

    patient: int
    enrollment_time: float
    response: bool
    toxicity: bool
    followup_time: float
    could_stop_next: bool


@dataclass(frozen=True)
class MultcLegacyDuration:
    """Legacy totals, actual clocks and consumed external random-draw counts.

    The original ``duration`` is last follow-up measured from enrollment zero.
    A stopping evaluation occurs at a later proposed arrival and can follow
    that time; ``decision_time`` exposes this distinction. At the enrollment
    cap it is the last enrollment time. No inference about acceptable treatment
    or exclusive stopping causes follows from reaching the cap.
    """

    patients: tuple[MultcLegacyDurationPatient, ...]
    responses: int
    toxicities: int
    balks: int
    duration: float
    decision_time: float
    decision: Literal["stop_response", "stop_toxicity", "stop_both", "cap_complete"]
    uniforms_consumed: int
    exponentials_consumed: int

    @property
    def sample_size(self) -> int:
        return len(self.patients)


def _real_scalar(value: object, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a positive finite real scalar")
    result = float(raw)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite real scalar")
    return result


def _stream(value: ArrayLike, name: str, maximum: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size > maximum or (raw.size and raw.dtype.kind not in "iuf"):
        raise ValueError(f"{name} must be a bounded one-dimensional real array")
    values = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _boundaries(value: ArrayLike, name: str, cap: int) -> tuple[int, ...]:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size > cap + 1 or (raw.size and raw.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array of length <= cap+1")
    if raw.size and (np.any(raw < 1) or np.any(raw > cap + 1)):
        raise ValueError(f"{name} values must be in [1, cap+1]; handle pretrial stops separately")
    return tuple(int(v) for v in raw)


def run_multc_legacy_duration(
    max_subjects: int,
    *,
    response_stop_at: ArrayLike,
    nontoxicity_stop_at: ArrayLike,
    joint_probabilities: ArrayLike,
    mean_interarrival: float,
    response_window: float,
    uniforms: ArrayLike,
    unit_exponentials: ArrayLike,
) -> MultcLegacyDuration:
    """Replay native duration behavior with explicit bounds and external draws.

    ``response_stop_at[r]`` is the earliest treated sample size that stops
    with r responses. ``nontoxicity_stop_at[s]`` is the equivalent toxicity
    stopping size with s NONtoxicities. Missing entries never stop; cap+1 is
    a never-stop sentinel. The caller supplies these compact boundaries after
    handling the native pretrial screen. This function does not infer posterior
    parameters, cohort conversion, or minimum-enrollment precedence.

    Joint categories are [both, response only, toxicity only, neither]. Each
    patient consumes a uniform in (0,1). Responders consume a unit exponential
    and have follow-up min(window, draw*window/log(20)); others use window.
    Arrivals consume further unit exponentials times mean_interarrival.

    BEFORE generating a patient, the routine uses previous LATENT complete
    counts to determine if the next patient could cross either boundary. If
    so, subsequent proposed arrivals before all treated follow-up finishes
    balk, consuming draws without treating patients. The actual stop is
    evaluated at the first remaining arrival. Arrivals are not frozen or
    queued. Both endpoints share the patient's follow-up time. This historical
    simulation is not an observation-aware rule for conducting trials.

    Caps are 3..1000 patients and 1,000,000 exponential inputs. Too-short
    streams, overflow and positive increments lost to clock rounding raise
    explicit errors instead of returning a truncated trial.
    """
    if (
        isinstance(max_subjects, (bool, np.bool_))
        or not isinstance(max_subjects, (int, np.integer))
        or not 3 <= max_subjects <= 1000
    ):
        raise ValueError("max_subjects must be an integer in [3,1000]")
    cap = int(max_subjects)
    response = _boundaries(response_stop_at, "response_stop_at", cap)
    nontoxicity = _boundaries(nontoxicity_stop_at, "nontoxicity_stop_at", cap)
    probabilities = _stream(joint_probabilities, "joint_probabilities", 4)
    if (
        probabilities.size != 4
        or np.any(probabilities < 0)
        or not np.isclose(probabilities.sum(), 1.0, rtol=0, atol=1e-12)
    ):
        raise ValueError("joint_probabilities must be four nonnegative probabilities summing to 1")
    mean = _real_scalar(mean_interarrival, "mean_interarrival")
    window = _real_scalar(response_window, "response_window")
    u = _stream(uniforms, "uniforms", cap)
    exponential = _stream(unit_exponentials, "unit_exponentials", _MAX_DRAWS)
    if np.any((u <= 0) | (u >= 1)):
        raise ValueError("uniforms must be in (0,1)")
    if np.any(exponential < 0):
        raise ValueError("unit_exponentials must be nonnegative")
    cumulative = np.cumsum(probabilities)
    # Native comparisons are inclusive at category boundaries.
    cumulative[-1] = 1.0
    uniform_index = 0
    exponential_index = 0

    def draw_exponential(scale: float, *, clipped_response: bool = False) -> float:
        nonlocal exponential_index
        if exponential_index >= exponential.size:
            raise ValueError("unit_exponentials exhausted before trial completion")
        draw = float(exponential[exponential_index])
        exponential_index += 1
        if clipped_response and draw >= _LOG_TWENTY:
            return window
        value = draw * scale
        if not np.isfinite(value) or (draw > 0 and value == 0):
            raise ArithmeticError("exponential time increment is not representable")
        return value

    def advance(clock: float, increment: float) -> float:
        value = clock + increment
        if not np.isfinite(value) or (increment > 0 and value == clock):
            raise ArithmeticError("positive increment cannot advance the trial clock")
        return value

    def hits(bounds: tuple[int, ...], events: int, size: int) -> bool:
        return events < len(bounds) and bounds[events] <= size

    patients: list[MultcLegacyDurationPatient] = []
    response_count = nontoxicity_count = balks = 0
    clock = last_followup = 0.0
    decision: Literal["stop_response", "stop_toxicity", "stop_both", "cap_complete"]
    decision = "cap_complete"
    for size in range(1, cap + 1):
        possible = hits(response, response_count, size) or hits(
            nontoxicity, nontoxicity_count, size
        )
        if uniform_index >= u.size:
            raise ValueError("uniforms exhausted before trial completion")
        category = int(np.searchsorted(cumulative, u[uniform_index], side="left"))
        uniform_index += 1
        responded = category in (0, 1)
        toxic = category in (0, 2)
        response_count += int(responded)
        nontoxicity_count += int(not toxic)
        delay = (
            draw_exponential(window / _LOG_TWENTY, clipped_response=True) if responded else window
        )
        followup = advance(clock, delay)
        last_followup = max(last_followup, followup)
        patients.append(
            MultcLegacyDurationPatient(size, clock, responded, toxic, followup, possible)
        )
        if size == cap:
            break
        clock = advance(clock, draw_exponential(mean))
        if possible:
            while clock < last_followup:
                balks += 1
                clock = advance(clock, draw_exponential(mean))
        response_hit = hits(response, response_count, size)
        toxicity_hit = hits(nontoxicity, nontoxicity_count, size)
        if response_hit or toxicity_hit:
            decision = (
                "stop_both"
                if response_hit and toxicity_hit
                else "stop_response"
                if response_hit
                else "stop_toxicity"
            )
            break
    return MultcLegacyDuration(
        tuple(patients),
        response_count,
        len(patients) - nontoxicity_count,
        balks,
        last_followup,
        clock,
        decision,
        uniform_index,
        exponential_index,
    )
