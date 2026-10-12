"""Current UAROET calendar data, explicit delay laws and as-of observation views."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike

from .uaroet import _integer, _real_matrix
from .uaroet_desktop_rules import _profile
from .uaroet_desktop_study import UAROETDesktopStudy


def _time(value: float, name: str) -> float:
    a = _real_matrix(value, name)
    if a.ndim or float(a) < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(a)


@dataclass(frozen=True)
class UAROETTimedPatient:
    identifier: int
    dose: int
    efficacy: int
    toxicity: int
    arrival: float
    efficacy_delay: float
    toxicity_delay: float

    def __post_init__(self) -> None:
        for name, low, high in (
            ("identifier", 1, 2**31 - 1),
            ("dose", 0, 4),
            ("efficacy", -1, 3),
            ("toxicity", -1, 3),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, low, high))
        for name in ("arrival", "efficacy_delay", "toxicity_delay"):
            object.__setattr__(self, name, _time(getattr(self, name), name))
        if not np.isfinite(self.arrival + max(self.efficacy_delay, self.toxicity_delay)):
            raise ValueError("outcome time exceeds floating-point range")

    @property
    def completion_time(self) -> float:
        return self.arrival + max(self.efficacy_delay, self.toxicity_delay)


def uaroet_outcome_delays(
    efficacy_window: ArrayLike,
    toxicity_window: ArrayLike,
    uniforms: ArrayLike,
    *,
    profile: str = "safe",
) -> tuple[float, float]:
    """Guide-continuous uniforms, or explicit native floor(uniform delay+0.5).

    The native rounding can return zero even with a positive lower endpoint.
    Supplied variates in[0,1] include deterministic boundary-reference cases;
    ordinary simulation draws use the NumPy Generator's half-open interval.
    """
    _profile(profile)
    intervals = [_real_matrix(x, "outcome window") for x in (efficacy_window, toxicity_window)]
    u = _real_matrix(uniforms, "uniforms")
    if u.shape != (2,) or np.any(u < 0) or np.any(u > 1):
        raise ValueError("uniforms must contain two probabilities")
    values = []
    for interval, value in zip(intervals, u, strict=True):
        if interval.shape != (2,) or not 0 <= interval[0] <= interval[1] <= 3600:
            raise ValueError("outcome windows require 0 <= minimum <= maximum <= 3600")
        delay = float(interval[0] + (interval[1] - interval[0]) * value)
        values.append(float(np.floor(delay + 0.5)) if profile == "legacy_1_8" else delay)
    return values[0], values[1]


def uaroet_study_as_of(
    study: UAROETDesktopStudy,
    patients: tuple[UAROETTimedPatient, ...],
    clock: float,
    *,
    profile: str = "safe",
) -> UAROETDesktopStudy:
    """Materialize observed/pending patient rows without altering the design.

    Require a design without existing patient rows to prevent silently replacing
    supplied conduct data. Future arrivals are excluded. Safe mode compares
    absolute outcome timestamps; legacy mode uses native clock-minus-arrival
    comparisons. Unknown(-1) outcomes remain unknown even after their time.
    """
    _profile(profile)
    if not isinstance(study, UAROETDesktopStudy) or study.patients:
        raise ValueError("as-of materialization requires a study without patient rows")
    time = _time(clock, "clock")
    if not isinstance(patients, tuple) or len(patients) > study.trial_parameters["NumPatients"]:
        raise ValueError("timed history must be a tuple within maximum accrual")
    rows = []
    previous_id = 0
    previous_time = -1.0
    for p in patients:
        if (
            not isinstance(p, UAROETTimedPatient)
            or p.dose >= study.dose_count
            or p.efficacy >= study.efficacy_levels
            or p.toxicity >= study.toxicity_levels
        ):
            raise ValueError("timed patient does not match the study")
        if (
            p.identifier <= previous_id
            or (previous_id == 0 and p.identifier != 1)
            or p.arrival < previous_time
        ):
            raise ValueError("timed history requires increasing IDs from1 and ordered arrivals")
        previous_id, previous_time = p.identifier, p.arrival
        if p.arrival > time:
            continue
        if profile == "legacy_1_8":
            seen_e = time - p.arrival >= p.efficacy_delay
            seen_t = time - p.arrival >= p.toxicity_delay
        else:
            seen_e = time >= p.arrival + p.efficacy_delay
            seen_t = time >= p.arrival + p.toxicity_delay
        rows.append(
            (p.identifier, p.dose, p.efficacy if seen_e else -1, p.toxicity if seen_t else -1)
        )
    return replace(study, patients=tuple(rows))


def _advance_arrival(previous, wait_until, next_interval, max_discarded):
    """Discard candidate arrivals before an optional native suspension deadline."""
    clock = _time(previous, "previous arrival")
    deadline = _time(wait_until, "wait_until")
    limit = _integer(max_discarded, "max_discarded", 0, 1_000_000)
    if not callable(next_interval):
        raise ValueError("next_interval must be callable")
    discarded = 0
    while True:
        clock += _time(next_interval(), "arrival interval")
        if not np.isfinite(clock):
            raise ArithmeticError("arrival clock exceeds floating-point range")
        if clock >= deadline:
            return clock, discarded
        if discarded >= limit:
            raise RuntimeError("discarded-arrival budget exhausted")
        discarded += 1
