"""ARAND 5.2.2 single-endpoint compatibility primitives.

These rules deliberately retain original controller edge cases. They are not
clinical recommendations. Independent posterior integration lives in
:mod:`arand_posterior`; no original runtime or random stream is reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite, scalar


class ArandNativeStatus(IntEnum):
    OPEN = 0
    TEMPORARILY_CLOSED = 1
    OTHER_ARM_WON = 2
    PERMANENTLY_CLOSED = 3
    ENROLLMENT_MAXIMUM = 4
    NOT_STARTED = 5
    DURATION_MAXIMUM = 9


def _probabilities(values: ArrayLike) -> FloatArray:
    p = finite(values, "probabilities")
    if p.ndim != 1 or not 1 <= len(p) <= 10 or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must contain 1 to 10 values in [0,1]")
    return p


def _unit(value: float, name: str) -> float:
    value = scalar(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0,1]")
    return value


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not boolean")
    x = scalar(value, name)
    if x != int(x) or not minimum <= x <= 2**31 - 1:
        raise ValueError(f"{name} must be an integer in [{minimum}, 2**31-1]")
    return int(x)


def _statuses(values: tuple[int, ...], size: int) -> list[int]:
    if len(values) != size or any(v not in set(ArandNativeStatus) for v in values):
        raise ValueError("one supported status is required per arm")
    return [int(v) for v in values]


def arand_native_allocation(
    probabilities: ArrayLike, *, power: float = 0.5, floor: float = 0.1
) -> FloatArray:
    """Power weights with iterative proportional floors on eligible arms.

    Reject infeasible floors and undefined all-zero weights rather than inherit
    native division by zero. No mixture with equal weights is applied.
    """
    p = _probabilities(probabilities)
    power = scalar(power, "power")
    floor = _unit(floor, "floor")
    if power < 0 or floor * len(p) > 1:
        raise ValueError("power must be nonnegative and floor feasible")
    if power == 0 or floor * len(p) == 1:
        return _freeze(np.full(len(p), 1 / len(p)))
    if not np.any(p > 0):
        raise ValueError("positive power requires a positive weight")
    with np.errstate(divide="ignore", under="ignore", over="ignore"):
        logp = np.log(p)
        weights = np.exp(power * (logp - logp.max()))
    result = np.zeros(len(p))
    free = np.ones(len(p), dtype=bool)
    for _ in range(len(p)):
        result[free] = weights[free] / weights[free].sum() * (1 - result[~free].sum())
        low = free & (result < floor)
        if not np.any(low):
            return _freeze(result)
        result[low] = floor
        free[low] = False
    raise ArithmeticError("allocation floor iteration failed")


@dataclass(frozen=True)
class ArandNativeMonitor:
    """Explicit scheduler state; modes 0..5 follow the recovered native enum.

    ``targets`` are times for mode 2, batch increments for mode 4, and observed
    counts for mode 5. ``batch_index`` is distinct from the discrete look index.
    """

    mode: int = 0
    next_time: float = 1
    next_enrollment: int = 1
    next_outcome: int = 1
    step: int = 1
    targets: tuple[float, ...] = (1, 2, 3)
    index: int = 0
    batch_index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _integer(self.mode, "mode"))
        if self.mode > 5:
            raise ValueError("mode must be in 0..5")
        if scalar(self.next_time, "next_time") < 0:
            raise ValueError("next_time must be nonnegative")
        for name in ("next_enrollment", "next_outcome", "index", "batch_index"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        object.__setattr__(self, "step", _integer(self.step, "step", 1))
        t = finite(self.targets, "targets")
        if t.ndim != 1 or np.any(t < 0):
            raise ValueError("targets must be a nonnegative vector")
        if self.mode in (2, 5) and np.any(np.diff(t) < 0):
            raise ValueError("discrete targets must be ordered")
        if self.mode in (4, 5) and np.any(t != np.floor(t)):
            raise ValueError("count targets must be integers")
        if self.mode == 4 and (not len(t) or np.any(t < 1) or self.batch_index >= len(t)):
            raise ValueError("batch monitoring requires positive increments and a valid index")
        object.__setattr__(self, "targets", tuple(float(x) for x in t))


@dataclass(frozen=True)
class ArandNativeMonitorResult:
    monitor: ArandNativeMonitor
    analyze: bool
    trial_status: ArandNativeStatus | None


def arand_native_monitor(
    monitor: ArandNativeMonitor, *, now: float, enrolled: int, observed: int, pending: bool
) -> ArandNativeMonitorResult:
    """Advance once at a candidate time; this does not run independent timers."""
    now = scalar(now, "now")
    enrolled = _integer(enrolled, "enrolled")
    observed = _integer(observed, "observed")
    if now < 0 or observed > enrolled or not isinstance(pending, bool):
        raise ValueError("invalid candidate time or observation counts")
    m = monitor
    analyze, status = False, None
    if m.mode == 0:
        analyze = True
    elif m.mode == 1 and now >= m.next_time:
        analyze = True
        m = replace(m, next_time=float(np.ceil(now)))
    elif m.mode in (2, 5):
        value = now if m.mode == 2 else observed
        threshold = m.next_time if m.mode == 2 else m.next_outcome
        if m.index < len(m.targets) and value >= threshold:
            analyze = True
            index = m.index + 1
            while index < len(m.targets) and m.targets[index] < value:
                index += 1
            m = replace(m, index=index)
            if index < len(m.targets):
                m = (
                    replace(m, next_time=m.targets[index])
                    if m.mode == 2
                    else replace(m, next_outcome=int(m.targets[index]))
                )
    elif m.mode == 3 and enrolled >= m.next_enrollment:
        analyze = True
        m = replace(m, next_enrollment=enrolled + m.step)
    elif m.mode == 4 and enrolled == m.next_enrollment:
        status = ArandNativeStatus.TEMPORARILY_CLOSED if pending else ArandNativeStatus.OPEN
        if not pending:
            analyze = True
            m = replace(
                m,
                next_enrollment=m.next_enrollment + int(m.targets[m.batch_index]),
                batch_index=min(m.batch_index + 1, len(m.targets) - 1),
            )
    return ArandNativeMonitorResult(m, analyze, status)


@dataclass(frozen=True)
class ArandNativeObservation:
    """One enrolled patient's absolute observation time and binary/event value."""

    arm: int
    arrival: float
    record_time: float
    indicator: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm", _integer(self.arm, "arm"))
        if (
            scalar(self.arrival, "arrival") < 0
            or scalar(self.record_time, "record_time") < self.arrival
        ):
            raise ValueError("record_time must be at or after nonnegative arrival")
        if _integer(self.indicator, "indicator") > 1:
            raise ValueError("indicator must be zero or one")


@dataclass(frozen=True)
class ArandNativeStatistics:
    enrolled: tuple[int, ...]
    observed: tuple[int, ...]
    positive: tuple[int, ...]
    exposure: FloatArray


def arand_native_statistics(
    records: tuple[ArandNativeObservation, ...], *, now: float, arms: int
) -> ArandNativeStatistics:
    """Visible zero outcomes count as observed; censored TTE records are not events."""
    arms = _integer(arms, "arms", 1)
    now = scalar(now, "now")
    if arms > 10 or now < 0:
        raise ValueError("require at most ten arms and a nonnegative time")
    enrolled, observed, positive = [0] * arms, [0] * arms, [0] * arms
    exposure = np.zeros(arms)
    for r in records:
        if r.arm >= arms or r.arrival > now:
            raise ValueError("records must belong to the trial and already be enrolled")
        enrolled[r.arm] += 1
        visible = now >= r.record_time
        observed[r.arm] += int(visible)
        positive[r.arm] += int(visible) * r.indicator
        exposure[r.arm] += min(now, r.record_time) - r.arrival
    return ArandNativeStatistics(
        tuple(enrolled), tuple(observed), tuple(positive), _freeze(exposure)
    )


def arand_native_statuses(
    probabilities: ArrayLike,
    statuses: tuple[int, ...],
    *,
    lower: float = 0.1,
    upper: float = 0.95,
    futility: ArrayLike | None = None,
    futility_cutoff: float = 0.2,
    min_action: int = 1,
    max_action: int = 2,
) -> tuple[int, ...]:
    """Ordered single-endpoint posterior status transition, including native overrides."""
    p = _probabilities(probabilities)
    s = _statuses(statuses, len(p))
    lower, upper = _unit(lower, "lower"), _unit(upper, "upper")
    cutoff = _unit(futility_cutoff, "futility_cutoff")
    if min_action not in (1, 3) or max_action not in (1, 2, 3):
        raise ValueError("unsupported closure action")
    f = None if futility is None else _probabilities(futility)
    if f is not None and len(f) != len(p):
        raise ValueError("futility probabilities require one value per arm")
    for i, probability in enumerate(p):
        status = s[i]
        if f is not None and status != 3 and cutoff > 0 and f[i] < cutoff:
            s[i], status, probability = min_action, min_action, -1
        if probability > upper and upper < 1:
            for j in range(len(p)):
                if i != j:
                    s[j] = max_action
                    # The original also overwrites the current arm's local status.
                    status = min_action
        if status not in (2, 3):
            s[i] = min_action if probability < lower and lower > 0 else 0
    return tuple(s)


@dataclass(frozen=True)
class ArandNativeSelection:
    winner: int | None
    early: bool


def arand_native_select(
    probabilities: ArrayLike,
    statuses: tuple[int, ...],
    *,
    cutoff: float = 0.5,
    upper: float = 0.95,
    completed: bool = True,
    group: bool = False,
) -> ArandNativeSelection:
    """Native whole-trial or group final selector; later arms win exact ties."""
    p = _probabilities(probabilities)
    s = _statuses(statuses, len(p))
    cutoff, upper = _unit(cutoff, "cutoff"), _unit(upper, "upper")
    winner, best, early = None, -1.0, False
    for i, probability in enumerate(p):
        if s[i] not in ((0,) if group else (0, 4)):
            continue
        if probability >= best and probability > cutoff:
            winner, best = i, float(probability)
        if probability > upper:
            early = group or not completed
            if group or probability > best:
                winner = i
                if not group:
                    best = float(probability)
    return ArandNativeSelection(winner, early)


@dataclass(frozen=True)
class ArandNativeArmLimits:
    minimum: int = 0
    maximum: int = 2**31 - 1
    group_minimum: int = 0
    group_maximum: int = 2**31 - 1
    start: float = 0
    duration: float = 0

    def __post_init__(self) -> None:
        for name in ("minimum", "maximum", "group_minimum", "group_maximum"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        for name in ("start", "duration"):
            if scalar(getattr(self, name), name) < 0:
                raise ValueError(f"{name} must be nonnegative")


def arand_native_constraints(
    statuses: tuple[int, ...],
    enrolled: tuple[int, ...],
    limits: tuple[ArandNativeArmLimits, ...],
    *,
    now: float,
) -> tuple[int, ...]:
    """Apply arm then group maxima/minima and finally start/duration gates.

    The advertised desktop has one group, so arm and group counts coincide.
    Conflicting limits remain representable to preserve the original ordering.
    """
    s = _statuses(statuses, len(limits))
    n = count(enrolled, "enrolled")
    now = scalar(now, "now")
    if n.shape != (len(s),) or now < 0:
        raise ValueError("invalid counts or time")
    for i, a in enumerate(limits):
        if n[i] >= a.maximum:
            s[i] = 4
        if n[i] < a.minimum and now >= a.start:
            s[i] = 0
        if n[i] >= a.group_maximum:
            s[i] = 4
        if n[i] < a.group_minimum:
            s[i] = 0
        if a.duration > 0:
            if now < a.start:
                s[i] = 5
            if s[i] == 0 and now > a.start + a.duration:
                s[i] = 9
            if s[i] == 5 and a.start <= now <= a.start + a.duration:
                s[i] = 0
    return tuple(s)
