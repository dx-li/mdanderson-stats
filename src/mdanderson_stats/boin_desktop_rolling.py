"""Desktop Rolling Six calendar, with safe exclusions and explicit legacy clocks."""

from dataclasses import dataclass

import numpy as np

from ._validation import count, finite, scalar
from .boin import _owned


@dataclass(frozen=True)
class BOINDesktopRollingTrial:
    patients: np.ndarray
    toxicities: np.ndarray
    doses: np.ndarray
    enrollment_times: np.ndarray
    delays: np.ndarray
    selected_dose: int | None
    reported_final_time: float
    actual_final_time: float
    suspension_checks: int
    decision_times: np.ndarray
    unsafe_assignments: np.ndarray
    selected_eliminated: bool
    profile: str


def _move(n, y, pending):
    if y >= 2:
        return -1
    if n < 3:
        return 0
    if n < 6:
        return 1 if y == 0 and pending == 0 else 0
    if (y == 0 and pending <= 1) or (y == 1 and pending == 0):
        return 1
    return None


def replay_boin_desktop_rolling_six(
    arrival_gaps,
    potential_delays,
    window,
    *,
    cohort_size=1,
    start_dose=1,
    profile="safe",
) -> BOINDesktopRollingTrial:
    """Replay bounded desktop Rolling Six, including lower-dose confirmation.

    Native cohorts may contain1,2,3 or6 patients. First enrollment is at zero;
    pending checks use further arrival gaps. Lower-dose confirmation can fill
    all remaining slots at once. The safe profile retains global exclusions and
    rechecks final complete outcomes; legacy selections and reported clock
    reversals remain explicit and diagnosable.
    """
    gaps = finite(arrival_gaps, "arrival_gaps")
    delays = np.asarray(potential_delays, dtype=float)
    duration = scalar(window, "window")
    if any(isinstance(v, (bool, np.bool_)) for v in (cohort_size, start_dose)):
        raise ValueError("cohort settings must be integer counts")
    size, start = count([cohort_size, start_dose], "cohort settings")
    if (
        profile not in ("safe", "desktop_legacy")
        or gaps.ndim != 1
        or not 1 <= gaps.size <= 100000
        or np.any(gaps <= 0)
        or delays.ndim != 2
        or not 2 <= delays.shape[1] <= 100
        or delays.shape[0] != 6 * delays.shape[1]
        or size not in (1, 2, 3, 6)
        or not 1 <= start <= delays.shape[1]
        or duration <= 0
        or np.any(
            ~(np.isposinf(delays) | (np.isfinite(delays) & (delays >= 0) & (delays <= duration)))
        )
    ):
        raise ValueError("invalid bounded Rolling Six calendar inputs")
    size, current = int(size), int(start)
    levels = delays.shape[1]
    n = np.zeros(levels, dtype=np.int64)
    y = np.zeros_like(n)
    pending = np.zeros_like(n)
    eliminated = np.zeros(levels, dtype=bool)
    entries: list[float] = []
    assignments: list[int] = []
    events: list[float] = []
    unsafe: list[bool] = []
    decisions: list[float] = []
    clock = 0.0
    cursor = 0
    pauses = 0
    confirm = False
    suspended = False
    selected = None

    def advance():
        nonlocal clock, cursor
        if cursor >= len(gaps):
            raise ValueError("arrival-gap tape exhausted before Rolling Six completion")
        updated = clock + float(gaps[cursor])
        cursor += 1
        if not np.isfinite(updated) or updated <= clock:
            raise ValueError("arrival gap does not advance a finite clock")
        clock = updated

    def observe():
        nonlocal y, pending, eliminated
        doses = np.asarray(assignments)
        age = clock - np.asarray(entries)
        delays = np.asarray(events)
        known = age >= np.minimum(delays, duration)
        y = np.bincount(doses[known & np.isfinite(delays)] - 1, minlength=levels)
        pending = np.bincount(doses[~known] - 1, minlength=levels)
        eliminated |= np.maximum.accumulate(y >= 2)
        decisions.append(clock)

    def enroll(amount):
        index = current - 1
        if amount < 1 or n[index] + amount > 6 or len(assignments) + amount > len(delays):
            raise ValueError("Rolling Six attempted to exceed six patients per dose")
        for position in range(amount):
            if position:
                advance()
            end = clock + duration
            if not np.isfinite(end) or end <= clock:
                raise ValueError("assessment window does not advance a finite clock")
            unsafe.append(bool(eliminated[index]))
            events.append(float(delays[len(assignments), index]))
            entries.append(clock)
            assignments.append(current)
            n[index] += 1

    for _ in range(100000):
        if not suspended:
            enroll(size)
        else:
            pauses += 1
        advance()
        observe()
        index = current - 1
        if (y[index] >= 2 and n[index] >= 3) or (profile == "safe" and eliminated.any()):
            confirm = True
        if not confirm:
            move = _move(n[index], y[index], pending[index])
            if move is None:
                suspended = True
                continue
            if current == levels and n[index] == 6 and move >= 0:
                selected = current
                break
            if current == levels and move >= 0:
                suspended = False
                continue
            if current == 1 and move == -1:
                break
            current += move
            suspended = False
            continue
        # Once a toxic level is exceeded, confirm a lower candidate with six
        # complete outcomes, rather than continuing ordinary escalation.
        while current:
            index = current - 1
            unsafe_level = eliminated[index] if profile == "safe" else y[index] >= 2
            if unsafe_level:
                current -= 1
                continue
            known_before = n[index] - pending[index]
            if n[index] < 6:
                enroll(6 - int(n[index]))
                observe()
                wait = n[index] - known_before if profile == "desktop_legacy" else pending[index]
            else:
                wait = pending[index]
            while wait > 0:
                pauses += 1
                advance()
                observe()
                wait = pending[index]
            admissible = not eliminated[index] if profile == "safe" else y[index] < 2
            if admissible:
                selected = current
                break
        break
    else:
        raise RuntimeError("bounded Rolling Six decision budget exhausted")
    final = max(entries) + duration
    actual = max(clock, final)
    doses = np.asarray(assignments)
    event = np.asarray(events)
    final_y = np.bincount(doses[np.isfinite(event)] - 1, minlength=levels)
    final_excluded = np.maximum.accumulate(final_y >= 2)
    selected_eliminated = selected is not None and bool(final_excluded[selected - 1])
    if profile == "safe" and selected_eliminated:
        selected = None
        selected_eliminated = False
    return BOINDesktopRollingTrial(
        _owned(n),
        _owned(final_y),
        _owned(doses),
        _owned(entries),
        _owned(event),
        selected,
        final if profile == "desktop_legacy" else actual,
        actual,
        pauses,
        _owned(decisions),
        _owned(unsafe),
        selected_eliminated,
        profile,
    )
