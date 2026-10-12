"""Desktop TITE-BOIN calendar replay with observable and explicit legacy profiles."""

from dataclasses import dataclass

import numpy as np
from scipy.special import betaincc

from ._validation import count, finite, scalar
from .boin import BOINDesign, _owned
from .tite_keyboard import toxicity_followup_weights


@dataclass(frozen=True)
class BOINDesktopTITETrial:
    patients: np.ndarray
    toxicities: np.ndarray
    doses: np.ndarray
    enrollment_times: np.ndarray
    delays: np.ndarray
    selected_dose: int | None
    final_time: float
    extra_checks: int
    stop_reason: str
    used_future_stop: bool
    profile: str
    decision_times: np.ndarray


def replay_boin_desktop_tite(
    design: BOINDesign,
    arrival_gaps,
    potential_delays,
    window: float,
    *,
    cohort_size: int = 3,
    start_dose: int = 1,
    max_pending_fraction: float = 0.5,
    trimester_probabilities=None,
    profile: str = "safe",
) -> BOINDesktopTITETrial:
    """Replay the desktop decision clock, using independent potential outcomes.

    Each first cohort member arrives at the previous decision time (initially
    zero). Gaps are consumed for later members and decision/suspension checks.
    Safe mode removes the native future-outcome stop and prevents clock reversal;
    native behavior is available only as desktop_legacy. Neither mode adopts the
    separate web app's minimum pending-follow-up rule.
    """
    if not isinstance(design, BOINDesign):
        raise ValueError("design must be BOINDesign")
    if profile not in ("safe", "desktop_legacy"):
        raise ValueError("profile must be safe or desktop_legacy")
    if any(isinstance(v, (bool, np.bool_)) for v in (cohort_size, start_dose)):
        raise ValueError("cohort settings must be integer counts, not booleans")
    if profile == "desktop_legacy" and (
        design.early_stop_patients is None or design.early_stop_patients < 6
    ):
        raise ValueError("native replay requires an explicit precision threshold of at least six")
    gaps = finite(arrival_gaps, "arrival_gaps")
    delays = np.asarray(potential_delays, dtype=float)
    duration = scalar(window, "window")
    pending_limit = scalar(max_pending_fraction, "max_pending_fraction")
    size, start = count([cohort_size, start_dose], "cohort settings")
    if (
        gaps.ndim != 1
        or not 1 <= gaps.size <= 100000
        or np.any(gaps <= 0)
        or delays.ndim != 2
        or not 2 <= delays.shape[1] <= 100
        or not 1 <= delays.shape[0] <= 1000
        or size < 1
        or delays.shape[0] % size
        or not 1 <= start <= delays.shape[1]
        or not duration > 0
        or not 0 <= pending_limit <= 0.65
        or np.any(
            ~(np.isposinf(delays) | (np.isfinite(delays) & (delays >= 0) & (delays <= duration)))
        )
    ):
        raise ValueError("invalid bounded desktop calendar inputs")
    toxicity_followup_weights([], duration, trimester_probabilities=trimester_probabilities)
    size, current = int(size), int(start)
    levels = delays.shape[1]
    enrolled: list[float] = []
    assigned: list[int] = []
    events: list[float] = []
    clock = 0.0
    gap_index = 0
    excluded = np.zeros(levels, dtype=bool)
    n = np.zeros(levels, dtype=np.int64)
    decision_times = []
    extra_checks = 0
    native_extra_checks = 0
    reason = "maximum_patients"
    future_stop = False
    stop = False

    def advance():
        nonlocal clock, gap_index
        if gap_index >= len(gaps):
            raise ValueError("arrival-gap tape exhausted before trial completion")
        updated = clock + float(gaps[gap_index])
        gap_index += 1
        if not np.isfinite(updated) or updated <= clock:
            raise ValueError("arrival gap does not advance a finite clock")
        clock = updated

    for cohort in range(delays.shape[0] // size):
        for offset in range(size):
            if offset:
                advance()
            assessment_end = clock + duration
            if not np.isfinite(assessment_end) or assessment_end <= clock:
                raise ValueError("assessment window does not advance a finite clock")
            index = len(assigned)
            enrolled.append(clock)
            assigned.append(current)
            events.append(float(delays[index, current - 1]))
            n[current - 1] += 1
        dose_vector = np.asarray(assigned)
        entry = np.asarray(enrolled)
        event = np.asarray(events)
        is_current = dose_vector == current
        total_current = int(n[current - 1])
        native_extra_checks -= 1
        if (
            profile == "desktop_legacy"
            and current == 1
            and design.early_stop_patients is not None
            and total_current >= design.early_stop_patients
            and np.count_nonzero(np.isfinite(event[is_current])) / total_current
            >= design.deescalation_boundary
        ):
            clock += duration
            reason = "legacy_future_outcome_stop"
            future_stop = True
            break
        checks_this_cohort = 0
        while True:
            if cohort == delays.shape[0] // size - 1:
                clock += duration
            else:
                advance()
            if not np.isfinite(clock):
                raise ValueError("calendar exceeded finite clock")
            decision_times.append(clock)
            checks_this_cohort += 1
            native_extra_checks += 1
            if checks_this_cohort > 1:
                extra_checks += 1
            known = entry + np.minimum(event, duration) <= clock
            observed_dlt = known & np.isfinite(event)
            y = np.bincount(dose_vector[observed_dlt] - 1, minlength=levels)
            current_y = int(y[current - 1])
            pending = is_current & ~known
            pending_count = int(pending.sum())
            followup = clock - entry[pending]
            stft = float(
                toxicity_followup_weights(
                    followup, duration, trimester_probabilities=trimester_probabilities
                ).sum()
            )
            safety = betaincc(y + 1, n - y + 1, design.target)
            excluded |= np.maximum.accumulate((n >= 3) & (safety > design.elimination_probability))
            if (
                design.extra_safe
                and current == 1
                and total_current >= 3
                and safety[0] > design.elimination_probability - design.safety_offset
            ):
                reason = "stop_safety"
                stop = True
                break
            if excluded[current - 1]:
                next_dose = int(np.flatnonzero(excluded)[0])
                if next_dose == 0:
                    reason = "stop_safety"
                    stop = True
                else:
                    current = next_dose
                break
            raw = current_y / total_current
            if raw >= design.deescalation_boundary:
                if design.stay_at_one_of_three and total_current == 3 and current_y == 1:
                    break
                if current > 1:
                    current -= 1
                    break
                if pending_count:
                    continue
                break
            if pending_count > total_current * pending_limit:
                continue
            if pending_count:
                posterior = (current_y + design.target / 2) / (total_current - pending_count + 1)
                rate = (
                    current_y + posterior / (1 - posterior) * (pending_count - stft)
                ) / total_current
            else:
                rate = raw
            le, ld = design.escalation_boundary, design.deescalation_boundary
            up_blocked = current == levels or excluded[min(current, levels - 1)]
            converged = (
                (
                    current == 1
                    and (
                        rate > le
                        or (rate <= le and raw > design.target)
                        or (
                            rate <= le
                            and raw <= design.target
                            and pending_count != total_current
                            and up_blocked
                        )
                    )
                )
                or (
                    current == levels
                    and (
                        (rate <= le and pending_count != total_current)
                        or le < rate < ld
                        or (rate >= ld and raw < design.target)
                    )
                )
                or (
                    1 < current < levels
                    and (
                        le < rate < ld
                        or (rate >= ld and raw < design.target)
                        or (rate <= le and raw > design.target)
                        or (
                            rate <= le
                            and raw <= design.target
                            and up_blocked
                            and pending_count != total_current
                        )
                    )
                )
            )
            if (
                design.early_stop_patients is not None
                and total_current >= design.early_stop_patients
                and converged
            ):
                reason = "stop_precision"
                stop = True
                break
            if rate <= le and raw <= design.target:
                if pending_count == total_current:
                    continue
                if not up_blocked:
                    current += 1
            elif rate >= ld and raw >= design.target:
                if current > 1:
                    current -= 1
                elif pending_count:
                    continue
            if excluded[current - 1]:
                current = int(np.flatnonzero(excluded)[0])
                if current == 0:
                    reason = "stop_safety"
                    stop = True
            break
        if stop:
            final_assessment = max(enrolled) + duration
            clock = (
                final_assessment if profile == "desktop_legacy" else max(clock, final_assessment)
            )
            break
    dose_vector = np.asarray(assigned)
    event = np.asarray(events)
    y = np.bincount(dose_vector[np.isfinite(event)] - 1, minlength=levels)
    selection = design.select_mtd(n, y)
    return BOINDesktopTITETrial(
        _owned(n),
        _owned(y),
        _owned(dose_vector),
        _owned(enrolled),
        _owned(event),
        None if reason == "stop_safety" else selection.dose,
        float(clock),
        native_extra_checks if profile == "desktop_legacy" else extra_checks,
        reason,
        future_stop,
        profile,
        _owned(decision_times),
    )
