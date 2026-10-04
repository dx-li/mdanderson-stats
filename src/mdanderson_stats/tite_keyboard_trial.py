"""Calendar-time TITE-Keyboard trial replay."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._tite_calendar import CalendarStep, CalendarTrial, run_calendar_trial
from ._validation import FloatArray, scalar
from .boin import _owned
from .keyboard import KeyboardDesign
from .tite_keyboard import TITEKeyboardDecision, tite_keyboard_decision, toxicity_followup_weights
from .tite_keyboard_adaptive import tite_keyboard_adaptive_weights
from .tite_keyboard_adaptive_calendar import (
    TITEKeyboardAdaptiveFitDiagnostics,
    TITEKeyboardAdaptiveSettings,
    TITEKeyboardAdaptiveStep,
    TITEKeyboardAdaptiveTrial,
)


@dataclass(frozen=True)
class TITEKeyboardStep(CalendarStep[TITEKeyboardDecision]):
    pass


@dataclass(frozen=True)
class TITEKeyboardTrial(CalendarTrial[TITEKeyboardStep]):
    pass


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
    adaptive_timing: TITEKeyboardAdaptiveSettings | None = None,
    adaptive_rng: np.random.Generator | None = None,
) -> TITEKeyboardTrial | TITEKeyboardAdaptiveTrial:
    """Conduct one trial from potential outcomes without exposing future events.

    dlt_delays[patient,dose] is time since enrollment to DLT, or +inf for no DLT
    within the window. Arrival gaps restart after suspensions; no queue builds up.
    Cohorts have a fixed dose and staggered enrollment. Decisions occur before
    each new cohort, and at outcome ascertainments while that cohort is waiting.
    """
    if not isinstance(design, KeyboardDesign):
        raise ValueError("design must be a KeyboardDesign")
    if adaptive_timing is not None and not isinstance(
        adaptive_timing, TITEKeyboardAdaptiveSettings
    ):
        raise TypeError("adaptive_timing must be TITEKeyboardAdaptiveSettings or None")
    if adaptive_timing is None and adaptive_rng is not None:
        raise ValueError("adaptive_rng requires adaptive_timing settings")
    if adaptive_timing is not None:
        if adaptive_rng is None or not isinstance(adaptive_rng, np.random.Generator):
            raise ValueError("adaptive calendar replay requires an explicit numpy Generator")
        if trimester_probabilities is not None:
            raise ValueError("trimester_probabilities cannot be combined with adaptive timing")
    toxicity_followup_weights([], window, trimester_probabilities=trimester_probabilities)
    if pending_fraction_limit is not None:
        limit = scalar(pending_fraction_limit, "pending_fraction_limit")
        if not 0 < limit <= 0.65:
            raise ValueError("pending_fraction_limit must be in (0,.65] or None")

    event_rows: tuple[FloatArray, ...] = tuple()
    last_fit: TITEKeyboardAdaptiveFitDiagnostics | None = None
    adaptive_work = 0
    adaptive_fit_count = 0

    def observe_events(rows: tuple[FloatArray, ...]) -> None:
        nonlocal event_rows
        event_rows = rows

    def decide(
        n: ArrayLike, y: ArrayLike, times: list[FloatArray], current: int, excluded: ArrayLike
    ) -> TITEKeyboardDecision:
        nonlocal last_fit, adaptive_work, adaptive_fit_count
        last_fit = None
        if adaptive_timing is not None and bool(sum(row.size for row in times)):
            assert adaptive_rng is not None
            ordinary = tite_keyboard_decision(
                design,
                n,
                y,
                times,
                current,
                window,
                pending_fraction_limit=pending_fraction_limit,
                eliminated=excluded,
            )
            current_index = int(current) - 1
            if ordinary.action in ("stop_safety", "suspend_pending") or (
                ordinary.action == "deescalate" and ordinary.eliminated[current_index]
            ):
                return ordinary
            remaining = adaptive_timing.max_total_work - adaptive_work
            if remaining <= 0:
                raise ValueError("adaptive trial exhausted max_total_work before the next fit")
            fit = tite_keyboard_adaptive_weights(
                event_rows,
                np.asarray(n, dtype=np.int64)
                - np.asarray(y, dtype=np.int64)
                - np.asarray([row.size for row in times], dtype=np.int64),
                times,
                window,
                lambda_prior=adaptive_timing.lambda_prior,
                gamma_prior=adaptive_timing.gamma_prior,
                chains=adaptive_timing.chains,
                draws=adaptive_timing.draws,
                warmup=adaptive_timing.warmup,
                rng=adaptive_rng,
                max_work=min(adaptive_timing.max_fit_work, remaining),
                max_split_rhat=adaptive_timing.max_split_rhat,
                max_weight_mcse=adaptive_timing.max_weight_mcse,
            )
            if not fit.diagnostics_passed:
                raise ArithmeticError(
                    "adaptive timing sampler did not meet configured R-hat/weight-MCSE checks"
                )
            adaptive_work += fit.work_units
            adaptive_fit_count += 1
            summary = fit.log_shape_summary
            finite_weight_rhat = fit.weight_split_rhat[np.isfinite(fit.weight_split_rhat)]
            last_fit = TITEKeyboardAdaptiveFitDiagnostics(
                _owned(summary.mean),
                _owned(summary.batch_mean_mcse),
                _owned(summary.split_rhat),
                _owned(fit.acceptance_rate),
                float(np.max(summary.split_rhat)),
                float(np.max(fit.weight_mcse, initial=0.0)),
                float(np.max(finite_weight_rhat, initial=1.0)),
                int(sum(row.size for row in event_rows)),
                int(sum(row.size for row in times)),
                fit.draws,
                fit.warmup,
                fit.evaluations,
                fit.work_units,
                fit.prior_only,
                fit.diagnostics_passed,
            )
            return tite_keyboard_decision(
                design,
                n,
                y,
                times,
                current,
                window,
                pending_weights=fit.pending_weights,
                pending_fraction_limit=pending_fraction_limit,
                eliminated=excluded,
            )
        return tite_keyboard_decision(
            design,
            n,
            y,
            times,
            current,
            window,
            trimester_probabilities=trimester_probabilities,
            pending_fraction_limit=pending_fraction_limit,
            eliminated=excluded,
        )

    def make_adaptive_step(
        time: float, current: int, decision: TITEKeyboardDecision
    ) -> TITEKeyboardAdaptiveStep:
        return TITEKeyboardAdaptiveStep(time, current, decision, last_fit)

    def make_standard_step(
        time: float, current: int, decision: TITEKeyboardDecision
    ) -> TITEKeyboardStep:
        return TITEKeyboardStep(time, current, decision)

    if adaptive_timing is not None:
        adaptive_result = run_calendar_trial(
            design,
            interarrival,
            dlt_delays,
            window,
            cohort_size,
            start_dose,
            decide,
            make_adaptive_step,
            on_observed_events=observe_events,
        )
        return TITEKeyboardAdaptiveTrial(
            adaptive_result.enrollment_times,
            adaptive_result.assigned_doses,
            adaptive_result.dlt_times,
            adaptive_result.patients,
            adaptive_result.toxicities,
            adaptive_result.selected_dose,
            adaptive_result.eliminated,
            adaptive_result.stop_reason,
            adaptive_result.final_time,
            adaptive_result.suspension_time,
            adaptive_result.steps,
            adaptive_work,
            adaptive_fit_count,
        )
    ordinary_result = run_calendar_trial(
        design,
        interarrival,
        dlt_delays,
        window,
        cohort_size,
        start_dose,
        decide,
        make_standard_step,
    )
    return TITEKeyboardTrial(
        ordinary_result.enrollment_times,
        ordinary_result.assigned_doses,
        ordinary_result.dlt_times,
        ordinary_result.patients,
        ordinary_result.toxicities,
        ordinary_result.selected_dose,
        ordinary_result.eliminated,
        ordinary_result.stop_reason,
        ordinary_result.final_time,
        ordinary_result.suspension_time,
        ordinary_result.steps,
    )
