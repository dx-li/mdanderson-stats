"""Calendar-time TITE-Keyboard simulations with calibrated DLT timing scenarios."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike

from ._tite_simulation import CalendarSimulation, simulate_calendar
from ._validation import FloatArray
from .keyboard import KeyboardDesign
from .tite_keyboard_adaptive_calendar import TITEKeyboardAdaptiveSettings
from .tite_keyboard_trial import (
    TITEKeyboardAdaptiveTrial,
    TITEKeyboardTrial,
    run_tite_keyboard_trial,
)


@dataclass(frozen=True)
class TITEKeyboardSimulation(CalendarSimulation):
    adaptive_work_units: int = 0
    adaptive_fit_count: int = 0
    adaptive_diagnostics_passed: int = 0
    max_adaptive_split_rhat: float | None = None
    max_adaptive_weight_mcse: float | None = None
    outcome_seed: int | None = None
    sampler_seed: int | None = None


def simulate_tite_keyboard(
    design: KeyboardDesign,
    true_toxicity: ArrayLike,
    window: float,
    accrual_rate: float,
    *,
    cohorts: int = 10,
    cohort_size: int = 3,
    trials: int = 1000,
    start_dose: int = 1,
    arrival: str = "fixed",
    event_distribution: str = "uniform",
    late_probability: ArrayLike | None = None,
    event_trimester_probabilities: ArrayLike | None = None,
    trimester_probabilities: ArrayLike | None = None,
    pending_fraction_limit: float | None = 0.5,
    rng: int | np.random.Generator | None = None,
    adaptive_timing: TITEKeyboardAdaptiveSettings | None = None,
) -> TITEKeyboardSimulation:
    """Simulate binary DLT incidence and conditional DLT time separately.

    accrual_rate is patients per time unit used for window. The first arrival
    also has an interarrival gap. 'fixed' uses gaps 1/rate; 'exponential' draws
    independent exponential gaps. True event timing and analysis weights are
    separate: both default to uniform but can have distinct trimester masses.
    Weibull/log-logistic scenarios use late_probability to calibrate the fraction
    of DLTs in the late half of the window, independently of analysis weights.
    """

    outcome_rng: int | np.random.Generator | None
    sampler_rng: np.random.Generator | None
    adaptive_seeds: tuple[int, int] | None
    if adaptive_timing is not None:
        if not isinstance(adaptive_timing, TITEKeyboardAdaptiveSettings):
            raise TypeError("adaptive_timing must be TITEKeyboardAdaptiveSettings or None")
        if rng is None:
            raise ValueError("adaptive calendar simulation requires an explicit rng")
        if isinstance(rng, np.random.Generator):
            raise ValueError("adaptive calendar simulation requires an integer seed")
        if trimester_probabilities is not None:
            raise ValueError("trimester_probabilities cannot be combined with adaptive timing")
        outcome_rng, sampler_rng, adaptive_seeds = _split_adaptive_rngs(rng)
    else:
        outcome_rng, sampler_rng = rng, None
        adaptive_seeds = None
    aggregate_work = 0
    aggregate_fits = 0
    aggregate_passed = 0
    max_split_rhat = 0.0
    max_weight_mcse = 0.0

    def replay(
        gaps: FloatArray, delays: FloatArray, duration: float, size: int, start: int
    ) -> TITEKeyboardTrial | TITEKeyboardAdaptiveTrial:
        nonlocal aggregate_work, aggregate_fits, aggregate_passed, max_split_rhat, max_weight_mcse
        if adaptive_timing is None:
            return run_tite_keyboard_trial(
                design,
                gaps,
                delays,
                duration,
                cohort_size=size,
                start_dose=start,
                trimester_probabilities=trimester_probabilities,
                pending_fraction_limit=pending_fraction_limit,
            )
        assert sampler_rng is not None
        remaining = adaptive_timing.max_total_work - aggregate_work
        if remaining <= 0:
            raise ValueError("adaptive simulation exhausted max_total_work before the next trial")
        trial_settings = replace(adaptive_timing, max_total_work=remaining)
        fit_rng = np.random.default_rng(int(sampler_rng.integers(0, 2**63, dtype=np.int64)))
        result = run_tite_keyboard_trial(
            design,
            gaps,
            delays,
            duration,
            cohort_size=size,
            start_dose=start,
            pending_fraction_limit=pending_fraction_limit,
            adaptive_timing=trial_settings,
            adaptive_rng=fit_rng,
        )
        assert isinstance(result, TITEKeyboardAdaptiveTrial)
        aggregate_work += result.adaptive_work_units
        aggregate_fits += result.adaptive_fit_count
        for step in result.steps:
            diagnostics = step.adaptive_fit
            if diagnostics is None:
                continue
            aggregate_passed += int(diagnostics.diagnostics_passed)
            max_split_rhat = max(max_split_rhat, diagnostics.max_log_shape_split_rhat)
            max_split_rhat = max(max_split_rhat, diagnostics.max_pending_weight_split_rhat)
            max_weight_mcse = max(max_weight_mcse, diagnostics.max_pending_weight_mcse)
        return result

    result = simulate_calendar(
        replay,
        true_toxicity,
        window,
        accrual_rate,
        cohorts,
        cohort_size,
        trials,
        start_dose,
        arrival,
        event_distribution,
        late_probability,
        event_trimester_probabilities,
        outcome_rng,
        max_output_cells=2_000_000 if adaptive_timing is not None else None,
    )
    return TITEKeyboardSimulation(
        result.patients,
        result.toxicities,
        result.selected_dose,
        result.selection_probability,
        result.selection_mcse,
        result.duration,
        result.suspension_time,
        result.stop_reason,
        aggregate_work,
        aggregate_fits,
        aggregate_passed,
        max_split_rhat if aggregate_fits else None,
        max_weight_mcse if aggregate_fits else None,
        None if adaptive_seeds is None else adaptive_seeds[0],
        None if adaptive_seeds is None else adaptive_seeds[1],
    )


def _split_adaptive_rngs(
    rng: int,
) -> tuple[np.random.Generator, np.random.Generator, tuple[int, int]]:
    if isinstance(rng, bool) or not isinstance(rng, (int, np.integer)) or int(rng) < 0:
        raise ValueError("adaptive simulation rng must be a nonnegative integer seed")
    children = np.random.SeedSequence(int(rng)).spawn(2)
    seeds = (
        int(children[0].generate_state(1, dtype=np.uint64)[0]),
        int(children[1].generate_state(1, dtype=np.uint64)[0]),
    )
    outcome_seed, sampler_seed = seeds
    return np.random.default_rng(outcome_seed), np.random.default_rng(sampler_seed), seeds
