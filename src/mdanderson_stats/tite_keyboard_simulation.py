"""Calendar-time TITE-Keyboard simulations with calibrated DLT timing scenarios."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite, scalar
from .boin import _owned
from .keyboard import KeyboardDesign
from .tite_keyboard import toxicity_followup_weights
from .tite_keyboard_trial import run_tite_keyboard_trial
from .toxicity_timing import toxicity_time_quantile


@dataclass(frozen=True)
class TITEKeyboardSimulation:
    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    selected_dose: NDArray[np.int64]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    duration: FloatArray
    suspension_time: FloatArray
    stop_reason: tuple[str, ...]


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
) -> TITEKeyboardSimulation:
    """Simulate binary DLT incidence and conditional DLT time separately.

    accrual_rate is patients per time unit used for window. The first arrival
    also has an interarrival gap. 'fixed' uses gaps 1/rate; 'exponential' draws
    independent exponential gaps. True event timing and analysis weights are
    separate: both default to uniform but can have distinct trimester masses.
    Weibull/log-logistic scenarios use late_probability to calibrate the fraction
    of DLTs in the late half of the window, independently of analysis weights.
    """
    p = finite(true_toxicity, "true_toxicity")
    if p.ndim != 1 or not 2 <= p.size <= 100 or np.any((p < 0) | (p > 1)):
        raise ValueError("require 2..100 toxicity probabilities in [0,1]")
    duration, rate = scalar(window, "window"), scalar(accrual_rate, "accrual_rate")
    if duration <= 0 or rate <= 0 or not np.isfinite(1 / rate):
        raise ValueError("window and accrual_rate must be positive with finite mean arrival gap")
    sizes = count([cohorts, cohort_size, trials, start_dose], "simulation settings")
    if sizes.shape != (4,) or np.any(sizes < 1):
        raise ValueError("simulation settings must be positive integers")
    nc, size, repetitions, start = map(int, sizes)
    maximum = nc * size
    if maximum > 200 or repetitions > 100000 or start > p.size:
        raise ValueError("require at most 200 patients, 100000 trials and a valid start dose")
    if arrival not in ("fixed", "exponential"):
        raise ValueError("arrival must be 'fixed' or 'exponential'")
    toxicity_followup_weights([], duration, trimester_probabilities=event_trimester_probabilities)
    event_prior = (
        None
        if event_trimester_probabilities is None
        else np.asarray(event_trimester_probabilities, dtype=float)
    )
    if event_prior is not None:
        event_prior = event_prior / event_prior.sum()
    toxicity_time_quantile(
        0.5, p, duration, distribution=event_distribution, late_probability=late_probability
    )
    if event_distribution != "uniform" and event_prior is not None:
        raise ValueError("trimester event masses require uniform timing")
    generator = np.random.default_rng(rng)
    n = np.zeros((repetitions, p.size), dtype=np.int64)
    y = np.zeros_like(n)
    selected = np.zeros(repetitions, dtype=np.int64)
    length = np.zeros(repetitions)
    pauses = np.zeros(repetitions)
    reasons = []
    for trial in range(repetitions):
        shape = (maximum, p.size)
        if event_distribution == "uniform":
            toxic = generator.random(shape) < p
            fractions = generator.random(shape)
            if event_prior is not None:
                category = generator.choice(3, size=shape, p=event_prior)
                fractions = (category + fractions) / 3
            delays = np.where(toxic, duration * fractions, np.inf)
        else:
            delays = toxicity_time_quantile(
                generator.random(shape),
                p,
                duration,
                distribution=event_distribution,
                late_probability=late_probability,
            )
        gaps = (
            np.full(maximum, 1 / rate)
            if arrival == "fixed"
            else generator.exponential(1 / rate, maximum)
        )
        result = run_tite_keyboard_trial(
            design,
            gaps,
            delays,
            duration,
            cohort_size=size,
            start_dose=start,
            trimester_probabilities=trimester_probabilities,
            pending_fraction_limit=pending_fraction_limit,
        )
        n[trial], y[trial] = result.patients, result.toxicities
        selected[trial] = 0 if result.selected_dose is None else result.selected_dose
        length[trial], pauses[trial] = result.final_time, result.suspension_time
        reasons.append(result.stop_reason)
    frequency = np.bincount(selected, minlength=p.size + 1) / repetitions
    return TITEKeyboardSimulation(
        _owned(n),
        _owned(y),
        _owned(selected),
        _owned(frequency),
        _owned(np.sqrt(frequency * (1 - frequency) / repetitions)),
        _owned(length),
        _owned(pauses),
        tuple(reasons),
    )
