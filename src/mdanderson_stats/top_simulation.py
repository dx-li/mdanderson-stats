"""TOP delayed-response simulation using existing calibrated event-time quantiles."""

import numpy as np

from ._validation import scalar
from .bayesian_monitoring import _integer
from .top_binary import TOPBinaryDesign
from .top_calendar import TOPBinarySimulation, _run_top
from .toxicity_timing import toxicity_time_quantile


def simulate_top_binary(
    design: TOPBinaryDesign,
    response_probability: float,
    window: float,
    accrual_rate: float,
    *,
    trials: int = 10000,
    arrival: str = "exponential",
    response_distribution: str = "uniform",
    late_probability: float | None = None,
    rng: int | np.random.Generator | None = None,
) -> TOPBinarySimulation:
    """Batched complete-calendar trials, using uniform TOP analysis weights.

    Arrival gaps are fixed or exponential with mean 1/accrual_rate, including
    the first gap. Timing alternatives use the shared Weibull/log-logistic
    quantile model with conditional late-half probability. Calibrating decisions
    to control type I error is a separate operation, not performed here.
    """
    gaps, delays, duration = _top_potential(
        design,
        response_probability,
        window,
        accrual_rate,
        trials=trials,
        arrival=arrival,
        response_distribution=response_distribution,
        late_probability=late_probability,
        rng=rng,
    )
    result, _, _, _ = _run_top(design, gaps, delays, duration)
    return result


def _top_potential(
    design: TOPBinaryDesign,
    response_probability: float,
    window: float,
    accrual_rate: float,
    *,
    trials: int,
    arrival: str,
    response_distribution: str,
    late_probability: float | None,
    rng: int | np.random.Generator | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    repetitions = _integer(trials, "trials")
    rate = scalar(accrual_rate, "accrual_rate")
    duration = scalar(window, "window")
    p = scalar(response_probability, "response_probability")
    if not 1 <= repetitions <= 100000 or repetitions * design.max_subjects > 2000000:
        raise ValueError("require 1..100000 trials and at most 2 million trial-patient cells")
    if (
        duration <= 0
        or rate <= 0
        or not np.isfinite(1 / rate)
        or not 0 <= p <= 1
        or arrival not in ("fixed", "exponential")
    ):
        raise ValueError(
            "require positive finite window/mean arrival gap, probability in [0,1] "
            "and fixed/exponential arrival"
        )
    # Validate timing before drawing any random numbers.
    toxicity_time_quantile(
        0.5, p, duration, distribution=response_distribution, late_probability=late_probability
    )
    generator = np.random.default_rng(rng)
    shape = (repetitions, design.max_subjects)
    delays = toxicity_time_quantile(
        generator.random(shape),
        p,
        duration,
        distribution=response_distribution,
        late_probability=late_probability,
    )
    gaps = (
        np.full(shape, 1 / rate) if arrival == "fixed" else generator.exponential(1 / rate, shape)
    )
    return gaps, delays, duration
