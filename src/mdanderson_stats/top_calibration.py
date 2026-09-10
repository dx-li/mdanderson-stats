"""Finite-grid TOP calibration with shared potential outcomes and independent validation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite, scalar
from .bayesian_monitoring import _integer
from .boin import _owned
from .top_binary import TOPBinaryDesign
from .top_calendar import _run_top
from .top_simulation import _top_potential, simulate_top_binary


class TOPInfeasibleError(ValueError):
    """No candidate meets the estimated type I error constraint on this grid."""


@dataclass(frozen=True)
class TOPBinaryOptimization:
    design: TOPBinaryDesign
    parameter_pairs: FloatArray
    calibration_probability: FloatArray
    calibration_mcse: FloatArray
    calibration_mean_patients: FloatArray
    calibration_mean_duration: FloatArray
    feasible: NDArray[np.bool_]
    selected_index: int
    validation_probability: FloatArray
    validation_mcse: FloatArray
    validation_mean_patients: FloatArray
    validation_mean_duration: FloatArray
    type1_error: float
    alternative_rate: float
    trials: int
    validation_trials: int
    calibration_seed: int
    validation_seed: int


def optimize_top_binary(
    max_subjects: int,
    null_rate: float,
    alternative_rate: float,
    window: float,
    accrual_rate: float,
    *,
    cutoff_scales: ArrayLike,
    gammas: ArrayLike,
    type1_error: float = 0.1,
    looks: ArrayLike | None = None,
    prior: ArrayLike | None = None,
    suspension: str = "table",
    trials: int = 10000,
    validation_trials: int = 10000,
    arrival: str = "exponential",
    response_distribution: str = "uniform",
    late_probability: float | None = None,
    rng: int | np.random.Generator | None = None,
) -> TOPBinaryOptimization:
    """Maximize simulated power subject to estimated null success <= type1_error.

    Rows follow cutoff-scale then gamma input order; probability columns are
    null and alternative. Ties prefer smaller null enrollment, then input order.
    Independent validation reports performance but never reselects the design.
    Neither the search nor its holdout is a guarantee of frequentist error control.
    """
    p0, p1, alpha = (
        scalar(v, name)
        for v, name in [
            (null_rate, "null_rate"),
            (alternative_rate, "alternative_rate"),
            (type1_error, "type1_error"),
        ]
    )
    if not 0 < p0 < p1 < 1 or not 0 < alpha < 1:
        raise ValueError("require 0<null_rate<alternative_rate<1 and type1_error in (0,1)")
    scales, powers = finite(cutoff_scales, "cutoff_scales"), finite(gammas, "gammas")
    if (
        scales.ndim != 1
        or powers.ndim != 1
        or not 1 <= scales.size * powers.size <= 500
        or np.any((scales <= 0) | (scales >= 1))
        or np.any((powers < 0) | (powers > 1))
    ):
        raise ValueError("require 1..500 grid pairs, cutoff scales in (0,1) and gammas in [0,1]")
    if np.unique(scales).size != scales.size or np.unique(powers).size != powers.size:
        raise ValueError("grid values must be distinct")
    max_subjects = _integer(max_subjects, "max_subjects")
    repetitions = _integer(trials, "trials")
    validation = _integer(validation_trials, "validation_trials")
    if (
        min(repetitions, validation) < 100
        or max(repetitions, validation) > 100000
        or max(repetitions, validation) * max_subjects > 2000000
    ):
        raise ValueError("require 100..100000 trials per stage and <=2 million trial-patient cells")
    pairs = np.array([(c, g) for c in scales for g in powers])
    designs = [
        TOPBinaryDesign(
            max_subjects, p0, float(c), float(g), prior=prior, looks=looks, suspension=suspension
        )
        for c, g in pairs
    ]
    generator = np.random.default_rng(rng)
    seed, holdout_seed = map(int, generator.integers(0, np.iinfo(np.int64).max, size=2))
    # Shared quantiles and gaps across both response scenarios and all candidates.
    potentials = [
        _top_potential(
            designs[0],
            p,
            window,
            accrual_rate,
            trials=repetitions,
            arrival=arrival,
            response_distribution=response_distribution,
            late_probability=late_probability,
            rng=seed,
        )
        for p in (p0, p1)
    ]
    probability = np.empty((len(designs), 2))
    mean_n, mean_duration = np.empty_like(probability), np.empty_like(probability)
    for i, candidate in enumerate(designs):
        for j, (gaps, delays, duration) in enumerate(potentials):
            result, _, _, _ = _run_top(candidate, gaps, delays, duration)
            probability[i, j] = result.success_probability
            mean_n[i, j] = result.patients.mean()
            mean_duration[i, j] = result.duration.mean()
    feasible = probability[:, 0] <= alpha
    candidates = np.flatnonzero(feasible)
    if not candidates.size:
        raise TOPInfeasibleError(
            "no candidate meets the estimated type I error target; "
            "extend the grid or revise the sample size"
        )
    order = np.lexsort((candidates, mean_n[candidates, 0], -probability[candidates, 1]))
    selected = int(candidates[order[0]])
    chosen = designs[selected]
    validated = [
        simulate_top_binary(
            chosen,
            p,
            window,
            accrual_rate,
            trials=validation,
            arrival=arrival,
            response_distribution=response_distribution,
            late_probability=late_probability,
            rng=holdout_seed,
        )
        for p in (p0, p1)
    ]
    return TOPBinaryOptimization(
        chosen,
        _owned(pairs),
        _owned(probability),
        _owned(np.sqrt(probability * (1 - probability) / repetitions)),
        _owned(mean_n),
        _owned(mean_duration),
        _owned(feasible),
        selected,
        _owned([v.success_probability for v in validated]),
        _owned([v.success_mcse for v in validated]),
        _owned([v.patients.mean() for v in validated]),
        _owned([v.duration.mean() for v in validated]),
        alpha,
        p1,
        repetitions,
        validation,
        seed,
        holdout_seed,
    )
