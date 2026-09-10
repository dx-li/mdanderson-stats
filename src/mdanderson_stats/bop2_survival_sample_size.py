"""Monte Carlo expected-enrollment and minimax BOP2 survival size searches."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._bop2_sample_size import SampleSizeResult, interim_schedule, looks_at, search_sizes
from .bop2_survival_optimization import BOP2SurvivalOptimization, optimize_bop2_survival


@dataclass(frozen=True)
class BOP2SurvivalSampleSizeOptimization(SampleSizeResult[BOP2SurvivalOptimization]):
    """Empirically feasible designs and per-size seeds in searched-size order."""

    simulation_seeds: NDArray[np.int64]


def optimize_bop2_survival_sample_size(
    sample_sizes: ArrayLike,
    null_median: float,
    alternative_median: float,
    *,
    minimum_power: float,
    accrual_rate: float,
    final_followup: float,
    type1_error: float = 0.1,
    objective: str = "expected_sample_size",
    interim_looks: ArrayLike | None = None,
    cutoff_scales: ArrayLike | None = None,
    gammas: ArrayLike | None = None,
    analysis_prior: ArrayLike | None = None,
    n_trials: int = 10000,
    n_validation: int = 10000,
    arrival: str = "fixed",
    rng: np.random.Generator | int | None = None,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2SurvivalSampleSizeOptimization:
    """Search all supplied sizes under empirical null-error and power constraints.

    Within each N, minimize null expected enrollment then maximize power. Across
    N, minimize null expected enrollment or maximum N ('minimax'). Independent
    validation is reported but never used for selection or feasibility filtering.
    These Monte Carlo constraints do not guarantee true error control or power.
    """
    schedule = interim_schedule(interim_looks)
    generator = np.random.default_rng(rng)
    seeds: list[int] = []

    def optimize(n: int, floor: float) -> BOP2SurvivalOptimization:
        # Separate streams prevent validation/prior changes from altering later N fits.
        seed = int(generator.integers(0, np.iinfo(np.int64).max))
        seeds.append(seed)
        return optimize_bop2_survival(
            n,
            null_median,
            alternative_median,
            accrual_rate=accrual_rate,
            final_followup=final_followup,
            type1_error=type1_error,
            looks=looks_at(schedule, n),
            cutoff_scales=cutoff_scales,
            gammas=gammas,
            analysis_prior=analysis_prior,
            objective="expected_sample_size",
            minimum_power=floor,
            n_trials=n_trials,
            n_validation=n_validation,
            arrival=arrival,
            rng=seed,
            min_subjects=min_subjects,
            cohort_size=cohort_size,
        )

    sizes, feasible, best = search_sizes(sample_sizes, minimum_power, objective, optimize)
    seed_array = np.array(seeds, dtype=np.int64)
    seed_array.flags.writeable = False
    return BOP2SurvivalSampleSizeOptimization(objective, sizes, feasible, best, seed_array)
