"""BOP2 binary expected-sample-size and minimax design searches."""

from dataclasses import dataclass

from numpy.typing import ArrayLike

from ._bop2_sample_size import SampleSizeResult, interim_schedule, looks_at, search_sizes
from .bop2_binary import BOP2BinaryOptimization, optimize_bop2_binary


@dataclass(frozen=True)
class BOP2BinarySampleSizeOptimization(SampleSizeResult[BOP2BinaryOptimization]):
    """Binary efficacy/toxicity sample-size search results."""


def optimize_bop2_binary_sample_size(
    sample_sizes: ArrayLike,
    null_rate: float,
    alternative_rate: float,
    *,
    minimum_power: float,
    type1_error: float = 0.1,
    endpoint: str = "efficacy",
    objective: str = "expected_sample_size",
    interim_looks: ArrayLike | None = None,
    cutoff_scales: ArrayLike | None = None,
    gammas: ArrayLike | None = None,
    analysis_prior: ArrayLike | None = None,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2BinarySampleSizeOptimization:
    """Search a declared size grid under strict null-error and minimum-power constraints.

    Minimize null expected sample size, or maximum sample size ('minimax'). Within
    each N, minimize null expected sample size then maximize power. For expected
    sample-size ties across N, prefer smaller N, then higher power. Always return
    the best feasible design at each searched N; infeasible sizes are omitted.
    """
    schedule = interim_schedule(interim_looks)

    def optimize(n: int, floor: float) -> BOP2BinaryOptimization:
        return optimize_bop2_binary(
            n,
            null_rate,
            alternative_rate,
            type1_error=type1_error,
            endpoint=endpoint,
            looks=looks_at(schedule, n),
            cutoff_scales=cutoff_scales,
            gammas=gammas,
            analysis_prior=analysis_prior,
            objective="expected_sample_size",
            minimum_power=floor,
            min_subjects=min_subjects,
            cohort_size=cohort_size,
        )

    return BOP2BinarySampleSizeOptimization(
        objective, *search_sizes(sample_sizes, minimum_power, objective, optimize)
    )
