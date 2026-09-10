"""BOP2 binary expected-sample-size and minimax design searches."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import count, scalar
from .bop2_binary import BOP2BinaryOptimization, BOP2InfeasibleError, optimize_bop2_binary


@dataclass(frozen=True)
class BOP2BinarySampleSizeOptimization:
    objective: str
    searched_sample_sizes: NDArray[np.int64]
    feasible_designs: tuple[BOP2BinaryOptimization, ...]
    best: BOP2BinaryOptimization

    @property
    def feasible_sample_sizes(self) -> NDArray[np.int64]:
        values = np.array(
            [r.calibration_design.max_subjects for r in self.feasible_designs], dtype=np.int64
        )
        values.flags.writeable = False
        return values


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
    requested = count(sample_sizes, "sample_sizes")
    if (
        requested.ndim != 1
        or not requested.size
        or np.any((requested < 1) | (requested > 200))
        or np.any(np.diff(requested) <= 0)
    ):
        raise ValueError("sample_sizes must be a strictly increasing vector of integers in [1,200]")
    sizes = requested.astype(np.int64)
    sizes.flags.writeable = False
    floor = scalar(minimum_power, "minimum_power")
    if not 0 < floor <= 1:
        raise ValueError("minimum_power must lie in (0,1]")
    if objective not in ("expected_sample_size", "minimax"):
        raise ValueError("objective must be expected_sample_size or minimax")
    schedule = None if interim_looks is None else count(interim_looks, "interim_looks")
    if schedule is not None and (
        schedule.ndim != 1 or np.any(schedule < 1) or np.any(np.diff(schedule) <= 0)
    ):
        raise ValueError("interim_looks must be a strictly increasing vector of positive integers")
    feasible: list[BOP2BinaryOptimization] = []
    for n in sizes:
        looks = None if schedule is None else np.r_[schedule[schedule < n], n]
        try:
            result = optimize_bop2_binary(
                int(n),
                null_rate,
                alternative_rate,
                type1_error=type1_error,
                endpoint=endpoint,
                looks=looks,
                cutoff_scales=cutoff_scales,
                gammas=gammas,
                analysis_prior=analysis_prior,
                objective="expected_sample_size",
                minimum_power=floor,
                min_subjects=min_subjects,
                cohort_size=cohort_size,
            )
        except BOP2InfeasibleError:
            continue
        feasible.append(result)
    if not feasible:
        raise BOP2InfeasibleError(
            "no sample size has a feasible design; expand the size or parameter grids"
        )

    def rank(result: BOP2BinaryOptimization) -> tuple[float, ...]:
        n = result.calibration_design.max_subjects
        en = float(result.calibration_oc.expected_sample_size[0])
        first = (en, n) if objective == "expected_sample_size" else (n, en)
        return (*first, -float(result.calibration_success_probability[1]))

    return BOP2BinarySampleSizeOptimization(
        objective, sizes, tuple(feasible), min(feasible, key=rank)
    )
