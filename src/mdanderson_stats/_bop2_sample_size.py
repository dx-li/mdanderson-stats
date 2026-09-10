"""Shared discrete-size search for categorical and survival BOP2 calibration families."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, scalar
from .bop2_binary import BOP2BinaryOptimization, BOP2InfeasibleError
from .bop2_efftox_optimization import BOP2EffToxOptimization
from .bop2_paired_optimization import BOP2PairedOptimization
from .bop2_survival_optimization import BOP2SurvivalOptimization

type _Fit = (
    BOP2BinaryOptimization
    | BOP2PairedOptimization
    | BOP2EffToxOptimization
    | BOP2SurvivalOptimization
)


@dataclass(frozen=True)
class SampleSizeResult[T: _Fit]:
    objective: str
    searched_sample_sizes: NDArray[np.int64]
    feasible_designs: tuple[T, ...]
    best: T

    @property
    def feasible_sample_sizes(self) -> NDArray[np.int64]:
        values = np.array(
            [r.calibration_design.max_subjects for r in self.feasible_designs], dtype=np.int64
        )
        values.flags.writeable = False
        return values


def interim_schedule(looks: ArrayLike | None) -> FloatArray | None:
    schedule = None if looks is None else count(looks, "interim_looks")
    if schedule is not None and (
        schedule.ndim != 1 or np.any(schedule < 1) or np.any(np.diff(schedule) <= 0)
    ):
        raise ValueError("interim_looks must be a strictly increasing vector of positive integers")
    return schedule


def looks_at(schedule: FloatArray | None, n: int) -> FloatArray | None:
    return None if schedule is None else np.r_[schedule[schedule < n], n]


def search_sizes[T: _Fit](
    sample_sizes: ArrayLike,
    minimum_power: float,
    objective: str,
    optimize: Callable[[int, float], T],
) -> tuple[NDArray[np.int64], tuple[T, ...], T]:
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
    feasible: list[T] = []
    for n in sizes:
        try:
            result = optimize(int(n), floor)
        except BOP2InfeasibleError:
            continue
        feasible.append(result)
    if not feasible:
        raise BOP2InfeasibleError(
            "no sample size has a feasible design; expand the size or parameter grids"
        )

    def rank(result: T) -> tuple[float, ...]:
        n = result.calibration_design.max_subjects
        en = float(result.calibration_oc.expected_sample_size[0])
        success = (
            result.calibration_success_probability
            if isinstance(result, BOP2BinaryOptimization)
            else result.calibration_oc.success_probability
        )
        first = (en, n) if objective == "expected_sample_size" else (n, en)
        return (*first, -float(success[-1]))

    return sizes, tuple(feasible), min(feasible, key=rank)
