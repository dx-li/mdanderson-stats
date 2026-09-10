"""Lee–Liu Phase II predictive-probability designs and exact finite-grid searches."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite, scalar
from .bayesian_monitoring import (
    BayesianMonitoringDesign,
    MonitoringOperatingCharacteristics,
    _owned,
    predictive_efficacy_design,
)


def phase2_predictive_design(
    max_subjects: int,
    null_rate: float,
    *,
    prior: ArrayLike = (1, 1),
    looks: ArrayLike | None = None,
    theta_lower: float = 0.1,
    theta_final: float = 0.9,
    theta_upper: float = 1,
    min_subjects: int = 10,
    cohort_size: int = 1,
) -> BayesianMonitoringDesign:
    """Stop for PP<theta_lower or PP>theta_upper; final efficacy requires P(p>p0)>theta_final."""
    p0 = scalar(null_rate, "null_rate")
    if not 0 < p0 < 1:
        raise ValueError("null_rate must lie in (0,1)")
    return predictive_efficacy_design(
        max_subjects,
        prior=prior,
        looks=looks,
        min_subjects=min_subjects,
        cohort_size=cohort_size,
        target_rate=p0,
        final_probability=theta_final,
        predictive_lower=theta_lower,
        predictive_upper=theta_upper,
        strict_thresholds=True,
    )


@dataclass(frozen=True)
class Phase2PredictiveCandidate:
    design: BayesianMonitoringDesign
    operating_characteristics: MonitoringOperatingCharacteristics
    threshold_pairs: FloatArray


@dataclass(frozen=True)
class Phase2PredictiveOptimization:
    objective: str
    searched_sample_sizes: NDArray[np.int64]
    feasible_designs: tuple[Phase2PredictiveCandidate, ...]
    best: Phase2PredictiveCandidate
    evaluated_boundaries: int


class Phase2PredictiveInfeasibleError(ValueError):
    """No supplied sample size and threshold pair meets the error and power limits."""


def optimize_phase2_predictive(
    sample_sizes: ArrayLike,
    null_rate: float,
    alternative_rate: float,
    *,
    type1_error: float = 0.1,
    minimum_power: float = 0.8,
    prior: ArrayLike = (1, 1),
    theta_lowers: ArrayLike | None = None,
    theta_finals: ArrayLike | None = None,
    theta_upper: float = 1,
    interim_looks: ArrayLike | None = None,
    objective: str = "power",
    min_subjects: int = 10,
    cohort_size: int = 1,
) -> Phase2PredictiveOptimization:
    """Best feasible boundary per N, with all declared-grid pairs producing that boundary.

    Maximize power or minimize null expected enrollment, under exact point-null error
    and alternative-power constraints. Every requested size is evaluated. Reported
    threshold pairs are discrete grid values, not inferred continuous rectangles.
    """
    sizes = count(sample_sizes, "sample_sizes").astype(np.int64)
    if (
        sizes.ndim != 1
        or not sizes.size
        or np.any((sizes < 1) | (sizes > 200))
        or np.any(np.diff(sizes) <= 0)
    ):
        raise ValueError("sample_sizes must be strictly increasing integers in [1,200]")
    p0, p1, alpha, power_floor = (
        scalar(x, name)
        for x, name in (
            (null_rate, "null_rate"),
            (alternative_rate, "alternative_rate"),
            (type1_error, "type1_error"),
            (minimum_power, "minimum_power"),
        )
    )
    if not 0 < p0 < p1 < 1 or not 0 < alpha < 1 or not 0 < power_floor <= 1:
        raise ValueError(
            "require 0<null_rate<alternative_rate<1, type1_error in (0,1), minimum_power in (0,1]"
        )
    if objective not in ("power", "expected_sample_size"):
        raise ValueError("objective must be power or expected_sample_size")
    lowers = finite(
        np.arange(1, 21) / 100 if theta_lowers is None else theta_lowers, "theta_lowers"
    )
    finals = finite(
        np.arange(80, 100) / 100 if theta_finals is None else theta_finals, "theta_finals"
    )
    upper = scalar(theta_upper, "theta_upper")
    if not 0 <= upper <= 1:
        raise ValueError("theta_upper must lie in [0,1]")
    if (
        lowers.ndim != 1
        or finals.ndim != 1
        or not lowers.size
        or not finals.size
        or lowers.size * finals.size > 10000
        or np.any((lowers < 0) | (lowers > upper))
        or np.any((finals < 0) | (finals > 1))
    ):
        raise ValueError(
            "require nonempty 1D threshold grids, 0<=lowers<=theta_upper, "
            "finals in [0,1], at most 10000 pairs"
        )
    schedule = None if interim_looks is None else count(interim_looks, "interim_looks")
    if schedule is not None and (
        schedule.ndim != 1 or np.any(schedule < 1) or np.any(np.diff(schedule) <= 0)
    ):
        raise ValueError("interim_looks must be a strictly increasing vector of positive counts")
    sizes.flags.writeable = False
    feasible = []
    evaluated = 0

    def rank(candidate: Phase2PredictiveCandidate) -> tuple[float, ...]:
        oc = candidate.operating_characteristics
        power, en = float(oc.positive_conclusion[1]), float(oc.expected_sample_size[0])
        return (
            (-power, en, candidate.design.max_subjects)
            if objective == "power"
            else (en, -power, candidate.design.max_subjects)
        )

    for n in sizes:
        looks = None if schedule is None else np.r_[schedule[schedule < n], n]
        final_cache: dict[int, BayesianMonitoringDesign] = {}
        candidates: dict[
            tuple[int, ...], tuple[BayesianMonitoringDesign, list[tuple[float, float]]]
        ] = {}
        template = phase2_predictive_design(
            int(n),
            p0,
            prior=prior,
            looks=looks,
            theta_lower=0,
            theta_final=float(finals[0]),
            theta_upper=upper,
            min_subjects=min_subjects,
            cohort_size=cohort_size,
        )
        final_cache[template.final_positive_min] = template
        for final in finals:
            final_min = (
                0
                if final == 0
                else int(np.count_nonzero(template.final_probability[n, : n + 1] <= final))
            )
            if final_min not in final_cache:
                final_cache[final_min] = phase2_predictive_design(
                    int(n),
                    p0,
                    prior=prior,
                    looks=looks,
                    theta_lower=0,
                    theta_final=float(final),
                    theta_upper=upper,
                    min_subjects=min_subjects,
                    cohort_size=cohort_size,
                )
            baseline = final_cache[final_min]
            for lower in lowers:
                low = np.array(
                    [
                        int(np.count_nonzero(baseline.low_probability[size, : size + 1] < lower))
                        - 1
                        for size in baseline.looks
                    ],
                    dtype=np.int64,
                )
                low.flags.writeable = False
                key = (
                    *map(int, low),
                    *map(int, baseline.positive_min),
                    baseline.final_positive_min,
                )
                if key not in candidates:
                    candidates[key] = (replace(baseline, futility_max=low), [])
                candidates[key][1].append((float(lower), float(final)))
        best: Phase2PredictiveCandidate | None = None
        for design, pairs in candidates.values():
            evaluated += 1
            oc = design.operating_characteristics([p0, p1])
            if oc.positive_conclusion[0] > alpha or oc.positive_conclusion[1] < power_floor:
                continue
            candidate = Phase2PredictiveCandidate(design, oc, _owned(pairs))
            if best is None or rank(candidate) < rank(best):
                best = candidate
        if best is not None:
            feasible.append(best)
    if not feasible:
        raise Phase2PredictiveInfeasibleError(
            "no size/threshold candidate satisfies the error and power constraints"
        )
    return Phase2PredictiveOptimization(
        objective, sizes, tuple(feasible), min(feasible, key=rank), evaluated
    )
