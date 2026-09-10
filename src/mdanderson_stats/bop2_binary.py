"""Binary-efficacy BOP2 monitoring and exact finite-grid calibration."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import finite, scalar
from .bayesian_monitoring import (
    BayesianMonitoringDesign,
    MonitoringOperatingCharacteristics,
    _inputs,
    _owned,
    _tail_table,
)


def bop2_binary_design(
    max_subjects: int,
    null_rate: float,
    *,
    cutoff_scale: float,
    gamma: float,
    looks: ArrayLike | None = None,
    prior: ArrayLike | None = None,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BayesianMonitoringDesign:
    """Stop for futility when Pr(p>p0|data) < scale*(n/N)**gamma.

    Equality continues; the final analysis uses the same rule. Tuning parameters
    are supplied, not automatically calibrated. Default prior is Beta(p0,1-p0).
    """
    p0 = scalar(null_rate, "null_rate")
    scale, exponent = scalar(cutoff_scale, "cutoff_scale"), scalar(gamma, "gamma")
    if not 0 < p0 < 1 or not 0 < scale < 1 or not 0 <= exponent <= 1:
        raise ValueError("require null_rate and cutoff_scale in (0,1), gamma in [0,1]")
    n, pair, schedule = _inputs(
        max_subjects, (p0, 1 - p0) if prior is None else prior, looks, min_subjects, cohort_size
    )
    go = _tail_table(n, pair, p0, upper=True)
    low = _tail_table(n, pair, p0, upper=False)
    cutoff = scale * (schedule / n) ** exponent
    if np.any(cutoff == 0):
        raise ArithmeticError("posterior cutoff underflows; increase cutoff_scale")
    bounds = np.array(
        [
            int(np.count_nonzero(go[size, : size + 1] < limit)) - 1
            for size, limit in zip(schedule, cutoff)
        ],
        dtype=np.int64,
    )
    bounds.flags.writeable = False
    high = schedule + 1
    high.flags.writeable = False
    return BayesianMonitoringDesign(
        "bop2_binary_efficacy",
        n,
        pair,
        schedule,
        bounds,
        high,
        int(bounds[-1] + 1),
        _owned(low),
        _owned(go),
        _owned(go),
    )


@dataclass(frozen=True)
class BOP2BinaryOptimization:
    cutoff_scale: float
    gamma: float
    calibration_design: BayesianMonitoringDesign
    analysis_design: BayesianMonitoringDesign
    calibration_oc: MonitoringOperatingCharacteristics
    analysis_oc: MonitoringOperatingCharacteristics
    distinct_boundaries: int
    parameter_pairs: int


def optimize_bop2_binary(
    max_subjects: int,
    null_rate: float,
    alternative_rate: float,
    *,
    type1_error: float = 0.1,
    looks: ArrayLike | None = None,
    cutoff_scales: ArrayLike | None = None,
    gammas: ArrayLike | None = None,
    analysis_prior: ArrayLike | None = None,
    error_control: str = "strict",
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2BinaryOptimization:
    """Maximize exact power over a declared finite grid, with null-centered calibration.

    'strict' enforces type I error <= nominal; 'closest' first minimizes its
    absolute distance to nominal and can exceed it. Break power ties by lower
    null expected sample size, then input grid order. Informative analysis priors
    never alter calibration and may change the achieved type I error.
    """
    p0, p1, alpha = (
        scalar(x, name)
        for x, name in (
            (null_rate, "null_rate"),
            (alternative_rate, "alternative_rate"),
            (type1_error, "type1_error"),
        )
    )
    if not 0 < p0 < p1 < 1 or not 0 < alpha < 1:
        raise ValueError("require 0<null<alternative<1 and type1_error in (0,1)")
    scales = finite(
        np.arange(50, 100) / 100 if cutoff_scales is None else cutoff_scales, "cutoff_scales"
    )
    powers = finite(np.arange(21) / 20 if gammas is None else gammas, "gammas")
    if (
        scales.ndim != 1
        or powers.ndim != 1
        or not scales.size
        or not powers.size
        or scales.size * powers.size > 100000
        or np.any((scales <= 0) | (scales >= 1))
        or np.any((powers < 0) | (powers > 1))
    ):
        raise ValueError(
            "require nonempty 1D grids, scales in (0,1), gammas in [0,1], <=100000 pairs"
        )
    if error_control not in ("strict", "closest"):
        raise ValueError("error_control must be strict or closest")
    baseline = bop2_binary_design(
        max_subjects,
        p0,
        cutoff_scale=float(scales[0]),
        gamma=float(powers[0]),
        looks=looks,
        min_subjects=min_subjects,
        cohort_size=cohort_size,
    )
    if baseline.max_subjects > 200:
        raise ValueError("exact grid calibration supports at most 200 subjects")
    schedule, go = baseline.looks, baseline.high_probability
    seen: set[tuple[int, ...]] = set()
    best_key: tuple[float, ...] | None = None
    best: (
        tuple[float, float, BayesianMonitoringDesign, MonitoringOperatingCharacteristics] | None
    ) = None
    for scale in scales:
        for exponent in powers:
            cutoff = scale * (schedule / max_subjects) ** exponent
            if np.any(cutoff == 0):
                raise ArithmeticError("posterior cutoff underflows; increase cutoff_scales")
            boundaries = tuple(
                int(np.count_nonzero(go[size, : size + 1] < limit)) - 1
                for size, limit in zip(schedule, cutoff)
            )
            if boundaries in seen:
                continue
            seen.add(boundaries)
            bounds = np.array(boundaries, dtype=np.int64)
            bounds.flags.writeable = False
            design = BayesianMonitoringDesign(
                baseline.method,
                baseline.max_subjects,
                baseline.prior,
                schedule,
                bounds,
                baseline.positive_min,
                boundaries[-1] + 1,
                baseline.low_probability,
                go,
                go,
            )
            oc = design.operating_characteristics([p0, p1])
            error, power = map(float, oc.positive_conclusion)
            if error_control == "strict" and error > alpha:
                continue
            key = ((abs(error - alpha),) if error_control == "closest" else ()) + (
                -power,
                float(oc.expected_sample_size[0]),
            )
            if best_key is None or key < best_key:
                best_key, best = key, (float(scale), float(exponent), design, oc)
    if best is None:
        raise ValueError("no grid candidate satisfies the type I error constraint; expand the grid")
    scale, exponent, calibration, calibration_oc = best
    analysis = (
        calibration
        if analysis_prior is None
        else bop2_binary_design(
            max_subjects,
            p0,
            cutoff_scale=scale,
            gamma=exponent,
            looks=schedule,
            prior=analysis_prior,
        )
    )
    analysis_oc = (
        calibration_oc if analysis is calibration else analysis.operating_characteristics([p0, p1])
    )
    return BOP2BinaryOptimization(
        scale,
        exponent,
        calibration,
        analysis,
        calibration_oc,
        analysis_oc,
        len(seen),
        int(scales.size * powers.size),
    )
