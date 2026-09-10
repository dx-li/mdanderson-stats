"""Binary efficacy/toxicity BOP2 monitoring and exact finite-grid calibration."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite, scalar
from .bayesian_monitoring import (
    BayesianMonitoringDesign,
    MonitoringOperatingCharacteristics,
    _inputs,
    _owned,
    _tail_table,
)


class BOP2InfeasibleError(ValueError):
    """No design in the declared grid meets the requested operating constraints."""


def bop2_binary_design(
    max_subjects: int,
    null_rate: float,
    *,
    cutoff_scale: float,
    gamma: float,
    endpoint: str = "efficacy",
    looks: ArrayLike | None = None,
    prior: ArrayLike | None = None,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BayesianMonitoringDesign:
    """Stop when posterior acceptability < scale*(n/N)**gamma.

    Acceptability is p>p0 for efficacy or p<p0 for toxicity. Toxicity counts
    remain adverse events; high/positive conclusions then mean unsafe.

    Equality continues; the final analysis uses the same rule. Tuning parameters
    are supplied, not automatically calibrated. Default prior is Beta(p0,1-p0).
    """
    if endpoint not in ("efficacy", "toxicity"):
        raise ValueError("endpoint must be efficacy or toxicity")
    p0 = scalar(null_rate, "null_rate")
    scale, exponent = scalar(cutoff_scale, "cutoff_scale"), scalar(gamma, "gamma")
    if not 0 < p0 < 1 or not 0 < scale < 1 or not 0 <= exponent <= 1:
        raise ValueError("require null_rate and cutoff_scale in (0,1), gamma in [0,1]")
    n, pair, schedule = _inputs(
        max_subjects, (p0, 1 - p0) if prior is None else prior, looks, min_subjects, cohort_size
    )
    upper = _owned(_tail_table(n, pair, p0, upper=True))
    lower = _owned(_tail_table(n, pair, p0, upper=False))
    baseline = BayesianMonitoringDesign(
        "bop2_binary_" + endpoint,
        n,
        pair,
        schedule,
        np.full(schedule.size, -1, dtype=np.int64),
        schedule + 1,
        n + 1,
        lower,
        upper,
        upper,
    )
    return _candidate(baseline, _boundaries(baseline, scale, exponent))


def _boundaries(
    baseline: BayesianMonitoringDesign,
    scale: float,
    exponent: float,
    *,
    equality_continues: bool = True,
) -> tuple[int, ...]:
    toxicity = baseline.method == "bop2_binary_toxicity"
    go = baseline.low_probability if toxicity else baseline.high_probability
    cutoff = scale * (baseline.looks / baseline.max_subjects) ** exponent
    if np.any(cutoff == 0):
        raise ArithmeticError("posterior cutoff underflows; increase cutoff_scale")
    return tuple(
        int(
            np.count_nonzero(
                go[size, : size + 1] >= limit
                if equality_continues
                else go[size, : size + 1] > limit
            )
        )
        if toxicity
        else int(
            np.count_nonzero(
                go[size, : size + 1] < limit
                if equality_continues
                else go[size, : size + 1] <= limit
            )
        )
        - 1
        for size, limit in zip(baseline.looks, cutoff)
    )


def _candidate(
    baseline: BayesianMonitoringDesign, boundaries: tuple[int, ...]
) -> BayesianMonitoringDesign:
    toxicity = baseline.method == "bop2_binary_toxicity"
    bounds = np.array(boundaries, dtype=np.int64)
    low = np.full(bounds.size, -1, dtype=np.int64) if toxicity else bounds
    high = bounds if toxicity else baseline.looks + 1
    low.flags.writeable = high.flags.writeable = False
    return BayesianMonitoringDesign(
        baseline.method,
        baseline.max_subjects,
        baseline.prior,
        baseline.looks,
        low,
        high,
        boundaries[-1] if toxicity else boundaries[-1] + 1,
        baseline.low_probability,
        baseline.high_probability,
        baseline.final_probability,
    )


def _success(
    design: BayesianMonitoringDesign, oc: MonitoringOperatingCharacteristics
) -> FloatArray:
    return (
        oc.complete_negative if design.method == "bop2_binary_toxicity" else oc.positive_conclusion
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
    objective: str = "power"
    minimum_power: float | None = None

    @property
    def calibration_success_probability(self) -> FloatArray:
        """Null type I error and alternative power (efficacious or safe)."""
        return _success(self.calibration_design, self.calibration_oc)

    @property
    def analysis_success_probability(self) -> FloatArray:
        """Success probabilities under the separately specified analysis prior."""
        return _success(self.analysis_design, self.analysis_oc)


def optimize_bop2_binary(
    max_subjects: int,
    null_rate: float,
    alternative_rate: float,
    *,
    type1_error: float = 0.1,
    endpoint: str = "efficacy",
    looks: ArrayLike | None = None,
    cutoff_scales: ArrayLike | None = None,
    gammas: ArrayLike | None = None,
    analysis_prior: ArrayLike | None = None,
    error_control: str = "strict",
    objective: str = "power",
    minimum_power: float | None = None,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2BinaryOptimization:
    """Optimize exact power or null expected sample size on a declared finite grid.

    Expected-sample-size optimization requires a minimum power and strict error
    control. All optimization uses the null-centered calibration prior.

    'strict' enforces type I error <= nominal; 'closest' first minimizes its
    absolute distance to nominal and can exceed it. Break power ties by lower
    null expected sample size, then input grid order. Informative analysis priors
    never alter calibration and may change the achieved type I error.
    """
    if objective not in ("power", "expected_sample_size"):
        raise ValueError("objective must be power or expected_sample_size")
    power_floor = None if minimum_power is None else scalar(minimum_power, "minimum_power")
    if power_floor is not None and not 0 < power_floor <= 1:
        raise ValueError("minimum_power must lie in (0,1]")
    if objective == "expected_sample_size" and (power_floor is None or error_control != "strict"):
        raise ValueError("expected_sample_size requires minimum_power and strict error control")
    p0, p1, alpha = (
        scalar(x, name)
        for x, name in (
            (null_rate, "null_rate"),
            (alternative_rate, "alternative_rate"),
            (type1_error, "type1_error"),
        )
    )
    if endpoint not in ("efficacy", "toxicity"):
        raise ValueError("endpoint must be efficacy or toxicity")
    ordered = 0 < p0 < p1 < 1 if endpoint == "efficacy" else 0 < p1 < p0 < 1
    if not ordered or not 0 < alpha < 1:
        raise ValueError(
            "require rates in (0,1), null<alternative for efficacy or alternative<null "
            "for toxicity, and type1_error in (0,1)"
        )
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
        endpoint=endpoint,
        cutoff_scale=float(scales[0]),
        gamma=float(powers[0]),
        looks=looks,
        min_subjects=min_subjects,
        cohort_size=cohort_size,
    )
    if baseline.max_subjects > 200:
        raise ValueError("exact grid calibration supports at most 200 subjects")
    schedule = baseline.looks
    seen: set[tuple[int, ...]] = set()
    best_key: tuple[float, ...] | None = None
    best: (
        tuple[float, float, BayesianMonitoringDesign, MonitoringOperatingCharacteristics] | None
    ) = None
    for scale in scales:
        for exponent in powers:
            boundaries = _boundaries(baseline, float(scale), float(exponent))
            if boundaries in seen:
                continue
            seen.add(boundaries)
            design = _candidate(baseline, boundaries)
            oc = design.operating_characteristics([p0, p1])
            error, power = map(float, _success(design, oc))
            if error_control == "strict" and error > alpha:
                continue
            if power_floor is not None and power < power_floor:
                continue
            en = float(oc.expected_sample_size[0])
            ranking = (-power, en) if objective == "power" else (en, -power)
            key = ((abs(error - alpha),) if error_control == "closest" else ()) + ranking
            if best_key is None or key < best_key:
                best_key, best = key, (float(scale), float(exponent), design, oc)
    if best is None:
        raise BOP2InfeasibleError(
            "no grid candidate satisfies the type I error and power constraints; expand the grid"
        )
    scale, exponent, calibration, calibration_oc = best
    analysis = (
        calibration
        if analysis_prior is None
        else bop2_binary_design(
            max_subjects,
            p0,
            endpoint=endpoint,
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
        objective,
        power_floor,
    )
