"""Exact finite-grid BOP2 calibration for nested or multiple efficacy endpoints."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import finite, scalar
from .bop2_binary import BOP2InfeasibleError, _boundaries, _candidate, _objective_constraints
from .bop2_paired import (
    BOP2PairedDesign,
    BOP2PairedOperatingCharacteristics,
    _cells,
    bop2_paired_design,
)


@dataclass(frozen=True)
class BOP2PairedOptimization:
    cutoff_scale: float
    gamma: float
    calibration_design: BOP2PairedDesign
    analysis_design: BOP2PairedDesign
    calibration_oc: BOP2PairedOperatingCharacteristics
    analysis_oc: BOP2PairedOperatingCharacteristics
    distinct_boundaries: int
    parameter_pairs: int
    objective: str = "power"
    minimum_power: float | None = None


def optimize_bop2_paired(
    max_subjects: int,
    null_rates: ArrayLike,
    alternative_rates: ArrayLike,
    *,
    endpoint: str = "ordinal",
    null_joint_rate: float | None = None,
    alternative_joint_rate: float | None = None,
    type1_error: float = 0.1,
    looks: ArrayLike | None = None,
    cutoff_scales: ArrayLike | None = None,
    gammas: ArrayLike | None = None,
    analysis_prior: ArrayLike | None = None,
    error_control: str = "strict",
    objective: str = "power",
    minimum_power: float | None = None,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2PairedOptimization:
    """Optimize power or null expected sample size on the supplied finite grid.

    Expected-sample-size optimization requires minimum_power and strict error control.

    Calibrate with an ESS-one Dirichlet prior centered on the null cell probabilities.
    Control error only at the specified joint null distribution. A separate analysis
    prior can change the achieved error. 'closest' permits error above nominal.
    """
    power_floor = _objective_constraints(objective, minimum_power, error_control)
    null = _cells(null_rates, null_joint_rate, endpoint)
    alternative = _cells(alternative_rates, alternative_joint_rate, endpoint)
    if not np.any(np.asarray(alternative_rates, dtype=float) > np.asarray(null_rates, dtype=float)):
        raise ValueError("alternative must improve at least one efficacy rate")
    scenarios = np.stack([null, alternative])
    alpha = scalar(type1_error, "type1_error")
    if not 0 < alpha < 1:
        raise ValueError("type1_error must lie in (0,1)")
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
    baseline = bop2_paired_design(
        max_subjects,
        null_rates,
        endpoint=endpoint,
        null_joint_rate=null_joint_rate,
        cutoff_scale=float(scales[0]),
        gamma=float(powers[0]),
        looks=looks,
        min_subjects=min_subjects,
        cohort_size=cohort_size,
    )
    seen: set[tuple[tuple[int, ...], ...]] = set()
    best_key: tuple[float, ...] | None = None
    best: tuple[float, float, BOP2PairedDesign, BOP2PairedOperatingCharacteristics] | None = None
    for scale in scales:
        for exponent in powers:
            boundaries = tuple(
                _boundaries(m, float(scale), float(exponent)) for m in baseline.marginals
            )
            if boundaries in seen:
                continue
            seen.add(boundaries)
            design = BOP2PairedDesign(
                endpoint,
                baseline.prior,
                (
                    _candidate(baseline.marginals[0], boundaries[0]),
                    _candidate(baseline.marginals[1], boundaries[1]),
                ),
            )
            oc = design.operating_characteristics(scenarios)
            error, power = map(float, oc.success_probability)
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
            "no grid candidate satisfies the error and power constraints; expand the grid"
        )
    scale, exponent, calibration, calibration_oc = best
    analysis = (
        calibration
        if analysis_prior is None
        else bop2_paired_design(
            max_subjects,
            null_rates,
            endpoint=endpoint,
            null_joint_rate=null_joint_rate,
            cutoff_scale=scale,
            gamma=exponent,
            looks=calibration.looks,
            prior=analysis_prior,
        )
    )
    analysis_oc = (
        calibration_oc if analysis is calibration else analysis.operating_characteristics(scenarios)
    )
    return BOP2PairedOptimization(
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
