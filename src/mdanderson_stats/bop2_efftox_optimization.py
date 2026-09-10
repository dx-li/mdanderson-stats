"""BOP2 joint monitoring calibration with global and partial null error constraints."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite
from .bayesian_monitoring import _owned
from .bop2_binary import BOP2InfeasibleError, _boundaries, _candidate, _objective_constraints
from .bop2_efftox import bop2_efftox_design
from .bop2_paired import BOP2PairedDesign, BOP2PairedOperatingCharacteristics, _cells


@dataclass(frozen=True)
class BOP2EffToxOptimization:
    cutoff_scales: FloatArray
    gamma: float
    type1_error: FloatArray
    calibration_design: BOP2PairedDesign
    analysis_design: BOP2PairedDesign
    calibration_oc: BOP2PairedOperatingCharacteristics
    analysis_oc: BOP2PairedOperatingCharacteristics
    distinct_boundaries: int
    parameter_triples: int
    objective: str = "power"
    minimum_power: float | None = None


def optimize_bop2_efftox(
    max_subjects: int,
    null_rates: ArrayLike,
    alternative_rates: ArrayLike,
    *,
    type1_error: ArrayLike = (0.05, 0.1, 0.1),
    joint_rates: ArrayLike | None = None,
    efficacy_looks: ArrayLike | None = None,
    toxicity_looks: ArrayLike | None = None,
    efficacy_scales: ArrayLike | None = None,
    toxicity_scales: ArrayLike | None = None,
    gammas: ArrayLike | None = None,
    analysis_prior: ArrayLike | None = None,
    objective: str = "power",
    minimum_power: float | None = None,
    toxicity_exponent_factor: float = 1 / 3,
    equality_continues: bool = False,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2EffToxOptimization:
    """Optimize power or null expected sample size with three null error constraints.

    Expected-sample-size optimization requires minimum_power. All constraints are strict.

    Scenario order: futile/toxic, futile/safe, efficacious/toxic, efficacious/safe.
    Default joint rates assume independence; supply all four joint probabilities
    to calibrate correlated scenarios. Constraints apply to these specified points.
    """
    power_floor = _objective_constraints(objective, minimum_power)
    null, alt = finite(null_rates, "null_rates"), finite(alternative_rates, "alternative_rates")
    if (
        null.shape != (2,)
        or alt.shape != (2,)
        or not (0 < null[0] < alt[0] <= 1 and 0 <= alt[1] < null[1] < 1)
    ):
        raise ValueError(
            "require 0<null efficacy<alternative efficacy<=1 "
            "and 0<=alternative toxicity<null toxicity<1"
        )
    rates = np.array([null, [null[0], alt[1]], [alt[0], null[1]], alt])
    joint = rates.prod(axis=-1) if joint_rates is None else finite(joint_rates, "joint_rates")
    if joint.shape != (4,):
        raise ValueError("joint_rates requires all four scenario joint probabilities")
    scenarios = np.stack([_cells(p, float(j), "multiple") for p, j in zip(rates, joint)])
    alpha = finite(type1_error, "type1_error")
    if alpha.shape != (3,) or np.any((alpha <= 0) | (alpha > 1)):
        raise ValueError("type1_error requires three limits in (0,1]; one disables a constraint")
    grids = [
        finite(np.arange(50, 100) / 100 if values is None else values, name)
        for values, name in [
            (efficacy_scales, "efficacy_scales"),
            (toxicity_scales, "toxicity_scales"),
        ]
    ]
    powers = finite(np.arange(21) / 20 if gammas is None else gammas, "gammas")
    if (
        any(g.ndim != 1 or not g.size or np.any((g <= 0) | (g >= 1)) for g in grids)
        or powers.ndim != 1
        or not powers.size
        or np.any((powers < 0) | (powers > 1))
    ):
        raise ValueError("require nonempty 1D scale grids in (0,1) and gammas in [0,1]")
    triples = int(grids[0].size * grids[1].size * powers.size)
    if triples > 100000:
        raise ValueError("at most 100000 parameter triples are supported")
    baseline = bop2_efftox_design(
        max_subjects,
        null,
        cutoff_scales=[g[0] for g in grids],
        gamma=float(powers[0]),
        null_joint_rate=float(joint[0]),
        efficacy_looks=efficacy_looks,
        toxicity_looks=toxicity_looks,
        toxicity_exponent_factor=toxicity_exponent_factor,
        equality_continues=equality_continues,
        min_subjects=min_subjects,
        cohort_size=cohort_size,
    )
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    best_key: tuple[float, ...] | None = None
    best: (
        tuple[float, float, float, BOP2PairedDesign, BOP2PairedOperatingCharacteristics] | None
    ) = None
    # Compute each marginal boundary sequence once per scale/exponent combination.
    e_bound = {
        (float(s), float(g)): _boundaries(
            baseline.marginals[0], float(s), float(g), equality_continues=equality_continues
        )
        for s, g in product(grids[0], powers)
    }
    t_bound = {
        (float(s), float(g)): _boundaries(
            baseline.marginals[1],
            float(s),
            float(g) * toxicity_exponent_factor,
            equality_continues=equality_continues,
        )
        for s, g in product(grids[1], powers)
    }
    for se, st, gamma in product(grids[0], grids[1], powers):
        boundaries = (e_bound[float(se), float(gamma)], t_bound[float(st), float(gamma)])
        if boundaries in seen:
            continue
        seen.add(boundaries)
        design = BOP2PairedDesign(
            "efficacy_toxicity",
            baseline.prior,
            (
                _candidate(baseline.marginals[0], boundaries[0]),
                _candidate(baseline.marginals[1], boundaries[1]),
            ),
        )
        oc = design.operating_characteristics(scenarios)
        if np.any(oc.success_probability[:3] > alpha):
            continue
        power, en = float(oc.success_probability[3]), float(oc.expected_sample_size[0])
        if power_floor is not None and power < power_floor:
            continue
        key = (-power, en) if objective == "power" else (en, -power)
        if best_key is None or key < best_key:
            best_key, best = key, (float(se), float(st), float(gamma), design, oc)
    if best is None:
        raise BOP2InfeasibleError(
            "no grid candidate satisfies the three errors and power constraint; expand the grid"
        )
    se, st, gamma, calibration, calibration_oc = best
    analysis = (
        calibration
        if analysis_prior is None
        else bop2_efftox_design(
            max_subjects,
            null,
            cutoff_scales=[se, st],
            gamma=gamma,
            null_joint_rate=float(joint[0]),
            efficacy_looks=baseline.marginals[0].looks,
            toxicity_looks=baseline.marginals[1].looks,
            prior=analysis_prior,
            toxicity_exponent_factor=toxicity_exponent_factor,
            equality_continues=equality_continues,
        )
    )
    analysis_oc = (
        calibration_oc if analysis is calibration else analysis.operating_characteristics(scenarios)
    )
    return BOP2EffToxOptimization(
        _owned([se, st]),
        gamma,
        _owned(alpha),
        calibration,
        analysis,
        calibration_oc,
        analysis_oc,
        len(seen),
        triples,
        objective,
        power_floor,
    )
