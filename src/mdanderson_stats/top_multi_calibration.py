"""Finite-grid TOP two-endpoint calibration with independent validation."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite, scalar
from .bayesian_monitoring import _integer
from .boin import _owned
from .top_calibration import TOPInfeasibleError
from .top_endpoints import TOPMultiEndpointDesign
from .top_multi_calendar import _run_top_multiendpoint_batch
from .top_multi_simulation import (
    _draw_multiendpoint_uniforms,
    _multiendpoint_potential_from_uniforms,
    _timing_probabilities,
    _TopMultiUniforms,
)

_DECISIONS = ("success", "stop_futility", "stop_toxicity", "stop_futility_toxicity")
_MAX_STAGE_CELLS = 2_000_000
_MAX_WORK_HARD = 100_000_000


@dataclass(frozen=True)
class TOPMultiEndpointOptimization:
    """Finite-grid search with separate calibration and holdout summaries.

    Probability arrays have a final scenario axis ordered as the supplied null
    rows followed by the alternative. Decision-probability arrays add a final
    action axis in ``decision_labels`` order. Feasibility applies to every
    supplied null scenario; it is not a guarantee over an unspecified composite
    null.
    """

    design: TOPMultiEndpointDesign
    parameter_pairs: FloatArray
    null_joint_probabilities: FloatArray
    alternative_joint_probabilities: FloatArray
    calibration_probability: FloatArray
    calibration_mcse: FloatArray
    calibration_mean_patients: FloatArray
    calibration_mean_duration: FloatArray
    calibration_decision_probability: FloatArray
    feasible: NDArray[np.bool_]
    selected_index: int
    validation_probability: FloatArray
    validation_mcse: FloatArray
    validation_mean_patients: FloatArray
    validation_mean_duration: FloatArray
    validation_decision_probability: FloatArray
    type1_error: float
    trials: int
    validation_trials: int
    calibration_seed: int
    validation_seed: int
    decision_labels: tuple[str, ...] = _DECISIONS


def _joint_scenarios(value: ArrayLike, name: str, *, rows: bool) -> FloatArray:
    shape = np.shape(value)
    valid_shapes = ((4,),) if not rows else ()
    if rows:
        if len(shape) != 2 or shape[1] != 4 or not 1 <= shape[0] <= 20:
            raise ValueError(f"{name} must have shape (scenarios, 4), with 1..20 scenarios")
    elif shape not in valid_shapes:
        raise ValueError(f"{name} must contain four joint-cell probabilities")
    probabilities = finite(value, name).copy()
    if probabilities.ndim == 1:
        probabilities = probabilities[None, :]
    if np.any(probabilities < 0):
        raise ValueError(f"{name} must contain nonnegative probabilities")
    totals = probabilities.sum(axis=1)
    if np.any(~np.isfinite(totals)) or np.any(np.abs(totals - 1) > 32 * np.finfo(float).eps):
        raise ValueError(f"each row of {name} must sum to one")
    probabilities /= totals[:, None]
    return probabilities


def _validate_hypotheses(
    design: TOPMultiEndpointDesign, null: FloatArray, alternative: FloatArray
) -> None:
    """Allow only float-roundoff slack at the marginal hypothesis boundaries."""
    null_margins = design.marginal_null
    tolerance = 32 * np.finfo(float).eps
    null_events = np.column_stack((null[:, 0] + null[:, 1], null[:, 0] + null[:, 2]))
    alternative_events = np.array(
        [alternative[0] + alternative[1], alternative[0] + alternative[2]]
    )
    if design.mode == "coprimary":
        null_valid = np.all(null_events <= null_margins + tolerance, axis=1)
        alternative_valid = np.any(alternative_events > null_margins + tolerance)
        description = (
            "coprimary null rows must be below both margins and the alternative above either"
        )
    else:
        null_valid = (null_events[:, 0] <= null_margins[0] + tolerance) | (
            null_events[:, 1] >= null_margins[1] - tolerance
        )
        alternative_valid = (alternative_events[0] > null_margins[0] + tolerance) and (
            alternative_events[1] < null_margins[1] - tolerance
        )
        description = (
            "efficacy/toxicity null rows must be adverse on an endpoint and the "
            "alternative acceptable on both"
        )
    if not np.all(null_valid) or not alternative_valid:
        raise ValueError(description)


def _run_scenario(
    design: TOPMultiEndpointDesign,
    joint: FloatArray,
    accrual_rate: float,
    arrival: str,
    timing: FloatArray,
    uniforms: _TopMultiUniforms,
    scan_budget: list[int],
) -> tuple[float, float, float, float, FloatArray]:
    gaps, delays = _multiendpoint_potential_from_uniforms(
        design, joint, accrual_rate, arrival, timing, uniforms
    )
    result = _run_top_multiendpoint_batch(design, gaps, delays, scan_budget=scan_budget)
    trials = gaps.shape[0]
    decisions = result.decisions
    probability = float(np.mean(decisions == "success"))
    mcse = float(np.sqrt(probability * (1 - probability) / trials))
    actions = np.asarray([np.mean(decisions == x) for x in _DECISIONS])
    return (
        probability,
        mcse,
        float(result.patients.mean()),
        float(result.final_times.mean()),
        actions,
    )


def optimize_top_multiendpoint(
    design: TOPMultiEndpointDesign,
    null_joint_probabilities: ArrayLike,
    alternative_joint_probabilities: ArrayLike,
    accrual_rate: float,
    *,
    cutoff_scales: ArrayLike,
    gammas: ArrayLike,
    type1_error: float = 0.1,
    trials: int = 10000,
    validation_trials: int = 10000,
    arrival: str = "exponential",
    truth_timing_probabilities: ArrayLike | None = None,
    rng: int | np.random.Generator | None = None,
    max_work: int = 50_000_000,
) -> TOPMultiEndpointOptimization:
    """Optimize finite TOP grids against explicit joint null and alternative cells.

    Joint probability rows use cell order ``(1,1), (1,0), (0,1), (0,0)``.
    Common arrival, cell, timing-component and within-third uniforms are shared
    across every scenario and candidate. A separate holdout stream evaluates
    the selected candidate without reselection. Candidate ties prefer smaller
    worst-null mean enrollment, then original grid order. This is finite-grid,
    Monte Carlo calibration against the supplied null scenarios only; it does
    not establish error control over an unspecified composite null.
    """
    if not isinstance(design, TOPMultiEndpointDesign):
        raise TypeError("design must be a TOPMultiEndpointDesign")
    null = _joint_scenarios(null_joint_probabilities, "null_joint_probabilities", rows=True)
    alt = _joint_scenarios(
        alternative_joint_probabilities, "alternative_joint_probabilities", rows=False
    )[0]
    _validate_hypotheses(design, null, alt)
    alpha = scalar(type1_error, "type1_error")
    if not 0 < alpha < 1:
        raise ValueError("type1_error must be in (0,1)")
    scale_shape, gamma_shape = np.shape(cutoff_scales), np.shape(gammas)
    if (
        len(scale_shape) != 1
        or len(gamma_shape) != 1
        or not 1 <= scale_shape[0] * gamma_shape[0] <= 100
    ):
        raise ValueError("require 1..100 grid pairs, cutoff scales in (0,1), and gammas in [0,1]")
    scales, powers = finite(cutoff_scales, "cutoff_scales"), finite(gammas, "gammas")
    if (
        scales.ndim != 1
        or powers.ndim != 1
        or not 1 <= scales.size * powers.size <= 100
        or np.any((scales <= 0) | (scales >= 1))
        or np.any((powers < 0) | (powers > 1))
    ):
        raise ValueError("require 1..100 grid pairs, cutoff scales in (0,1), and gammas in [0,1]")
    if np.unique(scales).size != scales.size or np.unique(powers).size != powers.size:
        raise ValueError("grid values must be distinct")
    repetitions = _integer(trials, "trials")
    validation = _integer(validation_trials, "validation_trials")
    if (
        not 100 <= repetitions <= 100_000
        or not 100 <= validation <= 100_000
        or repetitions * design.max_subjects * 2 > _MAX_STAGE_CELLS
        or validation * design.max_subjects * 2 > _MAX_STAGE_CELLS
    ):
        raise ValueError("require 100..100000 trials per stage and <=2 million cells per stage")
    candidate_count = scales.size * powers.size
    scenarios = null.shape[0] + 1
    look_scan_per_trial = sum(2 * int(n) for n in np.asarray(design.looks))
    estimated_work = look_scan_per_trial * (
        candidate_count * scenarios * repetitions + scenarios * validation
    )
    work_limit = _integer(max_work, "max_work")
    if not 1 <= work_limit <= _MAX_WORK_HARD or estimated_work > work_limit:
        raise ValueError("calibration exceeds max_work or the 100-million-cell hard limit")
    rate = scalar(accrual_rate, "accrual_rate")
    if rate <= 0 or not np.isfinite(1 / rate):
        raise ValueError("accrual_rate must be positive with a representable mean gap")
    if arrival not in ("fixed", "exponential"):
        raise ValueError("arrival must be 'fixed' or 'exponential'")
    timing = (
        np.array(design.timing_probabilities, copy=True)
        if truth_timing_probabilities is None
        else _timing_probabilities(truth_timing_probabilities, "truth_timing_probabilities")
    )
    generator = np.random.default_rng(rng)
    calibration_seed, validation_seed = map(
        int, generator.integers(0, np.iinfo(np.int64).max, size=2)
    )
    candidates = [
        replace(design, cutoff_scale=float(c), gamma=float(g)) for c in scales for g in powers
    ]
    scenario_truths = np.vstack((null, alt))
    scan_budget = [work_limit]

    def evaluate_stage(
        stage_trials: int, stage_seed: int, candidate_indices: NDArray[np.int64]
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        stage_rng = np.random.default_rng(stage_seed)
        uniforms = _draw_multiendpoint_uniforms(stage_rng, (stage_trials, design.max_subjects))
        shape = (candidate_indices.size, scenarios)
        probability = np.empty(shape)
        mcse = np.empty(shape)
        mean_patients = np.empty(shape)
        mean_duration = np.empty(shape)
        decision_probability = np.empty((*shape, len(_DECISIONS)))
        for candidate_position, candidate_index in enumerate(candidate_indices):
            candidate = candidates[int(candidate_index)]
            for scenario, truth in enumerate(scenario_truths):
                summary = _run_scenario(
                    candidate, truth, rate, arrival, timing, uniforms, scan_budget
                )
                probability[candidate_position, scenario] = summary[0]
                mcse[candidate_position, scenario] = summary[1]
                mean_patients[candidate_position, scenario] = summary[2]
                mean_duration[candidate_position, scenario] = summary[3]
                decision_probability[candidate_position, scenario] = summary[4]
        return probability, mcse, mean_patients, mean_duration, decision_probability

    candidate_indices = np.arange(candidate_count, dtype=np.int64)
    calibration = evaluate_stage(repetitions, calibration_seed, candidate_indices)
    (
        calibration_probability,
        calibration_mcse,
        calibration_patients,
        calibration_duration,
        calibration_actions,
    ) = calibration
    feasible = np.all(calibration_probability[:, : null.shape[0]] <= alpha, axis=1)
    feasible_indices = np.flatnonzero(feasible)
    if not feasible_indices.size:
        raise TOPInfeasibleError(
            "no candidate meets the estimated type I error target for every supplied null; "
            "extend the grid or revise the sample size"
        )
    worst_null_patients = np.max(calibration_patients[:, : null.shape[0]], axis=1)
    selected_order = np.lexsort(
        (
            feasible_indices,
            worst_null_patients[feasible_indices],
            -calibration_probability[feasible_indices, -1],
        )
    )
    selected_index = int(feasible_indices[selected_order[0]])
    chosen = candidates[selected_index]
    validation_outputs = evaluate_stage(
        validation, validation_seed, np.asarray([selected_index], dtype=np.int64)
    )
    val_probability, val_mcse, val_patients, val_duration, val_actions = validation_outputs
    return TOPMultiEndpointOptimization(
        chosen,
        _owned([(c, g) for c in scales for g in powers]),
        _owned(null),
        _owned(alt),
        _owned(calibration_probability),
        _owned(calibration_mcse),
        _owned(calibration_patients),
        _owned(calibration_duration),
        _owned(calibration_actions),
        _owned(feasible),
        selected_index,
        _owned(val_probability[0]),
        _owned(val_mcse[0]),
        _owned(val_patients[0]),
        _owned(val_duration[0]),
        _owned(val_actions[0]),
        alpha,
        repetitions,
        validation,
        calibration_seed,
        validation_seed,
    )
