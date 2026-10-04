"""Complete-outcome cohort simulation for the CiBolus adaptive design.

The paper's motivating trial has delayed toxicity ascertainment. This bounded
Python workflow assumes every patient's interval-response category and binary
toxicity are available together at the end of their cohort; it does not model
calendar time or pending outcomes.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray
from .cibolus import (
    CiBolusObservation,
    CiBolusPrior,
    _prediction_inputs,
    cibolus_predict,
)
from .cibolus_decision import CiBolusDecision, cibolus_decision
from .cibolus_fit import CiBolusFit, fit_cibolus
from .cibolus_scenarios import _joint_grid
from .hierarchical_binomial import ChainSummary
from .uaroet import _integer

_MAX_PATIENTS = 200
_MAX_TOTAL_EVALUATIONS = 2_000_000
_MAX_TOTAL_WORK = 50_000_000
_MAX_RETAINED_CELLS = 2_000_000


def _readonly_float(value: ArrayLike) -> FloatArray:
    array = np.ascontiguousarray(value, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


def _readonly_int(value: ArrayLike) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.int64)
    return np.frombuffer(array.tobytes(), dtype=np.int64).reshape(array.shape)


def _scalar(value: object, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _uniform_tape(count: int, rng: np.random.Generator, supplied: ArrayLike | None) -> FloatArray:
    if supplied is None:
        return _readonly_float(rng.random(count))
    if isinstance(supplied, np.ndarray):
        if supplied.shape != (count,):
            raise ValueError("outcome_uniforms must be a real vector of length n_patients")
    elif isinstance(supplied, (tuple, list)):
        if len(supplied) != count or any(
            not np.isscalar(item) or np.asarray(item).dtype.kind not in "iuf" for item in supplied
        ):
            raise ValueError("outcome_uniforms must be a real vector of length n_patients")
    else:
        raise ValueError("outcome_uniforms must be a real vector of length n_patients")
    raw = np.asarray(supplied)
    if raw.dtype.kind not in "iuf":
        raise ValueError("outcome_uniforms must be a real vector of length n_patients")
    tape = np.array(raw, dtype=float, copy=True)
    if np.any(~np.isfinite(tape)) or np.any((tape < 0) | (tape >= 1)):
        raise ValueError("outcome_uniforms must lie in [0, 1)")
    return _readonly_float(tape)


def _sample_cell(probabilities: FloatArray, uniform: float) -> tuple[int, int]:
    flat = probabilities.reshape(-1)
    total = float(np.sum(flat))
    if not isfinite(total) or total <= 0 or np.any(flat < 0):
        raise ArithmeticError("truth joint probabilities are invalid")
    cumulative = np.cumsum(flat / total)
    # Force the final edge to one so roundoff cannot turn the last category
    # into an impossible tail or return an out-of-range index.
    cumulative[-1] = 1.0
    selected = min(int(np.searchsorted(cumulative, uniform, side="right")), flat.size - 1)
    return divmod(selected, 2)


@dataclass(frozen=True)
class CiBolusTrialPatient:
    """One assigned patient and the complete generated observation."""

    patient: int
    cohort: int
    regimen: tuple[int, int]
    observation: CiBolusObservation
    outcome_uniform: float


@dataclass(frozen=True)
class CiBolusTrialStep:
    """One completed cohort and the decision made from its complete outcomes."""

    cohort: int
    first_patient: int
    last_patient: int
    regimen: tuple[int, int]
    decision: CiBolusDecision
    likelihood_evaluations: int
    work_units: int
    parameter_rhat_max: float | None
    utility_rhat_max: float | None
    utility_mcse_max: float


@dataclass(frozen=True)
class CiBolusTrial:
    """Immutable patient-level replay, compact cohort decisions and final choice."""

    patients: tuple[CiBolusTrialPatient, ...]
    steps: tuple[CiBolusTrialStep, ...]
    treated: np.ndarray
    outcome_uniforms: FloatArray
    stop_reason: str
    final_pair: tuple[int, int] | None
    final_decision: CiBolusDecision | None
    likelihood_evaluations: int
    work_units: int
    n_requested: int
    cohort_size: int
    early_stopped: bool


def _summary_max_defined(summary: ChainSummary, name: str) -> float | None:
    values = np.asarray(getattr(summary, name))
    defined = values[~np.isnan(values)]
    return float(np.max(defined)) if defined.size else None


def _fit_boundary(
    observations: list[CiBolusObservation],
    prior: CiBolusPrior,
    concentrations: FloatArray,
    bolus_fractions: FloatArray,
    endpoints: FloatArray,
    utility: FloatArray,
    treated: np.ndarray,
    *,
    thresholds: tuple[float, float, float, float],
    starting: tuple[int, int],
    final: bool,
    draws: int,
    warmup: int,
    chains: int,
    rng: np.random.Generator,
    remaining_evaluations: int,
    remaining_work: int,
) -> tuple[CiBolusDecision, CiBolusFit]:
    fit = fit_cibolus(
        observations,
        prior,
        concentrations,
        bolus_fractions,
        endpoints,
        utility=utility,
        draws=draws,
        warmup=warmup,
        chains=chains,
        rng=rng,
        max_evaluations=remaining_evaluations,
        max_work=remaining_work,
    )
    decision = cibolus_decision(
        fit,
        treated,
        toxicity_limit=thresholds[0],
        toxicity_cutoff=thresholds[1],
        efficacy_limit=thresholds[2],
        efficacy_cutoff=thresholds[3],
        starting=starting,
        final=final,
    )
    return decision, fit


def simulate_cibolus_trial(
    truth_log_parameters: ArrayLike | None,
    prior: CiBolusPrior,
    concentrations: ArrayLike,
    bolus_fractions: ArrayLike,
    endpoints: ArrayLike,
    *,
    utility: ArrayLike,
    n_patients: int,
    cohort_size: int,
    toxicity_limit: float,
    toxicity_cutoff: float,
    efficacy_limit: float,
    efficacy_cutoff: float,
    starting: tuple[int, int] = (0, 0),
    draws: int,
    warmup: int,
    chains: int,
    rng: np.random.Generator,
    outcome_uniforms: ArrayLike | None = None,
    truth_joint_probabilities: ArrayLike | None = None,
    max_total_evaluations: int = 500_000,
    max_total_work: int = 20_000_000,
) -> CiBolusTrial:
    """Simulate complete observed outcome cells and conduct adaptive cohorts.

    Each patient's outcome is drawn from observed-data joint cell
    probabilities at their assigned regimen: bolus response, response in an
    observation interval, or failure by time one, crossed with toxicity. This
    matches the paper's interval likelihood convention, including toxicity at
    the interval's upper endpoint. The first cohort uses ``starting``. After
    each completed cohort, one posterior fit drives either an interim
    allocation or, at ``n_patients``, unrestricted final selection. If no
    regimen is acceptable at an interim look, enrollment ends with no final
    recommendation; a later unrestricted choice does not reverse that stop.

    Outcomes are assumed to be complete together at cohort boundaries. The
    real application assessed toxicity later than efficacy, so this has no
    calendar or pending-outcome parity. A supplied uniform tape is used for
    outcome cells and returned in full for exact replay; otherwise ``rng``
    generates that tape before posterior sampling. Exactly one truth source is
    required: model log parameters or explicit joint response-category/toxicity
    probabilities on the regimen grid. Explicit joint truth supports scenarios
    defined outside the fitted parameter model and does not change that model.
    The same explicit ``Generator`` then drives serial MCMC fits. These
    short/long chain settings are caller-controlled; convergence is not implied.
    """
    if (truth_log_parameters is None) == (truth_joint_probabilities is None):
        raise ValueError("supply exactly one of truth_log_parameters or truth_joint_probabilities")
    if not isinstance(prior, CiBolusPrior):
        raise ValueError("prior must be a CiBolusPrior")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit NumPy Generator")
    n = _integer(n_patients, "n_patients", 1, _MAX_PATIENTS)
    cohort = _integer(cohort_size, "cohort_size", 1, n)
    draw_count = _integer(draws, "draws", 8, 100_000)
    warmup_count = _integer(warmup, "warmup", 0, 100_000)
    chain_count = _integer(chains, "chains", 2, 8)
    evaluation_limit = _integer(
        max_total_evaluations, "max_total_evaluations", 1, _MAX_TOTAL_EVALUATIONS
    )
    work_limit = _integer(max_total_work, "max_total_work", 1, _MAX_TOTAL_WORK)
    truth: FloatArray | None = None
    if truth_log_parameters is not None:
        truth_raw = np.asarray(truth_log_parameters)
        if truth_raw.shape != (11,) or truth_raw.dtype.kind not in "iuf":
            raise ValueError("truth_log_parameters must be eleven finite real log parameters")
        truth = np.array(truth_raw, dtype=float, copy=True)
        if np.any(~np.isfinite(truth)):
            raise ValueError("truth_log_parameters must be finite")
    c_grid, q_grid, endpoint_grid, utility_grid = _prediction_inputs(
        concentrations, bolus_fractions, endpoints, utility
    )
    start_raw = np.asarray(starting)
    if start_raw.shape != (2,) or start_raw.dtype.kind not in "iu" or start_raw.dtype.kind == "b":
        raise ValueError("starting must be an integer (concentration, bolus) pair")
    start = (int(start_raw[0]), int(start_raw[1]))
    grid_shape = (c_grid.size, q_grid.size)
    if not 0 <= start[0] < grid_shape[0] or not 0 <= start[1] < grid_shape[1]:
        raise ValueError("starting is outside the regimen grid")
    threshold_values = tuple(
        _scalar(value, name)
        for value, name in (
            (toxicity_limit, "toxicity_limit"),
            (toxicity_cutoff, "toxicity_cutoff"),
            (efficacy_limit, "efficacy_limit"),
            (efficacy_cutoff, "efficacy_cutoff"),
        )
    )
    thresholds: tuple[float, float, float, float] = (
        threshold_values[0],
        threshold_values[1],
        threshold_values[2],
        threshold_values[3],
    )
    if any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("toxicity/efficacy limits and cutoffs must lie in [0,1]")

    categories = endpoint_grid.size + 2
    grid_cells = grid_shape[0] * grid_shape[1]
    per_fit_cells = chain_count * draw_count * (12 + 2 * grid_cells * categories + 5 * grid_cells)
    steps_bound = (n + cohort - 1) // cohort
    history_cells = steps_bound * grid_cells * 5
    truth_cells = 3 * grid_cells * categories + 5 * grid_cells
    if (
        per_fit_cells > _MAX_RETAINED_CELLS
        or history_cells + per_fit_cells + truth_cells > _MAX_RETAINED_CELLS
    ):
        raise ValueError("retained fit and cohort decision history exceed two million cells")
    minimum_evaluations = chain_count * (1 + warmup_count + draw_count)
    initial_n = min(n, cohort)
    truth_work = (2 if truth_joint_probabilities is not None else 1) * grid_cells * categories + n
    minimum_work = (
        truth_work
        + minimum_evaluations * initial_n
        + chain_count * draw_count * grid_cells * categories
    )
    if evaluation_limit < minimum_evaluations:
        raise ValueError("max_total_evaluations is below one required cohort fit")
    if work_limit < minimum_work:
        raise ValueError("max_total_work is below the minimum first-cohort fit work")

    # Validate truth before outcome RNG is consumed. The retained truth surface
    # is just the model's bounded regimen/category/toxicity grid. The supplied
    # joint path uses the same fixed outcome cells without fitting its own model.
    if truth_joint_probabilities is None:
        assert truth is not None
        truth_joint = cibolus_predict(
            truth, c_grid, q_grid, endpoint_grid, utility=utility_grid
        ).joint
    else:
        truth_joint = _joint_grid(truth_joint_probabilities, (*grid_shape, categories, 2))
    tape = _uniform_tape(n, rng, outcome_uniforms)
    counts = np.zeros(grid_shape, dtype=np.int64)
    patients: list[CiBolusTrialPatient] = []
    steps: list[CiBolusTrialStep] = []
    observations: list[CiBolusObservation] = []
    total_evaluations = 0
    total_work = truth_work
    cohort_number = 0
    next_regimen = start
    stop_reason = "maximum_sample_size"
    final_decision: CiBolusDecision | None = None
    final_pair: tuple[int, int] | None = None
    early_stopped = False

    while len(patients) < n:
        cohort_number += 1
        regimen = next_regimen
        first_patient = len(patients)
        cohort_n = min(cohort, n - first_patient)
        probability_cells = np.asarray(truth_joint[regimen])
        for offset in range(cohort_n):
            patient_index = first_patient + offset
            cell_category, toxicity_index = _sample_cell(
                probability_cells, float(tape[patient_index])
            )
            if cell_category == 0:
                observation = CiBolusObservation(
                    float(c_grid[regimen[0]]),
                    float(q_grid[regimen[1]]),
                    "bolus",
                    bool(toxicity_index),
                )
            elif cell_category == categories - 1:
                observation = CiBolusObservation(
                    float(c_grid[regimen[0]]),
                    float(q_grid[regimen[1]]),
                    "failure",
                    bool(toxicity_index),
                )
            else:
                end_index = cell_category - 1
                lower = 0.0 if end_index == 0 else float(endpoint_grid[end_index - 1])
                upper = float(endpoint_grid[end_index])
                observation = CiBolusObservation(
                    float(c_grid[regimen[0]]),
                    float(q_grid[regimen[1]]),
                    "interval",
                    bool(toxicity_index),
                    lower=lower,
                    upper=upper,
                )
            observations.append(observation)
            patients.append(
                CiBolusTrialPatient(
                    patient_index,
                    cohort_number,
                    regimen,
                    observation,
                    float(tape[patient_index]),
                )
            )
            counts[regimen] += 1

        remaining_evaluations = evaluation_limit - total_evaluations
        remaining_work = work_limit - total_work
        required_eval = minimum_evaluations
        required_work = required_eval * len(observations) + (
            chain_count * draw_count * grid_cells * categories
        )
        if remaining_evaluations < required_eval or remaining_work < required_work:
            raise RuntimeError("cumulative CiBolus fitting budget exhausted at a cohort boundary")
        decision, fit = _fit_boundary(
            observations,
            prior,
            c_grid,
            q_grid,
            endpoint_grid,
            utility_grid,
            counts,
            thresholds=thresholds,
            starting=start,
            final=len(patients) == n,
            draws=draw_count,
            warmup=warmup_count,
            chains=chain_count,
            rng=rng,
            remaining_evaluations=remaining_evaluations,
            remaining_work=remaining_work,
        )
        total_evaluations += fit.likelihood_evaluations
        total_work += fit.work_units
        steps.append(
            CiBolusTrialStep(
                cohort_number,
                first_patient,
                len(patients) - 1,
                regimen,
                decision,
                fit.likelihood_evaluations,
                fit.work_units,
                _summary_max_defined(fit.parameter_summary, "split_rhat"),
                _summary_max_defined(fit.utility_summary, "split_rhat"),
                float(np.nanmax(fit.utility_summary.batch_mean_mcse)),
            )
        )
        # Drop the draw-heavy fit at the cohort boundary; the decision contains
        # only compact grid summaries retained for audit/replay.
        del fit
        if len(patients) == n:
            final_decision = decision
            if decision.action == "select":
                final_pair = decision.pair
            else:
                stop_reason = "no_regimen_acceptable_at_final_analysis"
            break
        if decision.action == "stop":
            stop_reason = (
                "all_regimens_unacceptable"
                if not np.any(decision.acceptable)
                else "no_acceptable_regimen_within_concentration_no_skip"
            )
            early_stopped = True
            break
        if decision.action != "treat" or decision.pair is None:
            raise ArithmeticError("CiBolus interim decision returned an invalid action")
        next_regimen = decision.pair

    return CiBolusTrial(
        tuple(patients),
        tuple(steps),
        _readonly_int(counts),
        tape,
        stop_reason,
        final_pair,
        final_decision,
        total_evaluations,
        total_work,
        n,
        cohort,
        early_stopped,
    )
