"""Bounded aggregate operating characteristics for complete-outcome CiBolus."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray
from .cibolus import CiBolusPrior, _prediction_inputs, cibolus_predict
from .cibolus_scenarios import _joint_grid
from .cibolus_trial import (
    _MAX_PATIENTS,
    _MAX_RETAINED_CELLS,
    _MAX_TOTAL_EVALUATIONS,
    _MAX_TOTAL_WORK,
    _scalar,
    simulate_cibolus_trial,
)
from .uaroet import _integer

_MAX_TRIALS = 10_000
_STOP_REASONS = (
    "maximum_sample_size",
    "all_regimens_unacceptable",
    "no_acceptable_regimen_within_concentration_no_skip",
    "no_regimen_acceptable_at_final_analysis",
)


def _readonly(value: ArrayLike, dtype: np.dtype | type = np.float64) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _mean_mcse(total: np.ndarray, squares: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    mean = total / count
    if count < 2:
        return mean, np.full_like(mean, np.nan)
    variance = np.maximum((squares - total * total / count) / (count - 1), 0.0)
    return mean, np.sqrt(variance / count)


def _ratio_mcse(
    numerator: np.ndarray,
    numerator_squares: np.ndarray,
    numerator_denominator: np.ndarray,
    denominator_total: np.ndarray,
    denominator_squares: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled proportion and trial-cluster robust Monte Carlo standard error."""
    expanded_denominator = denominator_total
    while expanded_denominator.ndim < numerator.ndim:
        expanded_denominator = expanded_denominator[..., None]
    probability = np.full_like(numerator, np.nan, dtype=float)
    np.divide(numerator, expanded_denominator, out=probability, where=expanded_denominator > 0)
    denominator_sq = denominator_squares
    while denominator_sq.ndim < numerator.ndim:
        denominator_sq = denominator_sq[..., None]
    mcse = np.full_like(probability, np.nan)
    if count < 2:
        return probability, mcse
    residual_squares = (
        numerator_squares
        - 2 * probability * numerator_denominator
        + probability * probability * denominator_sq
    )
    residual_squares = np.maximum(residual_squares, 0.0)
    denom_sq = expanded_denominator * expanded_denominator
    np.divide(
        count * residual_squares,
        (count - 1) * denom_sq,
        out=mcse,
        where=expanded_denominator > 0,
    )
    np.sqrt(mcse, out=mcse)
    return probability, mcse


@dataclass(frozen=True)
class CiBolusOperatingCharacteristics:
    """Aggregate selection, enrollment and observed-outcome summaries.

    ``mean_allocation`` contains mean patient counts per regimen and trial.
    Event proportions pool patient assignments; their MCSEs use trial-clustered
    ratio estimates. Selection and stop probabilities use binomial MCSEs.
    ``trial_seeds[i]`` can be passed to ``np.random.default_rng`` and then to
    ``simulate_cibolus_trial`` to reproduce replicate ``i`` exactly. No
    patient/fit histories are retained. Rhat and MCSE maxima preserve infinity;
    ``diagnostic_undefined_steps`` counts steps with any undefined diagnostic.
    """

    trials: int
    grid_shape: tuple[int, int]
    response_categories: int
    selection_count: np.ndarray
    selection_probability: FloatArray
    selection_mcse: FloatArray
    no_selection_count: int
    no_selection_probability: float
    no_selection_mcse: float
    early_stop_count: int
    early_stop_probability: float
    early_stop_mcse: float
    stop_reasons: tuple[str, ...]
    stop_reason_count: np.ndarray
    stop_reason_probability: FloatArray
    stop_reason_mcse: FloatArray
    mean_enrollment: float
    enrollment_mcse: float
    parameter_rhat_max: float | None
    utility_rhat_max: float | None
    utility_mcse_max: float | None
    diagnostic_undefined_steps: int
    mean_allocation: FloatArray
    allocation_mcse: FloatArray
    assigned_patients: np.ndarray
    toxicities: np.ndarray
    toxicity_probability: FloatArray
    toxicity_mcse: FloatArray
    responses: np.ndarray
    response_probability: FloatArray
    response_mcse: FloatArray
    response_category_count: np.ndarray
    response_category_probability: FloatArray
    response_category_mcse: FloatArray
    trial_seeds: np.ndarray
    likelihood_evaluations: int
    work_units: int


def simulate_cibolus_operating_characteristics(
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
    trials: int,
    starting: tuple[int, int] = (0, 0),
    draws: int,
    warmup: int,
    chains: int,
    rng: np.random.Generator,
    truth_joint_probabilities: ArrayLike | None = None,
    max_total_evaluations: int = 500_000,
    max_total_work: int = 20_000_000,
) -> CiBolusOperatingCharacteristics:
    """Run serial complete-outcome CiBolus trials and aggregate their results.

    This uses the same patient-level joint response-category/toxicity model and
    cohort conduct as :func:`simulate_cibolus_trial`. All outcomes are assumed
    complete at each cohort boundary; no calendar or pending-outcome behavior
    is represented. MCMC settings are explicit and diagnostics from individual
    fits are not treated as convergence guarantees. Total likelihood and work
    limits are shared across all replicates rather than reset per trial.
    """
    if not isinstance(prior, CiBolusPrior):
        raise ValueError("prior must be a CiBolusPrior")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit NumPy Generator")
    trial_count = _integer(trials, "trials", 1, _MAX_TRIALS)
    n = _integer(n_patients, "n_patients", 1, _MAX_PATIENTS)
    cohort = _integer(cohort_size, "cohort_size", 1, n)
    draw_count = _integer(draws, "draws", 8, 100_000)
    warmup_count = _integer(warmup, "warmup", 0, 100_000)
    chain_count = _integer(chains, "chains", 2, 8)
    evaluation_limit = _integer(
        max_total_evaluations, "max_total_evaluations", 1, _MAX_TOTAL_EVALUATIONS
    )
    work_limit = _integer(max_total_work, "max_total_work", 1, _MAX_TOTAL_WORK)

    concentration_grid, bolus_grid, endpoint_grid, utility_grid = _prediction_inputs(
        concentrations, bolus_fractions, endpoints, utility
    )
    grid_shape = (int(concentration_grid.size), int(bolus_grid.size))
    grid_cells = grid_shape[0] * grid_shape[1]
    category_count = int(endpoint_grid.size) + 2
    # Bound returned arrays together with the online sufficient statistics and
    # one trial's temporary outcome counts. Python integer arithmetic avoids
    # overflow in the preflight itself.
    retained_cells = (
        24 * grid_cells + 7 * grid_cells * category_count + trial_count + 4 * len(_STOP_REASONS)
    )
    if retained_cells > _MAX_RETAINED_CELLS:
        raise ValueError("CiBolus aggregate summaries exceed two million cells")

    per_fit_cells = (
        chain_count * draw_count * (12 + 2 * grid_cells * category_count + 5 * grid_cells)
    )
    step_bound = (n + cohort - 1) // cohort
    truth_cells = 3 * grid_cells * category_count + 5 * grid_cells
    persistent_joint_cells = (
        2 * grid_cells * category_count if truth_joint_probabilities is not None else 0
    )
    if (
        retained_cells
        + step_bound * grid_cells * 5
        + per_fit_cells
        + truth_cells
        + persistent_joint_cells
        > _MAX_RETAINED_CELLS
    ):
        raise ValueError("aggregate summaries, trial fit and history exceed two million cells")

    minimum_evaluations = chain_count * (1 + warmup_count + draw_count)
    first_cohort = min(n, cohort)
    truth_cell_work = (
        (2 if truth_joint_probabilities is not None else 1) * grid_cells * category_count
    )
    per_trial_truth_work = truth_cell_work + n
    per_trial_minimum_work = (
        per_trial_truth_work
        + minimum_evaluations * first_cohort
        + chain_count * draw_count * grid_cells * category_count
    )
    if trial_count * minimum_evaluations > evaluation_limit:
        raise ValueError("max_total_evaluations cannot fit the first cohort of every trial")
    if trial_count * per_trial_minimum_work > work_limit:
        raise ValueError("max_total_work cannot fit the first cohort of every trial")

    if (truth_log_parameters is None) == (truth_joint_probabilities is None):
        raise ValueError("supply exactly one of truth_log_parameters or truth_joint_probabilities")
    if truth_joint_probabilities is None:
        assert truth_log_parameters is not None
        truth_shape = np.asarray(truth_log_parameters).shape
        if truth_shape != (11,):
            raise ValueError("truth_log_parameters must contain eleven coordinates")
    start_raw = np.asarray(starting)
    if start_raw.shape != (2,) or start_raw.dtype.kind not in "iu" or start_raw.dtype.kind == "b":
        raise ValueError("starting must be an integer (concentration, bolus) pair")
    start = (int(start_raw[0]), int(start_raw[1]))
    if not (0 <= start[0] < grid_shape[0] and 0 <= start[1] < grid_shape[1]):
        raise ValueError("starting regimen is outside the prediction grid")
    threshold_values = tuple(
        _scalar(value, name)
        for value, name in (
            (toxicity_limit, "toxicity_limit"),
            (toxicity_cutoff, "toxicity_cutoff"),
            (efficacy_limit, "efficacy_limit"),
            (efficacy_cutoff, "efficacy_cutoff"),
        )
    )
    if any(not 0 <= value <= 1 for value in threshold_values):
        raise ValueError("toxicity/efficacy limits and cutoffs must be finite values in [0,1]")

    validation_work = truth_cell_work
    if validation_work + trial_count * per_trial_minimum_work > work_limit:
        raise ValueError("max_total_work cannot cover truth validation and each first cohort")
    # Validate the truth parameters and their actual grid calculations before
    # advancing the caller's RNG. This extra prediction work is included in the
    # reported total and the cumulative work budget.
    validated_truth_joint: FloatArray | None = None
    if truth_joint_probabilities is None:
        assert truth_log_parameters is not None
        truth_check = cibolus_predict(
            truth_log_parameters,
            concentration_grid,
            bolus_grid,
            endpoint_grid,
            utility=utility_grid,
        )
        del truth_check
    else:
        validated_truth_joint = _joint_grid(
            truth_joint_probabilities,
            (*grid_shape, category_count, 2),
        )

    # Derive independent replayable trial seeds only after all lower-bound and
    # storage checks pass. The generated uint32 entropy has a fixed layout.
    root_entropy = [int(value) for value in rng.integers(0, 2**32, size=4, dtype=np.uint32)]
    children = np.random.SeedSequence(root_entropy).spawn(trial_count)
    trial_seeds = np.array(
        [child.generate_state(1, dtype=np.uint64)[0] for child in children], dtype=np.uint64
    )

    selection_count = np.zeros(grid_shape, dtype=np.int64)
    no_selection = early_stop = total_evaluations = 0
    total_work = validation_work
    reason_count = np.zeros(len(_STOP_REASONS), dtype=np.int64)
    enrollment_sum = enrollment_squares = 0.0
    parameter_rhat_max: float | None = None
    utility_rhat_max: float | None = None
    utility_mcse_max: float | None = None
    diagnostic_undefined_steps = 0
    allocation_sum = np.zeros(grid_shape)
    allocation_squares = np.zeros(grid_shape)
    assigned_total = np.zeros(grid_shape)
    assigned_squares = np.zeros(grid_shape)
    toxic_total = np.zeros(grid_shape)
    toxic_squares = np.zeros(grid_shape)
    toxic_assigned_cross = np.zeros(grid_shape)
    response_total = np.zeros(grid_shape)
    response_squares = np.zeros(grid_shape)
    response_assigned_cross = np.zeros(grid_shape)
    response_category_total = np.zeros((*grid_shape, category_count))
    response_category_squares = np.zeros_like(response_category_total)
    response_category_assigned_cross = np.zeros_like(response_category_total)

    for seed_value in trial_seeds:
        remaining_evaluations = evaluation_limit - total_evaluations
        remaining_work = work_limit - total_work
        trial = simulate_cibolus_trial(
            truth_log_parameters,
            prior,
            concentration_grid,
            bolus_grid,
            endpoint_grid,
            utility=utility_grid,
            n_patients=n,
            cohort_size=cohort,
            toxicity_limit=toxicity_limit,
            toxicity_cutoff=toxicity_cutoff,
            efficacy_limit=efficacy_limit,
            efficacy_cutoff=efficacy_cutoff,
            starting=starting,
            draws=draw_count,
            warmup=warmup_count,
            chains=chain_count,
            rng=np.random.default_rng(int(seed_value)),
            truth_joint_probabilities=validated_truth_joint,
            max_total_evaluations=remaining_evaluations,
            max_total_work=remaining_work,
        )
        total_evaluations += trial.likelihood_evaluations
        total_work += trial.work_units
        if total_evaluations > evaluation_limit or total_work > work_limit:
            raise RuntimeError("aggregate CiBolus work exceeded its cumulative budget")
        enrollment = len(trial.patients)
        enrollment_sum += enrollment
        enrollment_squares += enrollment * enrollment
        for step in trial.steps:
            has_undefined_diagnostic = False
            if step.parameter_rhat_max is None:
                has_undefined_diagnostic = True
            else:
                parameter_rhat_max = (
                    step.parameter_rhat_max
                    if parameter_rhat_max is None
                    else max(parameter_rhat_max, step.parameter_rhat_max)
                )
            if step.utility_rhat_max is None or np.isnan(step.utility_mcse_max):
                has_undefined_diagnostic = True
            if step.utility_rhat_max is not None:
                utility_rhat_max = (
                    step.utility_rhat_max
                    if utility_rhat_max is None
                    else max(utility_rhat_max, step.utility_rhat_max)
                )
            if not np.isnan(step.utility_mcse_max):
                utility_mcse_max = (
                    step.utility_mcse_max
                    if utility_mcse_max is None
                    else max(utility_mcse_max, step.utility_mcse_max)
                )
            diagnostic_undefined_steps += int(has_undefined_diagnostic)
        early_stop += int(trial.early_stopped)
        if trial.final_pair is None:
            no_selection += 1
        else:
            selection_count[trial.final_pair] += 1
        try:
            reason_count[_STOP_REASONS.index(trial.stop_reason)] += 1
        except ValueError as exc:
            raise ArithmeticError(f"unknown CiBolus stop reason {trial.stop_reason!r}") from exc

        assigned = np.asarray(trial.treated, dtype=float)
        toxic = np.zeros(grid_shape)
        response = np.zeros(grid_shape)
        categories = np.zeros((*grid_shape, category_count))
        for patient in trial.patients:
            i, j = patient.regimen
            observation = patient.observation
            if observation.kind == "bolus":
                category = 0
            elif observation.kind == "failure":
                category = category_count - 1
            else:
                assert observation.upper is not None
                category = 1 + int(np.searchsorted(endpoint_grid, observation.upper, side="left"))
            categories[i, j, category] += 1
            toxic[i, j] += int(observation.toxicity)
            response[i, j] += int(observation.kind != "failure")

        allocation_sum += assigned
        allocation_squares += assigned * assigned
        assigned_total += assigned
        assigned_squares += assigned * assigned
        toxic_total += toxic
        toxic_squares += toxic * toxic
        toxic_assigned_cross += toxic * assigned
        response_total += response
        response_squares += response * response
        response_assigned_cross += response * assigned
        response_category_total += categories
        response_category_squares += categories * categories
        response_category_assigned_cross += categories * assigned[..., None]
        del trial

    selection_probability = selection_count / trial_count
    selection_mcse = np.sqrt(selection_probability * (1.0 - selection_probability) / trial_count)
    no_selection_probability = no_selection / trial_count
    no_selection_mcse = float(
        np.sqrt(no_selection_probability * (1.0 - no_selection_probability) / trial_count)
    )
    early_probability = early_stop / trial_count
    early_mcse = float(np.sqrt(early_probability * (1.0 - early_probability) / trial_count))
    reason_probability = reason_count / trial_count
    reason_mcse = np.sqrt(reason_probability * (1.0 - reason_probability) / trial_count)
    mean_allocation, allocation_mcse = _mean_mcse(allocation_sum, allocation_squares, trial_count)
    mean_enrollment = enrollment_sum / trial_count
    if trial_count < 2:
        enrollment_mcse = float("nan")
    else:
        enrollment_variance = max(
            (enrollment_squares - enrollment_sum * enrollment_sum / trial_count)
            / (trial_count - 1),
            0.0,
        )
        enrollment_mcse = float(np.sqrt(enrollment_variance / trial_count))
    toxicity_probability, toxicity_mcse = _ratio_mcse(
        toxic_total,
        toxic_squares,
        toxic_assigned_cross,
        assigned_total,
        assigned_squares,
        trial_count,
    )
    response_probability, response_mcse = _ratio_mcse(
        response_total,
        response_squares,
        response_assigned_cross,
        assigned_total,
        assigned_squares,
        trial_count,
    )
    category_denominator = assigned_total[..., None]
    category_denominator_squares = assigned_squares[..., None]
    response_category_probability, response_category_mcse = _ratio_mcse(
        response_category_total,
        response_category_squares,
        response_category_assigned_cross,
        category_denominator,
        category_denominator_squares,
        trial_count,
    )
    return CiBolusOperatingCharacteristics(
        trial_count,
        grid_shape,
        category_count,
        _readonly(selection_count, np.int64),
        _readonly(selection_probability),
        _readonly(selection_mcse),
        no_selection,
        no_selection_probability,
        no_selection_mcse,
        early_stop,
        early_probability,
        early_mcse,
        _STOP_REASONS,
        _readonly(reason_count, np.int64),
        _readonly(reason_probability),
        _readonly(reason_mcse),
        mean_enrollment,
        enrollment_mcse,
        parameter_rhat_max,
        utility_rhat_max,
        utility_mcse_max,
        diagnostic_undefined_steps,
        _readonly(mean_allocation),
        _readonly(allocation_mcse),
        _readonly(assigned_total, np.int64),
        _readonly(toxic_total, np.int64),
        _readonly(toxicity_probability),
        _readonly(toxicity_mcse),
        _readonly(response_total, np.int64),
        _readonly(response_probability),
        _readonly(response_mcse),
        _readonly(response_category_total, np.int64),
        _readonly(response_category_probability),
        _readonly(response_category_mcse),
        _readonly(trial_seeds, np.uint64),
        total_evaluations,
        total_work,
    )
