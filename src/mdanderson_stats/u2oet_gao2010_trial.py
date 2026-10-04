"""Cohort-level conduct for the original 2010 U2OET GAO model."""

import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .u2oet import _real
from .u2oet_decision import _integer
from .u2oet_gao2010_decision import U2OETGAO2010Decision, u2oet_gao2010_decision
from .u2oet_gao2010_fit import (
    _MAX_LIKELIHOOD_EVALUATIONS,
    _MAX_RETAINED_CELLS,
    _MAX_WORK_UNITS,
    _InvalidGAO2010Domain,
    _likelihood,
    fit_u2oet_gao2010,
    u2oet_gao2010_parameter_names,
)
from .u2oet_patients import U2OETPatients, u2oet_patients

_MAX_PATIENTS = 2500
_MAX_TRIAL_CELLS = 25_000_000


@dataclass(frozen=True)
class U2OETGAO2010Look:
    """Compact diagnostics and action at one completed cohort boundary."""

    patients: int
    treated_pair: tuple[int, int]
    decision: U2OETGAO2010Decision
    maximum_parameter_split_rhat: float
    maximum_utility_mcse: float
    likelihood_evaluations: int
    likelihood_work_units: int


@dataclass(frozen=True)
class U2OETGAO2010Trial:
    """A replayable cohort trial without retained per-look posterior draws."""

    patients: U2OETPatients
    outcome_uniforms: FloatArray
    looks: tuple[U2OETGAO2010Look, ...]
    final_decision: U2OETGAO2010Decision
    selected_pair: tuple[int, int] | None
    stopped_for_global_toxicity: bool
    stopped_early: bool
    stop_reason: str
    posterior_fits: int
    likelihood_evaluations: int
    likelihood_work_units: int
    maximum_parameter_split_rhat: float
    maximum_utility_mcse: float
    data_seed: int
    posterior_seed: int
    design_json: str


def _shape(value: ArrayLike, name: str, limit: int, dimensions: int) -> tuple[int, ...]:
    if isinstance(value, np.ndarray):
        if value.size > limit:
            raise ValueError(f"{name} exceeds its retained-cell bound")
        return value.shape
    if isinstance(value, (list, tuple)):

        def visit(item: object, level: int) -> tuple[tuple[int, ...], int]:
            if isinstance(item, np.ndarray):
                if item.size > limit or item.ndim != dimensions - level:
                    raise ValueError(f"{name} must be bounded rectangular numeric data")
                return item.shape, int(item.size)
            if isinstance(item, (list, tuple)):
                if level >= dimensions or len(item) > limit:
                    raise ValueError(f"{name} must be bounded rectangular numeric data")
                children = [visit(child, level + 1) for child in item]
                if children and any(shape != children[0][0] for shape, _ in children):
                    raise ValueError(f"{name} must be rectangular")
                size = sum(cells for _, cells in children)
                if size > limit:
                    raise ValueError(f"{name} exceeds its retained-cell bound")
                return (len(item),) + (children[0][0] if children else ()), size
            if level != dimensions:
                raise ValueError(f"{name} must have {dimensions} dimensions")
            return (), 1

        shape, _ = visit(value, 0)
        return shape
    shape_attr = getattr(value, "shape", None)
    size = getattr(value, "size", None)
    if shape_attr is None or size is None or int(size) > limit:
        raise ValueError(f"{name} must expose bounded shape and size before conversion")
    try:
        return tuple(int(dim) for dim in shape_attr)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must have a valid array shape") from exc


def _uniform_tape(value: ArrayLike, n_patients: int) -> FloatArray:
    if isinstance(value, np.ndarray):
        shape = value.shape
        if value.size > 2 * _MAX_PATIENTS:
            raise ValueError("outcome_uniforms exceeds the patient tape bound")
    elif isinstance(value, (list, tuple)):
        if len(value) != n_patients:
            raise ValueError("outcome_uniforms must have shape (n_patients, 2)")
        # Check every row width before walking any entries, bounding malformed
        # tapes without traversing an arbitrarily wide first row.
        for row in value:
            valid_row = (
                row.ndim == 1 and row.shape == (2,) and row.size == 2
                if isinstance(row, np.ndarray)
                else isinstance(row, (list, tuple)) and len(row) == 2
            )
            if not valid_row:
                raise ValueError("outcome_uniforms must have shape (n_patients, 2)")
        for row in value:
            for item in row:
                if isinstance(item, (list, tuple, np.ndarray)):
                    raise ValueError("outcome_uniforms must be a two-column numeric tape")
        shape = (n_patients, 2)
    else:
        shape = getattr(value, "shape", None)
        size = getattr(value, "size", None)
        if shape is None or size is None or int(size) > 2 * _MAX_PATIENTS:
            raise ValueError("outcome_uniforms must expose a bounded shape and size")
    if tuple(shape) != (n_patients, 2):
        raise ValueError("outcome_uniforms must have shape (n_patients, 2)")
    if np.iscomplexobj(value):
        raise ValueError("outcome_uniforms must be real")
    tape = _real(value, "outcome_uniforms")
    if np.any((tape < 0.0) | (tape >= 1.0)):
        raise ValueError("outcome_uniforms must lie in [0, 1)")
    return tape


def _maximum_defined(values: ArrayLike) -> float:
    array = np.asarray(values)
    defined = array[~np.isnan(array)]
    return float(np.max(defined)) if defined.size else float("nan")


def _minimum_fit_evaluations(
    *, chains: int, warmup: int, draws: int, active_blocks: int, free_association: bool
) -> int:
    per_iteration = active_blocks + int(free_association)
    return chains * (1 + (warmup + draws) * per_iteration)


def simulate_u2oet_gao2010_trial(
    doses1: ArrayLike,
    doses2: ArrayLike,
    scenario_joint: ArrayLike,
    utility: ArrayLike,
    *,
    prior_mean: ArrayLike,
    prior_sd: ArrayLike,
    starting: tuple[int, int],
    n_patients: int,
    efficacy_evaluability: float,
    toxicity_limit: float,
    cohort_size: int = 3,
    stopping_probability: float = 0.80,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 4,
    initial_parameters: ArrayLike | None = None,
    fixed_association: float | None = None,
    outcome_uniforms: ArrayLike | None = None,
    max_likelihood_evaluations: int = _MAX_LIKELIHOOD_EVALUATIONS,
    max_work: int = _MAX_WORK_UNITS,
    rng: np.random.Generator,
) -> U2OETGAO2010Trial:
    """Simulate complete-cohort decisions under the original 2010 GAO rules.

    ``scenario_joint`` is an explicit dose-by-dose-by-efficacy-by-toxicity
    truth array; it does not construct a truth scenario from fitted parameters.
    The first cohort uses ``starting`` without a prior safety analysis. After
    each completed cohort the 2010 fitter is applied to complete and
    toxicity-only outcomes. The global all-dose safety rule precedes an
    interim allocation; the final Eq. 11 choice is over the full grid. A
    constant, outcome-independent `efficacy_evaluability` probability models
    the source's `zeta`; within-patient dose changes and their clinical
    adjudication are outside this cohort-level simulator.

    ``outcome_uniforms`` optionally supplies two uniforms per planned patient:
    the flattened joint-outcome CDF draw and the independent efficacy-
    evaluability draw. Its full tape is retained for replay.
    """
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit NumPy Generator")
    dose1_shape = _shape(doses1, "doses1", 5, 1)
    dose2_shape = _shape(doses2, "doses2", 5, 1)
    if len(dose1_shape) != 1 or len(dose2_shape) != 1:
        raise ValueError("doses1 and doses2 must each be one-dimensional")
    d1, d2 = _real(doses1, "doses1"), _real(doses2, "doses2")
    if d1.ndim != 1 or d2.ndim != 1 or not 2 <= d1.size <= 5 or not 2 <= d2.size <= 5:
        raise ValueError("doses1 and doses2 must each have 2–5 values")
    if (
        np.any(d1 < 0.0)
        or np.any(d2 < 0.0)
        or np.any(np.diff(d1) <= 0.0)
        or np.any(np.diff(d2) <= 0.0)
    ):
        raise ValueError("dose grids must be strictly increasing and nonnegative")

    joint_shape = _shape(scenario_joint, "scenario_joint", _MAX_RETAINED_CELLS, 4)
    if (
        len(joint_shape) != 4
        or joint_shape[:2] != (d1.size, d2.size)
        or any(not 2 <= level <= 4 for level in joint_shape[2:])
    ):
        raise ValueError("scenario_joint must match dose grids and 2–4 outcome levels")
    if np.iscomplexobj(scenario_joint):
        raise ValueError("scenario_joint must be real")
    truth = _real(scenario_joint, "scenario_joint").copy()
    if np.any((truth < 0.0) | (truth > 1.0)) or np.any(
        np.abs(truth.sum(axis=(-2, -1)) - 1.0) > 1e-12
    ):
        raise ValueError("scenario_joint must contain probabilities summing to one per pair")
    truth /= truth.sum(axis=(-2, -1), keepdims=True)
    truth.flags.writeable = False

    efficacy_levels, toxicity_levels = joint_shape[2:]
    utility_shape = _shape(utility, "utility", 16, 2)
    if utility_shape != (efficacy_levels, toxicity_levels):
        raise ValueError("utility must have efficacy-by-toxicity shape")
    utility_array = _real(utility, "utility")
    prior_names = u2oet_gao2010_parameter_names(efficacy_levels, toxicity_levels)
    dimension = len(prior_names) - 1
    if _shape(prior_mean, "prior_mean", dimension, 1) != (dimension,):
        raise ValueError("prior_mean must be a bounded vector matching the Gaussian coordinates")
    if _shape(prior_sd, "prior_sd", dimension, 1) != (dimension,):
        raise ValueError("prior_sd must be a bounded vector matching the Gaussian coordinates")
    mean, sd = _real(prior_mean, "prior_mean"), _real(prior_sd, "prior_sd")
    if mean.shape != (dimension,) or sd.shape != mean.shape or np.any(sd < 0.0):
        raise ValueError("prior_mean and prior_sd must match the 2010 Gaussian coordinates")

    n = _integer(n_patients, "n_patients", 1, _MAX_PATIENTS)
    cohort = _integer(cohort_size, "cohort_size", 1, _MAX_PATIENTS)
    if _shape(starting, "starting", 2, 1) != (2,):
        raise ValueError("starting must contain two zero-based dose indices")
    first = (
        _integer(starting[0], "starting dose1 index", 0, d1.size - 1),
        _integer(starting[1], "starting dose2 index", 0, d2.size - 1),
    )
    zeta = _real(efficacy_evaluability, "efficacy_evaluability")
    limit = _real(toxicity_limit, "toxicity_limit")
    p_upper = _real(stopping_probability, "stopping_probability")
    if zeta.ndim or not 0.0 <= float(zeta) <= 1.0:
        raise ValueError("efficacy_evaluability must be a probability")
    if limit.ndim or not 0.0 <= float(limit) <= 1.0:
        raise ValueError("toxicity_limit must be a probability")
    if p_upper.ndim or not 0.0 <= float(p_upper) <= 1.0:
        raise ValueError("stopping_probability must be a probability")
    draws = _integer(draws, "draws", 8, 100_000)
    warmup = _integer(warmup, "warmup", 0, 100_000)
    chains = _integer(chains, "chains", 2, 16)
    max_likelihood_evaluations = _integer(
        max_likelihood_evaluations,
        "max_likelihood_evaluations",
        1,
        _MAX_LIKELIHOOD_EVALUATIONS,
    )
    max_work = _integer(max_work, "max_work", 1, _MAX_WORK_UNITS)
    fixed_rho: float | None
    if fixed_association is None:
        fixed_rho = None
    else:
        rho = _real(fixed_association, "fixed_association")
        if rho.ndim or not -1.0 <= float(rho) <= 1.0:
            raise ValueError("fixed_association must be in [-1,1]")
        fixed_rho = float(rho)
    if initial_parameters is None:
        initial_values = None
    else:
        if isinstance(initial_parameters, (list, tuple)) and initial_parameters:
            initial_dimensions = (
                2 if isinstance(initial_parameters[0], (list, tuple, np.ndarray)) else 1
            )
        else:
            initial_dimensions = 1
        start_shape = _shape(
            initial_parameters,
            "initial_parameters",
            16 * (dimension + 1),
            initial_dimensions,
        )
        if start_shape not in ((dimension + 1,), (chains, dimension + 1)):
            raise ValueError(
                "initial_parameters must be one full coordinate vector or one per chain"
            )
        if np.iscomplexobj(initial_parameters):
            raise ValueError("initial_parameters must be real")
        initial_values = _real(initial_parameters, "initial_parameters").copy()

    tape = None if outcome_uniforms is None else _uniform_tape(outcome_uniforms, n)
    look_count = (n + cohort - 1) // cohort
    joint_cells = int(np.prod(joint_shape, dtype=np.int64))
    dose_pairs = d1.size * d2.size
    active_blocks = int(np.any(sd[: 4 * (efficacy_levels - 1) + 2] > 0)) + int(
        np.any(sd[4 * (efficacy_levels - 1) + 2 :] > 0)
    )
    min_evaluations_per_fit = _minimum_fit_evaluations(
        chains=chains,
        warmup=warmup,
        draws=draws,
        active_blocks=active_blocks,
        free_association=fixed_rho is None,
    )
    min_work_per_fit = min_evaluations_per_fit * joint_cells
    start_validation_evaluations = chains
    start_validation_work = chains * joint_cells
    if (
        start_validation_evaluations + min_evaluations_per_fit * look_count
        > max_likelihood_evaluations
    ):
        raise ValueError("minimum U2OET 2010 trial evaluations exceed cumulative budget")
    if start_validation_work + min_work_per_fit * look_count > max_work:
        raise ValueError("minimum U2OET 2010 trial work exceeds cumulative budget")
    fit_storage = chains * draws * (14 * (dimension + 1) + 2 * joint_cells + 2)
    fit_storage += joint_cells + dose_pairs * toxicity_levels
    # Decision summaries retain seven float grids plus one boolean eligibility
    # grid per look; use eight float-sized cells per pair conservatively.
    decision_workspace = 12 * chains * draws * dose_pairs + 10 * dose_pairs
    history_cells = look_count * (8 * dose_pairs + 20) + n * 12
    tape_cells = 2 * n
    total_cells = fit_storage + decision_workspace + history_cells + tape_cells + joint_cells
    if fit_storage > _MAX_RETAINED_CELLS or total_cells > _MAX_TRIAL_CELLS:
        raise ValueError("combined 2010 GAO fit, decision and trial storage exceeds its bound")

    # Validate every sampler start before consuming the caller's Generator.
    if initial_values is None:
        starts = np.empty((chains, dimension + 1), dtype=float)
        starts[:, :dimension] = mean
        starts[:, -1] = 0.0 if fixed_rho is None else fixed_rho
    else:
        starts = initial_values.copy()
        if starts.shape == (dimension + 1,):
            starts = np.broadcast_to(starts, (chains, dimension + 1)).copy()
    if np.any(np.abs(starts[:, -1]) > 1.0):
        raise ValueError("initial association must lie in [-1,1]")
    if fixed_rho is not None and np.any(starts[:, -1] != fixed_rho):
        raise ValueError("initial association must equal fixed_association")
    if np.any(starts[:, :-1][:, sd == 0] != mean[sd == 0]):
        raise ValueError("initial fixed Gaussian coordinates must equal prior_mean")
    zero_complete = np.zeros((d1.size, d2.size, efficacy_levels, toxicity_levels))
    zero_toxicity = np.zeros((d1.size, d2.size, toxicity_levels))
    for chain, row in enumerate(starts):
        try:
            _likelihood(
                d1,
                d2,
                zero_complete,
                zero_toxicity,
                row,
                efficacy_levels,
                toxicity_levels,
            )
        except _InvalidGAO2010Domain as exc:
            raise ValueError(
                f"initial state is outside the valid dose-grid domain in chain {chain}"
            ) from exc

    design = {
        "format_version": 1,
        "model": "gao2010",
        "doses1": d1.tolist(),
        "doses2": d2.tolist(),
        "scenario_joint": truth.tolist(),
        "utility": utility_array.tolist(),
        "prior_names": prior_names[:-1],
        "prior_mean": mean.tolist(),
        "prior_sd": sd.tolist(),
        "starting": first,
        "n_patients": n,
        "cohort_size": cohort,
        "efficacy_evaluability": float(zeta),
        "toxicity_limit": float(limit),
        "stopping_probability": float(p_upper),
        "draws": draws,
        "warmup": warmup,
        "chains": chains,
        "initial_parameters": starts.tolist(),
        "fixed_association": fixed_rho,
        "max_likelihood_evaluations": max_likelihood_evaluations,
        "max_work": max_work,
    }
    design_json = json.dumps(design, sort_keys=True, separators=(",", ":"), allow_nan=False)

    seeds = rng.integers(0, 2**63, size=2, dtype=np.int64)
    data_seed, posterior_seed = (int(value) for value in seeds)
    data_rng = np.random.default_rng(data_seed)
    posterior_rng = np.random.default_rng(posterior_seed)
    if tape is None:
        tape = data_rng.random((n, 2))
    tape = _freeze(tape)

    rows: list[list[int]] = []
    treated_counts = np.zeros((d1.size, d2.size), dtype=float)
    complete_counts = np.zeros((d1.size, d2.size, efficacy_levels, toxicity_levels), dtype=float)
    toxicity_counts = np.zeros((d1.size, d2.size, toxicity_levels), dtype=float)
    current = first
    look_records: list[U2OETGAO2010Look] = []
    final_decision: U2OETGAO2010Decision | None = None
    selected_pair: tuple[int, int] | None = None
    stopped_safety = False
    posterior_fits = 0
    total_evaluations = start_validation_evaluations
    total_work = start_validation_work
    max_parameter_rhat = float("nan")
    max_utility_mcse = float("nan")

    completed = 0
    while completed < n:
        cohort_end = min(n, completed + cohort)
        for patient_index in range(completed, cohort_end):
            distribution = truth[current]
            cumulative = np.cumsum(distribution.ravel())
            cumulative /= cumulative[-1]
            category = int(np.searchsorted(cumulative, tape[patient_index, 0], side="right"))
            efficacy, toxicity = np.unravel_index(category, distribution.shape)
            observed_efficacy = int(efficacy) if tape[patient_index, 1] < float(zeta) else -1
            rows.append(
                [
                    patient_index + 1,
                    current[0] + 1,
                    current[1] + 1,
                    observed_efficacy,
                    int(toxicity),
                ]
            )
            treated_counts[current] += 1.0
            if observed_efficacy < 0:
                toxicity_counts[*current, toxicity] += 1.0
            else:
                complete_counts[*current, observed_efficacy, toxicity] += 1.0
        completed = cohort_end
        remaining_evaluations = max_likelihood_evaluations - total_evaluations
        remaining_work = max_work - total_work
        if remaining_evaluations < min_evaluations_per_fit or remaining_work < min_work_per_fit:
            raise ArithmeticError("cumulative U2OET 2010 trial fit budget exhausted")
        fit = fit_u2oet_gao2010(
            d1,
            d2,
            complete_counts,
            prior_mean=mean,
            prior_sd=sd,
            toxicity_only=toxicity_counts,
            draws=draws,
            warmup=warmup,
            chains=chains,
            initial=initial_values,
            fixed_association=fixed_rho,
            rng=posterior_rng,
            max_likelihood_evaluations=remaining_evaluations,
            max_work=remaining_work,
        )
        posterior_fits += 1
        total_evaluations += fit.likelihood_evaluations
        total_work += fit.likelihood_work_units
        fit_rhat = _maximum_defined(fit.parameter_summary.split_rhat)
        max_parameter_rhat = (
            max(max_parameter_rhat, fit_rhat) if not np.isnan(max_parameter_rhat) else fit_rhat
        )

        at_cap = completed >= n
        decision = u2oet_gao2010_decision(
            fit.joint,
            utility_array,
            toxicity_limit=float(limit),
            stopping_probability=float(p_upper),
            current_pair=None if at_cap else current,
            treated=None if at_cap else treated_counts,
        )
        max_fit_utility_mcse = _maximum_defined(decision.utility_mcse)
        max_utility_mcse = (
            max(max_utility_mcse, max_fit_utility_mcse)
            if not np.isnan(max_utility_mcse)
            else max_fit_utility_mcse
        )
        look_records.append(
            U2OETGAO2010Look(
                patients=completed,
                treated_pair=current,
                decision=decision,
                maximum_parameter_split_rhat=fit_rhat,
                maximum_utility_mcse=max_fit_utility_mcse,
                likelihood_evaluations=fit.likelihood_evaluations,
                likelihood_work_units=fit.likelihood_work_units,
            )
        )
        del fit
        final_decision = decision
        if decision.stopped_for_global_toxicity:
            stopped_safety = True
            break
        if at_cap:
            selected_pair = decision.selected_pair
            break
        if decision.selected_pair is None:
            raise RuntimeError("2010 GAO decision returned no pair without a global safety stop")
        current = decision.selected_pair

    if final_decision is None:
        raise RuntimeError("U2OET 2010 trial completed without a posterior decision")
    patient_data = u2oet_patients(
        np.asarray(rows, dtype=float),
        dose_counts=(d1.size, d2.size),
        efficacy_levels=efficacy_levels,
        toxicity_levels=toxicity_levels,
    )
    return U2OETGAO2010Trial(
        patients=patient_data,
        outcome_uniforms=tape,
        looks=tuple(look_records),
        final_decision=final_decision,
        selected_pair=selected_pair,
        stopped_for_global_toxicity=stopped_safety,
        stopped_early=stopped_safety and completed < n,
        stop_reason="all_dose_pairs_too_toxic" if stopped_safety else "patient_budget_reached",
        posterior_fits=posterior_fits,
        likelihood_evaluations=total_evaluations,
        likelihood_work_units=total_work,
        maximum_parameter_split_rhat=max_parameter_rhat,
        maximum_utility_mcse=max_utility_mcse,
        data_seed=data_seed,
        posterior_seed=posterior_seed,
        design_json=design_json,
    )
