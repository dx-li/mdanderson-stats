"""Serial operating-characteristic simulation for CRM trial replay."""

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray, scalar
from .bmacrm import _raw_numeric
from .crm_trial import CRMTrial, _count_setting, run_crm_trial
from .dacrm import (
    _MAX_RETAINED_CELLS,
    _MAX_TRANSITION_WORK,
    DACRMPrior,
)
from .dacrm import (
    _integer_setting as _da_integer_setting,
)
from .toxicity_timing import toxicity_time_quantile

_MAX_PATIENTS = 200
_MAX_DOSES = 20
_MAX_MODELS = 5
_MAX_TRIALS = 10_000
_MAX_DECISION_EVALUATIONS = 2_000_000
_MAX_TRIAL_EVALUATIONS = 20_000_000
_MAX_SIMULATION_EVALUATIONS = 200_000_000


@dataclass(frozen=True)
class CRMSimulation:
    """Compact operating characteristics across simulated CRM trials."""

    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    selected_dose: NDArray[np.int64]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    no_selection_probability: float
    no_selection_mcse: float
    early_stop_probability: float
    early_stop_mcse: float
    mean_patients: FloatArray
    mean_toxicities: FloatArray
    duration: FloatArray
    decision_time: FloatArray
    suspension_time: FloatArray
    stop_reason: tuple[str, ...]
    trial_evaluations: NDArray[np.int64]
    evaluations: int
    max_dose_mcse: FloatArray
    max_split_rhat: FloatArray


def _freeze_int(values: NDArray[np.int64]) -> NDArray[np.int64]:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


def simulate_crm(
    skeletons: ArrayLike,
    true_toxicity: ArrayLike,
    window: float,
    accrual_rate: float,
    *,
    target: float,
    cohorts: int = 10,
    cohort_size: int = 3,
    trials: int = 100,
    starting_dose: int = 0,
    method: Literal["bmacrm", "dacrm"] = "bmacrm",
    da_prior: DACRMPrior | None = None,
    model_prior: ArrayLike | None = None,
    prior_sd: float | None = None,
    safety_cutoff: float = 0.9,
    minimum_observed: int | None = None,
    rng: int | np.random.Generator | None = None,
    sampler_rng: np.random.Generator | None = None,
    draws: int = 2000,
    warmup: int = 1000,
    chains: int = 2,
    arrival: str = "exponential",
    event_distribution: str = "weibull",
    late_probability: ArrayLike | None = None,
    max_completions: int = 128,
    max_evaluations: int = 200_000,
    max_trial_evaluations: int = 2_000_000,
    max_total_evaluations: int = 20_000_000,
) -> CRMSimulation:
    """Simulate serial fixed-cohort CRM trials and summarize selection.

    Arrival gaps include the first patient and are fixed or exponential with
    mean ``1 / accrual_rate``. Potential toxicities are independent by patient
    and dose, and event times use the shared calibrated timing distributions.
    The ``rng`` stream generates arrivals and potential outcomes. DA-CRM
    requires a separate ``sampler_rng`` so posterior sampling never changes
    future trial scenarios.
    """
    if method not in {"bmacrm", "dacrm"}:
        raise ValueError("method must be 'bmacrm' or 'dacrm'")
    if sampler_rng is not None and not isinstance(sampler_rng, np.random.Generator):
        raise TypeError("sampler_rng must be a numpy.random.Generator")
    if arrival not in {"fixed", "exponential"}:
        raise ValueError("arrival must be 'fixed' or 'exponential'")
    true_p = _raw_numeric(true_toxicity, "true_toxicity", _MAX_DOSES)
    if true_p.ndim != 1 or not 2 <= true_p.size <= _MAX_DOSES:
        raise ValueError("true_toxicity must contain 2..20 dose probabilities")
    if np.any((true_p < 0) | (true_p > 1)):
        raise ValueError("true_toxicity probabilities must lie in [0,1]")
    raw_skeletons = _raw_numeric(skeletons, "skeletons", _MAX_MODELS * _MAX_DOSES)
    if raw_skeletons.ndim == 1:
        skeleton_matrix = raw_skeletons[None, :]
    elif raw_skeletons.ndim == 2:
        skeleton_matrix = raw_skeletons
    else:
        raise ValueError("skeletons must be one- or two-dimensional")
    model_count, dose_count = skeleton_matrix.shape
    if dose_count != true_p.size:
        raise ValueError("skeletons and true_toxicity must have the same dose count")
    if (
        not 1 <= model_count <= _MAX_MODELS
        or np.any((skeleton_matrix <= 0) | (skeleton_matrix >= 1))
        or np.any(np.diff(skeleton_matrix, axis=1) < 0)
    ):
        raise ValueError("skeletons must be 1..5 nondecreasing rows in (0,1)")
    if method == "dacrm" and model_count != 1:
        raise ValueError("DA-CRM requires exactly one skeleton")

    duration = scalar(window, "window")
    rate = scalar(accrual_rate, "accrual_rate")
    target_value = scalar(target, "target")
    cutoff = scalar(safety_cutoff, "safety_cutoff")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("window must be positive and finite")
    if not np.isfinite(rate) or rate <= 0 or not np.isfinite(1 / rate) or 1 / rate == 0:
        raise ValueError("accrual_rate must have a positive finite reciprocal")
    if not 0 < target_value < 1:
        raise ValueError("target must lie strictly between 0 and 1")
    if not 0 <= cutoff <= 1:
        raise ValueError("safety_cutoff must lie in [0,1]")
    late_value: FloatArray | None = None
    if late_probability is not None:
        late_value = _raw_numeric(late_probability, "late_probability", dose_count)
        if late_value.ndim != 0 and late_value.shape != (dose_count,):
            raise ValueError("late_probability must be scalar or one value per dose")
    cohort_count = _count_setting(cohorts, "cohorts", 1, _MAX_PATIENTS)
    size = _count_setting(cohort_size, "cohort_size", 1, 4)
    planned_patients = cohort_count * size
    if planned_patients > _MAX_PATIENTS:
        raise ValueError("planned enrollment must not exceed 200 patients")
    starting = _count_setting(starting_dose, "starting_dose", 0, dose_count - 1)
    repetition_count = _count_setting(trials, "trials", 1, _MAX_TRIALS)
    completion_limit = _count_setting(max_completions, "max_completions", 1, 1024)
    decision_limit = _count_setting(
        max_evaluations, "max_evaluations", 1, _MAX_DECISION_EVALUATIONS
    )
    trial_limit = _count_setting(
        max_trial_evaluations, "max_trial_evaluations", 1, _MAX_TRIAL_EVALUATIONS
    )
    simulation_limit = _count_setting(
        max_total_evaluations, "max_total_evaluations", 1, _MAX_SIMULATION_EVALUATIONS
    )

    if method == "bmacrm":
        if da_prior is not None or minimum_observed is not None:
            raise ValueError("da_prior and minimum_observed apply only to method='dacrm'")
        if prior_sd is not None and not 1e-3 <= scalar(prior_sd, "prior_sd") <= 10:
            raise ValueError("prior_sd must lie in [1e-3,10]")
        if model_prior is not None:
            prior_weights = _raw_numeric(model_prior, "model_prior", _MAX_MODELS)
            if (
                prior_weights.ndim != 1
                or prior_weights.shape != (model_count,)
                or np.any(prior_weights < 0)
                or not np.any(prior_weights > 0)
            ):
                raise ValueError("model_prior must give nonnegative weights with positive mass")
    else:
        if model_prior is not None or prior_sd is not None:
            raise ValueError("model_prior and prior_sd apply only to method='bmacrm'")
        if not isinstance(da_prior, DACRMPrior) or da_prior.breaks[-1] != duration:
            raise ValueError("DA-CRM requires a prior ending at window")
        if minimum_observed is None:
            raise ValueError("DA-CRM requires explicit minimum_observed")
        _count_setting(minimum_observed, "minimum_observed", 0, _MAX_PATIENTS)
        draw_count = _da_integer_setting(draws, "draws", 8, 10_000)
        warmup_count = _da_integer_setting(warmup, "warmup", 0, 10_000)
        chain_count = _da_integer_setting(chains, "chains", 2, 4)
        if not isinstance(sampler_rng, np.random.Generator):
            raise ValueError("DA-CRM simulation requires a separate sampler_rng Generator")
        if isinstance(rng, np.random.Generator) and rng.bit_generator is sampler_rng.bit_generator:
            raise ValueError("rng and sampler_rng must use distinct bit generators")
        transition_work = (
            chain_count
            * (draw_count + warmup_count)
            * (planned_patients + dose_count + da_prior.shape.size)
        )
        retained_cells = (
            chain_count * draw_count * (1 + da_prior.shape.size + dose_count + planned_patients)
        )
        if transition_work > _MAX_TRANSITION_WORK:
            raise ValueError("worst-case DA-CRM transition work exceeds its bounded limit")
        if retained_cells > _MAX_RETAINED_CELLS:
            raise ValueError("worst-case DA-CRM retained draws exceed their bounded limit")
    if rng is not None and not isinstance(rng, (int, np.integer, np.random.Generator)):
        raise TypeError("rng must be an integer seed or numpy.random.Generator")
    if isinstance(rng, (bool, np.bool_)):
        raise TypeError("rng must not be boolean")
    timing_check = toxicity_time_quantile(
        0.5,
        true_p,
        duration,
        distribution=event_distribution,
        late_probability=late_value,
    )
    if timing_check.shape != true_p.shape:
        raise ValueError("event timing settings do not broadcast to the dose probabilities")

    outcome_rng = np.random.default_rng(rng)
    patient_counts = np.zeros((repetition_count, dose_count), dtype=np.int64)
    toxicity_counts = np.zeros_like(patient_counts)
    selected = np.full(repetition_count, -1, dtype=np.int64)
    durations = np.empty(repetition_count)
    decision_times = np.empty(repetition_count)
    suspension_times = np.empty(repetition_count)
    trial_evaluations = np.zeros(repetition_count, dtype=np.int64)
    diagnostic_mcse = np.full(repetition_count, np.nan)
    diagnostic_rhat = np.full(repetition_count, np.nan)
    reasons: list[str] = []
    total_evaluations = 0

    for trial_index in range(repetition_count):
        if total_evaluations >= simulation_limit:
            raise RuntimeError("CRM simulation exhausted max_total_evaluations")
        if arrival == "fixed":
            gaps = np.full(planned_patients, 1 / rate)
        else:
            gaps = outcome_rng.exponential(1 / rate, planned_patients)
        quantiles = outcome_rng.random((planned_patients, dose_count))
        potential_delays = toxicity_time_quantile(
            quantiles,
            true_p,
            duration,
            distribution=event_distribution,
            late_probability=late_value,
        )
        remaining = min(trial_limit, simulation_limit - total_evaluations)
        if remaining < 1:
            raise RuntimeError("CRM simulation exhausted max_total_evaluations")
        result: CRMTrial = run_crm_trial(
            raw_skeletons,
            gaps,
            potential_delays,
            duration,
            target=target_value,
            cohort_size=size,
            starting_dose=starting,
            method=method,
            da_prior=da_prior,
            model_prior=model_prior,
            prior_sd=prior_sd,
            safety_cutoff=cutoff,
            minimum_observed=minimum_observed,
            rng=sampler_rng,
            draws=draws,
            warmup=warmup,
            chains=chains,
            max_completions=completion_limit,
            max_evaluations=decision_limit,
            max_total_evaluations=remaining,
        )
        patient_counts[trial_index] = result.treated_counts
        toxicity_counts[trial_index] = result.toxicities
        selected[trial_index] = -1 if result.selected_dose is None else result.selected_dose
        durations[trial_index] = result.final_time
        decision_times[trial_index] = result.decision_time
        suspension_times[trial_index] = result.suspension_time
        reasons.append(result.stop_reason)
        trial_evaluations[trial_index] = result.evaluations
        total_evaluations += result.evaluations
        if result.max_dose_mcse is not None:
            diagnostic_mcse[trial_index] = result.max_dose_mcse
        if result.max_split_rhat is not None:
            diagnostic_rhat[trial_index] = result.max_split_rhat

    selection = np.bincount(selected[selected >= 0], minlength=dose_count) / repetition_count
    no_selection = float(np.count_nonzero(selected < 0) / repetition_count)
    early = np.asarray(
        [
            reason == "safety_stop" and int(patient_counts[i].sum()) < planned_patients
            for i, reason in enumerate(reasons)
        ],
        dtype=bool,
    )
    early_probability = float(np.mean(early))
    return CRMSimulation(
        _freeze_int(patient_counts),
        _freeze_int(toxicity_counts),
        _freeze_int(selected),
        _freeze(selection),
        _freeze(np.sqrt(selection * (1 - selection) / repetition_count)),
        no_selection,
        sqrt(no_selection * (1 - no_selection) / repetition_count),
        early_probability,
        sqrt(early_probability * (1 - early_probability) / repetition_count),
        _freeze(np.mean(patient_counts, axis=0)),
        _freeze(np.mean(toxicity_counts, axis=0)),
        _freeze(durations),
        _freeze(decision_times),
        _freeze(suspension_times),
        tuple(reasons),
        _freeze_int(trial_evaluations),
        total_evaluations,
        _freeze(diagnostic_mcse),
        _freeze(diagnostic_rhat),
    )
