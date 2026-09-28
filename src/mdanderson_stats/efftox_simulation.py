"""Bounded operating-characteristic simulation for completed-outcome EffTox."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from ._validation import FloatArray, finite, scalar
from .efftox_decision import EffToxContour, EffToxDecision, efftox_decision
from .efftox_legacy_contour import EffToxLegacyContour
from .efftox_model import EffToxPrior, efftox_standardize, fit_efftox

_MAX_TRIALS = 2_000
_MAX_COHORTS = 100
_MAX_PATIENTS = 500
_MAX_OUTPUT_CELLS = 1_000_000
_MAX_TOTAL_FIT_WORK = 20_000_000


def _frozen(value: ArrayLike, dtype: DTypeLike | None = None) -> NDArray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _setting(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return result


def _rng_pair(
    rng: int | np.random.Generator | None,
    sampler_rng: int | np.random.Generator | None,
) -> tuple[np.random.Generator, np.random.Generator]:
    if isinstance(rng, (bool, np.bool_)) or isinstance(sampler_rng, (bool, np.bool_)):
        raise ValueError("random-state values must not be booleans")
    if rng is not None and not isinstance(rng, (int, np.integer, np.random.Generator)):
        raise TypeError("rng must be an integer seed, Generator or None")
    if sampler_rng is not None and not isinstance(
        sampler_rng, (int, np.integer, np.random.Generator)
    ):
        raise TypeError("sampler_rng must be an integer seed, Generator or None")
    if isinstance(rng, (int, np.integer)) and int(rng) < 0:
        raise ValueError("rng seed must be nonnegative")
    if isinstance(sampler_rng, (int, np.integer)) and int(sampler_rng) < 0:
        raise ValueError("sampler_rng seed must be nonnegative")
    if isinstance(rng, np.random.Generator):
        outcome = rng
    else:
        outcome_sequence = np.random.SeedSequence(rng).spawn(2)[0]
        outcome = np.random.default_rng(outcome_sequence)
    if isinstance(sampler_rng, np.random.Generator):
        if isinstance(rng, np.random.Generator) and rng.bit_generator is sampler_rng.bit_generator:
            raise ValueError("outcome and sampler generators must be distinct")
        sampler = sampler_rng
    elif sampler_rng is not None:
        sampler_sequence = np.random.SeedSequence(sampler_rng).spawn(2)[1]
        sampler = np.random.default_rng(sampler_sequence)
    elif not isinstance(rng, np.random.Generator):
        sampler_sequence = np.random.SeedSequence(rng).spawn(2)[1]
        sampler = np.random.default_rng(sampler_sequence)
    else:
        # Derive a separate stream once, before simulating outcomes, so MCMC
        # work cannot alter later trial scenarios.
        sampler = np.random.default_rng(int(outcome.integers(0, 2**63, dtype=np.int64)))
    return outcome, sampler


@dataclass(frozen=True)
class EffToxSimulation:
    """Compact trial outcomes, dose allocation and operating characteristics.

    Dose labels are one-based; zero in ``selected_dose`` means no selection.
    Outcome cell axes are efficacy then toxicity, each ordered 0, 1.
    ``max_split_rhat`` and ``max_batch_mean_mcse`` summarize per-fit diagnostics;
    they are descriptive and do not certify sampler convergence.
    """

    doses: FloatArray
    true_joint_probabilities: FloatArray
    dose_patients: NDArray[np.int64]
    outcome_counts: NDArray[np.int64]
    selected_dose: NDArray[np.int64]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    allocation_probability: FloatArray
    mean_patients_per_dose: FloatArray
    observed_joint_probability: FloatArray
    cohorts_completed: NDArray[np.int64]
    stop_reason: tuple[str, ...]
    stop_reason_levels: tuple[str, ...]
    stop_reason_probability: FloatArray
    early_stop_probability: float
    early_stop_mcse: float
    no_selection_probability: float
    no_selection_mcse: float
    posterior_fit_count: NDArray[np.int64]
    likelihood_evaluations: NDArray[np.int64]
    max_split_rhat: FloatArray
    max_batch_mean_mcse: FloatArray
    planned_cohorts: int
    cohort_size: int
    starting_dose: int
    efficacy_limit: float
    toxicity_limit: float
    efficacy_probability: float
    toxicity_probability: float
    allow_untried_exploration: bool
    skip_policy: str
    zero_dose_shift: bool
    draws: int
    warmup: int
    chains: int
    work_estimate: int
    random_state: int | None
    sampler_random_state: int | None


def _fit_decision(
    doses: FloatArray,
    counts: NDArray[np.int64],
    prior: EffToxPrior,
    contour: EffToxContour | EffToxLegacyContour,
    sampler: np.random.Generator,
    *,
    draws: int,
    warmup: int,
    chains: int,
    efficacy_limit: float,
    toxicity_limit: float,
    efficacy_probability: float,
    toxicity_probability: float,
    starting_dose: int,
    last_dose: int | None,
    phase: Literal["interim", "final"],
    allow_untried_exploration: bool,
    skip_policy: Literal["both", "escalation"],
    zero_dose_shift: bool,
) -> tuple[EffToxDecision, int, float, float]:
    fit = fit_efftox(
        doses,
        counts,
        prior=prior,
        draws=draws,
        warmup=warmup,
        chains=chains,
        zero_dose_shift=zero_dose_shift,
        rng=sampler,
    )
    decision = efftox_decision(
        fit,
        contour,
        efficacy_limit=efficacy_limit,
        toxicity_limit=toxicity_limit,
        efficacy_probability=efficacy_probability,
        toxicity_probability=toxicity_probability,
        starting_dose=starting_dose,
        last_dose=last_dose,
        phase=phase,
        allow_untried_exploration=allow_untried_exploration,
        skip_policy=skip_policy,
    )
    rhat = np.asarray(fit.summary.split_rhat)
    mcse = np.asarray(fit.summary.batch_mean_mcse)
    valid_rhat = rhat[~np.isnan(rhat)]
    valid_mcse = mcse[~np.isnan(mcse)]
    max_rhat = float(np.max(valid_rhat)) if valid_rhat.size else float("nan")
    max_mcse = float(np.max(valid_mcse)) if valid_mcse.size else float("nan")
    return decision, fit.likelihood_evaluations, max_rhat, max_mcse


def simulate_efftox(
    doses: ArrayLike,
    true_joint_probabilities: ArrayLike,
    *,
    prior: EffToxPrior,
    contour: EffToxContour | EffToxLegacyContour,
    efficacy_limit: float,
    toxicity_limit: float,
    efficacy_probability: float,
    toxicity_probability: float,
    starting_dose: int = 1,
    cohorts: int = 6,
    cohort_size: int = 3,
    trials: int = 100,
    draws: int = 100,
    warmup: int = 50,
    chains: int = 2,
    allow_untried_exploration: bool = True,
    skip_policy: Literal["both", "escalation"] = "both",
    zero_dose_shift: bool = True,
    rng: int | np.random.Generator | None = 0,
    sampler_rng: int | np.random.Generator | None = None,
    max_total_fit_work: int = _MAX_TOTAL_FIT_WORK,
) -> EffToxSimulation:
    """Simulate completed-outcome cohorts under explicit joint truth tables.

    ``true_joint_probabilities[d,e,t]`` is the full efficacy/toxicity cell
    probability at each dose; the bivariate association is preserved. Each
    initial and interim assignment uses ``efftox_decision`` with a fresh
    posterior fit. Trials reaching their cohort limit receive a final fit and
    selection. Interim stops have no selected dose. Outcome and posterior
    streams are independent when ``rng`` is a seed; separate NumPy streams do
    not promise native Windows RNG parity.
    """
    if not isinstance(prior, EffToxPrior) or not isinstance(
        contour, (EffToxContour, EffToxLegacyContour)
    ):
        raise ValueError(
            "prior must be EffToxPrior and contour EffToxContour or EffToxLegacyContour"
        )
    if not isinstance(allow_untried_exploration, (bool, np.bool_)):
        raise ValueError("allow_untried_exploration must be boolean")
    if skip_policy not in ("both", "escalation"):
        raise ValueError("skip_policy must be 'both' or 'escalation'")
    raw_doses = np.asarray(doses)
    if raw_doses.ndim != 1 or not 2 <= raw_doses.size <= 20:
        raise ValueError("doses must contain 2..20 values")
    dose_values = finite(raw_doses, "doses")
    # Reuse core dose validation before entering the trial loop.
    dose_codes = efftox_standardize(dose_values, zero_dose_shift=zero_dose_shift)
    del dose_codes
    raw_truth = np.asarray(true_joint_probabilities)
    if raw_truth.shape != (dose_values.size, 2, 2):
        raise ValueError("true_joint_probabilities must have shape (doses,2,2)")
    truth = finite(raw_truth, "true_joint_probabilities")
    if np.any((truth < 0.0) | (truth > 1.0)):
        raise ValueError("joint cell probabilities must lie in [0,1]")
    if not np.allclose(truth.sum(axis=(1, 2)), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("joint cell probabilities must sum to one at every dose")
    repetition_count = _setting(trials, "trials", 1, _MAX_TRIALS)
    cohort_count = _setting(cohorts, "cohorts", 1, _MAX_COHORTS)
    size = _setting(cohort_size, "cohort_size", 1, _MAX_PATIENTS)
    if cohort_count * size > _MAX_PATIENTS:
        raise ValueError("planned enrollment must not exceed 500 patients per trial")
    start = _setting(starting_dose, "starting_dose", 1, dose_values.size)
    draw_count = _setting(draws, "draws", 8, 10_000)
    warm_count = _setting(warmup, "warmup", 0, 10_000)
    chain_count = _setting(chains, "chains", 2, 4)
    if chain_count * dose_values.size * draw_count > 200_000:
        raise ValueError("retained dose-probability draws exceed the EffTox fit limit")
    per_fit_work = chain_count * dose_values.size * (draw_count + warm_count) * 6
    if per_fit_work > 2_000_000:
        raise ValueError("one posterior fit exceeds the EffTox work limit")
    total_work = repetition_count * (cohort_count + 1) * per_fit_work
    total_limit = _setting(max_total_fit_work, "max_total_fit_work", 1, _MAX_TOTAL_FIT_WORK)
    if total_work > total_limit:
        raise ValueError("worst-case trial/refit work exceeds max_total_fit_work")
    if repetition_count * dose_values.size * 6 > _MAX_OUTPUT_CELLS:
        raise ValueError("trial outcome summaries exceed the bounded output-cell limit")
    # Let efftox_decision apply the same scalar/range validation to the other
    # probability thresholds before any trial work begins.
    e_limit = scalar(efficacy_limit, "efficacy_limit")
    t_limit = scalar(toxicity_limit, "toxicity_limit")
    e_cut = scalar(efficacy_probability, "efficacy_probability")
    t_cut = scalar(toxicity_probability, "toxicity_probability")
    if any(not 0.0 <= value <= 1.0 for value in (e_limit, t_limit, e_cut, t_cut)):
        raise ValueError("decision thresholds must be probabilities in [0,1]")
    outcome_rng, posterior_rng = _rng_pair(rng, sampler_rng)

    dose_count = int(dose_values.size)
    dose_patients = np.zeros((repetition_count, dose_count), dtype=np.int64)
    outcomes = np.zeros((repetition_count, dose_count, 2, 2), dtype=np.int64)
    selected = np.zeros(repetition_count, dtype=np.int64)
    completed = np.zeros(repetition_count, dtype=np.int64)
    fit_count = np.zeros(repetition_count, dtype=np.int64)
    likelihood_evaluations = np.zeros(repetition_count, dtype=np.int64)
    max_rhat = np.full(repetition_count, np.nan)
    max_mcse = np.full(repetition_count, np.nan)
    reasons: list[str] = []
    for trial in range(repetition_count):
        counts = np.zeros((dose_count, 2, 2), dtype=np.int64)
        decision, evaluations, rhat, mcse = _fit_decision(
            dose_values,
            counts,
            prior,
            contour,
            posterior_rng,
            draws=draw_count,
            warmup=warm_count,
            chains=chain_count,
            efficacy_limit=e_limit,
            toxicity_limit=t_limit,
            efficacy_probability=e_cut,
            toxicity_probability=t_cut,
            starting_dose=start,
            last_dose=None,
            phase="interim",
            allow_untried_exploration=bool(allow_untried_exploration),
            skip_policy=skip_policy,
            zero_dose_shift=zero_dose_shift,
        )
        fit_count[trial] += 1
        likelihood_evaluations[trial] += evaluations
        max_rhat[trial] = rhat
        max_mcse[trial] = mcse
        if decision.action != "start" or decision.dose is None:
            raise ArithmeticError("empty-data EffTox interim decision did not start at a dose")
        current_dose = int(decision.dose)
        reason = "max_cohorts"
        for cohort in range(cohort_count):
            cells = outcome_rng.multinomial(size, truth[current_dose - 1].reshape(4))
            cell_table = cells.reshape(2, 2)
            counts[current_dose - 1] += cell_table
            outcomes[trial, current_dose - 1] += cell_table
            dose_patients[trial, current_dose - 1] += size
            completed[trial] = cohort + 1
            if cohort + 1 == cohort_count:
                decision, evaluations, rhat, mcse = _fit_decision(
                    dose_values,
                    counts,
                    prior,
                    contour,
                    posterior_rng,
                    draws=draw_count,
                    warmup=warm_count,
                    chains=chain_count,
                    efficacy_limit=e_limit,
                    toxicity_limit=t_limit,
                    efficacy_probability=e_cut,
                    toxicity_probability=t_cut,
                    starting_dose=start,
                    last_dose=current_dose,
                    phase="final",
                    allow_untried_exploration=bool(allow_untried_exploration),
                    skip_policy=skip_policy,
                    zero_dose_shift=zero_dose_shift,
                )
                fit_count[trial] += 1
                likelihood_evaluations[trial] += evaluations
                max_rhat[trial] = np.fmax(max_rhat[trial], rhat)
                max_mcse[trial] = np.fmax(max_mcse[trial], mcse)
                if decision.action == "select" and decision.dose is not None:
                    selected[trial] = int(decision.dose)
                elif decision.action == "stop_no_admissible":
                    reason = decision.action
                else:
                    raise ArithmeticError("final EffTox decision returned an unexpected action")
                break
            decision, evaluations, rhat, mcse = _fit_decision(
                dose_values,
                counts,
                prior,
                contour,
                posterior_rng,
                draws=draw_count,
                warmup=warm_count,
                chains=chain_count,
                efficacy_limit=e_limit,
                toxicity_limit=t_limit,
                efficacy_probability=e_cut,
                toxicity_probability=t_cut,
                starting_dose=start,
                last_dose=current_dose,
                phase="interim",
                allow_untried_exploration=bool(allow_untried_exploration),
                skip_policy=skip_policy,
                zero_dose_shift=zero_dose_shift,
            )
            fit_count[trial] += 1
            likelihood_evaluations[trial] += evaluations
            max_rhat[trial] = np.fmax(max_rhat[trial], rhat)
            max_mcse[trial] = np.fmax(max_mcse[trial], mcse)
            if decision.action == "assign" and decision.dose is not None:
                current_dose = int(decision.dose)
                continue
            if decision.action in ("stop_no_admissible", "stop_no_reachable"):
                reason = decision.action
                break
            raise ArithmeticError("interim EffTox decision returned an unexpected action")
        reasons.append(reason)

    selection_frequency = np.bincount(selected, minlength=dose_count + 1) / repetition_count
    total_patients_by_dose = dose_patients.sum(axis=0)
    total_patients = int(total_patients_by_dose.sum())
    allocation = total_patients_by_dose / total_patients if total_patients else np.zeros(dose_count)
    pooled_cells = outcomes.sum(axis=0)
    observed_joint = np.full((dose_count, 2, 2), np.nan)
    for dose in range(dose_count):
        if total_patients_by_dose[dose]:
            observed_joint[dose] = pooled_cells[dose] / total_patients_by_dose[dose]
    reason_levels = ("max_cohorts", "stop_no_admissible", "stop_no_reachable")
    reason_frequency = (
        np.asarray([reasons.count(reason) for reason in reason_levels]) / repetition_count
    )
    early = completed < cohort_count
    early_probability = float(np.mean(early))
    none_probability = float(selection_frequency[0])
    return EffToxSimulation(
        _frozen(dose_values, np.float64),
        _frozen(truth, np.float64),
        _frozen(dose_patients, np.int64),
        _frozen(outcomes, np.int64),
        _frozen(selected, np.int64),
        _frozen(selection_frequency, np.float64),
        _frozen(np.sqrt(selection_frequency * (1.0 - selection_frequency) / repetition_count)),
        _frozen(allocation, np.float64),
        _frozen(dose_patients.mean(axis=0), np.float64),
        _frozen(observed_joint, np.float64),
        _frozen(completed, np.int64),
        tuple(reasons),
        reason_levels,
        _frozen(reason_frequency, np.float64),
        early_probability,
        sqrt(early_probability * (1.0 - early_probability) / repetition_count),
        none_probability,
        sqrt(none_probability * (1.0 - none_probability) / repetition_count),
        _frozen(fit_count, np.int64),
        _frozen(likelihood_evaluations, np.int64),
        _frozen(max_rhat, np.float64),
        _frozen(max_mcse, np.float64),
        cohort_count,
        size,
        start,
        e_limit,
        t_limit,
        e_cut,
        t_cut,
        bool(allow_untried_exploration),
        skip_policy,
        bool(zero_dose_shift),
        draw_count,
        warm_count,
        chain_count,
        total_work,
        int(rng) if isinstance(rng, (int, np.integer)) else None,
        int(sampler_rng) if isinstance(sampler_rng, (int, np.integer)) else None,
    )
