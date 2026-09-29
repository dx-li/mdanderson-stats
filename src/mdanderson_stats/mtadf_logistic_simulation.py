"""Bounded complete-cohort simulation for MTADF logistic OBD policies."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar
from .mtadf import MTADFPrior, mtadf_decision, mtadf_toxicity_prior
from .mtadf_logistic import (
    MTADFLocalLogisticDecision,
    MTADFLocalLogisticPosterior,
    MTADFLogisticDecision,
    MTADFLogisticPosterior,
    _mcmc_options,
    _preflight_work,
    mtadf_local_logistic_decision,
    mtadf_logistic_decision,
    mtadf_logistic_posterior,
)

_MAX_DOSES = 20
_MAX_COHORTS = 1_000
_MAX_TRIALS = 10_000
_MAX_PATIENTS = 1_000
_MAX_TOTAL_WORK = 500_000_000
_MAX_TOTAL_STORAGE = 1_000_000_000


def _freeze_int(value: ArrayLike) -> NDArray[np.int64]:
    array = np.ascontiguousarray(value, dtype=np.int64)
    return np.frombuffer(array.tobytes(), dtype=np.int64).reshape(array.shape)


def _freeze_uint(value: ArrayLike) -> NDArray[np.uint64]:
    array = np.ascontiguousarray(value, dtype=np.uint64)
    return np.frombuffer(array.tobytes(), dtype=np.uint64).reshape(array.shape)


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iu" or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    result = int(raw)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return result


def _probabilities(value: ArrayLike, name: str) -> FloatArray:
    shape = np.shape(value)
    if len(shape) != 1 or not 1 <= shape[0] <= _MAX_DOSES:
        raise ValueError(f"{name} must be a vector with 1..{_MAX_DOSES} entries")
    result = finite(value, name)
    if result.ndim != 1 or np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} values must be probabilities in [0,1]")
    return result


@dataclass(frozen=True)
class MTADFLogisticSimulationConfig:
    """Complete-cohort design and explicit Python conduct settings.

    `true_toxicity` and `true_efficacy` are independent dose-specific binary
    marginals, so reported operating characteristics are conditional on the
    explicit `independent_marginals` patient-level outcome model. The local
    design starts at dose zero and ramps through one
    cohort at each of its first `window_length` levels.
    """

    model: Literal["global", "local"]
    true_toxicity: ArrayLike
    true_efficacy: ArrayLike
    doses: ArrayLike
    outcome_model: Literal["independent_marginals"] = "independent_marginals"
    cohorts: int = 6
    cohort_size: int = 3
    toxicity_limit: float = 0.3
    safety_cutoff: float = 0.8
    margin: float = 0.05
    concentration: float = 0.5
    prior: MTADFPrior | None = None
    window_length: int = 2
    efficacy_escalation_cutoff: float = 0.4
    efficacy_deescalation_cutoff: float = 0.3
    draws: int = 2_000
    warmup: int = 1_000
    chains: int = 4


@dataclass(frozen=True)
class MTADFLogisticTrial:
    """Compact output for one replayable complete-cohort trial."""

    subjects: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    responses: NDArray[np.int64]
    selected_dose: int
    stop_reason: str
    posterior_fit_count: int
    mean_acceptance_rate: float
    maximum_split_rhat: float
    mean_positive_slope_mcse: float
    maximum_positive_slope_rhat: float


@dataclass(frozen=True)
class MTADFLogisticSimulation:
    """Per-trial count summaries and compact Monte Carlo operating characteristics."""

    subjects: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    responses: NDArray[np.int64]
    selected_dose: NDArray[np.int64]
    stop_reason: tuple[str, ...]
    trial_seeds: NDArray[np.uint64]
    posterior_fit_count: NDArray[np.int64]
    mean_acceptance_rate: FloatArray
    maximum_split_rhat: FloatArray
    mean_positive_slope_mcse: FloatArray
    maximum_positive_slope_rhat: FloatArray
    selection_probability: FloatArray
    selection_mcse: FloatArray
    no_selection_probability: float
    no_selection_mcse: float
    early_stop_probability: float
    early_stop_mcse: float
    mean_subjects: FloatArray
    mean_toxicities: FloatArray
    mean_responses: FloatArray


@dataclass(frozen=True)
class _Prepared:
    model: Literal["global", "local"]
    tox: FloatArray
    eff: FloatArray
    doses: FloatArray
    cohorts: int
    cohort_size: int
    toxicity_limit: float
    safety_cutoff: float
    prior: MTADFPrior
    window_length: int
    escalation: float
    deescalation: float
    draws: int
    warmup: int
    chains: int


def _prepare(config: MTADFLogisticSimulationConfig) -> _Prepared:
    if not isinstance(config, MTADFLogisticSimulationConfig):
        raise TypeError("config must be an MTADFLogisticSimulationConfig")
    if config.model not in ("global", "local"):
        raise ValueError("model must be 'global' or 'local'")
    if config.outcome_model != "independent_marginals":
        raise ValueError("outcome_model must be 'independent_marginals'")
    tox = _probabilities(config.true_toxicity, "true_toxicity")
    eff = _probabilities(config.true_efficacy, "true_efficacy")
    if tox.shape != eff.shape:
        raise ValueError("true_toxicity and true_efficacy must have matching dose counts")
    doses = finite(config.doses, "doses")
    if doses.ndim != 1 or doses.shape != tox.shape or np.any(np.diff(doses) <= 0):
        raise ValueError("doses must be finite, strictly increasing, and match probabilities")
    cohorts = _integer(config.cohorts, "cohorts", 1, _MAX_COHORTS)
    cohort_size = _integer(config.cohort_size, "cohort_size", 1, _MAX_PATIENTS)
    if cohorts * cohort_size > _MAX_PATIENTS:
        raise ValueError(f"planned patients per trial must not exceed {_MAX_PATIENTS}")
    phi = scalar(config.toxicity_limit, "toxicity_limit")
    cutoff = scalar(config.safety_cutoff, "safety_cutoff")
    if not 0 < phi < 1 or not 0 < cutoff < 1:
        raise ValueError("toxicity_limit and safety_cutoff must lie in (0,1)")
    prior = config.prior
    if prior is None:
        prior = mtadf_toxicity_prior(
            phi,
            cutoff,
            scalar(config.margin, "margin"),
            scalar(config.concentration, "concentration"),
        )
    elif not isinstance(prior, MTADFPrior):
        raise ValueError("prior must be an MTADFPrior")
    if config.model == "local":
        length = _integer(config.window_length, "window_length", 2, tox.size)
        escalation = scalar(config.efficacy_escalation_cutoff, "efficacy_escalation_cutoff")
        deescalation = scalar(config.efficacy_deescalation_cutoff, "efficacy_deescalation_cutoff")
        if not 0 <= deescalation < escalation <= 1:
            raise ValueError("local cutoffs must satisfy 0 <= de-escalation < escalation <= 1")
        if cohorts < length:
            raise ValueError(
                "local simulation requires at least window_length cohorts for its initial ramp"
            )
    else:
        length = 2
        escalation, deescalation = 0.4, 0.3
    draws, warmup, chains = _mcmc_options(config.draws, config.warmup, config.chains)
    fit_doses = tox.size if config.model == "global" else length
    fit_parameters = 3 if config.model == "global" else 2
    _preflight_work(draws, warmup, chains, fit_doses, fit_parameters)
    return _Prepared(
        config.model,
        _freeze(tox),
        _freeze(eff),
        _freeze(doses),
        cohorts,
        cohort_size,
        phi,
        cutoff,
        prior,
        length,
        escalation,
        deescalation,
        draws,
        warmup,
        chains,
    )


def _seed(value: object, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer seed in [0, 2**64-1]")
    result = int(value)
    if not 0 <= result <= np.iinfo(np.uint64).max:
        raise ValueError(f"{name} must be an integer seed in [0, 2**64-1]")
    return result


def _record_diagnostic(
    posterior: MTADFLogisticPosterior | MTADFLocalLogisticPosterior,
) -> tuple[float, float, float, float]:
    acceptance = float(np.mean(posterior.acceptance_rate))
    if isinstance(posterior, MTADFLogisticPosterior):
        rhat = posterior.efficacy_summary.split_rhat
        slope_mcse = slope_rhat = float("nan")
    else:
        rhat = posterior.parameter_summary.split_rhat
        slope_mcse = posterior.probability_positive_slope_mcse
        slope_rhat_values = np.asarray(posterior.positive_slope_summary.split_rhat, dtype=float)
        slope_rhat = float(np.max(slope_rhat_values))
    values = np.asarray(rhat, dtype=float)
    finite_or_infinite = values[~np.isnan(values)]
    maximum = float(np.max(finite_or_infinite)) if finite_or_infinite.size else float("nan")
    return acceptance, maximum, slope_mcse, slope_rhat


def _maximum_trial_fits(prepared: _Prepared) -> int:
    if prepared.model == "global":
        return prepared.cohorts
    completed_ramp_interims = max(0, prepared.cohorts - prepared.window_length)
    if completed_ramp_interims == 0:
        return 0
    if prepared.tox.size == prepared.window_length:
        return completed_ramp_interims
    # The first post-ramp analysis has no previously treated next dose. Each
    # later local interim can retain a distinct bounce-guard fit.
    return 2 * completed_ramp_interims - 1


def _worst_trial_work(prepared: _Prepared) -> int:
    evaluation_doses = prepared.tox.size if prepared.model == "global" else prepared.window_length
    return (
        _maximum_trial_fits(prepared)
        * prepared.chains
        * (prepared.draws + prepared.warmup)
        * evaluation_doses
    )


def _simulate_trial(
    prepared: _Prepared, outcome_seed: int, sampler_seed: int
) -> MTADFLogisticTrial:
    d = prepared.tox.size
    n = np.zeros(d, dtype=np.int64)
    tox = np.zeros(d, dtype=np.int64)
    response = np.zeros(d, dtype=np.int64)
    outcome_rng = np.random.default_rng(outcome_seed)
    sampler_rng = np.random.default_rng(sampler_seed)
    acceptance_total = 0.0
    max_rhat = float("nan")
    slope_mcse_total = 0.0
    slope_mcse_count = 0
    slope_rhat_max = float("nan")
    fit_count = 0

    def record(posterior: MTADFLogisticPosterior | MTADFLocalLogisticPosterior) -> None:
        nonlocal acceptance_total, max_rhat, fit_count
        nonlocal slope_mcse_total, slope_mcse_count, slope_rhat_max
        acceptance, rhat, slope_mcse, slope_rhat = _record_diagnostic(posterior)
        acceptance_total += acceptance
        max_rhat = rhat if np.isnan(max_rhat) else max(max_rhat, rhat)
        if np.isfinite(slope_mcse):
            slope_mcse_total += slope_mcse
            slope_mcse_count += 1
        if not np.isnan(slope_rhat):
            slope_rhat_max = (
                slope_rhat if np.isnan(slope_rhat_max) else max(slope_rhat_max, slope_rhat)
            )
        fit_count += 1

    if prepared.model == "global":
        initial: MTADFLogisticDecision | MTADFLocalLogisticDecision = mtadf_logistic_decision(
            n,
            tox,
            response,
            prepared.doses,
            current_dose=None,
            toxicity_limit=prepared.toxicity_limit,
            safety_cutoff=prepared.safety_cutoff,
            prior=prepared.prior,
        )
    else:
        initial = mtadf_local_logistic_decision(
            n,
            tox,
            response,
            prepared.doses,
            current_dose=None,
            window_length=prepared.window_length,
            efficacy_escalation_cutoff=prepared.escalation,
            efficacy_deescalation_cutoff=prepared.deescalation,
            toxicity_limit=prepared.toxicity_limit,
            safety_cutoff=prepared.safety_cutoff,
            prior=prepared.prior,
        )
    if initial.action == "stop":
        return MTADFLogisticTrial(
            _freeze_int(n),
            _freeze_int(tox),
            _freeze_int(response),
            -1,
            "no_admissible_start",
            0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )
    assert initial.dose is not None
    current = initial.dose
    reason = "maximum_enrollment_no_safe_dose"
    selected = -1
    for cohort_index in range(prepared.cohorts):
        size = prepared.cohort_size
        tox[current] += outcome_rng.binomial(size, prepared.tox[current])
        response[current] += outcome_rng.binomial(size, prepared.eff[current])
        n[current] += size
        final_look = cohort_index == prepared.cohorts - 1

        decision: MTADFLogisticDecision | MTADFLocalLogisticDecision
        if prepared.model == "global":
            if final_look:
                final_safety = mtadf_decision(
                    n,
                    tox,
                    response,
                    final=True,
                    toxicity_limit=prepared.toxicity_limit,
                    safety_cutoff=prepared.safety_cutoff,
                    prior=prepared.prior,
                )
                if not np.any(final_safety.admissible):
                    decision = mtadf_logistic_decision(
                        n,
                        tox,
                        response,
                        prepared.doses,
                        current_dose=current,
                        toxicity_limit=prepared.toxicity_limit,
                        safety_cutoff=prepared.safety_cutoff,
                        prior=prepared.prior,
                        final=True,
                    )
                else:
                    posterior = mtadf_logistic_posterior(
                        n,
                        response,
                        prepared.doses,
                        draws=prepared.draws,
                        warmup=prepared.warmup,
                        chains=prepared.chains,
                        rng=sampler_rng,
                    )
                    record(posterior)
                    decision = mtadf_logistic_decision(
                        n,
                        tox,
                        response,
                        prepared.doses,
                        current_dose=current,
                        posterior=posterior,
                        toxicity_limit=prepared.toxicity_limit,
                        safety_cutoff=prepared.safety_cutoff,
                        prior=prepared.prior,
                        final=True,
                    )
            else:
                safety = mtadf_decision(
                    n,
                    tox,
                    response,
                    current_dose=current,
                    toxicity_limit=prepared.toxicity_limit,
                    safety_cutoff=prepared.safety_cutoff,
                    prior=prepared.prior,
                )
                if not np.any(safety.admissible) or not safety.admissible[current]:
                    decision = mtadf_logistic_decision(
                        n,
                        tox,
                        response,
                        prepared.doses,
                        current_dose=current,
                        toxicity_limit=prepared.toxicity_limit,
                        safety_cutoff=prepared.safety_cutoff,
                        prior=prepared.prior,
                    )
                else:
                    posterior = mtadf_logistic_posterior(
                        n,
                        response,
                        prepared.doses,
                        draws=prepared.draws,
                        warmup=prepared.warmup,
                        chains=prepared.chains,
                        rng=sampler_rng,
                    )
                    record(posterior)
                    decision = mtadf_logistic_decision(
                        n,
                        tox,
                        response,
                        prepared.doses,
                        current_dose=current,
                        posterior=posterior,
                        toxicity_limit=prepared.toxicity_limit,
                        safety_cutoff=prepared.safety_cutoff,
                        prior=prepared.prior,
                    )
        else:
            decision = mtadf_local_logistic_decision(
                n,
                tox,
                response,
                prepared.doses,
                current_dose=current,
                window_length=prepared.window_length,
                efficacy_escalation_cutoff=prepared.escalation,
                efficacy_deescalation_cutoff=prepared.deescalation,
                draws=prepared.draws,
                warmup=prepared.warmup,
                chains=prepared.chains,
                rng=sampler_rng,
                toxicity_limit=prepared.toxicity_limit,
                safety_cutoff=prepared.safety_cutoff,
                prior=prepared.prior,
                final=final_look,
            )
            assert isinstance(decision, MTADFLocalLogisticDecision)
            if decision.posterior is not None:
                record(decision.posterior)
            if decision.bounce_guard_posterior is not None and (
                decision.bounce_guard_posterior is not decision.posterior
            ):
                record(decision.bounce_guard_posterior)

        if final_look:
            if decision.action == "select_obd":
                assert decision.dose is not None
                selected = decision.dose
                reason = "maximum_enrollment"
            else:
                reason = "maximum_enrollment_no_safe_dose"
            break
        if decision.action == "stop":
            reason = "early_safety_stop"
            break
        assert decision.dose is not None
        current = decision.dose
        del decision
        if prepared.model == "global" and "posterior" in locals():
            del posterior

    return MTADFLogisticTrial(
        _freeze_int(n),
        _freeze_int(tox),
        _freeze_int(response),
        selected,
        reason,
        fit_count,
        acceptance_total / fit_count if fit_count else float("nan"),
        max_rhat,
        slope_mcse_total / slope_mcse_count if slope_mcse_count else float("nan"),
        slope_rhat_max,
    )


def simulate_mtadf_logistic_trial(
    config: MTADFLogisticSimulationConfig,
    *,
    outcome_seed: int,
    sampler_seed: int,
) -> MTADFLogisticTrial:
    """Replay one trial from independent outcome and posterior-sampler seeds."""
    prepared = _prepare(config)
    if _worst_trial_work(prepared) > _MAX_TOTAL_WORK:
        raise ValueError("worst-case single-trial posterior work exceeds the simulation work bound")
    return _simulate_trial(
        prepared,
        _seed(outcome_seed, "outcome_seed"),
        _seed(sampler_seed, "sampler_seed"),
    )


def _seed_entropy(rng: int | np.integer | np.random.SeedSequence | None) -> np.random.SeedSequence:
    if isinstance(rng, np.random.SeedSequence):
        return rng
    if rng is None or (
        isinstance(rng, (int, np.integer)) and not isinstance(rng, (bool, np.bool_))
    ):
        return np.random.SeedSequence(None if rng is None else int(rng))
    raise ValueError("rng must be an integer seed, SeedSequence, or None")


def simulate_mtadf_logistic(
    config: MTADFLogisticSimulationConfig,
    *,
    trials: int = 100,
    rng: int | np.integer | np.random.SeedSequence | None = None,
    max_total_work: int = 100_000_000,
    max_total_storage_bytes: int = 512_000_000,
) -> MTADFLogisticSimulation:
    """Run bounded serial complete-cohort trials and summarize selection.

    Each trial gets independent outcome and sampler seeds, retained in
    `trial_seeds[:, 0:2]` for exact single-trial replay. Work preflight
    includes one global fit per cohort or the maximum local fits after the
    required initial ramp; a local interim can require a distinct bounce-guard
    fit. Identical local windows are charged once. No posterior draws or
    patient-level histories are retained.
    """
    prepared = _prepare(config)
    trial_count = _integer(trials, "trials", 1, _MAX_TRIALS)
    work_limit = _integer(max_total_work, "max_total_work", 1, _MAX_TOTAL_WORK)
    storage_limit = _integer(
        max_total_storage_bytes, "max_total_storage_bytes", 1, _MAX_TOTAL_STORAGE
    )
    dose_count = prepared.tox.size
    worst_work = trial_count * _worst_trial_work(prepared)
    if worst_work > work_limit:
        raise ValueError("worst-case posterior work exceeds max_total_work")

    # Three count matrices are retained in mutable and frozen form at once;
    # seeds, decisions, reasons, diagnostics and per-trial working counts are
    # included conservatively.
    bytes_per_trial = 2 * 3 * dose_count * np.dtype(np.int64).itemsize + 96 + 4 * dose_count * 8
    draw_cells = (
        prepared.chains
        * prepared.draws
        * (
            3 + 2 * dose_count + 3 * max(3, dose_count)
            if prepared.model == "global"
            else 2 + 2 * prepared.window_length + 3 * max(2, prepared.window_length)
        )
    )
    distinct_live_fits = (
        1 if prepared.model == "global" or prepared.tox.size == prepared.window_length else 2
    )
    posterior_peak = distinct_live_fits * draw_cells * np.dtype(np.float64).itemsize
    estimated_storage = trial_count * bytes_per_trial + 1024 * dose_count + posterior_peak
    if estimated_storage > storage_limit:
        raise ValueError(
            "aggregate results, freeze-time copies, and peak posterior fit exceed "
            "max_total_storage_bytes"
        )

    children = _seed_entropy(rng).spawn(2)
    outcome_seeds = children[0].generate_state(trial_count, dtype=np.uint64)
    sampler_seeds = children[1].generate_state(trial_count, dtype=np.uint64)
    trial_seeds = np.column_stack((outcome_seeds, sampler_seeds))
    subjects = np.zeros((trial_count, dose_count), dtype=np.int64)
    toxicities = np.zeros_like(subjects)
    responses = np.zeros_like(subjects)
    selected = np.full(trial_count, -1, dtype=np.int64)
    reasons: list[str] = []
    fit_counts = np.zeros(trial_count, dtype=np.int64)
    acceptance = np.full(trial_count, np.nan, dtype=float)
    max_rhat = np.full(trial_count, np.nan, dtype=float)
    slope_mcse = np.full(trial_count, np.nan, dtype=float)
    slope_rhat = np.full(trial_count, np.nan, dtype=float)
    planned = prepared.cohorts * prepared.cohort_size

    for index in range(trial_count):
        result = _simulate_trial(prepared, int(outcome_seeds[index]), int(sampler_seeds[index]))
        subjects[index] = result.subjects
        toxicities[index] = result.toxicities
        responses[index] = result.responses
        selected[index] = result.selected_dose
        reasons.append(result.stop_reason)
        fit_counts[index] = result.posterior_fit_count
        acceptance[index] = result.mean_acceptance_rate
        max_rhat[index] = result.maximum_split_rhat
        slope_mcse[index] = result.mean_positive_slope_mcse
        slope_rhat[index] = result.maximum_positive_slope_rhat
        del result

    selected_freq = np.bincount(selected[selected >= 0], minlength=dose_count).astype(float)
    selected_freq /= trial_count
    no_select = float(np.count_nonzero(selected < 0) / trial_count)
    early = float(np.count_nonzero(subjects.sum(axis=1) < planned) / trial_count)
    return MTADFLogisticSimulation(
        _freeze_int(subjects),
        _freeze_int(toxicities),
        _freeze_int(responses),
        _freeze_int(selected),
        tuple(reasons),
        _freeze_uint(trial_seeds),
        _freeze_int(fit_counts),
        _freeze(acceptance),
        _freeze(max_rhat),
        _freeze(slope_mcse),
        _freeze(slope_rhat),
        _freeze(selected_freq),
        _freeze(np.sqrt(selected_freq * (1 - selected_freq) / trial_count)),
        no_select,
        sqrt(no_select * (1 - no_select) / trial_count),
        early,
        sqrt(early * (1 - early) / trial_count),
        _freeze(np.mean(subjects, axis=0)),
        _freeze(np.mean(toxicities, axis=0)),
        _freeze(np.mean(responses, axis=0)),
    )
