"""Explicit-prior posterior fit for variable-dose triangular-hazard schedules."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .dose_schedule import (
    _MAX_ADMINISTRATIONS,
    DoseSchedulePatient,
    dose_schedule_cumulative_hazard,
    dose_schedule_parameter_names,
    dose_schedule_patient_loglikelihood,
    validate_schedules,
)
from .dose_schedule_prior import DoseSchedulePrior
from .hierarchical_binomial import ChainSummary, summarize_chains
from .uaroet import _integer

_MAX_PATIENTS = 200
_MAX_CHAINS = 8
_MAX_DRAWS = 100_000
_MAX_WARMUP = 100_000
_MAX_RETAINED = 2_000_000
_MAX_EVALUATIONS = 2_000_000
_MAX_WORK = 50_000_000


def _scalar(value: object, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _physical(
    parameters: FloatArray, dose_count: int, ordered_areas: bool
) -> tuple[FloatArray, ...]:
    if parameters.shape != (3 * dose_count,) or not np.all(np.isfinite(parameters)):
        raise ValueError("parameter row must match the dose grid and be finite")
    with np.errstate(over="ignore", under="ignore"):
        components = np.exp(parameters.reshape(dose_count, 3))
    if not np.all(np.isfinite(components)) or np.any(components == 0):
        raise ArithmeticError("lognormal parameters exceed representable physical values")
    area = components[:, 0]
    if ordered_areas:
        area = np.cumsum(area, dtype=float)
        if not np.all(np.isfinite(area)):
            raise ArithmeticError("cumulative ordered areas exceed floating-point range")
    peak, tail = components[:, 1], components[:, 2]
    with np.errstate(over="ignore"):
        total = peak + tail
    if not np.all(np.isfinite(total)):
        raise ArithmeticError("peak plus tail time is not representable")
    return area, peak, tail


def _regimen_risk(
    areas: FloatArray,
    peaks: FloatArray,
    tails: FloatArray,
    schedules: tuple[FloatArray, ...],
    horizon: float,
) -> FloatArray:
    risks = np.empty((areas.size, len(schedules)), dtype=float)
    for schedule_index, times in enumerate(schedules):
        elapsed = horizon - times
        cumulative = dose_schedule_cumulative_hazard(
            elapsed[None, :], areas[:, None], peaks[:, None], tails[:, None]
        )
        cumulative_total = np.sum(cumulative, axis=1, dtype=float)
        if np.any(~np.isfinite(cumulative_total)):
            raise ArithmeticError("candidate regimen cumulative hazard is not representable")
        risks[:, schedule_index] = -np.expm1(-cumulative_total)
    if np.any(~np.isfinite(risks)) or np.any((risks < 0) | (risks > 1)):
        raise ArithmeticError("candidate regimen risk is not a probability")
    return risks


@dataclass(frozen=True)
class DoseScheduleFit:
    """Posterior draws and risk summaries with chain/draw/dose/schedule axes."""

    parameter_names: tuple[str, ...]
    prior: DoseSchedulePrior
    schedules: tuple[FloatArray, ...]
    horizon: float
    log_parameters: FloatArray
    areas: FloatArray
    peak_times: FloatArray
    tail_times: FloatArray
    regimen_risk: FloatArray
    log_likelihood: FloatArray
    parameter_summary: ChainSummary
    risk_summary: ChainSummary
    likelihood_evaluations: int
    work_units: int
    warmup: int
    direct_prior: bool


def fit_dose_schedule(
    patients: tuple[DoseSchedulePatient, ...] | list[DoseSchedulePatient],
    prior: DoseSchedulePrior,
    schedules: tuple[ArrayLike, ...] | list[ArrayLike],
    horizon: float,
    *,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 4,
    rng: np.random.Generator,
    max_evaluations: int = 200_000,
    max_work: int = 20_000_000,
) -> DoseScheduleFit:
    """Sample posterior dose-hazard parameters and evaluate regimen risks.

    Histories remain individual because event-time likelihoods depend on the
    actual administration times and dose indices. Elliptical slice sampling
    targets explicit independent normal priors on log parameters. Event times
    outside finite triangular-hazard support have genuine zero likelihood and
    are ordinary posterior rejections; numerical overflow or invalid arithmetic
    raises instead of being converted to a rejection.
    """
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit NumPy Generator")
    if not isinstance(prior, DoseSchedulePrior):
        raise ValueError("prior must be a DoseSchedulePrior")
    if not isinstance(patients, (tuple, list)) or len(patients) > _MAX_PATIENTS:
        raise ValueError(f"patients must be a sequence of at most {_MAX_PATIENTS} histories")
    if any(not isinstance(patient, DoseSchedulePatient) for patient in patients):
        raise ValueError("every patient must be a DoseSchedulePatient")
    if any(np.any(patient.dose_indices >= prior.dose_count) for patient in patients):
        raise ValueError("patient history refers to a dose outside the prior grid")
    administration_count = sum(patient.dose_indices.size for patient in patients)
    if administration_count > _MAX_ADMINISTRATIONS:
        raise ValueError("actual patient administrations exceed 10,000")
    horizon_value = _scalar(horizon, "horizon")
    if horizon_value <= 0:
        raise ValueError("horizon must be positive")
    candidate_schedules = validate_schedules(schedules, horizon_value)
    draw_count = _integer(draws, "draws", 8, _MAX_DRAWS)
    warmup_count = _integer(warmup, "warmup", 0, _MAX_WARMUP)
    chain_count = _integer(chains, "chains", 2, _MAX_CHAINS)
    evaluation_limit = _integer(max_evaluations, "max_evaluations", 1, _MAX_EVALUATIONS)
    work_limit = _integer(max_work, "max_work", 1, _MAX_WORK)
    dose_count = prior.dose_count
    parameter_count = 3 * dose_count
    output_shape = (chain_count, draw_count, dose_count, len(candidate_schedules))
    retained = (
        chain_count
        * draw_count
        * (parameter_count + 3 * dose_count + dose_count * len(candidate_schedules) + 1)
    )
    if retained > _MAX_RETAINED:
        raise ValueError("retained parameter and regimen-risk arrays exceed two million cells")
    direct = len(patients) == 0
    eval_min = 0 if direct else chain_count * (1 + warmup_count + draw_count)
    risk_work = chain_count * draw_count * dose_count * sum(row.size for row in candidate_schedules)
    min_work = eval_min * administration_count + risk_work
    if eval_min > evaluation_limit:
        raise ValueError("max_evaluations is below the minimum likelihood-call count")
    if min_work > work_limit:
        raise ValueError("max_work is below the minimum administration and prediction workload")

    parameter_draws = np.empty((chain_count, draw_count, parameter_count), dtype=float)
    area_draws = np.empty((chain_count, draw_count, dose_count), dtype=float)
    peak_draws = np.empty_like(area_draws)
    tail_draws = np.empty_like(area_draws)
    risk_draws = np.empty(output_shape, dtype=float)
    likelihood_draws = np.empty((chain_count, draw_count), dtype=float)
    evaluations = 0
    work = 0

    def evaluate(theta: FloatArray) -> tuple[float, FloatArray, tuple[FloatArray, ...]]:
        nonlocal evaluations, work
        if evaluations >= evaluation_limit:
            raise RuntimeError("dose-schedule likelihood calls exceeded max_evaluations")
        if work + administration_count > work_limit:
            raise RuntimeError("dose-schedule administration workload exceeded max_work")
        work += administration_count
        evaluations += 1
        areas, peaks, tails = _physical(theta, dose_count, prior.ordered_areas)
        log_likelihood = 0.0
        for patient in patients:
            value = dose_schedule_patient_loglikelihood(patient, areas, peaks, tails)
            if value == -np.inf:
                log_likelihood = -np.inf
                break
            log_likelihood += value
            if not np.isfinite(log_likelihood):
                raise ArithmeticError("summed dose-schedule log likelihood is not representable")
        if not np.isfinite(log_likelihood) and log_likelihood != -np.inf:
            raise ArithmeticError("dose-schedule log likelihood is not representable")
        return float(log_likelihood), areas, (peaks, tails)

    def save_draw(
        chain: int,
        draw: int,
        theta: FloatArray,
        log_like: float,
        area: FloatArray,
        peak: FloatArray,
        tail: FloatArray,
    ) -> None:
        nonlocal work
        risk_units = dose_count * sum(row.size for row in candidate_schedules)
        if work + risk_units > work_limit:
            raise RuntimeError("dose-schedule regimen work exceeded max_work")
        work += risk_units
        risk = _regimen_risk(area, peak, tail, candidate_schedules, horizon_value)
        parameter_draws[chain, draw] = theta
        area_draws[chain, draw] = area
        peak_draws[chain, draw] = peak
        tail_draws[chain, draw] = tail
        risk_draws[chain, draw] = risk
        likelihood_draws[chain, draw] = log_like

    for chain in range(chain_count):
        if direct:
            theta = rng.normal(prior.mean, prior.sd, size=(draw_count, parameter_count))
            if not np.all(np.isfinite(theta)):
                raise ArithmeticError("normal log-parameter prior draw is not representable")
            for draw in range(draw_count):
                area, peak, tail = _physical(theta[draw], dose_count, prior.ordered_areas)
                log_like = 0.0
                save_draw(chain, draw, theta[draw], log_like, area, peak, tail)
            continue

        for attempt in range(1000):
            state = rng.normal(prior.mean, prior.sd)
            log_like, areas, (peaks, tails) = evaluate(state)
            if np.isfinite(log_like):
                break
        else:
            raise ArithmeticError("could not find a finite-likelihood prior initialization")
        for iteration in range(warmup_count + draw_count):
            centered = state - prior.mean
            direction = rng.normal(size=parameter_count) * prior.sd
            height = log_like + np.log1p(-rng.random())
            angle = float(rng.uniform(0, 2 * np.pi))
            lower, upper = angle - 2 * np.pi, angle
            for _ in range(1000):
                proposal = prior.mean + centered * np.cos(angle) + direction * np.sin(angle)
                candidate_ll, candidate_area, (candidate_peak, candidate_tail) = evaluate(proposal)
                if candidate_ll >= height:
                    state, log_like = proposal, candidate_ll
                    areas, peaks, tails = candidate_area, candidate_peak, candidate_tail
                    break
                if angle < 0:
                    lower = angle
                else:
                    upper = angle
                angle = float(rng.uniform(lower, upper))
            else:
                raise ArithmeticError(f"dose-schedule elliptical slice failed in chain {chain}")
            if iteration >= warmup_count:
                save_draw(
                    chain,
                    iteration - warmup_count,
                    state,
                    log_like,
                    areas,
                    peaks,
                    tails,
                )

    return DoseScheduleFit(
        dose_schedule_parameter_names(dose_count, ordered_areas=prior.ordered_areas),
        prior,
        candidate_schedules,
        horizon_value,
        _freeze(parameter_draws),
        _freeze(area_draws),
        _freeze(peak_draws),
        _freeze(tail_draws),
        _freeze(risk_draws),
        _freeze(likelihood_draws),
        summarize_chains(parameter_draws),
        summarize_chains(risk_draws),
        evaluations,
        work,
        0 if direct else warmup_count,
        direct,
    )
