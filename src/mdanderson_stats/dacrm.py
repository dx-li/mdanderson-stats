"""Data-augmented CRM posterior for piecewise exponential event times."""

from dataclasses import dataclass
from math import prod, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray, scalar
from .bmacrm import _logistic_terms
from .hierarchical_binomial import ChainSummary, summarize_chains

_MAX_INTERVALS = 20
_MAX_DOSES = 20
_MAX_PATIENTS = 200
_MAX_RETAINED_CELLS = 2_000_000
_MAX_TRANSITION_WORK = 20_000_000
_MAX_EVALUATIONS = 2_000_000
_MAX_ELLIPSE_PROPOSALS = 1000
_MAX_GAMMA_PARAMETER = 1e8


def _freeze_typed(value: ArrayLike, dtype: np.dtype) -> NDArray:
    raw = np.asarray(value, dtype=dtype)
    return np.frombuffer(raw.tobytes(), dtype=dtype).reshape(raw.shape)


def _real_vector(value: ArrayLike, name: str, limit: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size > limit or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric vector of at most {limit} values")
    result = np.asarray(raw, dtype=float)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def _integer_setting(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    integer = int(value)
    if not low <= integer <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return integer


@dataclass(frozen=True)
class DACRMPrior:
    """Explicit DA-CRM gamma hazard and normal power-model priors.

    ``breaks`` starts at zero and ends at the toxicity window; ``shape`` and
    ``rate`` give independent Gamma(shape, rate) priors by interval.
    """

    breaks: FloatArray
    shape: FloatArray
    rate: FloatArray
    alpha_sd: float = sqrt(2)

    def __post_init__(self) -> None:
        breaks = _real_vector(self.breaks, "breaks", _MAX_INTERVALS + 1)
        shape = _real_vector(self.shape, "shape", _MAX_INTERVALS)
        rate = _real_vector(self.rate, "rate", _MAX_INTERVALS)
        if breaks.size < 2 or breaks[0] != 0 or np.any(np.diff(breaks) <= 0):
            raise ValueError("breaks must strictly increase from zero to a positive window")
        intervals = breaks.size - 1
        if (
            not 1 <= intervals <= _MAX_INTERVALS
            or shape.shape != (intervals,)
            or rate.shape != (intervals,)
        ):
            raise ValueError("shape and rate must have one value per 1..20 intervals")
        if (
            np.any(shape <= 0)
            or np.any(rate <= 0)
            or np.any(shape > _MAX_GAMMA_PARAMETER)
            or np.any(rate > _MAX_GAMMA_PARAMETER)
        ):
            raise ValueError("gamma shape/rate values must lie in (0,1e8]")
        if isinstance(self.alpha_sd, (bool, np.bool_)):
            raise ValueError("alpha_sd must be numeric, not boolean")
        alpha_sd = scalar(self.alpha_sd, "alpha_sd")
        if not 1e-3 <= alpha_sd <= 10:
            raise ValueError("alpha_sd must lie in [1e-3,10]")
        object.__setattr__(self, "breaks", _freeze(breaks))
        object.__setattr__(self, "shape", _freeze(shape))
        object.__setattr__(self, "rate", _freeze(rate))
        object.__setattr__(self, "alpha_sd", alpha_sd)


@dataclass(frozen=True)
class DACRMPosterior:
    """Retained DA-CRM draws and summaries, preserving chain/draw axes."""

    skeleton: FloatArray
    doses: FloatArray
    outcomes: FloatArray
    times: FloatArray
    prior: DACRMPrior
    target: float
    alpha_draws: FloatArray
    hazard_draws: FloatArray
    dose_probability: FloatArray
    pending_probability_draws: FloatArray
    pending_indices: NDArray[np.int64]
    dose_mean: FloatArray
    overdose_probability: FloatArray
    pending_probability: FloatArray
    parameter_summary: ChainSummary
    dose_summary: ChainSummary
    pending_summary: ChainSummary | None
    evaluations: int
    warmup: int


class _EvaluationBudget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.count = 0

    def use(self) -> None:
        if self.count >= self.limit:
            raise RuntimeError("DA-CRM sampler exceeded max_evaluations")
        self.count += 1


def _piecewise_exposure(times: FloatArray, breaks: FloatArray) -> FloatArray:
    start = breaks[:-1]
    end = breaks[1:]
    return np.maximum(0.0, np.minimum(times[:, None], end) - start)


def dacrm_pending_probability(probability: ArrayLike, cumulative_hazard: ArrayLike) -> FloatArray:
    """Stable conditional probability of eventual DLT given no DLT yet.

    Computes ``pi * exp(-H) / (1-pi + pi*exp(-H))`` in log space.
    """
    raw_p, raw_h = np.asarray(probability), np.asarray(cumulative_hazard)
    if raw_p.size > _MAX_PATIENTS or raw_h.size > _MAX_PATIENTS:
        raise ValueError("pending-probability inputs exceed 200 values")
    if raw_p.dtype.kind not in "iuf" or raw_h.dtype.kind not in "iuf":
        raise ValueError("probability and cumulative_hazard must be real numeric")
    try:
        output_shape = np.broadcast_shapes(raw_p.shape, raw_h.shape)
    except ValueError as exc:
        raise ValueError("probability and cumulative_hazard cannot be broadcast") from exc
    if prod(output_shape) > _MAX_PATIENTS:
        raise ValueError("pending-probability broadcast exceeds 200 values")
    p, hazard = np.broadcast_arrays(np.asarray(raw_p, dtype=float), np.asarray(raw_h, dtype=float))
    if np.any(~np.isfinite(p) | ~np.isfinite(hazard) | (p <= 0) | (p >= 1) | (hazard < 0)):
        raise ValueError("probability must lie in (0,1) and cumulative_hazard be nonnegative")
    log_pi = np.log(p)
    log_one_minus_pi = np.log1p(-p)
    log_numerator = log_pi - hazard
    log_denominator = np.logaddexp(log_one_minus_pi, log_numerator)
    return _freeze(np.exp(log_numerator - log_denominator))


def _alpha_log_likelihood(
    alpha: float,
    skeleton: FloatArray,
    doses: NDArray[np.int64],
    augmented_outcomes: NDArray[np.int64],
    budget: _EvaluationBudget,
) -> float:
    budget.use()
    if doses.size == 0:
        return 0.0
    log_neglog = np.log(-np.log(skeleton[doses]))
    t, log_failure, _, _ = _logistic_terms(log_neglog + alpha)
    toxic = augmented_outcomes == 1
    nontoxic = ~toxic
    with np.errstate(over="ignore", invalid="ignore"):
        toxic_part = float(np.sum(-t[toxic])) if np.any(toxic) else 0.0
        nontoxic_part = float(np.sum(log_failure[nontoxic])) if np.any(nontoxic) else 0.0
    return toxic_part + nontoxic_part


def _elliptical_alpha_step(
    current: float,
    prior_sd: float,
    skeleton: FloatArray,
    doses: NDArray[np.int64],
    augmented_outcomes: NDArray[np.int64],
    rng: np.random.Generator,
    budget: _EvaluationBudget,
) -> tuple[float, float]:
    current_log_likelihood = _alpha_log_likelihood(
        current, skeleton, doses, augmented_outcomes, budget
    )
    height = current_log_likelihood + np.log(max(rng.random(), np.finfo(float).tiny))
    direction = prior_sd * rng.normal()
    angle = rng.uniform(0, 2 * np.pi)
    lower, upper = angle - 2 * np.pi, angle
    for _ in range(_MAX_ELLIPSE_PROPOSALS):
        proposal = current * np.cos(angle) + direction * np.sin(angle)
        proposal_ll = _alpha_log_likelihood(proposal, skeleton, doses, augmented_outcomes, budget)
        if proposal_ll >= height:
            return float(proposal), proposal_ll
        if angle < 0:
            lower = angle
        else:
            upper = angle
        angle = rng.uniform(lower, upper)
    raise ArithmeticError("DA-CRM elliptical-slice bracket failed to accept a draw")


def _pending_probability_from_state(
    skeleton: FloatArray,
    doses: NDArray[np.int64],
    pending_indices: NDArray[np.int64],
    exposures: FloatArray,
    alpha: float,
    hazards: FloatArray,
) -> FloatArray:
    if pending_indices.size == 0:
        return np.empty(0)
    cumulative_hazard = exposures[pending_indices] @ hazards
    log_neglog = np.log(-np.log(skeleton[doses[pending_indices]]))
    t, log_failure, _, _ = _logistic_terms(log_neglog + alpha)
    log_numerator = -t - cumulative_hazard
    log_denominator = np.logaddexp(log_failure, log_numerator)
    return np.exp(log_numerator - log_denominator)


def _freeze_summary(summary: ChainSummary) -> ChainSummary:
    return ChainSummary(
        *[
            _freeze(value)
            for value in (
                summary.mean,
                summary.median,
                summary.standard_deviation,
                summary.interval,
                summary.split_rhat,
                summary.batch_mean_mcse,
            )
        ]
    )


def _gamma_draw(
    rng: np.random.Generator, shape: FloatArray, rate: FloatArray, *, name: str
) -> FloatArray:
    if np.any(~np.isfinite(shape) | (shape <= 0)) or np.any(~np.isfinite(rate) | (rate <= 0)):
        raise ArithmeticError(f"{name} posterior shape/rate is not representable")
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        scale = 1 / rate
    if np.any(~np.isfinite(scale) | (scale <= 0)):
        raise ArithmeticError(f"{name} gamma scale is not representable")
    sample = rng.gamma(shape, scale=scale)
    if np.any(~np.isfinite(sample) | (sample < 0)):
        raise ArithmeticError(f"{name} gamma draw is not representable")
    return np.asarray(sample, dtype=float)


def fit_dacrm(
    skeleton: ArrayLike,
    doses: ArrayLike,
    outcomes: ArrayLike,
    times: ArrayLike,
    *,
    prior: DACRMPrior,
    target: float,
    rng: np.random.Generator,
    draws: int = 2000,
    warmup: int = 1000,
    chains: int = 2,
    max_evaluations: int = _MAX_EVALUATIONS,
) -> DACRMPosterior:
    """Fit DA-CRM with imputed pending outcomes and piecewise gamma hazards.

    Completed outcomes use ``0`` for no DLT and ``1`` for observed DLT; pending
    outcomes use ``-1`` and ``times`` records their current follow-up. Latent
    pending DLTs update the CRM and contribute censoring exposure to hazard
    rates, but only observed DLT events increment gamma shapes.
    """
    if not isinstance(prior, DACRMPrior):
        raise TypeError("prior must be a DACRMPrior")
    skeleton_values = _real_vector(skeleton, "skeleton", _MAX_DOSES)
    dose_values = _real_vector(doses, "doses", _MAX_PATIENTS)
    outcome_values = _real_vector(outcomes, "outcomes", _MAX_PATIENTS)
    time_values = _real_vector(times, "times", _MAX_PATIENTS)
    dose_count, interval_count = skeleton_values.size, prior.shape.size
    if not 1 <= dose_count <= _MAX_DOSES:
        raise ValueError("skeleton must contain 1..20 doses")
    if np.any((skeleton_values <= 0) | (skeleton_values >= 1)) or np.any(
        np.diff(skeleton_values) < 0
    ):
        raise ValueError("skeleton must be nondecreasing in (0,1)")
    if dose_values.shape != outcome_values.shape or dose_values.shape != time_values.shape:
        raise ValueError("doses, outcomes, and times must have equal lengths")
    if np.any(dose_values != np.floor(dose_values)) or np.any(
        (dose_values < 0) | (dose_values >= dose_count)
    ):
        raise ValueError("doses must be zero-based integer indices in the skeleton")
    if np.any((outcome_values != -1) & (outcome_values != 0) & (outcome_values != 1)):
        raise ValueError("outcomes must be -1 (pending), 0 (no DLT), or 1 (DLT)")
    window = float(prior.breaks[-1])
    pending = outcome_values == -1
    no_dlt = outcome_values == 0
    dlt = outcome_values == 1
    if np.any((time_values[pending] < 0) | (time_values[pending] >= window)):
        raise ValueError("pending follow-up times must lie in [0, window)")
    if np.any(time_values[no_dlt] != window):
        raise ValueError("completed no-DLT times must equal the full window")
    if np.any((time_values[dlt] < 0) | (time_values[dlt] > window)):
        raise ValueError("observed DLT times must lie in [0, window]")

    raw_target = np.asarray(target)
    if raw_target.ndim != 0 or raw_target.dtype.kind not in "iuf":
        raise ValueError("target must be a finite real scalar")
    target_value = scalar(float(raw_target), "target")
    if not 0 < target_value < 1:
        raise ValueError("target must lie strictly between 0 and 1")
    draws = _integer_setting(draws, "draws", 8, 10_000)
    warmup = _integer_setting(warmup, "warmup", 0, 10_000)
    chains = _integer_setting(chains, "chains", 2, 4)
    max_evaluations = _integer_setting(max_evaluations, "max_evaluations", 1, _MAX_EVALUATIONS)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    transition_count = chains * (draws + warmup)
    if transition_count * (dose_values.size + dose_count + interval_count) > _MAX_TRANSITION_WORK:
        raise ValueError("DA-CRM transition work exceeds 20000000 units")
    pending_indices = np.flatnonzero(pending).astype(np.int64)
    retained_cells = chains * draws * (1 + interval_count + dose_count + pending_indices.size)
    if retained_cells > _MAX_RETAINED_CELLS:
        raise ValueError("retained DA-CRM draws exceed 2000000 cells")

    integer_doses = dose_values.astype(np.int64)
    integer_outcomes = outcome_values.astype(np.int64)
    exposures = _piecewise_exposure(time_values, prior.breaks)
    event_counts = np.zeros(interval_count, dtype=float)
    if np.any(dlt):
        event_interval = np.searchsorted(prior.breaks[1:], time_values[dlt], side="right")
        event_interval = np.minimum(event_interval, interval_count - 1)
        event_counts = np.bincount(event_interval, minlength=interval_count).astype(float)

    alpha_draws = np.empty((chains, draws))
    hazard_draws = np.empty((chains, draws, interval_count))
    pending_draws = np.empty((chains, draws, pending_indices.size))
    budget = _EvaluationBudget(max_evaluations)
    prior_only = not np.any(~pending) and not np.any(time_values[pending] > 0)
    if prior_only:
        alpha_draws[:] = rng.normal(0, prior.alpha_sd, size=(chains, draws))
        if np.any(~np.isfinite(alpha_draws)):
            raise ArithmeticError("normal alpha draws are not representable")
        for chain in range(chains):
            for draw in range(draws):
                hazard_draws[chain, draw] = _gamma_draw(rng, prior.shape, prior.rate, name="prior")
        for chain in range(chains):
            for draw in range(draws):
                pending_draws[chain, draw] = _pending_probability_from_state(
                    skeleton_values,
                    integer_doses,
                    pending_indices,
                    exposures,
                    float(alpha_draws[chain, draw]),
                    hazard_draws[chain, draw],
                )
    else:
        total_iterations = warmup + draws
        for chain in range(chains):
            alpha = float(rng.normal(0, prior.alpha_sd))
            hazards = _gamma_draw(rng, prior.shape, prior.rate, name="prior")
            augmented = integer_outcomes.copy()
            for iteration in range(total_iterations):
                pending_probability = _pending_probability_from_state(
                    skeleton_values,
                    integer_doses,
                    pending_indices,
                    exposures,
                    alpha,
                    hazards,
                )
                augmented[pending_indices] = rng.binomial(1, pending_probability)
                alpha, _ = _elliptical_alpha_step(
                    alpha,
                    prior.alpha_sd,
                    skeleton_values,
                    integer_doses,
                    augmented,
                    rng,
                    budget,
                )
                positive_exposure = (augmented == 1) & (integer_outcomes != 0)
                with np.errstate(over="ignore", invalid="ignore"):
                    rate_increment = (
                        np.sum(exposures[positive_exposure], axis=0)
                        if np.any(positive_exposure)
                        else np.zeros(interval_count)
                    )
                    posterior_rate = prior.rate + rate_increment
                hazards = _gamma_draw(
                    rng, prior.shape + event_counts, posterior_rate, name="posterior"
                )
                if not np.isfinite(alpha) or np.any(~np.isfinite(hazards)) or np.any(hazards < 0):
                    raise ArithmeticError("DA-CRM sampler reached an unrepresentable state")
                if iteration >= warmup:
                    draw = iteration - warmup
                    alpha_draws[chain, draw] = alpha
                    hazard_draws[chain, draw] = hazards
                    pending_draws[chain, draw] = _pending_probability_from_state(
                        skeleton_values,
                        integer_doses,
                        pending_indices,
                        exposures,
                        alpha,
                        hazards,
                    )

    log_neglog = np.log(-np.log(skeleton_values))
    log_t = log_neglog[None, None, :] + alpha_draws[:, :, None]
    with np.errstate(over="ignore", under="ignore"):
        t = np.exp(np.minimum(log_t, 700.0))
    t = np.where(log_t > 700, np.inf, t)
    dose_probability = _freeze(np.exp(-t))
    threshold = np.log(-np.log(target_value)) - log_neglog
    overdose_probability = np.mean(alpha_draws[:, :, None] < threshold[None, None, :], axis=(0, 1))
    parameter_draws = np.concatenate((alpha_draws[:, :, None], hazard_draws), axis=-1)
    parameter_summary = _freeze_summary(summarize_chains(parameter_draws))
    dose_summary = _freeze_summary(summarize_chains(dose_probability))
    pending_summary = (
        _freeze_summary(summarize_chains(pending_draws)) if pending_indices.size else None
    )
    return DACRMPosterior(
        _freeze(skeleton_values),
        _freeze(dose_values),
        _freeze(outcome_values),
        _freeze(time_values),
        prior,
        target_value,
        _freeze(alpha_draws),
        _freeze(hazard_draws),
        dose_probability,
        _freeze(pending_draws),
        _freeze_typed(pending_indices, np.dtype(np.int64)),
        _freeze(np.mean(dose_probability, axis=(0, 1))),
        _freeze(overdose_probability),
        _freeze(np.mean(pending_draws, axis=(0, 1))),
        parameter_summary,
        dose_summary,
        pending_summary,
        budget.count,
        warmup,
    )
