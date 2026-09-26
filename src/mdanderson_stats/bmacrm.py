"""Posterior calculations for the Bayesian model-averaged CRM."""

from dataclasses import dataclass
from math import log, pi, sqrt

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad_vec
from scipy.special import gammaln, logsumexp
from scipy.stats import norm

from ._cdflib import _freeze
from ._validation import FloatArray

_MAX_MODELS = 5
_MAX_DOSES = 20
_MAX_SUBJECTS = 10_000
_MAX_EVALUATIONS = 200_000
_LOG_2PI = log(2 * pi)
_SQRT2PI = sqrt(2 * pi)
_EPS = np.finfo(float).eps


@dataclass(frozen=True)
class BMACRMPosterior:
    """Immutable model-specific and model-averaged BMA-CRM posterior summaries."""

    skeletons: FloatArray
    events: FloatArray
    subjects: FloatArray
    target: float
    prior_sd: float
    prior_model_weights: FloatArray
    posterior_model_weights: FloatArray
    model_log_evidence: FloatArray
    model_dose_mean: FloatArray
    model_overdose_probability: FloatArray
    dose_mean: FloatArray
    overdose_probability: FloatArray
    alpha_mean: FloatArray
    alpha_sd: FloatArray
    integration_error: FloatArray
    evaluations: int


class _Budget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.count = 0

    def use(self) -> None:
        if self.count >= self.limit:
            raise RuntimeError("BMA-CRM integration exceeded max_evaluations")
        self.count += 1


def _raw_numeric(value: ArrayLike, name: str, max_size: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.size > max_size:
        raise ValueError(f"{name} exceeds its size limit")
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=float)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _scalar_value(value: ArrayLike, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _logistic_terms(log_t: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return t, log(1-exp(-t)), t/(exp(t)-1), and its curvature term."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        t = np.exp(np.minimum(log_t, 700.0))
    t = np.where(log_t > 700, np.inf, t)
    small = t < 1e-5
    mid = (t >= 1e-5) & (t < 50)
    r = np.zeros_like(t)
    r[small] = 1 - t[small] / 2 + t[small] ** 2 / 12
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        r[mid] = t[mid] / np.expm1(t[mid])
    curvature_term = np.zeros_like(t)
    curvature_term[small] = -t[small] / 2 + t[small] ** 2 / 6
    curvature_term[mid] = r[mid] * (1 - t[mid] - r[mid])

    log_failure = np.empty_like(t)
    log_failure[small] = log_t[small] + np.log1p(-t[small] / 2 + t[small] ** 2 / 6)
    other = ~small
    log_failure[other] = np.log(-np.expm1(-t[other]))
    log_failure[np.isinf(t)] = 0.0
    return t, log_failure, r, curvature_term


def _log_likelihood(
    alpha: float,
    log_neglog_skeleton: FloatArray,
    events: FloatArray,
    subjects: FloatArray,
    log_binomial: float,
) -> float:
    log_t = log_neglog_skeleton + alpha
    t, log_failure, _, _ = _logistic_terms(log_t)
    toxic = events > 0
    nontoxic = subjects > events
    value = log_binomial
    if np.any(toxic):
        with np.errstate(over="ignore", invalid="ignore"):
            toxic_term = np.sum(events[toxic] * -t[toxic])
        value += float(toxic_term)
    if np.any(nontoxic):
        value += float(np.sum((subjects[nontoxic] - events[nontoxic]) * log_failure[nontoxic]))
    return value


def _posterior_derivatives(
    alpha: float,
    log_neglog_skeleton: FloatArray,
    events: FloatArray,
    subjects: FloatArray,
    prior_sd: float,
    budget: _Budget,
) -> tuple[float, float]:
    budget.use()
    t, _, r, curvature_term = _logistic_terms(log_neglog_skeleton + alpha)
    toxic = events > 0
    nontoxic = subjects > events
    with np.errstate(over="ignore", invalid="ignore"):
        toxic_slope = float(np.sum(events[toxic] * t[toxic])) if np.any(toxic) else 0.0
        nontoxic_slope = (
            float(np.sum((subjects[nontoxic] - events[nontoxic]) * r[nontoxic]))
            if np.any(nontoxic)
            else 0.0
        )
        toxic_curvature = float(np.sum(events[toxic] * t[toxic])) if np.any(toxic) else 0.0
        nontoxic_curvature = (
            float(np.sum((subjects[nontoxic] - events[nontoxic]) * curvature_term[nontoxic]))
            if np.any(nontoxic)
            else 0.0
        )
    slope = -toxic_slope + nontoxic_slope - alpha / (prior_sd * prior_sd)
    curvature = -toxic_curvature + nontoxic_curvature - 1 / (prior_sd * prior_sd)
    return slope, curvature


def _mode_and_scale(
    log_neglog_skeleton: FloatArray,
    events: FloatArray,
    subjects: FloatArray,
    prior_sd: float,
    budget: _Budget,
) -> tuple[float, float]:
    def slope(alpha: float) -> float:
        return _posterior_derivatives(
            alpha, log_neglog_skeleton, events, subjects, prior_sd, budget
        )[0]

    step = prior_sd
    lower, upper = -step, step
    lower_value, upper_value = slope(lower), slope(upper)
    for _ in range(80):
        if lower_value > 0 and upper_value < 0:
            break
        step *= 2
        if lower_value <= 0:
            lower = -step
            lower_value = slope(lower)
        if upper_value >= 0:
            upper = step
            upper_value = slope(upper)
    else:
        raise ArithmeticError("could not bracket the unique BMA-CRM posterior mode")
    from scipy.optimize import brentq

    mode = float(brentq(slope, lower, upper, xtol=1e-12, rtol=1e-12))
    _, curvature = _posterior_derivatives(
        mode, log_neglog_skeleton, events, subjects, prior_sd, budget
    )
    if not np.isfinite(curvature) or curvature >= 0:
        raise ArithmeticError("posterior curvature is not strictly negative at its mode")
    scale = 1 / sqrt(-curvature)
    if not np.isfinite(scale) or scale <= 0:
        raise ArithmeticError("posterior integration scale is not representable")
    return mode, scale


def _fit_model(
    skeleton: FloatArray,
    events: FloatArray,
    subjects: FloatArray,
    target: float,
    prior_sd: float,
    budget: _Budget,
) -> tuple[float, FloatArray, FloatArray, float, float, float]:
    log_neglog = np.log(-np.log(skeleton))
    log_binomial = float(
        np.sum(gammaln(subjects + 1) - gammaln(events + 1) - gammaln(subjects - events + 1))
    )
    mode, scale = _mode_and_scale(log_neglog, events, subjects, prior_sd, budget)
    mode_logpost = (
        _log_likelihood(mode, log_neglog, events, subjects, log_binomial)
        - 0.5 * (mode / prior_sd) ** 2
        - np.log(prior_sd)
        - 0.5 * _LOG_2PI
    )
    thresholds = np.log(-np.log(target)) - log_neglog
    scaled_thresholds = (thresholds - mode) / scale
    max_cut = float(np.max(np.abs(scaled_thresholds)))
    if not np.isfinite(max_cut):
        raise ArithmeticError("overdose threshold is not representable in posterior coordinates")
    scaffold_power = int(np.ceil(np.log2(max(1.0, max_cut))))
    scaffold = np.ldexp(1.0, np.arange(scaffold_power + 1, dtype=int))
    cuts = np.unique(
        np.concatenate(([-4.0, -1.0, 0.0, 1.0, 4.0], scaffold, -scaffold, scaled_thresholds))
    )
    dose_count = skeleton.size
    output_size = dose_count + 3

    def integrand(x: float) -> FloatArray:
        budget.use()
        alpha = mode + scale * x
        logpost = (
            _log_likelihood(alpha, log_neglog, events, subjects, log_binomial)
            - 0.5 * (alpha / prior_sd) ** 2
            - np.log(prior_sd)
            - 0.5 * _LOG_2PI
        )
        log_weight = logpost - mode_logpost
        if not np.isfinite(log_weight) or log_weight < -745:
            return np.zeros(output_size)
        weight = float(np.exp(log_weight))
        t, _, _, _ = _logistic_terms(log_neglog + alpha)
        pi_value = np.exp(-t)
        return np.concatenate(([weight, x * weight, x * x * weight], pi_value * weight))

    endpoints = np.concatenate(([-np.inf], cuts, [np.inf]))
    parts: list[FloatArray] = []
    total_error = 0.0
    for left, right in zip(endpoints[:-1], endpoints[1:], strict=True):
        integral, error, info = quad_vec(
            integrand,
            float(left),
            float(right),
            epsabs=1e-11,
            epsrel=2e-9,
            norm="max",
            cache_size=1_048_576,
            limit=300,
            workers=1,
            full_output=True,
        )
        if not info.success:
            raise ArithmeticError(f"BMA-CRM posterior quadrature failed: {info.message}")
        parts.append(np.asarray(integral, dtype=float))
        total_error += float(error)
    segment = np.stack(parts)
    total = np.sum(segment, axis=0)
    mass = float(total[0])
    if not np.isfinite(mass) or mass <= 0:
        raise ArithmeticError("posterior normalizing integral is not representable")
    relative_error = total_error / mass
    if not np.isfinite(relative_error) or relative_error > 2e-7:
        raise ArithmeticError("BMA-CRM posterior quadrature did not meet its error tolerance")

    model_mean = total[3:] / mass
    mean_x, second_x = float(total[1] / mass), float(total[2] / mass)
    alpha_mean = mode + scale * mean_x
    alpha_variance = scale * scale * (second_x - mean_x * mean_x)
    if alpha_variance < -128 * _EPS * max(prior_sd * prior_sd, 1.0):
        raise ArithmeticError("posterior alpha variance is negative beyond rounding error")
    alpha_sd = sqrt(max(0.0, alpha_variance))
    cumulative_mass = np.cumsum(segment[:, 0])
    overdose = np.empty(dose_count)
    for j, threshold in enumerate(thresholds):
        cut_index = int(np.searchsorted(cuts, (threshold - mode) / scale))
        overdose[j] = cumulative_mass[cut_index] / mass
    if np.all(subjects == 0):
        overdose = norm.cdf(thresholds / prior_sd)
        model_log_evidence = 0.0
    else:
        model_log_evidence = mode_logpost + np.log(scale) + np.log(mass)
    if np.any(~np.isfinite(model_mean) | (model_mean < 0) | (model_mean > 1)):
        raise ArithmeticError("posterior dose mean is not representable")
    if np.any(~np.isfinite(overdose) | (overdose < -1e-12) | (overdose > 1 + 1e-12)):
        raise ArithmeticError("posterior overdose probability is not representable")
    return (
        float(model_log_evidence),
        model_mean,
        np.clip(overdose, 0, 1),
        float(alpha_mean),
        alpha_sd,
        relative_error,
    )


def fit_bmacrm(
    skeletons: ArrayLike,
    events: ArrayLike,
    subjects: ArrayLike,
    *,
    target: float,
    model_prior: ArrayLike | None = None,
    prior_sd: float = sqrt(2),
    max_evaluations: int = _MAX_EVALUATIONS,
) -> BMACRMPosterior:
    """Fit model-averaged CRM posterior summaries for binomial dose outcomes.

    One-dimensional ``skeletons`` denotes one model; a two-dimensional array
    denotes rows of candidate models. Current BMA-CRM skeletons are prior
    medians and enter the power model directly. ``prior_sd`` may vary from
    the documented native value ``sqrt(2)`` as an explicit Python extension.
    The adaptive integrations cover the full real prior domain and share the
    ``max_evaluations`` budget across all candidate models. The Python
    extension allows ``prior_sd`` in ``[1e-3, 10]``; the native default is
    ``sqrt(2)``.
    """
    raw_skeletons = _raw_numeric(skeletons, "skeletons", _MAX_MODELS * _MAX_DOSES)
    raw_events = _raw_numeric(events, "events", _MAX_DOSES)
    raw_subjects = _raw_numeric(subjects, "subjects", _MAX_DOSES)
    if raw_skeletons.ndim == 1:
        raw_skeletons = raw_skeletons[None, :]
    if raw_skeletons.ndim != 2:
        raise ValueError("skeletons must be one- or two-dimensional")
    model_count, dose_count = raw_skeletons.shape
    if not 1 <= model_count <= _MAX_MODELS or not 1 <= dose_count <= _MAX_DOSES:
        raise ValueError("skeletons must contain 1..5 models and 1..20 doses")
    if (
        raw_events.ndim != 1
        or raw_subjects.ndim != 1
        or raw_events.shape != (dose_count,)
        or raw_subjects.shape != (dose_count,)
    ):
        raise ValueError("events and subjects must have one value per dose")
    if np.any((raw_skeletons <= 0) | (raw_skeletons >= 1)) or np.any(
        np.diff(raw_skeletons, axis=1) < 0
    ):
        raise ValueError("each skeleton must be nondecreasing in (0,1)")
    if (
        np.any(raw_events < 0)
        or np.any(raw_subjects < 0)
        or np.any(raw_events > raw_subjects)
        or np.any(raw_events != np.floor(raw_events))
        or np.any(raw_subjects != np.floor(raw_subjects))
        or np.any(raw_subjects >= 2**53)
        or np.sum(raw_subjects) > _MAX_SUBJECTS
    ):
        raise ValueError(
            "events/subjects must be integer counts with events<=subjects and total<=10000"
        )
    target_value = _scalar_value(target, "target")
    sd_value = _scalar_value(prior_sd, "prior_sd")
    if not 0 < target_value < 1:
        raise ValueError("target must lie strictly between 0 and 1")
    if not np.isfinite(sd_value) or not 1e-3 <= sd_value <= 10:
        raise ValueError("prior_sd must lie in [1e-3,10]")
    if (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, (int, np.integer))
        or not 1 <= max_evaluations <= _MAX_EVALUATIONS
    ):
        raise ValueError(f"max_evaluations must be an integer from 1 to {_MAX_EVALUATIONS}")
    if model_prior is None:
        log_prior_weights = np.full(model_count, -np.log(model_count))
    else:
        raw_prior = _raw_numeric(model_prior, "model_prior", _MAX_MODELS)
        if raw_prior.ndim != 1 or raw_prior.shape != (model_count,) or np.any(raw_prior < 0):
            raise ValueError("model_prior must be a nonnegative weight per model")
        if not np.any(raw_prior > 0):
            raise ValueError("model_prior weights must have positive sum")
        raw_log_prior = np.log(raw_prior, where=raw_prior > 0, out=np.full_like(raw_prior, -np.inf))
        log_prior_weights = raw_log_prior - logsumexp(raw_log_prior)
    prior_weights = np.exp(log_prior_weights)

    budget = _Budget(int(max_evaluations))
    log_evidence = np.empty(model_count)
    model_means = np.empty((model_count, dose_count))
    model_overdose = np.empty_like(model_means)
    alpha_means = np.empty(model_count)
    alpha_sds = np.empty(model_count)
    errors = np.empty(model_count)
    for k in range(model_count):
        (
            log_evidence[k],
            model_means[k],
            model_overdose[k],
            alpha_means[k],
            alpha_sds[k],
            errors[k],
        ) = _fit_model(raw_skeletons[k], raw_events, raw_subjects, target_value, sd_value, budget)
    if np.any(~np.isfinite(log_evidence)):
        raise ArithmeticError("model evidence is not representable")
    log_model_mass = log_prior_weights + log_evidence
    posterior_weights = np.exp(log_model_mass - logsumexp(log_model_mass))
    posterior_weights /= np.sum(posterior_weights)
    dose_mean = posterior_weights @ model_means
    overdose_probability = posterior_weights @ model_overdose
    for name, value in (
        ("model-averaged dose mean", dose_mean),
        ("model-averaged overdose probability", overdose_probability),
    ):
        if np.any(~np.isfinite(value) | (value < -1e-12) | (value > 1 + 1e-12)):
            raise ArithmeticError(f"{name} is not representable in [0,1]")
    dose_mean = np.clip(dose_mean, 0, 1)
    overdose_probability = np.clip(overdose_probability, 0, 1)
    return BMACRMPosterior(
        _freeze(raw_skeletons),
        _freeze(raw_events),
        _freeze(raw_subjects),
        target_value,
        sd_value,
        _freeze(prior_weights),
        _freeze(posterior_weights),
        _freeze(log_evidence),
        _freeze(model_means),
        _freeze(model_overdose),
        _freeze(dose_mean),
        _freeze(overdose_probability),
        _freeze(alpha_means),
        _freeze(alpha_sds),
        _freeze(errors),
        budget.count,
    )
