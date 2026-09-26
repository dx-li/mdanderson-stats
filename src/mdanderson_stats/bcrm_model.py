"""Numerical single-outcome core for the bCRM Goodman CRM model.

This module implements the bounded logistic curve and deterministic posterior
summaries for a uniform, bounded prior on its nonnegative slope. It does not
implement allocation rules or a joint toxicity/efficacy association model.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, log_expit, logit

from ._validation import count, finite, scalar

FloatArray = NDArray[np.float64]


def _readonly(values: ArrayLike) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _bounds(lower: float, upper: float) -> tuple[float, float]:
    lo = scalar(lower, "lower")
    hi = scalar(upper, "upper")
    if not 0 <= lo < hi <= 1:
        raise ValueError("asymptotes must satisfy 0 <= lower < upper <= 1")
    return lo, hi


@dataclass(frozen=True)
class BCRMCurve:
    """A Goodman CRM dose skeleton with explicit probability asymptotes."""

    skeleton: FloatArray
    alpha: float = 3.0
    lower: float = 0.0
    upper: float = 1.0
    standardized_doses: FloatArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        alpha = scalar(self.alpha, "alpha")
        lower, upper = _bounds(self.lower, self.upper)
        raw = finite(self.skeleton, "skeleton")
        if raw.ndim != 1 or not 1 <= raw.size <= 100:
            raise ValueError("skeleton must be a one-dimensional array of 1 to 100 levels")
        if np.any(np.diff(raw) <= 0) or np.any((raw <= lower) | (raw >= upper)):
            raise ValueError("skeleton must be strictly increasing inside its asymptotes")
        scaled = (raw - lower) / (upper - lower)
        doses = (logit(scaled) - alpha).astype(np.float64)
        object.__setattr__(self, "skeleton", _readonly(raw))
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "standardized_doses", _readonly(doses))

    def probabilities(self, beta: ArrayLike) -> FloatArray:
        """Return rows for slope values and columns for dose levels."""
        return bcrm_probabilities(
            self.standardized_doses,
            beta,
            alpha=self.alpha,
            lower=self.lower,
            upper=self.upper,
        )


def bcrm_probabilities(
    x: ArrayLike,
    beta: ArrayLike,
    *,
    alpha: float = 3.0,
    lower: float = 0.0,
    upper: float = 1.0,
) -> FloatArray:
    """Return bounded logistic event probabilities for every beta and dose pair."""
    raw_x = np.asarray(x)
    raw_beta = np.asarray(beta)
    if raw_x.ndim != 1 or raw_x.size > 100:
        raise ValueError("x must be one-dimensional with at most 100 doses")
    if raw_beta.size * raw_x.size > 200_000:
        raise ValueError("beta-dose prediction exceeds 200000 pairs")
    doses = finite(raw_x, "x")
    slopes = finite(raw_beta, "beta")
    if np.any(slopes < 0):
        raise ValueError("beta must be nonnegative")
    intercept = scalar(alpha, "alpha")
    lo, hi = _bounds(lower, upper)
    linear = intercept + slopes.reshape((-1, 1)) * doses.reshape((1, -1))
    return np.asarray(lo + (hi - lo) * expit(linear), dtype=np.float64)


def bcrm_log_probabilities(
    x: ArrayLike,
    beta: ArrayLike,
    *,
    alpha: float = 3.0,
    lower: float = 0.0,
    upper: float = 1.0,
) -> FloatArray:
    """Return final axis (no-event, event) log probabilities without clipping."""
    raw_x = np.asarray(x)
    raw_beta = np.asarray(beta)
    if raw_x.ndim != 1 or raw_x.size > 100:
        raise ValueError("x must be one-dimensional with at most 100 doses")
    if raw_beta.size * raw_x.size > 200_000:
        raise ValueError("beta-dose prediction exceeds 200000 pairs")
    doses = finite(raw_x, "x")
    slopes = finite(raw_beta, "beta")
    if np.any(slopes < 0):
        raise ValueError("beta must be nonnegative")
    intercept = scalar(alpha, "alpha")
    lo, hi = _bounds(lower, upper)
    z = intercept + slopes.reshape((-1, 1)) * doses.reshape((1, -1))
    span = hi - lo
    log_span = np.log(span)
    log_event = np.logaddexp(
        -np.inf if lo == 0 else np.log(lo), log_span + log_expit(z)
    )
    log_nonevent = np.logaddexp(
        -np.inf if hi == 1 else np.log1p(-hi), log_span + log_expit(-z)
    )
    return np.stack((log_nonevent, log_event), axis=-1)


def bcrm_log_likelihood(
    x: ArrayLike,
    events: ArrayLike,
    subjects: ArrayLike,
    beta: ArrayLike,
    *,
    alpha: float = 3.0,
    lower: float = 0.0,
    upper: float = 1.0,
) -> FloatArray:
    """Grouped Bernoulli log likelihood, omitting binomial coefficients."""
    # Check shapes and allocation bounds before conversions that copy or mask.
    raw_x, raw_events, raw_subjects, raw_beta = map(np.asarray, (x, events, subjects, beta))
    if raw_x.ndim != 1 or raw_x.size > 100:
        raise ValueError("x must be one-dimensional with at most 100 rows")
    if raw_events.shape != raw_x.shape or raw_subjects.shape != raw_x.shape:
        raise ValueError("x, events and subjects must have matching one-dimensional shapes")
    if raw_beta.size * raw_x.size > 200_000:
        raise ValueError("beta-row likelihood exceeds 200000 pairs")
    doses = finite(raw_x, "x")
    successes = count(raw_events, "events")
    totals = count(raw_subjects, "subjects")
    if np.any(successes > totals) or np.sum(totals) > 10_000:
        raise ValueError("events must not exceed subjects and total subjects must be <= 10000")
    logs = bcrm_log_probabilities(
        doses, raw_beta, alpha=alpha, lower=lower, upper=upper
    )
    event_term = np.zeros_like(logs[..., 1])
    nonevent_term = np.zeros_like(logs[..., 0])
    np.multiply(successes, logs[..., 1], out=event_term, where=successes != 0)
    np.multiply(totals - successes, logs[..., 0], out=nonevent_term, where=totals != successes)
    return np.sum(event_term + nonevent_term, axis=-1)


@dataclass(frozen=True)
class BCRMPosterior:
    """Posterior slope and dose-level predictive summaries."""

    beta_mean: float
    beta_sd: float
    beta_interval: FloatArray
    dose_mean: FloatArray
    dose_interval: FloatArray
    plugin_dose_probability: FloatArray
    log_evidence: float
    integration_error: float


def fit_bcrm(
    curve: BCRMCurve,
    events: ArrayLike,
    subjects: ArrayLike,
    *,
    prior_bounds: tuple[float, float] = (0.0, 3.0),
    probability: float = 0.95,
    alpha: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> BCRMPosterior:
    """Integrate a one-dimensional uniform-slope posterior deterministically.

    Adaptive QUADPACK integrations are scaled by the maximum log likelihood;
    their reported error is checked before summaries are returned.
    """
    if not isinstance(curve, BCRMCurve):
        raise TypeError("curve must be a BCRMCurve")
    if len(prior_bounds) != 2:
        raise ValueError("prior_bounds must contain lower and upper values")
    beta_lo = scalar(prior_bounds[0], "prior lower bound")
    beta_hi = scalar(prior_bounds[1], "prior upper bound")
    if beta_lo < 0 or beta_hi <= beta_lo:
        raise ValueError("prior bounds must satisfy 0 <= lower < upper")
    level = scalar(probability, "probability")
    if not 0 < level < 1:
        raise ValueError("probability must lie strictly between zero and one")
    intercept = curve.alpha if alpha is None else scalar(alpha, "alpha")
    lo, hi = _bounds(
        curve.lower if lower is None else lower,
        curve.upper if upper is None else upper,
    )
    doses = curve.standardized_doses
    raw_events, raw_subjects = np.asarray(events), np.asarray(subjects)
    if raw_events.shape != doses.shape or raw_subjects.shape != doses.shape:
        raise ValueError("events and subjects must match the curve dose levels")
    successes, totals = count(raw_events, "events"), count(raw_subjects, "subjects")
    if np.any(successes > totals) or np.sum(totals) > 10_000:
        raise ValueError("events must not exceed subjects and total subjects must be <= 10000")

    def loglike(beta: float) -> float:
        return float(
            bcrm_log_likelihood(
                doses, successes, totals, np.array([beta]), alpha=intercept, lower=lo, upper=hi
            )[0]
        )

    optimized = minimize_scalar(
        lambda b: -loglike(float(b)), bounds=(beta_lo, beta_hi), method="bounded",
        options={"xatol": 1e-12},
    )
    candidates = [(beta_lo, loglike(beta_lo)), (beta_hi, loglike(beta_hi))]
    if optimized.success:
        candidates.append((float(optimized.x), loglike(float(optimized.x))))
    peak_beta, peak = max(candidates, key=lambda pair: pair[1])
    del peak_beta
    if not np.isfinite(peak):
        raise ArithmeticError("posterior likelihood is not finite")

    def density(beta: float) -> float:
        return float(np.exp(loglike(beta) - peak))

    def integrate(function) -> tuple[float, float]:
        value, error = quad(
            lambda b: density(b) * function(b), beta_lo, beta_hi,
            epsabs=2e-12, epsrel=2e-10, limit=250,
        )
        tolerance = 2e-8 * max(abs(value), 1e-12)
        if not np.isfinite(value) or not np.isfinite(error) or error > tolerance:
            raise ArithmeticError("posterior quadrature did not meet its error tolerance")
        return float(value), float(error)

    normalizer, norm_error = integrate(lambda _b: 1.0)
    if normalizer <= 0:
        raise ArithmeticError("posterior normalization underflowed")
    first, first_error = integrate(lambda b: b)
    second, second_error = integrate(lambda b: b * b)
    mean = first / normalizer
    variance = max(0.0, second / normalizer - mean * mean)
    tail = (1 - level) / 2

    def cdf(beta: float) -> float:
        val, err = quad(
            density, beta_lo, beta, epsabs=2e-12, epsrel=2e-10, limit=250
        )
        if err > 2e-8 * max(abs(val), 1e-12):
            raise ArithmeticError("posterior quantile integration did not converge")
        return float(val / normalizer)

    interval = [
        brentq(lambda b: cdf(b) - tail, beta_lo, beta_hi, xtol=1e-12),
        brentq(lambda b: cdf(b) - (1 - tail), beta_lo, beta_hi, xtol=1e-12),
    ]
    dose_mean = np.empty(doses.size)
    dose_interval = np.empty((doses.size, 2))
    for index, dose in enumerate(doses):
        dose_mean[index] = integrate(
            lambda b, x=float(dose): lo + (hi - lo) * expit(intercept + b * x)
        )[0] / normalizer
        dose_interval[index] = [
            lo + (hi - lo) * expit(intercept + interval[0] * dose),
            lo + (hi - lo) * expit(intercept + interval[1] * dose),
        ]
    plugin = bcrm_probabilities(
        doses, np.array([mean]), alpha=intercept, lower=lo, upper=hi
    )[0]
    return BCRMPosterior(
        beta_mean=mean,
        beta_sd=float(np.sqrt(variance)),
        beta_interval=_readonly(interval),
        dose_mean=_readonly(dose_mean),
        dose_interval=_readonly(dose_interval),
        plugin_dose_probability=_readonly(plugin),
        log_evidence=float(peak + np.log(normalizer / (beta_hi - beta_lo))),
        integration_error=float(norm_error + first_error + second_error),
    )
