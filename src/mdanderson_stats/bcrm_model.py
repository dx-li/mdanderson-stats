"""Numerical single-outcome core for the bCRM Goodman CRM model.

This module implements the bounded logistic curve and deterministic posterior
summaries for a uniform, bounded prior on its nonnegative slope. It does not
implement allocation rules or a joint toxicity/efficacy association model.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, log_expit

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


@dataclass(frozen=True, init=False)
class BCRMCurve:
    """A Goodman CRM dose skeleton with explicit probability asymptotes."""

    skeleton: FloatArray
    alpha: float = 3.0
    lower: float = 0.0
    upper: float = 1.0
    standardized_doses: FloatArray = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        skeleton: ArrayLike,
        *,
        alpha: float = 3.0,
        lower: float = 0.0,
        upper: float = 1.0,
    ) -> None:
        alpha = scalar(alpha, "alpha")
        if abs(alpha) > 50:
            raise ValueError("curve alpha must satisfy abs(alpha) <= 50")
        lower, upper = _bounds(lower, upper)
        input_skeleton = np.asarray(skeleton)
        if input_skeleton.ndim != 1 or not 1 <= input_skeleton.size <= 100:
            raise ValueError("skeleton must be a one-dimensional array of 1 to 100 levels")
        raw = finite(input_skeleton, "skeleton")
        if np.any(np.diff(raw) <= 0) or np.any((raw <= lower) | (raw >= upper)):
            raise ValueError("skeleton must be strictly increasing inside its asymptotes")
        with np.errstate(divide="ignore", invalid="ignore"):
            doses = np.log(raw - lower) - np.log(upper - raw) - alpha
        if not np.all(np.isfinite(doses)):
            raise ValueError("standardized doses must be finite")
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
    if raw_beta.size > 200_000 or raw_beta.size * raw_x.size > 200_000:
        raise ValueError("beta-dose prediction exceeds 200000 pairs")
    doses = finite(raw_x, "x")
    slopes = finite(raw_beta, "beta")
    if np.any(slopes < 0):
        raise ValueError("beta must be nonnegative")
    intercept = scalar(alpha, "alpha")
    lo, hi = _bounds(lower, upper)
    with np.errstate(over="ignore", invalid="ignore"):
        linear = intercept + slopes[..., np.newaxis] * doses
    if np.any(np.isnan(linear)):
        raise ValueError("beta-dose linear predictor produced an undefined value")
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
    if raw_beta.size > 200_000 or raw_beta.size * raw_x.size > 200_000:
        raise ValueError("beta-dose prediction exceeds 200000 pairs")
    doses = finite(raw_x, "x")
    slopes = finite(raw_beta, "beta")
    if np.any(slopes < 0):
        raise ValueError("beta must be nonnegative")
    intercept = scalar(alpha, "alpha")
    lo, hi = _bounds(lower, upper)
    with np.errstate(over="ignore", invalid="ignore"):
        z = intercept + slopes[..., np.newaxis] * doses
    if np.any(np.isnan(z)):
        raise ValueError("beta-dose linear predictor produced an undefined value")
    span = hi - lo
    log_span = np.log(span)
    log_event = np.logaddexp(-np.inf if lo == 0 else np.log(lo), log_span + log_expit(z))
    log_nonevent = np.logaddexp(-np.inf if hi == 1 else np.log1p(-hi), log_span + log_expit(-z))
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
    if raw_beta.size > 200_000 or raw_beta.size * raw_x.size > 200_000:
        raise ValueError("beta-row likelihood exceeds 200000 pairs")
    doses = finite(raw_x, "x")
    successes = count(raw_events, "events")
    totals = count(raw_subjects, "subjects")
    if np.any(successes > totals) or np.sum(totals) > 10_000:
        raise ValueError("events must not exceed subjects and total subjects must be <= 10000")
    logs = bcrm_log_probabilities(doses, raw_beta, alpha=alpha, lower=lower, upper=upper)
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
) -> BCRMPosterior:
    """Integrate a one-dimensional uniform-slope posterior deterministically.

    Adaptive QUADPACK integrations are scaled by local likelihood maxima;
    their reported error is checked before summaries are returned. The bounded
    prior support is limited to [0, 3] and the intercept to [-50, 50] for
    predictable numerical work. Integration stops after 200000 likelihood
    evaluations.
    """
    if not isinstance(curve, BCRMCurve):
        raise TypeError("curve must be a BCRMCurve")
    if len(prior_bounds) != 2:
        raise ValueError("prior_bounds must contain lower and upper values")
    beta_lo = scalar(prior_bounds[0], "prior lower bound")
    beta_hi = scalar(prior_bounds[1], "prior upper bound")
    if beta_lo < 0 or beta_hi <= beta_lo or beta_hi > 3:
        raise ValueError("prior bounds must satisfy 0 <= lower < upper <= 3")
    level = scalar(probability, "probability")
    if not 0 < level < 1:
        raise ValueError("probability must lie strictly between zero and one")
    intercept, lo, hi = curve.alpha, curve.lower, curve.upper
    if abs(intercept) > 50:
        raise ValueError("fit_bcrm requires abs(curve.alpha) <= 50")
    doses = curve.standardized_doses
    raw_events, raw_subjects = np.asarray(events), np.asarray(subjects)
    if raw_events.shape != doses.shape or raw_subjects.shape != doses.shape:
        raise ValueError("events and subjects must match the curve dose levels")
    successes, totals = count(raw_events, "events"), count(raw_subjects, "subjects")
    if np.any(successes > totals) or np.sum(totals) > 10_000:
        raise ValueError("events must not exceed subjects and total subjects must be <= 10000")

    evaluations = 0

    def loglike(beta: float) -> float:
        nonlocal evaluations
        evaluations += 1
        if evaluations > 200_000:
            raise ArithmeticError("posterior integration exceeded 200000 likelihood evaluations")
        return float(
            bcrm_log_likelihood(
                doses, successes, totals, np.array([beta]), alpha=intercept, lower=lo, upper=hi
            )[0]
        )

    # Segment the bounded support before optimizing. This catches separated
    # local modes and gives the quadrature explicit breaks at each mode.
    boundaries = np.linspace(beta_lo, beta_hi, 17)
    candidates = [(float(b), loglike(float(b))) for b in boundaries]
    mode_points: list[float] = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        optimized = minimize_scalar(
            lambda b: -loglike(float(b)),
            bounds=(float(left), float(right)),
            method="bounded",
            options={"xatol": 1e-12},
        )
        if optimized.success:
            mode = float(optimized.x)
            candidates.append((mode, loglike(mode)))
            mode_points.append(mode)
    peak_beta, peak = max(candidates, key=lambda pair: pair[1])
    mode_points = sorted({b for b in mode_points if beta_lo < b < beta_hi})
    if not np.isfinite(peak):
        raise ArithmeticError("posterior likelihood is not finite")

    # Explicitly bracket the likelihood's material mass around the global
    # maximum. This resolves sharply concentrated endpoint and interior modes.
    shoulder_points: list[float] = []
    grid_values = [(b, value) for b, value in candidates if b in boundaries]
    for side in (-1, 1):
        side_values = [pair for pair in grid_values if (pair[0] - peak_beta) * side > 0]
        side_values.sort(key=lambda pair: abs(pair[0] - peak_beta))
        bracket = next((pair for pair in side_values if pair[1] <= peak - 40), None)
        if bracket is not None:
            shoulder = brentq(
                lambda b: loglike(b) - peak + 40,
                min(peak_beta, bracket[0]),
                max(peak_beta, bracket[0]),
                xtol=1e-14,
            )
            if beta_lo < shoulder < beta_hi:
                shoulder_points.append(shoulder)
    integration_points = sorted(set(mode_points + shoulder_points))

    def density(beta: float) -> float:
        return float(np.exp(loglike(beta) - peak))

    normalizer_scale: float | None = None

    def integrate(
        function: Callable[[float], float], left: float = beta_lo, right: float = beta_hi
    ) -> tuple[float, float]:
        points = [p for p in integration_points if left < p < right]
        value, error = quad(
            lambda b: density(b) * function(b),
            left,
            right,
            epsabs=2e-12,
            epsrel=2e-10,
            limit=300,
            points=points or None,
        )
        scale = max(abs(value), normalizer_scale or 1e-12)
        tolerance = 2e-8 * scale
        if not np.isfinite(value) or not np.isfinite(error) or error > tolerance:
            raise ArithmeticError("posterior quadrature did not meet its error tolerance")
        return float(value), float(error)

    normalizer, norm_error = integrate(lambda _b: 1.0)
    if normalizer <= 0:
        raise ArithmeticError("posterior normalization underflowed")
    normalizer_scale = normalizer
    first, first_error = integrate(lambda b: b)
    mean = first / normalizer
    centered_second, second_error = integrate(lambda b: (b - mean) ** 2)
    variance = centered_second / normalizer
    tail = (1 - level) / 2

    def cdf(beta: float) -> float:
        val, _ = integrate(lambda _b: 1.0, beta_lo, beta)
        return float(val / normalizer)

    interval = [
        brentq(lambda b: cdf(b) - tail, beta_lo, beta_hi, xtol=1e-12),
        brentq(lambda b: cdf(b) - (1 - tail), beta_lo, beta_hi, xtol=1e-12),
    ]
    dose_mean = np.empty(doses.size)
    dose_interval = np.empty((doses.size, 2))
    for index, dose in enumerate(doses):
        dose_value = float(dose)

        def dose_probability(beta_value: float) -> float:
            return float(lo + (hi - lo) * expit(intercept + beta_value * dose_value))

        dose_mean[index] = integrate(dose_probability)[0] / normalizer
        dose_interval[index] = [
            *sorted(
                (
                    lo + (hi - lo) * expit(intercept + interval[0] * dose),
                    lo + (hi - lo) * expit(intercept + interval[1] * dose),
                )
            ),
        ]
    plugin = bcrm_probabilities(doses, np.array([mean]), alpha=intercept, lower=lo, upper=hi)[0]
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
