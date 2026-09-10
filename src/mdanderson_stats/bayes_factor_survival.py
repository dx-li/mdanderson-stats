"""Exponential/iMOM Bayes Factor TTE monitoring and time-on-test boundaries."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import expit, logit, logsumexp, roots_legendre

from ._validation import FloatArray, count, finite, scalar
from .bayesian_monitoring import _prob
from .beta_binomial import _owned


@lru_cache(maxsize=6)
def _nodes(order: int) -> tuple[FloatArray, FloatArray]:
    nodes, weights = roots_legendre(order)
    return _owned(nodes), _owned(np.log(weights))


def _integral(events: int, exposure: float, log_scale: float, order: int) -> float:
    # Dimensionless mean ratio r=theta/theta0=1+exp(log_scale+t).
    # Prior t density is 2*exp(-2t-exp(-2t)) on the real line.
    # Prior mass on [-.5,.5] provides a lower bound for the Bayes factor.
    log_mass = np.log(np.exp(-np.exp(-1)) - np.exp(-np.exp(1)))
    lower_log_bf = (
        log_mass - events * np.logaddexp(0, log_scale + 0.5) + exposure * expit(log_scale - 0.5)
    )
    cutoff = exposure - lower_log_bf + 40
    lower, upper = -0.5 * np.log(cutoff), 0.5 * cutoff
    if not np.isfinite(upper) or upper - lower > 2048:
        raise ArithmeticError("survival Bayes factor requires more than 2048 integration panels")
    panels = max(1, int(np.ceil(upper - lower)))
    edges = np.linspace(lower, upper, panels + 1)
    half = np.diff(edges) / 2
    nodes, logweights = _nodes(order)
    t = ((edges[:-1] + half)[:, None] + half[:, None] * nodes).ravel()
    log_weights = (np.log(half)[:, None] + logweights).ravel()
    log_r = np.logaddexp(0, log_scale + t)
    log_values = (
        np.log(2)
        - 2 * t
        - np.exp(-2 * t)
        - events * log_r
        + exposure * expit(log_scale + t)
        + log_weights
    )
    return float(logsumexp(log_values))


def _log_bf(events: int, exposure: float, log_scale: float) -> float:
    if events == 0 and exposure == 0:
        return 0.0
    previous = _integral(events, exposure, log_scale, 8)
    for order in (16, 32, 64, 128, 256):
        current = _integral(events, exposure, log_scale, order)
        if np.isfinite(current) and abs(current - previous) < 2e-10:
            return current
        previous = current
    raise ArithmeticError("survival iMOM quadrature failed to converge")


def _parameters(
    null_median: float,
    alternative_median_mode: float,
    inferiority_cutoff: float,
    superiority_cutoff: float,
) -> tuple[float, float, float, float]:
    m0 = scalar(null_median, "null_median")
    m1 = scalar(alternative_median_mode, "alternative_median_mode")
    low = _prob(inferiority_cutoff, "inferiority_cutoff")
    high = _prob(superiority_cutoff, "superiority_cutoff")
    if not 0 < m0 < m1 or not low < high:
        raise ValueError("require 0<null_median<alternative_median_mode and low cutoff<high cutoff")
    log_scale = float(0.5 * np.log(1.5) + np.log(m1 - m0) - np.log(m0))
    return m0, log_scale, low, high


@dataclass(frozen=True)
class BayesFactorSurvivalState:
    events: FloatArray
    total_time: FloatArray
    log_bayes_factor: FloatArray
    alternative_probability: FloatArray
    decision: NDArray[np.str_]


@dataclass(frozen=True)
class BayesFactorSurvivalBoundaries:
    """Strict exposure boundaries in the same time unit as the input medians.

    Inferiority requires total_time < inferiority_time; superiority requires
    total_time > superiority_time. Negative/positive infinity denotes absent
    or always-satisfied boundaries according to those comparisons.
    """

    events: FloatArray
    inferiority_time: FloatArray
    superiority_time: FloatArray
    null_median: float
    alternative_median_mode: float
    inferiority_cutoff: float
    superiority_cutoff: float


def bayes_factor_survival(
    events: ArrayLike,
    total_time: ArrayLike,
    *,
    null_median: float,
    alternative_median_mode: float,
    inferiority_cutoff: float = 0.15,
    superiority_cutoff: float = 0.8,
    final: bool = False,
) -> BayesFactorSurvivalState:
    """Monitor independent exponential survival observations, with right censoring.

    total_time is the SUM of event times and observed follow-up for censored
    patients. Medians are converted to the mean-survival parameter internally.
    k=1, nu=2; equal prior model odds. Inputs broadcast. No accrual model is assumed.
    """
    m0, log_scale, low, high = _parameters(
        null_median, alternative_median_mode, inferiority_cutoff, superiority_cutoff
    )
    if not isinstance(final, bool):
        raise ValueError("final must be boolean")
    d, exposure = np.broadcast_arrays(count(events, "events"), finite(total_time, "total_time"))
    if np.any(d > 500) or np.any(exposure < 0) or np.any((d > 0) & (exposure == 0)):
        raise ValueError(
            "require events<=500, nonnegative total_time and positive time with events"
        )
    with np.errstate(over="ignore"):
        h = exposure / m0 * np.log(2)
    if np.any(~np.isfinite(h)):
        raise ArithmeticError("dimensionless time-on-test is not representable")
    values = np.empty(d.shape)
    for i in np.ndindex(d.shape):
        values[i] = _log_bf(int(d[i]), float(h[i]), log_scale)
    decisions = np.full(d.shape, "inconclusive" if final else "continue", dtype="U16")
    decisions[values < logit(low)] = "inferiority"
    decisions[values > logit(high)] = "superiority"
    decisions.flags.writeable = False
    return BayesFactorSurvivalState(
        _owned(d), _owned(exposure), _owned(values), _owned(expit(values)), decisions
    )


def bayes_factor_survival_boundaries(
    events: ArrayLike,
    *,
    null_median: float,
    alternative_median_mode: float,
    inferiority_cutoff: float = 0.15,
    superiority_cutoff: float = 0.8,
) -> BayesFactorSurvivalBoundaries:
    """Invert monotone log Bayes factors for total-time-on-test decision boundaries."""
    m0, log_scale, low, high = _parameters(
        null_median, alternative_median_mode, inferiority_cutoff, superiority_cutoff
    )
    d = count(events, "events")
    if np.any(d > 500):
        raise ValueError("events must not exceed 500")
    lower, upper = np.empty(d.shape), np.empty(d.shape)

    def root(events: int, target: float, at_zero: float) -> float:
        if at_zero > target:
            return -np.inf
        if at_zero == target:
            return 0.0
        high = max(1.0, float(events))
        for _ in range(30):
            if _log_bf(events, high, log_scale) >= target:
                break
            high *= 2
        else:
            raise ArithmeticError("could not bracket time-on-test boundary")
        h, info = brentq(
            lambda h: _log_bf(events, h, log_scale) - target,
            0,
            high,
            xtol=2e-11,
            rtol=2e-12,
            maxiter=100,
            full_output=True,
            disp=False,
        )
        if not info.converged or abs(_log_bf(events, h, log_scale) - target) > 2e-8:
            raise ArithmeticError("time-on-test boundary failed its residual check")
        # Multiply in log space to avoid overflowing an intermediate unit conversion.
        with np.errstate(over="ignore", under="ignore"):
            time = float(np.exp(np.log(h) + np.log(m0) - np.log(np.log(2))))
        if not np.isfinite(time) or time <= 0:
            raise ArithmeticError("time-on-test boundary is not representable")
        return time

    for index in np.ndindex(d.shape):
        n = int(d[index])
        at_zero = _log_bf(n, 0, log_scale)
        lower[index] = -np.inf if low == 0 else root(n, float(logit(low)), at_zero)
        upper[index] = np.inf if high == 1 else root(n, float(logit(high)), at_zero)
    return BayesFactorSurvivalBoundaries(
        _owned(d), _owned(lower), _owned(upper), m0, alternative_median_mode, low, high
    )
