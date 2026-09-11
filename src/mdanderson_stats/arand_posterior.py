"""Multi-arm posterior probabilities for Adaptive Randomization's two models."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.special import betainc, betaincc, gammainc, gammaincc, gammainccinv, gammaln

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite, scalar
from .beta_comparison import _cdf_from_logs, _logit_quantile
from .inequality import inequality_probability
from .parameter_distribution import ParameterDistribution


@dataclass(frozen=True)
class ArandBestProbability:
    probability: FloatArray
    absolute_error: FloatArray
    family: str
    maximize: bool


def _parameters(parameters: ArrayLike, family: str) -> FloatArray:
    if family not in ("beta", "inverse_gamma"):
        raise ValueError("family must be beta or inverse_gamma")
    p = finite(parameters, "parameters")
    if p.ndim != 2 or p.shape[1] != 2 or not 1 <= len(p) <= 10 or np.any(p <= 0):
        raise ValueError("parameters must have 1 to 10 positive rows of (shape, shape/scale)")
    if family == "beta" and not np.all(np.isfinite(p.sum(axis=1))):
        raise ValueError("beta shape sums must be finite")
    return p


def _coordinate(a: float, b: float, u: float, family: str) -> float:
    if family == "beta":
        value = _logit_quantile(a, b, u)
    else:
        # If theta ~ IG(a,b), b/theta ~ Gamma(a,1). Work in log(theta)
        # so enormous or tiny time scales do not require representable quantiles.
        gamma = float(gammainccinv(a, u))
        if gamma > 0 and np.isfinite(gamma):
            log_gamma = float(np.log(gamma))
        elif gamma == 0:
            log_gamma = float((np.log1p(-u) + gammaln(a + 1)) / a)
            if log_gamma > -25:
                raise ArithmeticError("inverse-gamma quantile approximation is unresolved")
        else:
            raise ArithmeticError("inverse-gamma quantile is unresolved")
        value = float(np.log(b) - log_gamma)
    if not np.isfinite(value):
        raise ArithmeticError("posterior quantile coordinate is unresolved")
    return value


def _tail(a: float, b: float, t: float, family: str, lower: bool) -> float:
    if family == "beta":
        lx, lc = -float(np.logaddexp(0, -t)), -float(np.logaddexp(0, t))
        return _cdf_from_logs(a, b, lx, lc) if lower else _cdf_from_logs(b, a, lc, lx)
    log_gamma = float(np.log(b) - t)
    if log_gamma < -700:
        log_lower = a * log_gamma - float(gammaln(a + 1))
        return float(-np.expm1(log_lower) if lower else np.exp(log_lower))
    if log_gamma > np.log(np.finfo(float).max):
        return float(not lower)
    value = float(np.exp(log_gamma))
    return float(gammaincc(a, value) if lower else gammainc(a, value))


def _tails(parameters: FloatArray, t: float, family: str, lower: bool) -> FloatArray:
    """Vectorize the repeated competing-arm CDF evaluations in each integral."""
    a, b = parameters.T
    if family == "beta":
        lx, lc = -float(np.logaddexp(0, -t)), -float(np.logaddexp(0, t))
        if min(lx, lc) < -700:
            return np.array([_tail(float(c), float(d), t, family, lower) for c, d in parameters])
        if not lower:
            a, b, lx, lc = b, a, lc, lx
        return np.asarray(betainc(a, b, np.exp(lx)) if lx < lc else betaincc(b, a, np.exp(lc)))
    log_gamma = np.log(b) - t
    tiny, large = log_gamma < -700, log_gamma > np.log(np.finfo(float).max)
    regular = ~(tiny | large)
    result = np.empty(len(parameters))
    result[large] = float(not lower)
    log_lower = a[tiny] * log_gamma[tiny] - gammaln(a[tiny] + 1)
    result[tiny] = -np.expm1(log_lower) if lower else np.exp(log_lower)
    gamma = np.exp(log_gamma[regular])
    result[regular] = gammaincc(a[regular], gamma) if lower else gammainc(a[regular], gamma)
    return result


def _integral(
    parameters: FloatArray, i: int, family: str, maximize: bool, tolerance: float
) -> tuple[float, float]:
    a, b = (float(v) for v in parameters[i])
    others = np.delete(parameters, i, axis=0)
    tail = tolerance / 16
    marks = [tail, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1 - tail]
    if _coordinate(a, b, tail, family) >= _coordinate(a, b, 1 - tail, family):
        raise ArithmeticError("posterior quantile interval cannot be resolved")
    points = sorted(
        {
            _tail(a, b, _coordinate(float(c), float(d), u, family), family, True)
            for c, d in others
            for u in marks
        }
    )
    # Break at other arms' CDF transitions, including narrow posterior peaks.
    # Merge only adjacent roundoff-scale positions; no integration mass is dropped.
    separated: list[float] = []
    previous = tail
    for point in points:
        if not np.isfinite(point) or not 0 <= point <= 1:
            raise ArithmeticError("posterior integration breakpoint is invalid")
        if (
            point - previous > 32 * np.finfo(float).eps
            and 1 - tail - point > 32 * np.finfo(float).eps
        ):
            separated.append(point)
            previous = point

    def integrand(u: float) -> float:
        t = _coordinate(a, b, u, family)
        probabilities = _tails(others, t, family, maximize)
        if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
            raise ArithmeticError("posterior integration CDF is invalid")
        return float(np.prod(probabilities))

    result = quad(
        integrand,
        tail,
        1 - tail,
        points=separated,
        epsabs=tolerance / 4,
        epsrel=tolerance / 4,
        limit=500,
        full_output=1,
    )
    value, error = result[:2]
    if len(result) != 3 or not np.isfinite(error) or error > tolerance / 2 or not 0 <= value <= 1:
        raise ArithmeticError("multi-arm posterior quadrature did not converge")
    return float(value), float(error + 2 * tail)


def arand_best_probability(
    parameters: ArrayLike,
    *,
    family: str = "beta",
    maximize: bool = True,
    absolute_tolerance: float = 1e-9,
) -> ArandBestProbability:
    """Probability each independent arm has the largest or smallest parameter.

    Rows are beta (alpha,beta) or inverse-gamma (shape,scale). Multi-arm values
    integrate a PRODUCT of CDFs, not a product of pairwise winning probabilities.
    Errors include bounded discarded mass and estimated quadrature error. No
    post-hoc probability normalization conceals a failed partition check.
    """
    p = _parameters(parameters, family)
    if not isinstance(maximize, (bool, np.bool_)):
        raise ValueError("maximize must be boolean")
    tolerance = scalar(absolute_tolerance, "absolute_tolerance")
    if not 1e-12 <= tolerance <= 1e-3:
        raise ValueError("absolute_tolerance must be in [1e-12,1e-3]")
    k = len(p)
    if np.all(p == p[0]):
        values, errors = np.full(k, 1 / k), np.zeros(k)
    elif k == 2:
        comparison = inequality_probability(
            ParameterDistribution(family, *p[0]),
            ParameterDistribution(family, *p[1]),
            absolute_tolerance=tolerance,
        )
        values = np.array([comparison.x_greater, comparison.shifted_y_greater])
        if not maximize:
            values = values[::-1]
        errors = np.full(k, comparison.absolute_error)
    else:
        values, errors = np.empty(k), np.empty(k)
        cache: dict[tuple[float, float], tuple[float, float]] = {}
        for i in range(k):
            key = (float(p[i, 0]), float(p[i, 1]))
            if key not in cache:
                cache[key] = _integral(p, i, family, bool(maximize), tolerance)
            values[i], errors[i] = cache[key]
        if abs(values.sum() - 1) > k * tolerance:
            raise ArithmeticError("best-arm probabilities failed the partition check")
    return ArandBestProbability(_freeze(values), _freeze(errors), family, bool(maximize))


@dataclass(frozen=True)
class ArandPosterior:
    parameters: FloatArray
    best: ArandBestProbability
    allocation_probability: FloatArray
    exceedance_probability: FloatArray | None
    threshold: float | None
    tuning: float
    parameter: str


def _posterior(
    parameters: FloatArray,
    family: str,
    maximize: bool,
    tuning: float,
    threshold: float | None,
    tolerance: float,
    parameter: str,
) -> ArandPosterior:
    tuning = scalar(tuning, "tuning")
    if tuning < 0:
        raise ValueError("tuning must be nonnegative")
    best = arand_best_probability(
        parameters, family=family, maximize=maximize, absolute_tolerance=tolerance
    )
    if tuning == 0:
        allocation = np.full(len(parameters), 1 / len(parameters))
    else:
        p = best.probability
        if np.any(p <= best.absolute_error):
            raise ArithmeticError("best-arm tail unresolved for allocation; tighten tolerance")
        # Subtract the maximum before multiplying, so arbitrarily large finite
        # tuning cannot overflow the largest weight. Zero tails retain zero weight.
        with np.errstate(divide="ignore", over="ignore", under="ignore"):
            logp = np.log(p)
            weights = np.exp(tuning * (logp - logp.max()))
        allocation = weights / weights.sum()
    exceedance = None
    if threshold is not None:
        threshold = scalar(threshold, "threshold")
        if family == "beta":
            if not 0 <= threshold <= 1:
                raise ValueError("binary threshold must be in [0,1]")
            if threshold in (0, 1):
                exceedance = np.full(len(parameters), 1 - threshold)
            else:
                t = float(np.log(threshold) - np.log1p(-threshold))
                exceedance = np.array(
                    [_tail(float(a), float(b), t, family, False) for a, b in parameters]
                )
        else:
            if threshold < 0:
                raise ValueError("time threshold must be nonnegative")
            exceedance = (
                np.ones(len(parameters))
                if threshold == 0
                else np.array(
                    [
                        _tail(float(a), float(b), float(np.log(threshold)), family, False)
                        for a, b in parameters
                    ]
                )
            )
        exceedance = _freeze(exceedance)
    return ArandPosterior(
        _freeze(parameters), best, _freeze(allocation), exceedance, threshold, tuning, parameter
    )


def arand_binary_posterior(
    successes: ArrayLike,
    failures: ArrayLike,
    *,
    prior: ArrayLike,
    maximize: bool = True,
    tuning: float = 0.5,
    threshold: float | None = None,
    absolute_tolerance: float = 1e-9,
) -> ArandPosterior:
    """Update observed binary outcomes and compute best-arm allocation targets.

    Pending patients do not contribute success or failure counts. This is the
    unmodified tuned allocation, before burn-in, suspension or allocation floors.
    """
    p = _parameters(prior, "beta")
    s, f = count(successes, "successes"), count(failures, "failures")
    if s.shape != (len(p),) or f.shape != s.shape:
        raise ValueError("counts must have one entry per prior row")
    posterior = p + np.column_stack((s, f))
    return _posterior(
        posterior, "beta", maximize, tuning, threshold, absolute_tolerance, "probability"
    )


def arand_survival_posterior(
    events: ArrayLike,
    exposure: ArrayLike,
    *,
    prior: ArrayLike,
    parameter: str = "mean",
    maximize: bool = True,
    tuning: float = 0.5,
    threshold: float | None = None,
    absolute_tolerance: float = 1e-9,
) -> ArandPosterior:
    """Exponential likelihood with inverse-gamma prior on mean or median time.

    exposure is total observed time at risk, including right-censored follow-up.
    Only observed events increment shape. For median time, exposure is multiplied
    by log(2); prior scale and threshold must already be in median-time units.
    """
    p = _parameters(prior, "inverse_gamma")
    d, e = count(events, "events"), finite(exposure, "exposure")
    if d.shape != (len(p),) or e.shape != d.shape or np.any(e < 0) or np.any((d > 0) & (e == 0)):
        raise ValueError("one event/exposure value per arm; events require positive exposure")
    if parameter not in ("mean", "median"):
        raise ValueError("parameter must be mean or median")
    scale = 1.0 if parameter == "mean" else float(np.log(2))
    with np.errstate(over="ignore"):
        posterior = p + np.column_stack((d, e * scale))
    if not np.all(np.isfinite(posterior)):
        raise ArithmeticError("survival posterior parameters overflow")
    return _posterior(
        posterior, "inverse_gamma", maximize, tuning, threshold, absolute_tolerance, parameter
    )
