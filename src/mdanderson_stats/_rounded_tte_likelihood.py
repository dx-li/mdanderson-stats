"""Stable likelihoods for continuous survival times observed in intervals."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import log_expit, log_ndtr

from .tte_family_bayesian_gof import _LOG_FLOAT_MAX, _LOG_FLOAT_MIN, _log_regularized_gamma_tail

_FAMILIES = {
    "exponential",
    "weibull",
    "lognormal",
    "gamma",
    "inverse_gamma",
    "log_logistic",
    "log_odds_rate",
}
_SMALL_LOG_PRODUCT = -36.0


def _log1mexp(value: float) -> float:
    """Log(1-exp(value)) for value <= 0, retaining close differences."""
    if value == 0.0:
        return -np.inf
    if value < -np.log(2.0):
        return float(np.log1p(-np.exp(value)))
    return float(np.log(-np.expm1(value)))


def _log_difference(larger: float, smaller: float) -> float | None:
    """Log(exp(larger)-exp(smaller)); None signals inadequate input precision."""
    if smaller == -np.inf:
        return larger
    if larger == smaller:
        return None
    gap = larger - smaller
    precision_floor = 32.0 * np.finfo(float).eps * max(1.0, abs(larger), abs(smaller))
    if gap <= precision_floor:
        return None
    return larger + _log1mexp(smaller - larger)


def _log_exp_minus_one_neg(log_x: np.ndarray) -> np.ndarray:
    """Log(1-exp(-x)) from log(x), including x outside float range."""
    result = np.empty_like(log_x)
    tiny = log_x < -36.0
    huge = log_x > _LOG_FLOAT_MAX
    ordinary = ~(tiny | huge)
    result[tiny] = log_x[tiny]
    result[huge] = -0.0
    if np.any(ordinary):
        x = np.exp(log_x[ordinary])
        result[ordinary] = np.log(-np.expm1(-x))
    return result


def _log_odds_tails(
    relative_time: np.ndarray, log_shape: float, log_scale: float, log_c: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return log CDF and log survival without materializing a tiny CDF."""
    shape = float(np.exp(log_shape))
    with np.errstate(over="ignore", invalid="ignore"):
        log_power = shape * (relative_time - log_scale)
        log_product = log_c + log_power
    log_q = np.empty_like(log_product)
    small = log_product < _SMALL_LOG_PRODUCT
    log_q[small] = log_power[small]
    regular = ~small
    if np.any(regular):
        log_q[regular] = np.log(np.logaddexp(0.0, log_product[regular])) - log_c
    log_cdf = _log_exp_minus_one_neg(log_q)
    q_overflow = log_q > _LOG_FLOAT_MAX
    log_survival = np.empty_like(log_q)
    log_survival[q_overflow] = -np.inf
    with np.errstate(over="ignore", under="ignore"):
        log_survival[~q_overflow] = -np.exp(log_q[~q_overflow])
    return log_cdf, log_survival


def _family_log_tails(
    relative_time: np.ndarray,
    coordinates: np.ndarray,
    family: str,
    *,
    work_counter: list[int] | None,
    work_limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Log-CDF and log-survival for one family's transformed parameters."""
    if family == "exponential":
        with np.errstate(over="ignore", invalid="ignore"):
            log_x = relative_time - coordinates[0]
        with np.errstate(over="ignore", under="ignore"):
            x = np.exp(log_x)
            log_sf = -x
        log_cdf = _log_exp_minus_one_neg(log_x)
        return log_cdf, log_sf

    if family == "lognormal":
        sigma = float(np.exp(coordinates[1]))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ArithmeticError("log-normal sigma is outside the representable range")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            z = (relative_time - coordinates[0]) / sigma
        return log_ndtr(z), log_ndtr(-z)

    if family in ("gamma", "inverse_gamma"):
        log_shape, log_scale = map(float, coordinates)
        if not _LOG_FLOAT_MIN <= log_shape <= _LOG_FLOAT_MAX:
            raise ArithmeticError("Gamma shape is outside the representable range")
        shape = float(np.exp(log_shape))
        with np.errstate(over="ignore", invalid="ignore"):
            log_x = relative_time - log_scale
        if family == "inverse_gamma":
            log_x = -log_x
        log_lower = _log_regularized_gamma_tail(
            shape,
            log_x,
            lower=family == "gamma",
            work_counter=work_counter,
            work_limit=work_limit,
        )
        log_upper = _log_regularized_gamma_tail(
            shape,
            log_x,
            lower=family == "inverse_gamma",
            work_counter=work_counter,
            work_limit=work_limit,
        )
        return log_lower, log_upper

    log_shape, log_scale = map(float, coordinates[:2])
    if not _LOG_FLOAT_MIN <= log_shape <= _LOG_FLOAT_MAX:
        raise ArithmeticError("distribution shape is outside the representable range")
    shape = float(np.exp(log_shape))
    with np.errstate(over="ignore", invalid="ignore"):
        log_ratio = relative_time - log_scale
    if family == "weibull":
        with np.errstate(over="ignore", invalid="ignore"):
            log_x = shape * log_ratio
        with np.errstate(over="ignore", under="ignore"):
            x = np.exp(log_x)
            log_sf = -x
        return _log_exp_minus_one_neg(log_x), log_sf
    if family == "log_logistic":
        with np.errstate(over="ignore", invalid="ignore"):
            z = shape * log_ratio
        return log_expit(z), log_expit(-z)
    if family == "log_odds_rate":
        return _log_odds_tails(relative_time, log_shape, log_scale, float(coordinates[2]))
    raise AssertionError("family checked by caller")


def rounded_tte_log_probabilities(
    relative_lower: ArrayLike,
    relative_upper: ArrayLike,
    coordinates: ArrayLike,
    family: str,
    *,
    work_counter: list[int] | None = None,
    work_limit: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate interval mass and endpoint log-CDFs for rounded TTE data.

    Bounds are centered log times ``log(t) - time_offset``; lower zero is
    represented by ``-inf``. All families are continuous, so endpoint
    inclusion does not alter interval probabilities. Coordinates are centered
    log scale for exponential; centered log median and log sigma for lognormal;
    and log shape plus centered log scale for Weibull, Gamma, inverse-Gamma,
    and log-logistic. Log-odds-rate adds log(c) as coordinate three.

    The interval probability is computed by subtracting log CDFs or log
    survivals, choosing the pair with the more separated log endpoints. If
    both tails collapse to identical floating-point values, the routine raises
    instead of replacing the interval mass with a density-times-width
    approximation. It also rejects a log-tail separation no larger than 32
    machine epsilons times the largest endpoint-log magnitude (or one), since
    the resulting difference would have unreliable relative precision. A truly
    unrepresentable extreme-tail mass may be ``-inf``. Gamma-tail fallback
    iterations share ``work_counter`` and ``work_limit`` with caller accounting.
    """
    if family not in _FAMILIES:
        raise ValueError(f"family must be one of {sorted(_FAMILIES)}")
    expected = {
        "exponential": 1,
        "lognormal": 2,
        "weibull": 2,
        "gamma": 2,
        "inverse_gamma": 2,
        "log_logistic": 2,
        "log_odds_rate": 3,
    }[family]
    if any(np.iscomplexobj(value) for value in (relative_lower, relative_upper, coordinates)):
        raise ValueError("interval endpoints and coordinates must be real")
    lower = np.asarray(relative_lower, dtype=np.float64)
    upper = np.asarray(relative_upper, dtype=np.float64)
    params = np.asarray(coordinates, dtype=np.float64)
    if lower.ndim != 1 or lower.size == 0 or upper.shape != lower.shape:
        raise ValueError("relative interval endpoints must be matching vectors")
    if np.any(np.isnan(lower)) or np.any(np.isposinf(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("lower bounds allow -inf only; upper bounds must be finite")
    if np.any(lower >= upper):
        raise ValueError("every lower endpoint must be strictly below its upper endpoint")
    if params.shape != (expected,) or np.any(~np.isfinite(params)):
        raise ValueError(f"coordinates must be a finite vector with length {expected}")
    if work_counter is not None and (
        not isinstance(work_counter, list)
        or len(work_counter) != 1
        or isinstance(work_counter[0], bool)
        or not isinstance(work_counter[0], int)
        or work_counter[0] < 0
    ):
        raise ValueError("work_counter must be a one-element list of nonnegative integers")
    if isinstance(work_limit, bool) or not isinstance(work_limit, int) or work_limit < 0:
        raise ValueError("work_limit must be a nonnegative integer")

    lower_cdf, lower_sf = _family_log_tails(
        lower,
        params,
        family,
        work_counter=work_counter,
        work_limit=work_limit,
    )
    upper_cdf, upper_sf = _family_log_tails(
        upper,
        params,
        family,
        work_counter=work_counter,
        work_limit=work_limit,
    )
    for tails in (lower_cdf, lower_sf, upper_cdf, upper_sf):
        if np.any(np.isnan(tails)) or np.any(tails > 0):
            raise ArithmeticError("distribution log-tail evaluation became invalid")
    mass = np.empty_like(lower)
    for index in range(lower.size):
        log_cdf_lo, log_cdf_hi = float(lower_cdf[index]), float(upper_cdf[index])
        log_sf_lo, log_sf_hi = float(lower_sf[index]), float(upper_sf[index])
        cdf_mass = _log_difference(log_cdf_hi, log_cdf_lo) if log_cdf_hi > log_cdf_lo else None
        sf_mass = _log_difference(log_sf_lo, log_sf_hi) if log_sf_lo > log_sf_hi else None
        if cdf_mass is not None and sf_mass is not None:
            cdf_gap = log_cdf_hi - log_cdf_lo
            sf_gap = log_sf_lo - log_sf_hi
            value = cdf_mass if cdf_gap >= sf_gap else sf_mass
        elif cdf_mass is not None:
            value = cdf_mass
        elif sf_mass is not None:
            value = sf_mass
        else:
            endpoint_logs = (log_cdf_lo, log_cdf_hi, log_sf_lo, log_sf_hi)
            unrepresentable_tail = all(
                np.isneginf(endpoint_log) or endpoint_log == 0.0 for endpoint_log in endpoint_logs
            )
            if not unrepresentable_tail:
                raise ArithmeticError(
                    "positive interval probability is unresolved in both log-CDF tails"
                )
            value = -np.inf
        if np.isnan(value) or value > 0:
            raise ArithmeticError("interval log probability is not finite and nonpositive")
        mass[index] = value
    return mass, lower_cdf, upper_cdf
