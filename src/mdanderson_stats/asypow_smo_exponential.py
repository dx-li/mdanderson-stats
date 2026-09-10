"""SMO power for exponential survival with uniform entry and censoring."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from ._validation import finite
from .asypow_regression import _log_information
from .asypow_smo import SMOPower, _poisson_log_kl
from .boin import _owned


def asypow_smo_exponential(
    rates: ArrayLike,
    duration: ArrayLike,
    *,
    null_rates: ArrayLike | None = None,
    group_size: ArrayLike = 1,
    subtract_df: bool = True,
) -> SMOPower:
    """SMO for positive rates with uniform follow-up from zero to duration.

    Omit null_rates to compare all rates (df=G-1), or supply a scalar/vector
    fixing every null rate (df=G). Group durations and allocations may differ.
    """
    p = finite(rates, "rates")
    if p.ndim != 1 or not 1 <= p.size <= 500 or np.any(p <= 0):
        raise ValueError("rates must be a positive vector of length 1..500")
    length = np.broadcast_to(finite(duration, "duration"), p.shape)
    weight = np.broadcast_to(finite(group_size, "group_size"), p.shape)
    if np.any(length <= 0) or np.any(weight <= 0):
        raise ValueError("duration and group_size must be positive")
    if not isinstance(subtract_df, (bool, np.bool_)):
        raise ValueError("subtract_df must be boolean")
    log_p = np.log(p)
    log_weight = np.log(weight) - logsumexp(np.log(weight))
    log_events = _log_information(log_p, "exponential", length)
    log_exposure = log_weight + log_events - log_p
    if null_rates is None:
        if p.size < 2:
            raise ValueError("equality testing requires at least two groups")
        # Expected deaths divided by expected observed person-time maximizes
        # the null likelihood. Center close rates to preserve small differences.
        if p.min() >= 0.5 * p.max():
            normalized = np.exp(log_exposure - logsumexp(log_exposure))
            normalized /= normalized.sum()
            mean = p.min() + normalized @ (p - p.min())
        else:
            log_mean = logsumexp(log_weight + log_events) - logsumexp(log_exposure)
            with np.errstate(over="ignore", under="ignore"):
                mean = np.exp(np.clip(log_mean, log_p.min(), log_p.max()))
        q = np.full_like(p, np.clip(mean, p.min(), p.max()))
        df = p.size - 1
    else:
        q = np.broadcast_to(finite(null_rates, "null_rates"), p.shape)
        if np.any(q <= 0):
            raise ValueError("null_rates must be positive")
        df = p.size
    # KL for the censored record = P(event)/p * KL(Pois(p), Pois(q)).
    with np.errstate(over="ignore", under="ignore"):
        w = float(np.exp(np.log(2) + logsumexp(log_exposure + _poisson_log_kl(p, q))))
    if not np.isfinite(w):
        raise ArithmeticError("exponential-survival divergence is not representable")
    return SMOPower(w, df, _owned(q), bool(subtract_df))
