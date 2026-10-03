"""Numerically stable log-odds-rate likelihood and distribution functions.

The parameterization follows Shen and Thall (1998):
``S(t) = (1 + c * (t / scale)**shape)**(-1 / c)`` with positive shape,
scale and ``c``. The internal coordinates are ``log(shape)``,
``log(scale / time_reference)`` and ``log(c)``. Densities returned here are
with respect to the centered time unit; callers restore the event-only
``-log(time_reference)`` Jacobian in right-censored likelihoods.

As ``c`` approaches zero, survival and density converge to their Weibull
limits. These helpers do not define priors or imply BCSTTE fitter parity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite, scalar

_LOG_FLOAT_MIN = float(np.log(np.nextafter(0.0, 1.0)))
_LOG_FLOAT_MAX = float(np.log(np.finfo(np.float64).max))
_SMALL_LOG_PRODUCT = -36.0


def log_odds_rate_components(
    relative_log_times: ArrayLike,
    log_shape: float,
    centered_log_scale: float,
    log_c: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return centered log-density, log-survival and CDF for each time.

    ``relative_log_times`` contains ``log(t / time_reference)`` and
    ``centered_log_scale`` is ``log(scale / time_reference)``. The returned
    log-density is therefore relative to that same time unit. Add
    ``-log(time_reference)`` for each event to obtain an absolute-time log
    likelihood. Censored observations use only the returned log-survival and
    do not receive that Jacobian adjustment.

    The shape and ``c`` parameters must be strictly positive; their natural
    logarithms are accepted to preserve the Weibull limit and avoid exponent
    overflow in intermediate expressions. A sufficiently small ``c *
    (t/scale)**shape`` uses the mathematically equivalent Weibull limit at
    machine precision, avoiding cancellation in division by ``c``.
    """
    if np.iscomplexobj(relative_log_times):
        raise ValueError("relative_log_times must be real")
    times = finite(relative_log_times, "relative_log_times")
    if times.ndim != 1 or times.size == 0:
        raise ValueError("relative_log_times must be a nonempty vector")

    log_shape_value = scalar(log_shape, "log_shape")
    centered_scale = scalar(centered_log_scale, "centered_log_scale")
    log_c_value = scalar(log_c, "log_c")
    if not _LOG_FLOAT_MIN <= log_shape_value <= _LOG_FLOAT_MAX:
        raise ValueError("log_shape must represent a positive finite shape")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        shape = float(np.exp(log_shape_value))
    if not np.isfinite(shape) or shape <= 0:
        raise ValueError("log_shape must represent a positive finite shape")

    with np.errstate(over="ignore", invalid="ignore"):
        log_ratio = times - centered_scale
        log_power = shape * log_ratio
        log_product = log_c_value + log_power
    if np.any(np.isnan(log_ratio)) or np.any(np.isnan(log_power)) or np.any(np.isnan(log_product)):
        raise ArithmeticError("log-odds-rate transformed time became indeterminate")

    # q = log(1 + c * exp(log_power)) / c is -log(S). When the product is
    # below exp(-36), q differs from exp(log_power) by under machine epsilon
    # relatively; taking that limit avoids cancellation for c near zero.
    log_q = np.empty_like(log_product)
    small = log_product < _SMALL_LOG_PRODUCT
    log_q[small] = log_power[small]
    regular = ~small
    if np.any(regular):
        with np.errstate(divide="ignore", invalid="ignore"):
            log_q[regular] = np.log(np.logaddexp(0.0, log_product[regular])) - log_c_value

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        q = np.exp(log_q)
        log_one_plus_product = np.logaddexp(0.0, log_product)
        log_survival = -q
        cdf = -np.expm1(-q)
        log_density = log_shape_value + log_power - times - q - log_one_plus_product
    log_density[np.isposinf(q)] = -np.inf
    if np.any(np.isnan(log_survival)) or np.any(np.isnan(cdf)) or np.any(np.isnan(log_density)):
        raise ArithmeticError("log-odds-rate likelihood evaluation became indeterminate")
    if np.any(np.isposinf(log_density)):
        raise ArithmeticError("log-odds-rate log-density exceeded representable range")

    return (
        np.asarray(log_density, dtype=np.float64),
        np.asarray(log_survival, dtype=np.float64),
        np.asarray(cdf, dtype=np.float64),
    )
