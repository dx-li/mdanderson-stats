"""ASYPOW information for ordered categorical outcomes."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from ._validation import FloatArray, finite
from .boin import _owned


def asypow_ordinal_information(cumulative: ArrayLike, *, group_size: ArrayLike = 1) -> FloatArray:
    """Raw cumulative-probability information; rows are independent groups.

    Supply K-1 strictly increasing cumulative probabilities, excluding the
    terminal 1. Group allocations are normalized across groups.
    """
    p = finite(cumulative, "cumulative")
    if p.ndim == 1:
        p = p[None, :]
    if p.ndim != 2 or not 1 <= p.size <= 500 or not p.shape[1]:
        raise ValueError("cumulative needs nonempty group rows and at most 500 parameters")
    mass = np.diff(np.column_stack((np.zeros(len(p)), p, np.ones(len(p)))), axis=1)
    if np.any(mass <= 0):
        raise ValueError("cumulative probabilities must increase strictly within (0,1)")
    weight = np.broadcast_to(finite(group_size, "group_size"), (len(p),))
    if np.any(weight <= 0):
        raise ValueError("group_size must be positive")
    log_weight = np.log(weight) - logsumexp(np.log(weight))
    result = np.zeros((p.size, p.size))
    k = p.shape[1]
    with np.errstate(over="ignore", under="ignore"):
        inverse = np.exp(log_weight[:, None] - np.log(mass))
    for group, inv in enumerate(inverse):
        block = np.diag(inv[:-1] + inv[1:])
        if k > 1:
            block += np.diag(-inv[1:-1], 1) + np.diag(-inv[1:-1], -1)
        result[group * k : (group + 1) * k, group * k : (group + 1) * k] = block
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("ordinal information exceeds floating-point range")
    return _owned(result)


def _category_logs(eta: FloatArray, link: str) -> tuple[FloatArray, FloatArray]:
    """Log category masses and cumulative-probability derivatives."""
    gap = np.diff(eta)
    if np.any(gap <= 0):
        raise ValueError("ordered thresholds are not distinguishable at this predictor")
    if link == "logistic":
        log_cdf, log_sf = -np.logaddexp(0, -eta), -np.logaddexp(0, eta)
        interior = log_cdf[1:] + log_sf[:-1] + np.log(-np.expm1(-gap))
        return np.r_[log_cdf[0], interior, log_sf[-1]], log_cdf + log_sf
    if link != "cloglog":
        raise ValueError("link must be logistic or cloglog")
    # Finite log probabilities are needed for derivative scaling, even where
    # their exponentials underflow. Reject beyond this representable log range.
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        z = np.exp(eta)
        first = eta[0] if eta[0] < -36 else np.log(-np.expm1(-z[0]))
        log_delta = eta[1:] + np.log(-np.expm1(-gap))
        delta = np.exp(log_delta)
        log_drop = np.where(log_delta < -36, log_delta, np.log(-np.expm1(-delta)))
        mass = np.r_[first, -z[:-1] + log_drop, -z[-1]]
    if not np.all(np.isfinite(mass)):
        raise ArithmeticError("cloglog category log probabilities exceed floating-point range")
    return mass, eta - z


def asypow_ordinal_regression_information(
    parameters: ArrayLike,
    covariates: ArrayLike,
    *,
    quadratic: bool = False,
    link: str = "logistic",
    observations: ArrayLike = 1,
    group_size: ArrayLike = 1,
) -> FloatArray:
    """Ordinal cumulative-link design with ordered intercepts and shared slopes.

    Each parameter row contains K-1 intercepts, then the coefficient of x and,
    for quadratic=True, x^2. Covariate/allocation broadcasting matches ASYPOW's
    regression designs. Output coefficients are ordered group by group.
    """
    if not isinstance(quadratic, (bool, np.bool_)):
        raise ValueError("quadratic must be boolean")
    slopes = 2 if quadratic else 1
    theta = finite(parameters, "parameters")
    if theta.ndim == 1:
        theta = theta[None, :]
    if theta.ndim != 2 or not slopes < theta.shape[1] or not 1 <= theta.size <= 500:
        raise ValueError("parameters need ordered intercepts and slope(s), up to 500 coefficients")
    groups, size = theta.shape
    thresholds = size - slopes
    if np.any(np.diff(theta[:, :thresholds], axis=1) <= 0):
        raise ValueError("intercepts must be strictly increasing")
    x = finite(covariates, "covariates")
    if x.ndim == 1:
        x = np.broadcast_to(x, (groups, x.size))
    if x.ndim != 2 or x.shape[0] != groups or not 1 <= x.size <= 10000:
        raise ValueError("covariates need one row per group and 1..10000 entries")
    count = np.broadcast_to(finite(observations, "observations"), x.shape)
    relative = np.broadcast_to(finite(group_size, "group_size"), (groups,))
    if np.any(count < 0) or np.any(relative <= 0) or np.any(np.sum(count > 0, axis=1) == 0):
        raise ValueError("allocations must be nonnegative with positive mass in every group")
    with np.errstate(divide="ignore"):
        allocation = np.log(count) + np.log(relative)[:, None]
    allocation -= logsumexp(allocation)
    result = np.zeros((theta.size, theta.size))
    for group in range(groups):
        block = np.zeros((size, size))
        for point in np.flatnonzero(count[group] > 0):
            value = x[group, point]
            with np.errstate(over="ignore", invalid="ignore"):
                predictors = np.array([value, value * value]) if quadratic else np.array([value])
                eta = theta[group, :thresholds] + predictors @ theta[group, thresholds:]
            if not np.all(np.isfinite(eta)) or not np.all(np.isfinite(predictors)):
                raise ArithmeticError("ordinal predictor overflow; rescale covariates")
            log_mass, log_derivative = _category_logs(eta, link)
            # Rows are category gradients divided by sqrt(category probability),
            # with sqrt(allocation) folded in before exponentiating.
            features = np.zeros((thresholds + 1, size))
            with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
                upper = log_derivative - log_mass[:-1] / 2 + allocation[group, point] / 2
                lower = log_derivative - log_mass[1:] / 2 + allocation[group, point] / 2
                rows = np.arange(thresholds)
                features[rows, rows] = np.exp(upper)
                features[rows + 1, rows] = -np.exp(lower)
                log_x = np.log(np.abs(predictors))
                features[:-1, thresholds:] += np.sign(predictors) * np.exp(upper[:, None] + log_x)
                features[1:, thresholds:] -= np.sign(predictors) * np.exp(lower[:, None] + log_x)
                block += features.T @ features
        result[group * size : (group + 1) * size, group * size : (group + 1) * size] = block
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("ordinal information exceeds floating-point range")
    return _owned(result)
