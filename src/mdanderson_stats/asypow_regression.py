"""ASYPOW information for linear/quadratic regression designs."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from ._validation import FloatArray, finite
from .boin import _owned
from .survival_sample_size import exponential_event_probability


def _log_information(eta: FloatArray, family: str, duration: FloatArray | None) -> FloatArray:
    if family == "logistic":
        return -np.logaddexp(0, eta) - np.logaddexp(0, -eta)
    if family == "poisson":
        return eta
    if family == "cloglog":
        result = np.full_like(eta, -np.inf)
        small = eta < -36
        result[small] = eta[small]
        ordinary = ~small & (eta < np.log(10_000))
        z = np.exp(eta[ordinary])
        result[ordinary] = 2 * eta[ordinary] - z - np.log(-np.expm1(-z))
        return result
    if family == "exponential":
        assert duration is not None
        log_z = eta + np.log(duration)
        result = np.zeros_like(eta)
        small = log_z < -36
        result[small] = log_z[small] - np.log(2)
        ordinary = ~small & (log_z < 36)
        z = np.exp(log_z[ordinary])
        result[ordinary] = np.log(exponential_event_probability(1, z, 0))
        return result
    raise ValueError("family must be logistic, cloglog, poisson, or exponential")


def asypow_regression_information(
    parameters: ArrayLike,
    covariates: ArrayLike,
    *,
    family: str = "logistic",
    observations: ArrayLike = 1,
    group_size: ArrayLike = 1,
    duration: ArrayLike | None = None,
) -> FloatArray:
    """Information per observation for ASYPOW linear/quadratic regressions.

    Parameter rows are independent groups, each with intercept/slope and optional
    quadratic coefficient. A covariate vector is shared across groups; a matrix
    has one row per group. observations broadcast over that matrix; group_size
    multiplies each group's observations BEFORE normalization over all groups.
    Coefficients in the result are ordered group by group (row-major flattening).
    """
    theta = finite(parameters, "parameters")
    if theta.ndim == 1:
        theta = theta[None, :]
    if theta.ndim != 2 or theta.shape[1] not in (2, 3) or not 1 <= theta.size <= 500:
        raise ValueError("parameters need 2 or 3 columns and at most 500 total coefficients")
    groups, coefficients = theta.shape
    x = finite(covariates, "covariates")
    if x.ndim == 1:
        x = np.broadcast_to(x, (groups, x.size))
    if x.ndim != 2 or x.shape[0] != groups or not 1 <= x.size <= 1_000_000:
        raise ValueError("covariates need one row per group and 1..1000000 total entries")
    count = np.broadcast_to(finite(observations, "observations"), x.shape)
    relative = np.broadcast_to(finite(group_size, "group_size"), (groups,))
    if np.any(count < 0) or np.any(relative <= 0) or np.any(np.sum(count > 0, axis=1) == 0):
        raise ValueError("group sizes must be positive; every group needs positive observations")
    if family == "exponential":
        if duration is None:
            raise ValueError("duration is required for exponential survival")
        length = np.broadcast_to(finite(duration, "duration"), (groups,))
        if np.any(length <= 0):
            raise ValueError("duration must be positive")
    else:
        length = None
        if duration is not None:
            raise ValueError("duration is only used for exponential survival")
    with np.errstate(divide="ignore"):
        allocation = np.log(count) + np.log(relative)[:, None]
    allocation -= logsumexp(allocation)
    result = np.zeros((theta.size, theta.size))
    for group in range(groups):
        used = count[group] > 0
        with np.errstate(over="ignore", invalid="ignore"):
            design = np.vander(x[group, used], N=coefficients, increasing=True)
            eta = design @ theta[group]
        if not np.all(np.isfinite(design)) or not np.all(np.isfinite(eta)):
            raise ArithmeticError("polynomial design or predictor overflow; rescale covariates")
        log_info = _log_information(eta, family, None if length is None else length[group])
        log_weight = allocation[group, used] + log_info
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            features = np.sign(design) * np.exp(np.log(np.abs(design)) + log_weight[:, None] / 2)
            block = features.T @ features
        if not np.all(np.isfinite(block)):
            raise ArithmeticError("regression information exceeds floating-point range")
        start = group * coefficients
        result[start : start + coefficients, start : start + coefficients] = block
    return _owned(result)
