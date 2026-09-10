"""Per-observation information for ASYPOW's independent-group models."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from ._validation import FloatArray, finite
from .boin import _owned
from .survival_sample_size import exponential_event_probability


def asypow_group_information(
    parameters: ArrayLike,
    *,
    model: str = "binomial",
    group_size: ArrayLike = 1,
    duration: ArrayLike | None = None,
) -> FloatArray:
    """Information for one subject spread over binomial, Poisson or survival groups.

    Parameters are binomial probabilities, Poisson means, or exponential rates.
    Exponential survival assumes uniform entry during duration and administrative
    censoring at its end. Positive relative group sizes are normalized to sum 1.
    """
    theta = finite(parameters, "parameters")
    if theta.ndim != 1 or not 1 <= theta.size <= 500 or np.any(theta <= 0):
        raise ValueError("parameters must be a positive vector of length 1..500")
    weights = np.broadcast_to(finite(group_size, "group_size"), theta.shape)
    if np.any(weights <= 0):
        raise ValueError("group sizes must be positive")
    log_weight = np.log(weights) - logsumexp(np.log(weights))
    if model != "exponential" and duration is not None:
        raise ValueError("duration is only used for exponential survival")
    if model == "binomial":
        if np.any(theta >= 1):
            raise ValueError("binomial probabilities must be strictly below one")
        log_info = log_weight - np.log(theta) - np.log1p(-theta)
    elif model == "poisson":
        log_info = log_weight - np.log(theta)
    elif model == "exponential":
        if duration is None:
            raise ValueError("duration is required for exponential survival")
        length = np.broadcast_to(finite(duration, "duration"), theta.shape)
        if np.any(length <= 0):
            raise ValueError("duration must be positive")
        probability = exponential_event_probability(theta, length, 0)
        with np.errstate(divide="ignore"):
            log_p = np.where(
                probability > 0, np.log(probability), np.log(theta) + np.log(length) - np.log(2)
            )
        log_info = log_weight + log_p - 2 * np.log(theta)
    else:
        raise ValueError("model must be binomial, poisson, or exponential")
    with np.errstate(over="ignore", under="ignore"):
        diagonal = np.exp(log_info)
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0):
        raise ArithmeticError("information is not representable; rescale parameters or allocations")
    return _owned(np.diag(diagonal))
