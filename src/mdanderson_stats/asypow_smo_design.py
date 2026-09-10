"""General-design SMO, including original ASYPOW multivariable logistic models."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from ._validation import finite, scalar
from .asypow_constraints import _components
from .asypow_smo import SMOPower
from .asypow_smo_regression import _fit_regression
from .boin import _owned


def asypow_smo_design(
    coefficients: ArrayLike,
    design: ArrayLike,
    *,
    constraints: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    family: str = "logistic",
    observations: ArrayLike = 1,
    duration: float | None = None,
    subtract_df: bool = True,
    tolerance: float = 1e-8,
) -> SMOPower:
    """SMO for an explicit design matrix; no intercept is automatically added.

    Each row has one covariate vector; one-based constraints index coefficients.
    The default logistic model is ASYPOW's noncent.mvlogistic. The shared engine
    also supports cloglog, Poisson and uniformly censored exponential survival.
    loglinear uses positive multiplicative coefficients and returns their null
    values on that same scale; its probabilities must remain strictly below one.
    """
    theta = finite(coefficients, "coefficients")
    if theta.ndim != 1 or not 1 <= theta.size <= 500:
        raise ValueError("coefficients must be a vector of length 1..500")
    x = finite(design, "design")
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != theta.size or not 1 <= x.size <= 1_000_000:
        raise ValueError("design must match coefficients and have 1..1000000 entries")
    count = np.broadcast_to(finite(observations, "observations"), (len(x),))
    if np.any(count < 0) or not np.any(count > 0):
        raise ValueError("observations must be nonnegative with a positive total")
    if family not in ("logistic", "cloglog", "poisson", "exponential", "loglinear"):
        raise ValueError("family must be logistic, cloglog, poisson, exponential, or loglinear")
    length = None
    if family == "exponential":
        if duration is None or scalar(duration, "duration") <= 0:
            raise ValueError("positive scalar duration is required for exponential survival")
        length = np.array([duration])
    elif duration is not None:
        raise ValueError("duration is only used for exponential survival")
    if family == "loglinear":
        lo = np.broadcast_to(finite(lower, "lower"), theta.shape)
        hi = np.broadcast_to(finite(upper, "upper"), theta.shape)
        if np.any(theta <= 0) or np.any(lo <= 0) or np.any(hi <= 0):
            raise ValueError("log-linear coefficients and bounds must be positive")
        # A monotone log transform preserves fixed/equality hypotheses and df.
        _components(constraints, theta.size)
        rows = finite(constraints, "constraints").reshape(-1, 3).copy()
        fixed = rows[:, 0] == 1
        if np.any(rows[fixed, 2] <= 0):
            raise ValueError("fixed log-linear coefficients must be positive")
        rows[fixed, 2] = np.log(rows[fixed, 2])
        constraints, lower, upper = rows, np.log(lo), np.log(hi)
        theta = np.log(theta)
    used = count > 0
    log_weight = np.log(count[used])
    log_weight -= logsumexp(log_weight)
    result = _fit_regression(
        theta[None, :],
        x[used],
        np.zeros(np.count_nonzero(used), dtype=np.intp),
        log_weight,
        family,
        length,
        constraints,
        lower,
        upper,
        subtract_df,
        tolerance,
    )
    if family == "loglinear":
        return SMOPower(
            result.divergence_per_observation,
            result.degrees_of_freedom,
            _owned(np.exp(result.null_parameters)),
            result.subtract_df,
        )
    return result
