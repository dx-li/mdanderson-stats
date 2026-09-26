"""Power approximations for STPLAN's continuous-outcome procedures."""

from math import prod
from typing import Final

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import chi2, f, nct, t

from ._validation import FloatArray, finite, scalar

_MAX_CASES: Final = 200_000


def _inputs(*values: tuple[ArrayLike, str]) -> tuple[FloatArray, ...]:
    """Validate finite numeric inputs and bounded broadcast size."""
    raw = tuple(np.asarray(value) for value, _ in values)
    if any(array.size > _MAX_CASES for array in raw):
        raise ValueError(f"each input is limited to {_MAX_CASES} values")
    try:
        shape = np.broadcast_shapes(*(array.shape for array in raw))
    except ValueError as exc:
        raise ValueError("inputs cannot be broadcast together") from exc
    if prod(shape) > _MAX_CASES:
        raise ValueError(f"broadcast result exceeds {_MAX_CASES} cases")
    arrays = tuple(finite(array, name) for array, (_, name) in zip(raw, values))
    return tuple(np.broadcast_arrays(*arrays))


def _alpha(alpha: float, sides: int) -> float:
    alpha = scalar(alpha, "alpha")
    if isinstance(sides, (bool, np.bool_)) or sides not in (1, 2) or not 0 < alpha < 0.5:
        raise ValueError("require 0 < alpha < 0.5 and sides in {1, 2}")
    return alpha / sides


def _power(value: ArrayLike) -> FloatArray:
    result = np.asarray(value, dtype=float)
    tolerance = 8 * np.finfo(float).eps
    if np.any(~np.isfinite(result) | (result < -tolerance) | (result > 1 + tolerance)):
        raise ArithmeticError("power calculation is not representable")
    return np.clip(result, 0.0, 1.0)


def stplan_normal_one_sample_power(
    difference: ArrayLike,
    sd: ArrayLike,
    sample_size: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """STPLAN one-sample unknown-variance normal mean test power."""
    a = _alpha(alpha, sides)
    delta, sigma, n = _inputs((difference, "difference"), (sd, "sd"), (sample_size, "sample_size"))
    if np.any(sigma <= 0) or np.any(n < 2):
        raise ValueError("sd must be positive and sample_size at least 2")
    df = n - 1
    nc = (np.abs(delta) / sigma) * np.sqrt(n)
    return _power(nct.sf(t.isf(a, df), df, nc))


def stplan_normal_two_sample_power(
    difference: ArrayLike,
    sd: ArrayLike,
    n1: ArrayLike,
    n2: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """STPLAN pooled equal-variance independent two-sample t-test power."""
    a = _alpha(alpha, sides)
    delta, sigma, first, second = _inputs(
        (difference, "difference"), (sd, "sd"), (n1, "n1"), (n2, "n2")
    )
    if np.any(sigma <= 0) or np.any(first < 2) or np.any(second < 2):
        raise ValueError("sd must be positive and each sample size at least 2")
    df = first + second - 2
    information = np.sqrt(1 / first + 1 / second)
    nc = (np.abs(delta) / sigma) / information
    return _power(nct.sf(t.isf(a, df), df, nc))


def stplan_welch_two_sample_power(
    difference: ArrayLike,
    sd1: ArrayLike,
    sd2: ArrayLike,
    n1: ArrayLike,
    n2: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """STPLAN unequal-variance two-sample t-test with Satterthwaite df."""
    a = _alpha(alpha, sides)
    delta, s1, s2, first, second = _inputs(
        (difference, "difference"), (sd1, "sd1"), (sd2, "sd2"), (n1, "n1"), (n2, "n2")
    )
    if np.any(s1 <= 0) or np.any(s2 <= 0) or np.any(first < 2) or np.any(second < 2):
        raise ValueError("standard deviations must be positive and sample sizes at least 2")
    e1, e2 = s1 / np.sqrt(first), s2 / np.sqrt(second)
    scale = np.maximum(e1, e2)
    w1, w2 = (e1 / scale) ** 2, (e2 / scale) ** 2
    df = (w1 + w2) ** 2 / (w1**2 / (first - 1) + w2**2 / (second - 1))
    nc = (np.abs(delta) / scale) / np.hypot(np.sqrt(w1), np.sqrt(w2))
    return _power(nct.sf(t.isf(a, df), df, nc))


def stplan_lognormal_two_sample_power(
    mean1: ArrayLike,
    mean2: ArrayLike,
    cv: ArrayLike,
    n1: ArrayLike,
    n2: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """STPLAN log-normal test, parameterized by arithmetic means and common CV."""
    m1, m2, coeff, first, second = _inputs(
        (mean1, "mean1"), (mean2, "mean2"), (cv, "cv"), (n1, "n1"), (n2, "n2")
    )
    if np.any(m1 <= 0) or np.any(m2 <= 0) or np.any(coeff <= 0):
        raise ValueError("means and cv must be positive")
    # log1p avoids loss of precision when the coefficient of variation is small.
    log_cv = np.log(coeff)
    log_sd = np.sqrt(np.logaddexp(0.0, 2 * log_cv))
    log_mean_difference = np.log(m1) - np.log(m2)
    return stplan_normal_two_sample_power(
        log_mean_difference, log_sd, first, second, alpha=alpha, sides=sides
    )


def stplan_exponential_one_sample_power(
    ratio: ArrayLike,
    sample_size: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """STPLAN one-sample exponential-mean test power (ratio=alternative/null)."""
    a = _alpha(alpha, sides)
    r, n = _inputs((ratio, "ratio"), (sample_size, "sample_size"))
    if np.any(r <= 0) or np.any(n <= 0):
        raise ValueError("ratio and sample_size must be positive")
    df = 2 * n
    upper = r > 1
    critical = np.where(upper, chi2.isf(a, df), chi2.ppf(a, df))
    value = np.where(upper, chi2.sf(critical / r, df), chi2.cdf(critical / r, df))
    return _power(value)


def stplan_exponential_two_sample_power(
    mean1: ArrayLike,
    mean2: ArrayLike,
    n1: ArrayLike,
    n2: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """STPLAN exponential two-sample F-ratio power (dominant tail)."""
    a = _alpha(alpha, sides)
    m1, m2, first, second = _inputs((mean1, "mean1"), (mean2, "mean2"), (n1, "n1"), (n2, "n2"))
    if np.any(m1 <= 0) or np.any(m2 <= 0) or np.any(first <= 0) or np.any(second <= 0):
        raise ValueError("means and sample sizes must be positive")
    reverse = m1 > m2
    log_ratio = np.log(np.maximum(m1, m2)) - np.log(np.minimum(m1, m2))
    dfn = np.where(reverse, first, second) * 2
    dfd = np.where(reverse, second, first) * 2
    reciprocal_cutoff = f.ppf(a, dfd, dfn)
    if np.any(~np.isfinite(reciprocal_cutoff) | (reciprocal_cutoff <= 0)):
        raise ArithmeticError("F critical value is not representable at this alpha")
    log_argument = log_ratio + np.log(reciprocal_cutoff)
    float_info = np.finfo(float)
    min_log = np.log(np.nextafter(0.0, 1.0))
    max_log = np.log(float_info.max)
    argument = np.exp(np.clip(log_argument, min_log, max_log))
    argument = np.where(log_argument < min_log, 0.0, argument)
    argument = np.where(log_argument > max_log, np.inf, argument)
    return _power(f.cdf(argument, dfd, dfn))
