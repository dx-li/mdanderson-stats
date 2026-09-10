"""CONFINT planning for normal confidence-interval length assurance."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammainc, gammaln
from scipy.stats import chi2, t

from ._validation import FloatArray, count, finite, scalar
from .bayesian_monitoring import _integer
from .boin import _owned


def _parameters(
    sample_size: ArrayLike,
    max_length: ArrayLike,
    confidence: ArrayLike,
    target: str,
    sample_size2: ArrayLike | None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    n, length, level = np.broadcast_arrays(
        count(sample_size, "sample_size"),
        finite(max_length, "max_length"),
        finite(confidence, "confidence"),
    )
    if target not in {"mean", "sd", "mean_difference"}:
        raise ValueError("target must be mean, sd, or mean_difference")
    if (sample_size2 is not None) != (target == "mean_difference"):
        raise ValueError("sample_size2 is required only for mean_difference")
    if np.any((n < 2) | (n > 100000000)):
        raise ValueError("sample_size must be in 2..100000000")
    if target == "mean_difference":
        assert sample_size2 is not None
        n, length, level, n2 = np.broadcast_arrays(
            n, length, level, count(sample_size2, "sample_size2")
        )
        if np.any((n2 < 2) | (n2 > 100000000)):
            raise ValueError("sample_size2 must be in 2..100000000")
        df = n + n2 - 2
        log_rate = np.log(1 / n + 1 / n2)
    else:
        df = n - 1
        log_rate = -np.log(np.maximum(n, 1))
    if n.size == 0 or n.size > 2000000 or np.any((n < 2) | (n > 100000000)):
        raise ValueError("require 1..2000000 designs with sample sizes in 2..100000000")
    if np.any(length <= 0) or np.any((level < 1e-6) | (level > 1 - 1e-12)):
        raise ValueError("require positive max_length and confidence in [1e-6,1-1e-12]")
    tail = (1 - level) / 2
    if target == "sd":
        lo, hi = chi2.ppf(tail, df), chi2.isf(tail, df)
        # 1/sqrt(lo)-1/sqrt(hi), avoiding subtraction of inverse square roots.
        log_factor = (
            np.log(hi - lo) - np.log(np.sqrt(hi) + np.sqrt(lo)) - (np.log(lo) + np.log(hi)) / 2
        )
    else:
        log_factor = np.log(2 * t.isf(tail, df)) + (log_rate - np.log(df)) / 2
    return df, np.log(length), log_factor


def confint_normal_probability(
    sample_size: ArrayLike,
    max_length: ArrayLike,
    population_sd: ArrayLike,
    *,
    confidence: ArrayLike = 0.95,
    target: str = "mean",
    sample_size2: ArrayLike | None = None,
) -> FloatArray:
    """Probability that a two-sided normal CI has total length <= max_length.

    Targets: mean (Student t), sd (chi-square), mean_difference (independent
    groups, common unknown variance and pooled Student t). The latter requires
    sample_size2. Inputs broadcast; outputs are read-only, including scalar 0d
    arrays. Population SD is not a standard error. This is width assurance,
    distinct from the CI's confidence level.
    """
    df, log_length, log_factor = _parameters(
        sample_size, max_length, confidence, target, sample_size2
    )
    df, log_length, log_factor, sd = np.broadcast_arrays(
        df, log_length, log_factor, finite(population_sd, "population_sd")
    )
    if sd.size == 0 or sd.size > 2000000 or np.any(sd <= 0):
        raise ValueError("require positive population_sd and at most 2000000 designs")
    log_x = 2 * (log_length - np.log(sd) - log_factor) - np.log(2)
    with np.errstate(over="ignore", under="ignore"):
        result = np.asarray(gammainc(df / 2, np.exp(log_x)))
        # A representable CDF can have an unrepresentably small argument.
        small = log_x < -700
        result[small] = np.exp((df[small] / 2) * log_x[small] - gammaln(df[small] / 2 + 1))
    return _owned(result)


def confint_normal_sd_limit(
    sample_size: ArrayLike,
    max_length: ArrayLike,
    *,
    assurance: ArrayLike = 0.9,
    confidence: ArrayLike = 0.95,
    target: str = "mean",
    sample_size2: ArrayLike | None = None,
) -> FloatArray:
    """Largest population SD attaining the specified CI-length assurance."""
    df, log_length, log_factor = _parameters(
        sample_size, max_length, confidence, target, sample_size2
    )
    df, log_length, log_factor, probability = np.broadcast_arrays(
        df, log_length, log_factor, finite(assurance, "assurance")
    )
    if (
        probability.size == 0
        or probability.size > 2000000
        or np.any((probability < 1e-12) | (probability > 1 - 1e-12))
    ):
        raise ValueError("require assurance in [1e-12,1-1e-12] and at most 2000000 designs")
    with np.errstate(over="ignore", under="ignore"):
        result = np.exp(log_length - log_factor - np.log(chi2.ppf(probability, df)) / 2)
    if np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise ArithmeticError("population SD limit is outside positive float64 range")
    return _owned(result)


def confint_normal_sample_size(
    max_length: float,
    population_sd: float,
    *,
    assurance: float = 0.9,
    confidence: float = 0.95,
    target: str = "mean",
    max_sample_size: int = 100000000,
) -> int:
    """Smallest integer n attaining assurance; per-group n for mean_difference.

    Two-sample planning uses equal group sizes. Unequal fixed group sizes are
    supported by the probability and SD-limit functions. Raises when the search
    bound is inadequate, rather than returning an underpowered boundary value.
    """
    length, sd = scalar(max_length, "max_length"), scalar(population_sd, "population_sd")
    level, probability = scalar(confidence, "confidence"), scalar(assurance, "assurance")
    limit = _integer(max_sample_size, "max_sample_size")
    if not 2 <= limit <= 100000000 or not 1e-12 <= probability <= 1 - 1e-12:
        raise ValueError("require max_sample_size in 2..100000000 and assurance in [1e-12,1-1e-12]")

    def sufficient(n: int) -> bool:
        return (
            float(
                confint_normal_probability(
                    n,
                    length,
                    sd,
                    confidence=level,
                    target=target,
                    sample_size2=n if target == "mean_difference" else None,
                )
            )
            >= probability
        )

    if sufficient(2):
        return 2
    lo, hi = 2, min(4, limit)
    while not sufficient(hi):
        if hi == limit:
            raise ValueError("requested assurance is not attained within max_sample_size")
        lo, hi = hi, min(2 * hi, limit)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if sufficient(mid):
            hi = mid
        else:
            lo = mid
    return hi
