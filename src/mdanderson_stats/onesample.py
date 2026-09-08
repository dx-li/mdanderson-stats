"""ONESAMPLE's inclusive binomial and Poisson one-sided tests."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betainc, betaincc, gammainc, gammaincc

from ._validation import FloatArray, count, finite


@dataclass(frozen=True)
class OneSampleTest:
    """p_less=P(X<=observed), p_greater=P(X>=observed) under the null."""

    estimate: FloatArray
    null_value: FloatArray
    p_less: FloatArray
    p_greater: FloatArray
    legacy_cutoffs: bool


def binomial_test(
    successes: ArrayLike,
    trials: ArrayLike,
    null_probability: ArrayLike,
    *,
    legacy_cutoffs: bool = False,
) -> OneSampleTest:
    """Inclusive one-sided binomial p-values, broadcasting independent tests.

    Exact tails are default. legacy_cutoffs reproduces ONESAMPLE's forced zero
    upper tail at p0<=1e-10 and lower tail at p0>=1-1e-10, except trivial boundaries.
    No two-sided p-value convention is inferred. A positive total is required.
    """
    k, n, p = np.broadcast_arrays(
        count(successes, "successes"),
        count(trials, "trials"),
        finite(null_probability, "null_probability"),
    )
    if np.any((n <= 0) | (k > n) | (p < 0) | (p > 1)):
        raise ValueError("Require 0 <= successes <= positive trials and null probability in [0,1]")
    if not isinstance(legacy_cutoffs, bool):
        raise ValueError("legacy_cutoffs must be boolean")
    greater, less = np.ones(k.shape), np.ones(k.shape)
    positive, below = k > 0, k < n
    greater[positive] = betainc(k[positive], n[positive] - k[positive] + 1, p[positive])
    less[below] = betaincc(k[below] + 1, n[below] - k[below], p[below])
    if legacy_cutoffs:
        greater[positive & (p <= 1e-10)] = 0
        less[below & (p >= 1 - 1e-10)] = 0
    return OneSampleTest(k / n, p, less, greater, legacy_cutoffs)


def poisson_test(
    events: ArrayLike,
    null_rate: ArrayLike,
    *,
    exposure: ArrayLike = 1,
    legacy_cutoffs: bool = False,
) -> OneSampleTest:
    """Inclusive one-sided Poisson rate-test p-values with a known exposure.

    The null mean is null_rate*exposure. legacy_cutoffs reproduces the source's
    forced zero upper tail for positive counts when that mean is <=1e-10.
    """
    k, rate, time = np.broadcast_arrays(
        count(events, "events"),
        finite(null_rate, "null_rate"),
        finite(exposure, "exposure"),
    )
    if np.any((rate < 0) | (time <= 0)):
        raise ValueError("null_rate must be nonnegative and exposure positive")
    if not isinstance(legacy_cutoffs, bool):
        raise ValueError("legacy_cutoffs must be boolean")
    with np.errstate(over="ignore"):
        mean = rate * time
        estimate = k / time
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(estimate)):
        raise ValueError("Null mean or estimated rate exceeds numerical range")
    greater = np.ones(k.shape)
    positive = k > 0
    greater[positive] = gammainc(k[positive], mean[positive])
    less = np.asarray(gammaincc(k + 1, mean))
    if legacy_cutoffs:
        greater[positive & (mean <= 1e-10)] = 0
    return OneSampleTest(estimate, rate, less, greater, legacy_cutoffs)
