"""CONFINT one-binomial Clopper–Pearson width assurance and inversion."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.stats import binom

from ._validation import FloatArray, count, finite, scalar
from .bayesian_monitoring import _integer
from .boin import _owned
from .intervals import binomial_interval


def _inputs(n: ArrayLike, length: ArrayLike, confidence: ArrayLike) -> tuple[FloatArray, ...]:
    n, length, confidence = np.broadcast_arrays(
        count(n, "sample_size"), finite(length, "max_length"), finite(confidence, "confidence")
    )
    if not 0 < n.size <= 2000000 or np.any((n < 1) | (n > 1000000)):
        raise ValueError("require 1..2000000 designs with integer sample sizes in 1..1000000")
    if np.any((length <= 0) | (length > 1)) or np.any(
        (confidence < 1e-6) | (confidence > 1 - 1e-12)
    ):
        raise ValueError("require max_length in (0,1] and confidence in [1e-6,1-1e-12]")
    return n, length, confidence


def _width(k: ArrayLike, n: ArrayLike, confidence: ArrayLike) -> FloatArray:
    lo, hi = binomial_interval(k, n, confidence)
    return hi - lo


def _cutoff(n: FloatArray, length: FloatArray, confidence: FloatArray) -> FloatArray:
    # -1 means no successful outcome; floor(n/2) means every outcome qualifies.
    low = np.full(n.shape, -1.0)
    high = np.asarray(np.floor(n / 2) + 1)
    while np.any(high - low > 1):
        active = high - low > 1
        mid = np.floor((low[active] + high[active]) / 2)
        good = _width(mid, n[active], confidence[active]) <= length[active]
        low[active] = np.where(good, mid, low[active])
        high[active] = np.where(good, high[active], mid)
    return low


def _mass(k: ArrayLike, n: ArrayLike, p: ArrayLike) -> FloatArray:
    k, n, p = np.broadcast_arrays(k, n, p)
    result = np.asarray(binom.cdf(k, n, p) + binom.sf(n - k - 1, n, p))
    result[k >= np.floor(n / 2)] = 1
    return np.minimum(result, 1)


def confint_binomial_probability(
    sample_size: ArrayLike,
    max_length: ArrayLike,
    event_probability: ArrayLike,
    *,
    confidence: ArrayLike = 0.95,
) -> FloatArray:
    """Probability the equal-tail Clopper–Pearson interval has total length <=L.

    Inputs broadcast; outputs are read-only arrays. Exact integer cutoffs include
    zero/all-event samples and handle all-qualifying outcomes without overlap.
    """
    n, length, level = _inputs(sample_size, max_length, confidence)
    n, length, level, p = np.broadcast_arrays(
        n, length, level, finite(event_probability, "event_probability")
    )
    if not 0 < p.size <= 2000000 or np.any((p < 0) | (p > 1)):
        raise ValueError("require event_probability in [0,1] and 1..2000000 designs")
    return _owned(_mass(_cutoff(n, length, level), n, p))


def _assurance(value: float) -> float:
    probability = scalar(value, "assurance")
    if not 1e-12 <= probability <= 1 - 1e-12:
        raise ValueError("assurance must be in [1e-12,1-1e-12]")
    return probability


def confint_binomial_length(
    sample_size: int, event_probability: float, *, assurance: float = 0.9, confidence: float = 0.95
) -> float:
    """Smallest attainable CI length whose cumulative probability >=assurance."""
    n, _, level = _inputs(scalar(sample_size, "sample_size"), 1, scalar(confidence, "confidence"))
    p, target = scalar(event_probability, "event_probability"), _assurance(assurance)
    if not 0 <= p <= 1:
        raise ValueError("event_probability must be in [0,1]")
    low, high = -1, int(n) // 2
    while high - low > 1:
        mid = (low + high) // 2
        if float(_mass(mid, n, p)) >= target:
            high = mid
        else:
            low = mid
    return float(_width(high, n, level))


def confint_binomial_event_limit(
    sample_size: int, max_length: float, *, assurance: float = 0.9, confidence: float = 0.95
) -> float | None:
    """Return p* so assurance holds for p<=p* or p>=1-p*.

    None means no event probability can attain assurance; 0.5 means all can.
    """
    n, length, level = _inputs(
        scalar(sample_size, "sample_size"),
        scalar(max_length, "max_length"),
        scalar(confidence, "confidence"),
    )
    target = _assurance(assurance)
    k = _cutoff(n, length, level)
    if k < 0:
        return None
    if float(_mass(k, n, 0.5)) >= target:
        return 0.5
    return float(brentq(lambda p: float(_mass(k, n, p)) - target, 0, 0.5, xtol=1e-14))


def confint_binomial_sample_size(
    max_length: float,
    event_probability: float,
    *,
    assurance: float = 0.9,
    confidence: float = 0.95,
    max_sample_size: int = 1000000,
) -> int:
    """First integer sample size attaining assurance, scanning in NumPy batches.

    Assurance is not monotone in n. A continuous root or binary search can miss
    an earlier qualifying size. This function checks every n from 1 to the first
    success (up to max_sample_size), in batches of at most 4096 designs.
    """
    length, p = scalar(max_length, "max_length"), scalar(event_probability, "event_probability")
    level, target = scalar(confidence, "confidence"), _assurance(assurance)
    limit = _integer(max_sample_size, "max_sample_size")
    if not 1 <= limit <= 1000000:
        raise ValueError("max_sample_size must be in 1..1000000")
    # Exponentially growing early batches avoid computing thousands of unused n.
    first, batch = 1, 32
    while first <= limit:
        sizes = np.arange(first, min(first + batch, limit + 1))
        probabilities = confint_binomial_probability(sizes, length, p, confidence=level)
        qualifying = np.flatnonzero(probabilities >= target)
        if qualifying.size:
            return int(sizes[qualifying[0]])
        first += batch
        batch = min(2 * batch, 4096)
    raise ValueError("requested assurance is not attained within max_sample_size")
