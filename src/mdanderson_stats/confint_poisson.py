"""CONFINT Poisson rate-interval width assurance with discrete event counts."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammainccinv
from scipy.stats import poisson

from ._validation import FloatArray, finite, scalar
from .bayesian_monitoring import _integer
from .boin import _owned
from .confint_binomial import _assurance
from .intervals import poisson_interval


def _inputs(time: ArrayLike, length: ArrayLike, confidence: ArrayLike) -> tuple[FloatArray, ...]:
    time, length, level = np.broadcast_arrays(
        finite(time, "exposure"), finite(length, "max_length"), finite(confidence, "confidence")
    )
    if not 0 < time.size <= 2000000 or np.any(time <= 0) or np.any(length <= 0):
        raise ValueError("require positive exposure/length and 1..2000000 designs")
    if np.any((level < 1e-6) | (level > 1 - 1e-12)):
        raise ValueError("confidence must be in [1e-6,1-1e-12]")
    return time, length, level


def _width(counts: ArrayLike, confidence: ArrayLike) -> FloatArray:
    lo, hi = poisson_interval(counts, confidence)
    return hi - lo


def _cutoff(time: FloatArray, length: FloatArray, level: FloatArray) -> FloatArray:
    with np.errstate(over="ignore", under="ignore"):
        extent = time * length
    if np.any(_width(100000001, level) <= extent):
        raise ValueError("qualifying event cutoff exceeds 100000000")
    low, high = np.full(time.shape, -1.0), np.full(time.shape, 100000001.0)
    while np.any(high - low > 1):
        active = high - low > 1
        mid = np.floor((low[active] + high[active]) / 2)
        good = _width(mid, level[active]) <= extent[active]
        low[active] = np.where(good, mid, low[active])
        high[active] = np.where(good, high[active], mid)
    return low


def confint_poisson_probability(
    exposure: ArrayLike,
    max_length: ArrayLike,
    event_rate: ArrayLike,
    *,
    confidence: ArrayLike = 0.95,
) -> FloatArray:
    """Probability the equal-tail Garwood rate interval has total length <=L.

    Exposure may be time or another unit of Poisson observation. The true rate
    is nonnegative. Inputs broadcast and outputs are read-only arrays.
    """
    time, length, level = _inputs(exposure, max_length, confidence)
    time, length, level, rate = np.broadcast_arrays(
        time, length, level, finite(event_rate, "event_rate")
    )
    if not 0 < rate.size <= 2000000 or np.any(rate < 0):
        raise ValueError("require nonnegative event_rate and 1..2000000 designs")
    k = _cutoff(time, length, level)
    with np.errstate(over="ignore", under="ignore"):
        mean = time * rate
    return _owned(poisson.cdf(k, mean))


def confint_poisson_length(
    exposure: float, event_rate: float, *, assurance: float = 0.9, confidence: float = 0.95
) -> float:
    """Smallest attainable rate-CI length with cumulative probability >=assurance."""
    time, _, level = _inputs(scalar(exposure, "exposure"), 1, scalar(confidence, "confidence"))
    rate, target = scalar(event_rate, "event_rate"), _assurance(assurance)
    if rate < 0:
        raise ValueError("event_rate must be nonnegative")
    with np.errstate(over="ignore", under="ignore"):
        k = float(poisson.ppf(target, time * rate))
    if not np.isfinite(k) or not 0 <= k <= 100000000:
        raise ValueError("required event quantile exceeds 100000000")
    with np.errstate(over="ignore", under="ignore"):
        result = float(_width(k, level) / time)
    if not np.isfinite(result) or result <= 0:
        raise ArithmeticError("CI length is outside positive float64 range")
    # Ensure the returned representable length includes its defining outcome.
    if result * time < _width(k, level):
        with np.errstate(over="ignore"):
            result = float(np.nextafter(result, np.inf))
        if not np.isfinite(result):
            raise ArithmeticError("rounded CI length is outside positive float64 range")
    return result


def confint_poisson_rate_limit(
    exposure: float, max_length: float, *, assurance: float = 0.9, confidence: float = 0.95
) -> float | None:
    """Largest true event rate attaining assurance; None if no count qualifies."""
    time, length, level = _inputs(
        scalar(exposure, "exposure"),
        scalar(max_length, "max_length"),
        scalar(confidence, "confidence"),
    )
    target = _assurance(assurance)
    k = float(_cutoff(time, length, level))
    if k < 0:
        return None
    with np.errstate(over="ignore", under="ignore"):
        result = float(gammainccinv(k + 1, target) / time)
    if not np.isfinite(result) or result <= 0:
        raise ArithmeticError("rate limit is outside positive float64 range")
    return result


def confint_poisson_exposure(
    max_length: float,
    event_rate: float,
    *,
    assurance: float = 0.9,
    confidence: float = 0.95,
    max_events: int = 1000000,
) -> float:
    """Earliest exposure attaining assurance, checking discrete width thresholds.

    For count k, the event becomes qualifying at exposure width(k)/max_length.
    Between thresholds assurance decreases, so its first success occurs at a
    threshold. Check counts from zero through max_events in vectorized batches.
    Later exposures need not all attain the same assurance.
    """
    length, rate = scalar(max_length, "max_length"), scalar(event_rate, "event_rate")
    _, _, level = _inputs(1, length, scalar(confidence, "confidence"))
    target, limit = _assurance(assurance), _integer(max_events, "max_events")
    if rate < 0 or not 0 <= limit <= 1000000:
        raise ValueError("require nonnegative event_rate and max_events in 0..1000000")
    first, batch = 0, 32
    while first <= limit:
        counts = np.arange(first, min(first + batch, limit + 1))
        widths = _width(counts, level)
        with np.errstate(over="ignore", under="ignore"):
            times = widths / length
        valid = np.isfinite(times) & (times > 0)
        probabilities = np.full(times.shape, -1.0)
        with np.errstate(over="ignore", under="ignore"):
            times[valid] = np.where(
                times[valid] * length < widths[valid],
                np.nextafter(times[valid], np.inf),
                times[valid],
            )
            valid &= np.isfinite(times)
            probabilities[valid] = poisson.cdf(counts[valid], rate * times[valid])
        good = np.flatnonzero(probabilities >= target)
        if good.size:
            if not np.all(valid[: good[0] + 1]):
                raise ArithmeticError(
                    "an earlier exposure threshold is outside positive float64 range"
                )
            return float(times[good[0]])
        if not np.all(valid):
            raise ArithmeticError("exposure thresholds are outside positive float64 range")
        first += batch
        batch = min(2 * batch, 4096)
    raise ValueError("requested assurance not attained at thresholds through max_events")
