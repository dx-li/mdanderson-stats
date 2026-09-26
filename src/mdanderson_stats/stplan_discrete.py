"""Power approximations and exact tests for STPLAN discrete outcomes."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr, ndtri
from scipy.stats import binom, chi2, ncx2, poisson

from ._validation import FloatArray, scalar
from .stplan_continuous import _alpha, _inputs, _power


def _integral(value: FloatArray, name: str, *, minimum: int = 0) -> FloatArray:
    if np.any((value < minimum) | (value != np.floor(value)) | (value >= 2**53)):
        raise ValueError(f"{name} must contain integers >= {minimum} and below 2**53")
    return value


def stplan_arcsine_binomial_two_sample_power(
    probability1: ArrayLike,
    probability2: ArrayLike,
    n1: ArrayLike,
    n2: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """STPLAN two-sample binomial power using the arcsine variance stabilizer."""
    a = _alpha(alpha, sides)
    p1, p2, first, second = _inputs(
        (probability1, "probability1"), (probability2, "probability2"), (n1, "n1"), (n2, "n2")
    )
    if np.any((p1 <= 0) | (p1 >= 1) | (p2 <= 0) | (p2 >= 1)):
        raise ValueError("probabilities must be strictly between 0 and 1")
    if np.any(first < 2) or np.any(second < 2):
        raise ValueError("each sample size must be at least 2")
    effect = np.abs(2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1)))
    se = np.sqrt(1 / first + 1 / second)
    return _power(ndtr(effect / se + ndtri(a)))


def stplan_median_split_power(
    overall_probability: ArrayLike,
    delta: ArrayLike,
    total_size: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """Power for the above-versus-below-median binomial comparison.

    ``delta`` is P(event | above median) minus the overall event probability;
    the two halves therefore use probabilities overall-delta and overall+delta.
    STPLAN allows continuous planning sample sizes and allocates half to each side.
    """
    overall, d, n = _inputs(
        (overall_probability, "overall_probability"), (delta, "delta"), (total_size, "total_size")
    )
    if np.any((overall <= 0) | (overall >= 1)) or np.any(n < 4):
        raise ValueError("overall_probability must be interior and total_size at least 4")
    p1, p2 = overall - d, overall + d
    if np.any((p1 <= 0) | (p1 >= 1) | (p2 <= 0) | (p2 >= 1)):
        raise ValueError("overall_probability +/- delta must lie strictly between 0 and 1")
    return stplan_arcsine_binomial_two_sample_power(p1, p2, n / 2, n / 2, alpha=alpha, sides=sides)


def stplan_historical_binomial_power(
    control_probability: ArrayLike,
    experimental_probability: ArrayLike,
    control_size: ArrayLike,
    experimental_size: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """Arcsine power for a new binomial group versus a fixed historic control."""
    a = _alpha(alpha, sides)
    pc, pe, nc, ne = _inputs(
        (control_probability, "control_probability"),
        (experimental_probability, "experimental_probability"),
        (control_size, "control_size"),
        (experimental_size, "experimental_size"),
    )
    if np.any((pc <= 0) | (pc >= 1) | (pe <= 0) | (pe >= 1)):
        raise ValueError("probabilities must be strictly between 0 and 1")
    if np.any(nc <= 0) or np.any(ne <= 0):
        raise ValueError("sample sizes must be positive")
    effect = np.abs(2 * np.arcsin(np.sqrt(pe)) - 2 * np.arcsin(np.sqrt(pc)))
    critical_se = np.sqrt(1 / ne + 1 / nc)
    alternative_se = 1 / np.sqrt(ne)
    z = (effect + ndtri(a) * critical_se) / alternative_se
    return _power(ndtr(z))


def stplan_responder_normal_approximation_power(
    conservative_probability: ArrayLike,
    expensive_probability: ArrayLike,
    conservative_size: ArrayLike,
    expensive_size: ArrayLike,
    margin: ArrayLike,
    *,
    confidence: float = 0.95,
) -> FloatArray:
    """STPLAN normal-approximation power for its conservative-responder criterion."""
    confidence = scalar(confidence, "confidence")
    if not 0.5 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0.5 and 1")
    pc, pe, nc, ne, d = _inputs(
        (conservative_probability, "conservative_probability"),
        (expensive_probability, "expensive_probability"),
        (conservative_size, "conservative_size"),
        (expensive_size, "expensive_size"),
        (margin, "margin"),
    )
    if np.any((pc <= 0) | (pc >= 1) | (pe <= 0) | (pe >= 1)):
        raise ValueError("response probabilities must be strictly between 0 and 1")
    if np.any(nc <= 0) or np.any(ne <= 0) or np.any(d < 0):
        raise ValueError("sample sizes must be positive and margin nonnegative")
    variance = pe * (1 - pe) / ne + pc * (1 - pc) / nc
    z = (d - pe + pc) / np.sqrt(variance) - ndtri(confidence)
    return _power(ndtr(z))


def stplan_binomial_k_sample_power(
    probabilities: ArrayLike,
    sample_sizes: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 2,
) -> FloatArray:
    """Noncentral-chi-square approximation for STPLAN's K-sample binomial test.

    Group is the final axis of both arrays. The native two-group one-sided option
    uses a chi-square cutoff at 2*alpha; for three or more groups STPLAN only
    defines the two-sided omnibus test.
    """
    a = _alpha(alpha, sides)
    raw_p, raw_n = np.asarray(probabilities), np.asarray(sample_sizes)
    if raw_p.ndim == 0 or raw_n.ndim == 0 or raw_p.shape[-1] != raw_n.shape[-1]:
        raise ValueError("probabilities and sample_sizes need a matching final group axis")
    p, n = _inputs((raw_p, "probabilities"), (raw_n, "sample_sizes"))
    groups = p.shape[-1]
    if groups < 2:
        raise ValueError("at least two groups are required")
    if groups > 2 and sides != 2:
        raise ValueError("STPLAN defines one-sided K-sample power only for two groups")
    if np.any((p <= 0) | (p >= 1)) or np.any(n <= 0):
        raise ValueError("probabilities must be interior and sample sizes positive")
    scale = np.max(n, axis=-1)
    weights = n / scale[..., None]
    pooled = np.sum(weights * p, axis=-1) / np.sum(weights, axis=-1)
    if np.any(~np.isfinite(pooled) | (pooled <= 0) | (pooled >= 1)):
        raise ArithmeticError("pooled probability is not representable")
    noncentrality = (
        scale * np.sum(weights * (p - pooled[..., None]) ** 2, axis=-1) / (pooled * (1 - pooled))
    )
    if np.any(~np.isfinite(noncentrality)):
        raise ArithmeticError("K-sample noncentrality is not representable")
    df = groups - 1
    critical_alpha = 2 * a
    critical = chi2.isf(critical_alpha, df)
    return _power(ncx2.sf(critical, df, noncentrality))


def stplan_retention_probability(
    initial_size: ArrayLike,
    minimum_remaining: ArrayLike,
    dropout_rate: ArrayLike,
    duration: ArrayLike,
) -> FloatArray:
    """Probability at least the requested number remain under STPLAN's loss model.

    Individual retention is ``(1-dropout_rate)**duration``; retained subjects
    are binomially distributed. Counts must be integers; duration may be fractional.
    """
    initial, minimum, dropout, time = _inputs(
        (initial_size, "initial_size"),
        (minimum_remaining, "minimum_remaining"),
        (dropout_rate, "dropout_rate"),
        (duration, "duration"),
    )
    _integral(initial, "initial_size", minimum=0)
    _integral(minimum, "minimum_remaining", minimum=0)
    if np.any((dropout < 0) | (dropout > 1)) or np.any(time < 0):
        raise ValueError("dropout_rate must be in [0,1] and duration nonnegative")
    safe_dropout = np.where(dropout == 1, 0.0, dropout)
    retention = np.exp(time * np.log1p(-safe_dropout))
    retention = np.where(dropout == 1, np.where(time == 0, 1.0, 0.0), retention)
    return _power(binom.sf(minimum - 1, initial, retention))


def stplan_fisher_exact_approx_power(
    probability1: ArrayLike,
    probability2: ArrayLike,
    sample_size: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """STPLAN's Casagrande–Pike–Smith approximation to Fisher-test power.

    This reproduces STPLAN's continuity-corrected normal approximation, including
    its absolute-value correction ``abs(n*|p1-p2|-1)`` and low-sample nonmonotonicity.
    It does not compute exact Fisher-test power.
    """
    a = _alpha(alpha, sides)
    p1, p2, n = _inputs(
        (probability1, "probability1"), (probability2, "probability2"), (sample_size, "sample_size")
    )
    if np.any((p1 <= 0) | (p1 >= 1) | (p2 <= 0) | (p2 >= 1)) or np.any(n < 2):
        raise ValueError("probabilities must be interior and sample_size at least 2")
    pbar = p1 / 2 + p2 / 2
    pooled_sd = np.sqrt(2 * pbar * (1 - pbar))
    alternative_sd = np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    correction = np.abs(n * np.abs(p2 - p1) - 1) / np.sqrt(n)
    z = (correction + ndtri(a) * pooled_sd) / alternative_sd
    return _power(ndtr(z))


def _binomial_exact_power(
    p0: FloatArray, pa: FloatArray, n: FloatArray, alpha: float
) -> FloatArray:
    """Find a nonrandomized exact binomial rejection tail with size at most alpha."""
    lower = pa <= p0
    lo = np.where(lower, -1.0, 0.0)
    hi = np.where(lower, n, n + 1)
    while np.any(hi - lo > 1):
        mid = np.floor(lo + (hi - lo) / 2)
        cdf = binom.cdf(mid, n, p0)
        sf = binom.sf(mid - 1, n, p0)
        lower_ok = cdf <= alpha
        upper_ok = sf <= alpha
        lo = np.where(lower & lower_ok, mid, lo)
        hi = np.where(lower & ~lower_ok, mid, hi)
        hi = np.where(~lower & upper_ok, mid, hi)
        lo = np.where(~lower & ~upper_ok, mid, lo)
    lower_power = np.where(lo < 0, 0.0, binom.cdf(lo, n, pa))
    upper_power = np.where(hi > n, 0.0, binom.sf(hi - 1, n, pa))
    return _power(np.where(lower, lower_power, upper_power))


def _poisson_exact_power(mean0: FloatArray, meana: FloatArray, alpha: float) -> FloatArray:
    """Find a one-sided exact Poisson region within exactly representable counts."""
    if np.any(mean0 >= 2**48) or np.any(meana >= 2**48):
        raise ValueError("Poisson means must be below 2**48 for exact count resolution")
    lower = meana <= mean0
    bound = np.ceil(mean0 + 8 * np.sqrt(mean0) + 16)
    while np.any((poisson.cdf(bound, mean0) <= alpha) | (poisson.sf(bound - 1, mean0) > alpha)):
        needs_room = (poisson.cdf(bound, mean0) <= alpha) | (poisson.sf(bound - 1, mean0) > alpha)
        bound = np.where(needs_room, 2 * bound + 1, bound)
        if np.any(bound >= 2**53):
            raise ArithmeticError("could not bracket the exact Poisson rejection region")
    if np.any(~np.isfinite(bound)):
        raise ArithmeticError("could not bracket the exact Poisson rejection region")
    lo = np.where(lower, -1.0, 0.0)
    hi = bound
    while np.any(hi - lo > 1):
        mid = np.floor(lo + (hi - lo) / 2)
        cdf = poisson.cdf(mid, mean0)
        sf = poisson.sf(mid - 1, mean0)
        lo = np.where(lower & (cdf <= alpha), mid, lo)
        hi = np.where(lower & (cdf > alpha), mid, hi)
        hi = np.where(~lower & (sf <= alpha), mid, hi)
        lo = np.where(~lower & (sf > alpha), mid, lo)
    lower_power = np.where(lo < 0, 0.0, poisson.cdf(lo, meana))
    upper_power = poisson.sf(hi - 1, meana)
    return _power(np.where(lower, lower_power, upper_power))


def stplan_exact_binomial_power(
    null_probability: ArrayLike,
    alternative_probability: ArrayLike,
    sample_size: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """Exact one-sided binomial power with the largest nonrandomized region <= alpha.

    Unlike the native routine's -1 sentinel, an empty rejection region returns 0.
    Sample sizes must be integers; probabilities may include 0 and 1.
    """
    a = _alpha(alpha, sides)
    p0, pa, n = _inputs(
        (null_probability, "null_probability"),
        (alternative_probability, "alternative_probability"),
        (sample_size, "sample_size"),
    )
    _integral(n, "sample_size", minimum=1)
    if np.any((p0 < 0) | (p0 > 1) | (pa < 0) | (pa > 1)):
        raise ValueError("probabilities must lie in [0,1]")
    return _binomial_exact_power(p0, pa, n, a)


def stplan_exact_poisson_power(
    null_rate: ArrayLike,
    alternative_rate: ArrayLike,
    exposure: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """Exact one-sided Poisson-rate power with nonrandomized size at most alpha.

    Empty rejection regions return 0 rather than the native -1 sentinel.
    """
    a = _alpha(alpha, sides)
    r0, ra, time = _inputs(
        (null_rate, "null_rate"), (alternative_rate, "alternative_rate"), (exposure, "exposure")
    )
    if np.any(r0 < 0) or np.any(ra < 0) or np.any(time <= 0):
        raise ValueError("rates must be nonnegative and exposure positive")
    return _poisson_exact_power(r0 * time, ra * time, a)
