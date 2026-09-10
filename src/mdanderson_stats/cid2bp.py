"""CID2BP intervals for the difference between independent binomial proportions."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtri

from ._validation import scalar
from .bayesian_monitoring import _integer
from .cdflib_elementary import rlog1
from .cid2bp_exact import _exact
from .cid2bp_peskun import _peskun
from .cid2bp_weighted import _weighted


@dataclass(frozen=True)
class BinomialDifferenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    method: str


def _profile_loss(delta: float, n1: int, x1: int, n2: int, x2: int) -> float:
    lo, hi = max(0.0, -delta), min(1.0, 1 - delta)

    def score(q: float) -> float:
        p = min(1.0, max(0.0, q + delta))
        terms = [(x1, p), (n1 - x1, 1 - p), (x2, q), (n2 - x2, 1 - q)]
        values = [0.0 if x == 0 else np.inf if p == 0 else x / p for x, p in terms]
        return float(values[0] - values[1] + values[2] - values[3])

    if lo == hi or score(lo) <= 0:
        q = lo
    elif score(hi) >= 0:
        q = hi
    else:
        q = brentq(score, lo, hi, xtol=1e-14)
    p = min(1.0, max(0.0, q + delta))
    loss = 0.0
    for n, x, probability in [(n1, x1, p), (n2, x2, q)]:
        observed = x / n
        if probability == observed:
            continue
        if probability in (0.0, 1.0):
            return np.inf
        if x == 0:
            loss -= n * np.log1p(-probability)
        elif x == n:
            loss -= n * np.log(probability)
        else:
            change = probability - observed
            remainders = rlog1([change / observed, -change / (1 - observed)])
            loss += n * (observed * remainders[0] + (1 - observed) * remainders[1])
    return float(loss)


def _adjust(
    n1: int, x1: int, n2: int, x2: int, tail: float, lower: float, upper: float
) -> tuple[float, float]:
    """Source Peskun boundary adjustments for normal-based intervals."""
    difference = x1 / n1 - x2 / n2
    log_tail = np.log(tail)
    if abs(difference) == 1:
        log_ratio = np.log(n1 / n2)
        if log_tail / n1 <= log_ratio <= -log_tail / n2:
            total = n1 + n2
            # log(sum n) cancels using sample-size fractions, not large powers.
            boundary = np.expm1(
                log_tail / total - (n1 * np.log(n1 / total) + n2 * np.log(n2 / total)) / total
            )
        else:
            boundary = np.exp(log_tail / min(n1, n2))
        return (float(boundary), 1.0) if difference == 1 else (-1.0, float(-boundary))
    power = 1 / max(n1, n2)
    if difference == power - 1:
        lower = -np.exp(np.log1p(-tail) * power)
    elif difference == 1 - power:
        upper = np.exp(np.log1p(-tail) * power)
    return float(lower), float(upper)


def cid2bp_interval(
    successes1: int,
    trials1: int,
    successes2: int,
    trials2: int,
    *,
    confidence: float = 0.95,
    method: str = "cox_snell",
) -> BinomialDifferenceInterval:
    """Estimate p1-p2 using a supported CID2BP method.

    Methods: wald, continuity_corrected, yates, cox_snell, peskun_native,
    weighted_likelihood, weighted_mid_p, auto, exact. The corrected normal
    method uses unbiased binomial variances plus 1/(2*min(n1,n2)); Yates adds
    1/(2*n1)+1/(2*n2) to the ordinary Wald half-width. Source boundary adjustments
    apply to all three normal methods. Final limits are clipped to [-1,1].
    """
    x1, n1, x2, n2 = (
        _integer(v, name)
        for v, name in zip(
            [successes1, trials1, successes2, trials2],
            ["successes1", "trials1", "successes2", "trials2"],
            strict=True,
        )
    )
    level = scalar(confidence, "confidence")
    if not (1 <= n1 <= 1000000 and 1 <= n2 <= 1000000 and 0 <= x1 <= n1 and 0 <= x2 <= n2):
        raise ValueError("require integer trials 1..1000000 and successes in 0..trials")
    if not 1e-6 <= level <= 1 - 1e-12:
        raise ValueError("confidence must be in [1e-6,1-1e-12]")
    if method == "auto":
        method = "weighted_likelihood" if n1 + n2 <= 50 else "cox_snell"
    if method not in {
        "wald",
        "continuity_corrected",
        "yates",
        "cox_snell",
        "peskun_native",
        "weighted_likelihood",
        "weighted_mid_p",
        "exact",
    }:
        raise ValueError("unknown CID2BP method")
    if method == "continuity_corrected" and min(n1, n2) < 2:
        raise ValueError("continuity_corrected requires at least two trials in each group")
    p1, p2 = x1 / n1, x2 / n2
    difference = p1 - p2
    tail = (1 - level) / 2
    z = -float(ndtri(tail))
    if method == "cox_snell":

        def crossing(delta: float) -> float:
            return z * z / 2 - _profile_loss(delta, n1, x1, n2, x2)

        lower = -1.0 if difference == -1 else brentq(crossing, -1, difference, xtol=1e-12)
        upper = 1.0 if difference == 1 else brentq(crossing, difference, 1, xtol=1e-12)
    elif method in {"weighted_likelihood", "weighted_mid_p"}:
        lower, upper = _weighted(n1, x1, n2, x2, tail, method == "weighted_mid_p")
    elif method == "exact":
        lower, upper = _exact(n1, x1, n2, x2, tail)
    elif method == "peskun_native":
        lower, upper = _peskun(n1, x1, n2, x2, z)
        lower, upper = _adjust(n1, x1, n2, x2, tail, lower, upper)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ArithmeticError(
                "native Peskun formula has no finite ordered interval for these inputs"
            )
    else:
        if method == "continuity_corrected":
            width = z * np.sqrt(p1 * (1 - p1) / (n1 - 1) + p2 * (1 - p2) / (n2 - 1)) + 0.5 / min(
                n1, n2
            )
        else:
            width = z * np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
            if method == "yates":
                width += 0.5 / n1 + 0.5 / n2
        lower, upper = _adjust(n1, x1, n2, x2, tail, difference - width, difference + width)
    return BinomialDifferenceInterval(
        difference, max(-1.0, float(lower)), min(1.0, float(upper)), level, method
    )
