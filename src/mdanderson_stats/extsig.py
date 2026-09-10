"""EXTSIG unconditional two-binomial tests with five outcome orderings."""

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp, xlogy
from scipy.stats import chi2

from ._validation import FloatArray, count, scalar
from .cta_fisher import fisher_exact
from .extsig_maximum import ExtsigMaximum, _maximum


@dataclass(frozen=True)
class ExtsigResult:
    ordering: str
    nuisance: str
    tie_rule: str
    direction: str
    one_sided: ExtsigMaximum
    mid_p_one_sided: ExtsigMaximum
    two_sided: ExtsigMaximum
    mid_p_two_sided: ExtsigMaximum
    fisher_one_sided: float
    fisher_two_sided: float
    chi_square: float
    chi_square_two_sided: float

    @property
    def chi_square_one_sided(self) -> float:
        """Original data-directed half of the two-sided chi-square tail."""
        return self.chi_square_two_sided / 2


def _conditional(n1: int, n2: int) -> tuple[FloatArray, FloatArray]:
    """Conditional table masses and smaller Fisher tails, grouped by total successes."""
    n = n1 + n2
    mass, log_tail = np.empty((n1 + 1, n2 + 1)), np.empty((n1 + 1, n2 + 1))
    factorial = gammaln(np.arange(n + 1) + 1)
    for total in range(n + 1):
        i = np.arange(max(0, total - n2), min(n1, total) + 1)
        j = total - i
        logs = (
            factorial[n1]
            - factorial[i]
            - factorial[n1 - i]
            + factorial[n2]
            - factorial[j]
            - factorial[n2 - j]
        )
        logs -= logsumexp(logs)
        mass[i, j] = np.exp(logs)
        log_tail[i, j] = np.minimum(
            np.logaddexp.accumulate(logs), np.logaddexp.accumulate(logs[::-1])[::-1]
        )
    log_tail[0, 0] = log_tail[-1, -1] = -np.log(2.0)
    return mass, np.minimum(log_tail, 0)


def _statistics(n1: int, n2: int, ordering: str, log_tail: FloatArray, native: bool) -> FloatArray:
    i, j = np.arange(n1 + 1)[:, None], np.arange(n2 + 1)[None, :]
    difference = np.abs(i * n2 - j * n1).astype(float)
    if ordering == "difference":
        return difference / (n1 * n2)
    if ordering == "fisher":
        return -np.expm1(log_tail) if native else -log_tail
    total = i + j
    if ordering == "chi_square":
        denominator = n1 * n2 * total.astype(float) * (n1 + n2 - total)
        return np.divide(
            difference**2 * (n1 + n2),
            denominator,
            out=np.zeros_like(difference),
            where=denominator > 0,
        )
    p1, p2 = i / n1, j / n2
    if ordering == "unpooled_z":
        sd = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        statistic = np.divide(
            difference / (n1 * n2), sd, out=np.zeros_like(difference), where=sd > 0
        )
        statistic[(sd == 0) & (difference > 0)] = 999 if native else np.inf
        return statistic
    # Half the likelihood-ratio deviance. Evaluate KL directly instead of
    # subtracting fitted log likelihoods with large additive terms.
    pooled = total / (n1 + n2)
    with np.errstate(divide="ignore", invalid="ignore"):
        if native:
            result = (
                xlogy(i, p1)
                + xlogy(n1 - i, 1 - p1)
                + xlogy(j, p2)
                + xlogy(n2 - j, 1 - p2)
                - xlogy(total, pooled)
                - xlogy(n1 + n2 - total, 1 - pooled)
            )
        else:
            result = (
                xlogy(i, p1 / pooled)
                + xlogy(n1 - i, (1 - p1) / (1 - pooled))
                + xlogy(j, p2 / pooled)
                + xlogy(n2 - j, (1 - p2) / (1 - pooled))
            )
    result[0, 0] = result[-1, -1] = 0
    return np.maximum(result, 0)


def extsig(
    successes1: int,
    size1: int,
    successes2: int,
    size2: int,
    *,
    ordering: str = "fisher",
    nuisance: str = "bounded",
    tie_rule: str = "inclusive",
    tolerance: float = 1e-9,
) -> ExtsigResult:
    """Return original EXTSIG one-/two-sided and mid-p analyses.

    One-sided results use the direction selected by the observed proportions,
    as in EXTSIG; they are not an a-priori directional test. ``native_grid``
    maximizes only over .01,...,.99. ``bounded`` returns a numerical upper
    bound over the complete [0,1] nuisance domain. Mid-p lacks exact size control.
    """
    data = count([successes1, size1, successes2, size2], "success counts and sizes")
    if data.shape != (4,):
        raise ValueError("counts and sizes must be scalar integers")
    s1, n1, s2, n2 = map(int, data)
    if not 1 <= n1 <= 1000 or not 1 <= n2 <= 1000 or s1 > n1 or s2 > n2:
        raise ValueError("group sizes must be in 1..1000 and successes cannot exceed size")
    if ordering not in ("difference", "log_likelihood", "chi_square", "unpooled_z", "fisher"):
        raise ValueError("unknown EXTSIG ordering")
    if nuisance not in ("bounded", "native_grid") or tie_rule not in ("inclusive", "native"):
        raise ValueError("unknown nuisance maximization or tie rule")
    tolerance = scalar(tolerance, "tolerance")
    if not 1e-12 <= tolerance <= 1e-3:
        raise ValueError("tolerance must be in [1e-12,1e-3]")
    if nuisance == "bounded" and n1 + n2 > 400:
        raise ValueError("bounded maximization supports at most 400 combined observations")
    direction = "p1 <= p2" if s1 * n2 <= s2 * n1 else "p1 >= p2"
    if s1 * n2 > s2 * n1:
        s1, n1, s2, n2 = s2, n2, s1, n1
    mass, log_tail = _conditional(n1, n2)
    statistic = _statistics(n1, n2, ordering, log_tail, tie_rule == "native")
    cutoff = statistic[s1, s2]
    with np.errstate(invalid="ignore"):
        tied = (statistic == cutoff) | (
            np.abs(statistic - cutoff) < 1e-12 * np.maximum(np.abs(statistic) + abs(cutoff), 1e-100)
        )
    selected = statistic >= cutoff
    if tie_rule == "inclusive":
        selected |= tied
    i, j = np.arange(n1 + 1)[:, None], np.arange(n2 + 1)[None, :]
    side = i * n2 <= j * n1
    totals = np.broadcast_to(i + j, mass.shape).ravel()
    masks = [selected & side, selected & side, selected, selected]
    results = []
    for index, mask in enumerate(masks):
        weight = mask.astype(float)
        if index % 2:
            weight *= np.where(tied, 0.5, 1.0)
        coefficients = np.bincount(totals, weights=(mass * weight).ravel(), minlength=n1 + n2 + 1)
        coefficients = np.clip(coefficients, 0, 1)
        if coefficients.max() == 0 and np.any(mask):
            raise ArithmeticError("selected outcome probabilities underflow float64")
        results.append(_maximum(coefficients, nuisance, tolerance))
    chi = float(_statistics(n1, n2, "chi_square", log_tail, False)[s1, s2])
    fish2 = float(fisher_exact([[s1, n1 - s1], [s2, n2 - s2]]).pvalue)
    return ExtsigResult(
        ordering,
        nuisance,
        tie_rule,
        direction,
        results[0],
        results[1],
        results[2],
        results[3],
        float(np.exp(log_tail[s1, s2])),
        fish2,
        chi,
        float(chi2.sf(chi, 1)),
    )
