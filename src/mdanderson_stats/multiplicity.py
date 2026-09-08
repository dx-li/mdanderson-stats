"""MULTI's adjustment and sharpened testing procedures.

Modified Python implementations; original copyright/terms are preserved in
notices/mdanderson-multi-LEGALITIES.txt. Labels follow the source's algorithms,
with ambiguous historical names made explicit. See docs/multiple-testing.md.
"""

from dataclasses import dataclass
from math import fsum
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln

from ._validation import FloatArray, finite, scalar

AdjustmentMethod = Literal[
    "bonferroni",
    "sidak",
    "holm",
    "holm-sidak",
    "finner",
    "hochberg",
    "multi-hommel",
    "simes",
    "rom",
]
METHODS = (
    "bonferroni",
    "sidak",
    "holm",
    "holm-sidak",
    "finner",
    "hochberg",
    "multi-hommel",
    "simes",
    "rom",
)


def _pvalues(pvalues: ArrayLike) -> FloatArray:
    values = finite(pvalues, "pvalues")
    if values.ndim < 1 or values.shape[-1] == 0:
        raise ValueError("pvalues must have a nonempty last dimension")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("pvalues must lie in [0, 1]")
    return values


def _alpha(alpha: float) -> float:
    result = scalar(alpha, "alpha")
    if not 0 <= result <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    return result


@dataclass(frozen=True)
class MultipleTestingResult:
    """Results use input order; order gives sorted ranks along the last axis.

    Rom returns critical values rather than adjusted p-values. The other eight
    procedures return adjusted p-values rather than alpha-dependent thresholds.
    """

    method: str
    alpha: float
    reject: NDArray[np.bool_]
    adjusted_pvalues: FloatArray | None
    critical_values: FloatArray | None
    order: NDArray[np.intp]


def rom_critical_values(number_of_tests: int, alpha: float = 0.05) -> FloatArray:
    """Rom's step-up thresholds, from smallest to largest p-value rank.

    Uses the original SURM recurrence. Binomial terms are evaluated in the log
    domain to avoid overflowing their coefficients before multiplication by a
    small power. Work is O(n^2), storage O(n). No adjusted p-values are implied.
    """
    if (
        isinstance(number_of_tests, bool)
        or not isinstance(number_of_tests, int)
        or number_of_tests < 1
    ):
        raise ValueError("number_of_tests must be a positive integer")
    alpha = _alpha(alpha)
    n = number_of_tests
    if alpha == 0:
        return np.zeros(n)
    c = np.empty(n)
    c[0] = alpha
    log_factorials = gammaln(np.arange(n + 1, dtype=float) + 1)
    total, power = 0.0, 1.0
    for i in range(2, n + 1):
        power *= alpha
        total += power
        j = np.arange(1, i - 1)
        terms = np.exp(
            log_factorials[i] - log_factorials[j] - log_factorials[i - j] + (i - j) * np.log(c[j])
        )
        c[i - 1] = (total - fsum(terms)) / i
        if not 0 < c[i - 1] <= 1:
            raise ArithmeticError("Rom recurrence lost numerical validity")
    return c[::-1].copy()


def multiple_testing(
    pvalues: ArrayLike, method: AdjustmentMethod = "holm", *, alpha: float = 0.05
) -> MultipleTestingResult:
    """Run one of MULTI's nine desktop p-value procedures.

    Families occupy the last axis; leading axes are independent batches.
    Results are restored to input order. The source's 'Hommel' option is named
    'multi-hommel' because it uses the harmonic step-up formula, not the usual
    closed-testing Hommel algorithm. 'simes' is the source's step-up adjustment,
    mathematically the Benjamini-Hochberg adjusted p-value formula.
    These methods have different error-control assumptions; see the guide.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose one of {METHODS}")
    x = _pvalues(pvalues)
    alpha = _alpha(alpha)
    order = np.argsort(x, axis=-1, kind="stable")
    inverse = np.argsort(order, axis=-1)
    p = np.take_along_axis(x, order, axis=-1)
    n = p.shape[-1]
    ranks = np.arange(1, n + 1, dtype=float)
    remaining = n + 1 - ranks
    if method == "rom":
        critical = np.broadcast_to(rom_critical_values(n, alpha), p.shape)
        reject = np.logical_or.accumulate((p <= critical)[..., ::-1], axis=-1)[..., ::-1]
        return MultipleTestingResult(
            method,
            alpha,
            np.take_along_axis(reject, inverse, axis=-1),
            None,
            np.take_along_axis(critical, inverse, axis=-1),
            order,
        )
    if method in ("sidak", "holm-sidak", "finner"):
        exponent = n if method == "sidak" else remaining if method == "holm-sidak" else n / ranks
        # log1p/expm1 retain p-values far smaller than machine epsilon.
        with np.errstate(divide="ignore"):
            adjusted = -np.expm1(exponent * np.log1p(-p))
    elif method == "bonferroni":
        adjusted = n * p
    elif method in ("holm", "hochberg"):
        adjusted = remaining * p
    else:
        adjusted = p * (n / ranks)
        if method == "multi-hommel":
            adjusted *= fsum(1 / ranks)
    if method in ("holm", "holm-sidak", "finner"):
        adjusted = np.maximum.accumulate(adjusted, axis=-1)
    elif method in ("hochberg", "multi-hommel", "simes"):
        adjusted = np.minimum.accumulate(adjusted[..., ::-1], axis=-1)[..., ::-1]
    adjusted = np.take_along_axis(np.minimum(adjusted, 1), inverse, axis=-1)
    return MultipleTestingResult(method, alpha, adjusted <= alpha, adjusted, None, order)


def sharpened_testing(
    pvalues: ArrayLike,
    null_estimate: float,
    method: Literal["holm", "hochberg"] = "holm",
    *,
    alpha: float = 0.05,
) -> NDArray[np.bool_]:
    """Reproduce MULTI's SHL or SHC decisions using a supplied null-count estimate.

    The original truncates the estimate to an integer and caps it at the family
    size. An estimate below one rejects everything, as in the source. This is
    an estimate-dependent procedure, not a general error-control guarantee.

    The Holm variant deliberately follows the executable's two thresholds: its
    loop returns before the later denominator update described in the comments.
    It is not replaced with a different adaptive step-down procedure.
    """
    values = _pvalues(pvalues)
    if values.ndim != 1:
        raise ValueError("Sharpened testing accepts one p-value vector at a time")
    if method not in ("holm", "hochberg"):
        raise ValueError("method must be 'holm' or 'hochberg'")
    alpha = _alpha(alpha)
    estimate = scalar(null_estimate, "null_estimate")
    if estimate < 0:
        raise ValueError("null_estimate must be nonnegative")
    n = values.size
    m0 = min(int(estimate), n)
    if m0 == 0:
        return np.ones(n, dtype=bool)
    order = np.argsort(values, kind="stable")
    p = values[order]
    if method == "holm":
        rejected = int(np.searchsorted(p, alpha / m0, side="right"))
        remaining = n - rejected
        if 0 < remaining < m0:
            rejected = int(np.searchsorted(p, alpha / remaining, side="right"))
    else:
        denominators = np.minimum(m0, np.arange(n, 0, -1))
        eligible = np.flatnonzero(p <= alpha / denominators)
        rejected = int(eligible[-1] + 1) if eligible.size else 0
    result = np.zeros(n, dtype=bool)
    result[order[:rejected]] = True
    return result
