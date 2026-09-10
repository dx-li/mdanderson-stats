"""Large-sample binary endpoint planning and paired/agreement designs."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr, ndtri

from ._validation import FloatArray, count, finite, scalar
from .continuous_sample_size import MeanObjective, _level, _probability, _search
from .cta_kappa import cohen_kappa


@dataclass(frozen=True)
class BinarySampleSize:
    """Enrollment, achieved planning power and preceding reference-size power."""

    group_sizes: tuple[int, ...]
    power: float
    previous_power: float | None
    target_power: float
    method: str

    @property
    def total_size(self) -> int:
        return sum(self.group_sizes)


def _rates(*values: ArrayLike) -> list[FloatArray]:
    arrays = [finite(v, "response rate") for v in values]
    if any(np.any((v <= 0) | (v >= 1)) for v in arrays):
        raise ValueError("response rates must lie strictly between zero and one")
    return arrays


def binary_proportion_power(
    sample_size: ArrayLike,
    control_rate: ArrayLike,
    treatment_rate: ArrayLike,
    *,
    treatment_size: ArrayLike | None = None,
    objective: MeanObjective = "equality",
    margin: ArrayLike = 0,
    method: Literal["wald", "score", "score-continuity"] = "wald",
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """Normal-approximation planning power, retaining the dominant equality tail.

    Sizes denote single sample or control and treatment. Wald variances use the
    specified alternative rates. Two-group score methods use the weighted pooled
    rate for the critical boundary; continuity correction adds half a count per
    arm. Equivalence uses the intersection of two level-alpha normal tests.
    """
    alpha = _level(alpha, sides)
    if objective not in ("equality", "equivalence", "noninferiority", "superiority"):
        raise ValueError("unknown objective")
    if method not in ("wald", "score", "score-continuity"):
        raise ValueError("unknown binary planning method")
    if method != "wald" and (objective != "equality" or treatment_size is None):
        raise ValueError("score methods require a two-group equality design")
    n, pc, pt, width = np.broadcast_arrays(
        count(sample_size, "sample_size"),
        *_rates(control_rate, treatment_rate),
        finite(margin, "margin"),
    )
    if np.any(n < 1):
        raise ValueError("sample_size must be positive")
    if (objective == "equality" and np.any(width != 0)) or (
        objective != "equality" and np.any((width <= 0) | (width >= 1))
    ):
        raise ValueError("margin must be zero for equality and in (0, 1) otherwise")
    correction: ArrayLike = 0
    if treatment_size is None:
        se = np.sqrt(pt * (1 - pt) / n)
        boundary_se = se
    else:
        n, nt, pc, pt, width = np.broadcast_arrays(
            n, count(treatment_size, "treatment_size"), pc, pt, width
        )
        if np.any(nt < 1) or np.any(n + nt >= 2**53):
            raise ValueError("positive group sizes and total below 2**53 are required")
        se = np.sqrt(pc * (1 - pc) / n + pt * (1 - pt) / nt)
        pooled = (n * pc + nt * pt) / (n + nt)
        boundary_se = se if method == "wald" else np.sqrt(pooled * (1 - pooled) * (1 / n + 1 / nt))
        if method == "score-continuity":
            correction = 0.5 * (1 / n + 1 / nt)
    delta = pt - pc
    critical = -ndtri(alpha / sides if objective == "equality" else alpha)
    if objective == "equivalence":
        # Stable interval probability under the working normal distribution.
        lo = (-width - delta) / se + critical
        hi = (width - delta) / se - critical
        value = np.where(lo > 0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))
        value = np.where(hi > lo, value, 0)
    else:
        effect = (
            np.abs(delta)
            if objective == "equality"
            else (delta + width if objective == "noninferiority" else delta - width)
        )
        value = ndtr((effect - critical * boundary_se - correction) / se)
    return _probability(value)


def binary_proportion_sample_size(
    control_rate: float,
    treatment_rate: float,
    *,
    allocation_ratio: float | None = None,
    objective: MeanObjective = "equality",
    margin: float = 0,
    method: Literal["wald", "score", "score-continuity"] = "wald",
    alpha: float = 0.05,
    sides: int = 1,
    target_power: float = 0.8,
    max_size: int = 10_000_000,
) -> BinarySampleSize:
    """Smallest qualifying single/control size with ceil(ratio*control) treatment."""
    pc, pt = scalar(control_rate, "control_rate"), scalar(treatment_rate, "treatment_rate")
    width = scalar(margin, "margin")
    ratio = None if allocation_ratio is None else scalar(allocation_ratio, "allocation_ratio")
    if ratio is not None and not 1e-6 <= ratio <= 1e6:
        raise ValueError("allocation_ratio must be in [1e-6, 1e6]")
    delta = pt - pc
    alternatives = {
        "equality": delta != 0,
        "equivalence": abs(delta) < width,
        "noninferiority": delta + width > 0,
        "superiority": delta - width > 0,
    }
    if objective not in alternatives or not alternatives[objective]:
        raise ValueError("response rates must lie inside the requested alternative")

    def sizes(n: int) -> tuple[int, ...]:
        return (n,) if ratio is None else (n, int(np.ceil(ratio * n)))

    def evaluate(n: int) -> float:
        sizes_at_n = sizes(n)
        return float(
            binary_proportion_power(
                n,
                pc,
                pt,
                treatment_size=sizes_at_n[1] if ratio is not None else None,
                objective=objective,
                margin=width,
                method=method,
                alpha=alpha,
                sides=sides,
            )
        )

    # Wald power is monotone as both variances decrease. For score methods the
    # pooled boundary changes with integer allocation: scan to retain oscillations.
    if method == "wald":
        n, achieved, previous = _search(evaluate, target_power, 1, max_size)
    else:
        target, limit = scalar(target_power, "target_power"), scalar(max_size, "max_size")
        if not 0 < target < 1 or limit != int(limit) or not 1 <= limit <= 10_000_000:
            raise ValueError("invalid target_power or max_size")
        previous = None
        for start in range(1, int(limit) + 1, 512):
            ns = np.arange(start, min(start + 512, int(limit) + 1))
            values = binary_proportion_power(
                ns,
                pc,
                pt,
                treatment_size=None if ratio is None else np.ceil(ratio * ns),
                objective=objective,
                margin=width,
                method=method,
                alpha=alpha,
                sides=sides,
            )
            passing = np.flatnonzero(values >= target)
            if passing.size:
                index = int(passing[0])
                n = int(ns[index])
                achieved = float(values[index])
                previous = float(values[index - 1]) if index else previous
                break
            previous = float(values[-1])
        else:
            raise ValueError("target power is not attained within max_size")
    return BinarySampleSize(
        sizes(n), achieved, previous, target_power, f"{objective}: {method} approximation"
    )


def mcnemar_power(
    sample_size: ArrayLike,
    loss_probability: ArrayLike,
    gain_probability: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """Source large-sample paired-shift power; one-sided direction is gain > loss."""
    alpha = _level(alpha, sides)
    n, loss, gain = np.broadcast_arrays(
        count(sample_size, "sample_size"),
        finite(loss_probability, "loss_probability"),
        finite(gain_probability, "gain_probability"),
    )
    discordance = loss + gain
    delta = gain - loss
    variance = discordance - delta**2
    if np.any((n < 1) | (loss < 0) | (gain < 0) | (discordance > 1) | (variance <= 0)):
        raise ValueError(
            "require positive size, valid discordant probabilities and positive variance"
        )
    effect = np.abs(delta) if sides == 2 else delta
    return _probability(
        ndtr(
            (np.sqrt(n) * effect + ndtri(alpha / sides) * np.sqrt(discordance)) / np.sqrt(variance)
        )
    )


def mcnemar_sample_size(
    loss_probability: float,
    gain_probability: float,
    *,
    alpha: float = 0.05,
    sides: int = 1,
    target_power: float = 0.8,
    max_size: int = 10_000_000,
) -> BinarySampleSize:
    """Number of complete pairs for the normal McNemar planning formula."""
    loss, gain = (
        scalar(loss_probability, "loss_probability"),
        scalar(gain_probability, "gain_probability"),
    )
    if gain == loss or (sides == 1 and gain < loss):
        raise ValueError("discordance probabilities must satisfy the alternative")
    n, achieved, previous = _search(
        lambda n: float(mcnemar_power(n, loss, gain, alpha=alpha, sides=sides)),
        target_power,
        1,
        max_size,
    )
    return BinarySampleSize((n,), achieved, previous, target_power, "McNemar: normal approximation")


def _kappa_variance(first: ArrayLike, second: ArrayLike, kappa: ArrayLike) -> FloatArray:
    a, b, k = np.broadcast_arrays(*_rates(first, second), finite(kappa, "kappa"))
    disagreement = a * (1 - b) + (1 - a) * b
    pp = a * b + k * disagreement / 2
    table = np.stack([pp, a - pp, b - pp, 1 - a - b + pp], axis=-1).reshape(a.shape + (2, 2))
    if np.any(table < -1e-14):
        raise ValueError("kappa is infeasible for the specified rater margins")
    table = np.maximum(table, 0)
    variance = cohen_kappa(table).variance
    if np.any(~np.isfinite(variance) | (variance <= 0)):
        raise ValueError("kappa planning requires a positive finite asymptotic variance")
    return variance


def kappa_power(
    sample_size: ArrayLike,
    first_positive: ArrayLike,
    second_positive: ArrayLike,
    null_kappa: ArrayLike,
    alternative_kappa: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """Cantor normal-approximation planning with fixed rater margins under both models."""
    alpha = _level(alpha, sides)
    n, k0, k1 = np.broadcast_arrays(
        count(sample_size, "sample_size"),
        finite(null_kappa, "null_kappa"),
        finite(alternative_kappa, "alternative_kappa"),
    )
    if np.any(n < 1):
        raise ValueError("sample_size must be positive")
    v0 = _kappa_variance(first_positive, second_positive, k0)
    v1 = _kappa_variance(first_positive, second_positive, k1)
    effect = np.abs(k1 - k0) if sides == 2 else k1 - k0
    return _probability(
        ndtr((np.sqrt(n) * effect + ndtri(alpha / sides) * np.sqrt(v0)) / np.sqrt(v1))
    )


def kappa_sample_size(
    first_positive: float,
    second_positive: float,
    null_kappa: float,
    alternative_kappa: float,
    *,
    alpha: float = 0.05,
    sides: int = 1,
    target_power: float = 0.8,
    max_size: int = 10_000_000,
) -> BinarySampleSize:
    """Number of subjects each assessed by both raters for a kappa comparison."""
    a, b, k0, k1 = (
        scalar(v, name)
        for v, name in zip(
            (first_positive, second_positive, null_kappa, alternative_kappa),
            ("first_positive", "second_positive", "null_kappa", "alternative_kappa"),
            strict=True,
        )
    )
    if k1 == k0 or (sides == 1 and k1 < k0):
        raise ValueError("kappas must satisfy the alternative")
    n, achieved, previous = _search(
        lambda n: float(kappa_power(n, a, b, k0, k1, alpha=alpha, sides=sides)),
        target_power,
        1,
        max_size,
    )
    return BinarySampleSize((n,), achieved, previous, target_power, "Kappa: Cantor approximation")
