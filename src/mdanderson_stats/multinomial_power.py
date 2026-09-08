"""Exact, nonrandomized one-sample multinomial power (MULTINOMPOW)."""

from dataclasses import dataclass
from itertools import chain, combinations
from math import comb
from operator import index

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln, xlogy

from ._validation import finite

FloatArray = NDArray[np.float64]


def _freeze(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _probabilities(values: ArrayLike, name: str) -> FloatArray:
    array = finite(values, name)
    if np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    totals = array.sum(axis=-1, keepdims=True)
    if np.any(np.abs(totals - 1) > 1e-12):
        raise ValueError(f"{name} must sum to one within 1e-12 along each row")
    return array / totals


@dataclass(frozen=True)
class MultinomialPower:
    """Immutable exact-test results; axes are statistic, alternative, alpha.

    ``critical_values`` and ``actual_size`` have shape (statistics, alpha).
    ``power`` has shape (statistics, alternatives, alpha). Infinite critical
    values denote empty rejection regions. Values at a numerical tie boundary
    belong to the same group, using the original adjacent relative tolerance.
    """

    n: int
    null: FloatArray
    alternatives: FloatArray
    alpha: FloatArray
    statistics: tuple[str, ...]
    critical_values: FloatArray
    actual_size: FloatArray
    power: FloatArray
    sample_space_size: int


def _mass(counts: FloatArray, coefficient: FloatArray, probabilities: FloatArray) -> FloatArray:
    mass = np.exp(coefficient + xlogy(counts, probabilities).sum(axis=1))
    total = float(np.sum(mass, dtype=np.longdouble))
    if not np.isfinite(total) or abs(total - 1) > 1e-8:
        raise ArithmeticError("enumerated multinomial probability does not sum to one")
    return mass / total


def multinomial_power(
    n: int,
    null: ArrayLike,
    alternatives: ArrayLike,
    alpha: ArrayLike = (0.05,),
    *,
    statistic: str = "both",
    max_points: int = 1_000_000,
) -> MultinomialPower:
    """Enumerate exact power with Pearson or likelihood-ratio tail ordering.

    Accept ``statistic='pearson'``, ``'likelihood_ratio'`` or ``'both'``.
    Null probabilities must be strictly positive; alternative probabilities
    may be zero. Vectors must sum to one within 1e-12 and are normalized.
    ``n`` is an integer in [1, 10000], with 2–10 categories. ``alpha`` is a
    scalar or nonempty vector in [0, 1]. A single alternative vector is accepted.

    Complete statistic tie groups are included in descending order while the
    cumulative null probability fits alpha. The comparison permits relative
    rounding error of 32 machine epsilons. Alpha zero always rejects nothing;
    alpha one rejects everything. Probabilities are evaluated in log space,
    checked for total mass, then normalized to remove accumulated roundoff.

    The sample space has comb(n + k - 1, k - 1) points. Its size is checked
    before allocation. NumPy evaluates all point statistics together; each
    alternative is processed separately to avoid a points × alternatives ×
    categories temporary. This corrects original uninitialized probabilities,
    single-precision coefficients, and empty-region sentinel behavior.
    """
    if isinstance(n, (bool, np.bool_)):
        raise ValueError("n must be an integer in [1, 10000]")
    try:
        n = index(n)
        budget = index(max_points)
    except TypeError as exc:
        raise ValueError("n and max_points must be integers") from exc
    if not 1 <= n <= 10000:
        raise ValueError("n must be an integer in [1, 10000]")
    if isinstance(max_points, (bool, np.bool_)) or budget < 1:
        raise ValueError("max_points must be a positive integer")
    if statistic not in ("both", "pearson", "likelihood_ratio"):
        raise ValueError("statistic must be 'both', 'pearson', or 'likelihood_ratio'")
    p0 = finite(null, "null")
    if p0.ndim != 1 or not 2 <= p0.size <= 10:
        raise ValueError("null must be a vector with 2–10 categories")
    p0 = _probabilities(p0, "null")
    if np.any(p0 <= 0):
        raise ValueError("null probabilities must be strictly positive")
    pa = finite(alternatives, "alternatives")
    if pa.ndim == 1:
        pa = pa[None, :]
    if pa.ndim != 2 or pa.shape[0] == 0 or pa.shape[1] != p0.size:
        raise ValueError("alternatives must have one or more rows with the same categories as null")
    pa = _probabilities(pa, "alternatives")
    levels = np.atleast_1d(finite(alpha, "alpha"))
    if levels.ndim != 1 or levels.size == 0 or np.any((levels < 0) | (levels > 1)):
        raise ValueError("alpha must be a nonempty scalar or vector in [0, 1]")
    size = comb(n + p0.size - 1, p0.size - 1)
    if size > budget:
        raise ValueError(f"sample space requires {size} points, exceeding max_points={budget}")
    # Stars and bars yields lexicographically ordered weak compositions.
    bars = np.fromiter(
        chain.from_iterable(combinations(range(n + p0.size - 1), p0.size - 1)),
        dtype=np.int64,
        count=size * (p0.size - 1),
    ).reshape(size, p0.size - 1)
    counts: FloatArray = (
        np.asarray(
            np.diff(np.pad(bars, ((0, 0), (1, 1)), constant_values=(-1, n + p0.size - 1))),
            dtype=np.float64,
        )
        - 1
    )
    coefficient = gammaln(n + 1) - gammaln(counts + 1).sum(axis=1)
    null_mass = _mass(counts, coefficient, p0)
    names = ("pearson", "likelihood_ratio") if statistic == "both" else (statistic,)
    critical = np.full((len(names), levels.size), np.inf)
    actual = np.zeros_like(critical)
    powers = np.zeros((len(names), pa.shape[0], levels.size))
    regions: list[tuple[NDArray[np.intp], NDArray[np.intp]]] = []
    expected = n * p0
    for method, name in enumerate(names):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if name == "pearson":
                values = np.sum((counts - expected) ** 2 / expected, axis=1)
            else:
                values = 2 * np.sum(xlogy(counts, counts / n) - xlogy(counts, p0), axis=1)
        if not np.all(np.isfinite(values)):
            raise ArithmeticError("multinomial statistics overflow for these null probabilities")
        order = np.argsort(-values, kind="stable")
        ordered = values[order]
        tied = np.abs(np.diff(ordered)) <= 1e-12 * np.maximum(
            np.abs(ordered[:-1]) + np.abs(ordered[1:]), 1e-20
        )
        ends = np.concatenate((np.flatnonzero(~tied), [size - 1]))
        cumulative = np.cumsum(null_mass[order], dtype=np.longdouble)[ends]
        cumulative[-1] = 1
        accepted = np.searchsorted(
            cumulative, levels * (1 + 32 * np.finfo(float).eps), side="right"
        )
        accepted[levels == 0] = 0
        accepted[levels == 1] = ends.size
        last = np.where(accepted > 0, ends[np.maximum(accepted - 1, 0)], -1)
        active = accepted > 0
        critical[method, active] = ordered[last[active]]
        actual[method, active] = cumulative[accepted[active] - 1]
        regions.append((order, last))
    for alternative, probabilities in enumerate(pa):
        mass = _mass(counts, coefficient, probabilities)
        for method, (order, last) in enumerate(regions):
            cumulative = np.cumsum(mass[order], dtype=np.longdouble)
            cumulative[-1] = 1
            active = last >= 0
            powers[method, alternative, active] = cumulative[last[active]]
    return MultinomialPower(
        n,
        _freeze(p0),
        _freeze(pa),
        _freeze(levels),
        names,
        _freeze(critical),
        _freeze(actual),
        _freeze(powers),
        size,
    )


def format_multinomial_power(result: MultinomialPower, *, digits: int = 8) -> str:
    """Render the problem, critical values, sizes and every alternative's power.

    Returns text for printing or saving with ``Path.write_text``. Category and
    alternative numbers are one-based. Empty regions are explicitly labeled.
    """
    if not isinstance(result, MultinomialPower):
        raise TypeError("result must be a MultinomialPower")
    if isinstance(digits, (bool, np.bool_)) or not isinstance(digits, (int, np.integer)):
        raise ValueError("digits must be an integer from 1 through 17")
    if not 1 <= digits <= 17:
        raise ValueError("digits must be an integer from 1 through 17")

    def number(value: float) -> str:
        return f"{value:.{digits}g}"

    lines = [
        "MULTINOMPOW exact multinomial power",
        f"Sample size: {result.n}",
        f"Categories: {result.null.size}",
        f"Possible outcomes: {result.sample_space_size}",
        "Null probabilities: " + " ".join(number(p) for p in result.null),
        "Nominal significance levels: " + " ".join(number(p) for p in result.alpha),
    ]
    labels = {"pearson": "Pearson chi-square", "likelihood_ratio": "Likelihood ratio"}
    for method, name in enumerate(result.statistics):
        lines.extend(["", labels[name], "Nominal alpha\tActual size\tCritical value"])
        for j, alpha in enumerate(result.alpha):
            critical = result.critical_values[method, j]
            label = "empty region" if np.isinf(critical) else number(critical)
            lines.append(f"{number(alpha)}\t{number(result.actual_size[method, j])}\t{label}")
        for alternative, probabilities in enumerate(result.alternatives):
            lines.extend(
                [
                    f"Alternative {alternative + 1}: " + " ".join(number(p) for p in probabilities),
                    "Nominal alpha\tActual size\tPower",
                ]
            )
            for j, alpha in enumerate(result.alpha):
                lines.append(
                    "\t".join(
                        number(value)
                        for value in (
                            alpha,
                            result.actual_size[method, j],
                            result.power[method, alternative, j],
                        )
                    )
                )
    return "\n".join(lines) + "\n"
