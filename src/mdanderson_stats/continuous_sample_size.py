"""Continuous-endpoint planning: mean tests, correlation and balanced ANOVA."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.special import ndtr, ndtri
from scipy.stats import chi2, f, ncf, nct, t

from ._validation import FloatArray, count, finite, scalar

MeanObjective = Literal["equality", "equivalence", "noninferiority", "superiority"]


def _level(alpha: float, sides: int) -> float:
    alpha = scalar(alpha, "alpha")
    if not 0 < alpha < 0.5 or sides not in (1, 2):
        raise ValueError("require 0 < alpha < .5 and sides in {1, 2}")
    return alpha


def _probability(value: ArrayLike) -> FloatArray:
    value = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(value) | (value < -1e-12) | (value > 1 + 1e-12)):
        raise ArithmeticError("power calculation failed to return a finite probability")
    return np.clip(value, 0, 1)


def _tost_power(df: float, effect: float, width: float, critical: float) -> float:
    """Integrate the conditional TOST acceptance interval over residual variance."""
    maximum = chi2.cdf(df * (width / critical) ** 2, df)

    def integrand(u: float) -> float:
        radius = width - critical * np.sqrt(chi2.ppf(u, df) / df)
        lo, hi = -radius - effect, radius - effect
        if radius <= 0:
            return 0.0
        return float(ndtr(-lo) - ndtr(-hi) if lo > 0 else ndtr(hi) - ndtr(lo))

    value, error = quad(integrand, 0, maximum, epsabs=2e-10, epsrel=2e-9, limit=200)
    if error > 1e-8:
        raise ArithmeticError("equivalence power integration did not reach absolute error 1e-8")
    return float(value)


def normal_mean_power(
    sample_size: ArrayLike,
    difference: ArrayLike,
    sd: ArrayLike = 1,
    *,
    treatment_size: ArrayLike | None = None,
    objective: MeanObjective = "equality",
    margin: ArrayLike = 0,
    alpha: float = 0.05,
    sides: int = 2,
    exact: bool = False,
) -> FloatArray:
    """Power for a one-sample/paired or pooled equal-variance two-sample t test.

    sample_size is the single sample size or control size. Supplying treatment_size
    selects the independent two-sample test. Equality uses |difference|, so the
    one-sided direction is the supplied effect's direction. Margin tests use the
    signed treatment-minus-control effect, with larger outcomes favorable.

    Default planning formulas retain only the dominant equality tail and use a
    nonnegative lower bound for TOST equivalence. exact=True adds the opposite
    equality tail or integrates the full TOST rejection region. Margin objectives
    are inherently one-sided (TOST has two one-sided tests); sides applies only
    to equality.
    """
    alpha = _level(alpha, sides)
    if not isinstance(exact, (bool, np.bool_)):
        raise ValueError("exact must be a boolean")
    if objective not in ("equality", "equivalence", "noninferiority", "superiority"):
        raise ValueError("unknown mean-test objective")
    n, delta, sigma, width = np.broadcast_arrays(
        count(sample_size, "sample_size"),
        finite(difference, "difference"),
        finite(sd, "sd"),
        finite(margin, "margin"),
    )
    if np.any(n < 2) or np.any(sigma <= 0):
        raise ValueError("sample sizes must be at least 2 and sd must be positive")
    if np.any(width < 0) or (objective != "equality" and np.any(width <= 0)):
        raise ValueError("margin must be positive for margin tests and nonnegative otherwise")
    if objective == "equality" and np.any(width != 0):
        raise ValueError("margin must be zero for equality")
    if treatment_size is None:
        df, information = n - 1, np.sqrt(n)
    else:
        n, nt, delta, sigma, width = np.broadcast_arrays(
            n, count(treatment_size, "treatment_size"), delta, sigma, width
        )
        if np.any(nt < 2) or np.any(n + nt >= 2**53):
            raise ValueError("each group needs at least 2 observations; total must be below 2**53")
        df, information = n + nt - 2, 1 / np.sqrt(1 / n + 1 / nt)
    effect, bound = delta / sigma * information, width / sigma * information
    if np.any(~np.isfinite(effect) | ~np.isfinite(bound)):
        raise ArithmeticError("standardized effect or margin is not representable")
    critical = t.isf(alpha / sides if objective == "equality" else alpha, df)
    if objective == "equivalence":
        if exact:
            output = np.empty(effect.shape)
            for index in np.ndindex(output.shape):
                output[index] = _tost_power(
                    float(df[index]),
                    float(effect[index]),
                    float(bound[index]),
                    float(critical[index]),
                )
        else:
            # Bonferroni lower bound for the intersection of the two rejections.
            output = np.maximum(
                0, nct.sf(critical, df, bound - effect) - nct.cdf(critical, df, bound + effect)
            )
    else:
        shift = (
            np.abs(effect)
            if objective == "equality"
            else (effect + bound if objective == "noninferiority" else effect - bound)
        )
        output = nct.sf(critical, df, shift)
        if exact and objective == "equality" and sides == 2:
            output = output + nct.cdf(-critical, df, shift)
    return _probability(output)


@dataclass(frozen=True)
class ContinuousSampleSize:
    """Integer enrollment with power at the chosen and preceding search size."""

    group_sizes: tuple[int, ...]
    power: float
    previous_power: float | None
    target_power: float
    method: str

    @property
    def total_size(self) -> int:
        return sum(self.group_sizes)


def _search(
    evaluate: Callable[[int], float], target: float, minimum: int, maximum: int
) -> tuple[int, float, float | None]:
    target = scalar(target, "target_power")
    limit = scalar(maximum, "max_size")
    if not 0 < target < 1:
        raise ValueError("target_power must lie in (0, 1)")
    if limit != int(limit) or not minimum <= limit <= 10_000_000:
        raise ValueError("max_size must be an integer between the minimum size and 10000000")
    maximum = int(limit)
    low, high = minimum - 1, minimum
    while evaluate(high) < target:
        if high == maximum:
            raise ValueError("target power is not attained within max_size")
        low, high = high, min(2 * high, maximum)
    while high - low > 1:
        middle = (high + low) // 2
        if evaluate(middle) >= target:
            high = middle
        else:
            low = middle
    return high, evaluate(high), evaluate(high - 1) if high > minimum else None


def normal_mean_sample_size(
    difference: float,
    sd: float = 1,
    *,
    allocation_ratio: float | None = None,
    objective: MeanObjective = "equality",
    margin: float = 0,
    alpha: float = 0.05,
    sides: int = 2,
    target_power: float = 0.8,
    exact: bool = False,
    max_size: int = 10_000_000,
) -> ContinuousSampleSize:
    """First qualifying single/control size; treatment size is ceil(ratio*control).

    allocation_ratio=None selects one sample (or paired differences). The ratio
    is treatment/control. With a ratio, each group has at least two observations.
    max_size bounds single/control enrollment, not total enrollment.
    """
    delta, sigma, width = (
        scalar(difference, "difference"),
        scalar(sd, "sd"),
        scalar(margin, "margin"),
    )
    ratio = None if allocation_ratio is None else scalar(allocation_ratio, "allocation_ratio")
    if ratio is not None and not 1e-6 <= ratio <= 1e6:
        raise ValueError("allocation_ratio must be between 1e-6 and 1e6")
    valid = {
        "equality": delta != 0,
        "equivalence": abs(delta) < width,
        "noninferiority": delta + width > 0,
        "superiority": delta - width > 0,
    }
    if objective not in valid or not valid[objective]:
        raise ValueError("the expected difference must lie inside the requested alternative")

    def sizes(n: int) -> tuple[int, ...]:
        return (n,) if ratio is None else (n, max(2, int(np.ceil(ratio * n))))

    def evaluate(n: int) -> float:
        groups = sizes(n)
        return float(
            normal_mean_power(
                n,
                delta,
                sigma,
                treatment_size=groups[1] if len(groups) == 2 else None,
                objective=objective,
                margin=width,
                alpha=alpha,
                sides=sides,
                exact=exact,
            )
        )

    n, achieved, previous = _search(evaluate, target_power, 2, max_size)
    mode = "exact" if exact else "conservative planning"
    return ContinuousSampleSize(sizes(n), achieved, previous, target_power, f"{objective}: {mode}")


def correlation_power(
    sample_size: ArrayLike,
    correlation: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 2,
    method: Literal["t", "z"] = "t",
) -> FloatArray:
    """Source Fisher-z planning approximations, with or without t critical values.

    Both are approximations, not exact finite-sample correlation-test power.
    Two-sided planning retains the dominant tail; the sign selects direction.
    """
    alpha = _level(alpha, sides)
    n, r = np.broadcast_arrays(
        count(sample_size, "sample_size"), finite(correlation, "correlation")
    )
    if np.any(n < 4) or np.any(np.abs(r) >= 1):
        raise ValueError("require sample_size >= 4 and abs(correlation) < 1")
    r = np.abs(r)
    center = np.arctanh(r)
    if method == "z":
        argument = center * np.sqrt(n - 3) + ndtri(alpha / sides)
    elif method == "t":
        critical = t.isf(alpha / sides, n - 2)
        # atanh(sqrt(t²/(t²+df))) = asinh(t/sqrt(df)), without rounding r to 1.
        transformed_critical = np.arcsinh(critical / np.sqrt(n - 2))
        argument = (center + r / (2 * (n - 1)) - transformed_critical) * np.sqrt(n - 3)
    else:
        raise ValueError("method must be 't' or 'z'")
    return _probability(ndtr(argument))


def correlation_sample_size(
    correlation: float,
    *,
    alpha: float = 0.05,
    sides: int = 2,
    method: Literal["t", "z"] = "t",
    target_power: float = 0.8,
    max_size: int = 10_000_000,
) -> ContinuousSampleSize:
    """First integer paired-observation count meeting the source planning power."""
    r = scalar(correlation, "correlation")
    if not 0 < abs(r) < 1:
        raise ValueError("require 0 < abs(correlation) < 1")
    n, achieved, previous = _search(
        lambda n: float(correlation_power(n, r, alpha=alpha, sides=sides, method=method)),
        target_power,
        4,
        max_size,
    )
    return ContinuousSampleSize(
        (n,), achieved, previous, target_power, f"correlation: {method} approximation"
    )


def anova_effect_size(means: ArrayLike, sd: ArrayLike) -> FloatArray:
    """Cohen's f from means on the final group axis and common within-group SD."""
    means, sigma = finite(means, "means"), finite(sd, "sd")
    if means.ndim < 1 or means.shape[-1] < 2 or np.any(sigma <= 0):
        raise ValueError("at least two group means and positive sd are required")
    means, sigma = np.broadcast_arrays(means, sigma[..., None])
    # Scale before squaring; a common offset is removed before division.
    shifted = means - means[..., :1]
    if np.any(~np.isfinite(shifted)):
        raise ArithmeticError("group mean differences are not representable")
    scale = np.max(np.abs(shifted), axis=-1, keepdims=True)
    standardized = shifted / np.where(scale == 0, 1, scale)
    rms = np.sqrt(np.mean((standardized - standardized.mean(axis=-1, keepdims=True)) ** 2, axis=-1))
    result = (rms * scale[..., 0]) / sigma[..., 0]
    if np.any(~np.isfinite(result)):
        raise ArithmeticError("ANOVA effect size is not representable")
    return np.asarray(result)


def anova_power(
    sample_size: ArrayLike,
    effect_size: ArrayLike,
    groups: int,
    *,
    alpha: float = 0.05,
) -> FloatArray:
    """Exact normal-model F-test power for equal integer sample sizes per group."""
    alpha = _level(alpha, 1)
    m = scalar(groups, "groups")
    if m != int(m) or not 2 <= m <= 100_000:
        raise ValueError("groups must be an integer in [2, 100000]")
    n, effect = np.broadcast_arrays(
        count(sample_size, "sample_size"), finite(effect_size, "effect_size")
    )
    if np.any(n < 2) or np.any(effect < 0) or np.any(m * n >= 2**53):
        raise ValueError("require sample_size >= 2, effect_size >= 0 and total < 2**53")
    noncentrality = (effect * np.sqrt(m * n)) ** 2
    if np.any(~np.isfinite(noncentrality)):
        raise ArithmeticError("ANOVA noncentrality is not representable")
    df = m * (n - 1)
    critical = f.isf(alpha, m - 1, df)
    # The central branch also avoids ncf.sf's special-case behavior at nc=0.
    values = ncf.sf(critical, m - 1, df, np.where(noncentrality == 0, 1, noncentrality))
    return _probability(np.where(noncentrality == 0, alpha, values))


def anova_sample_size(
    effect_size: float,
    groups: int,
    *,
    alpha: float = 0.05,
    target_power: float = 0.8,
    max_size: int = 10_000_000,
) -> ContinuousSampleSize:
    """First qualifying per-group sample size for a balanced one-way ANOVA."""
    effect = scalar(effect_size, "effect_size")
    if effect <= 0:
        raise ValueError("effect_size must be positive")
    n, achieved, previous = _search(
        lambda n: float(anova_power(n, effect, groups, alpha=alpha)),
        target_power,
        2,
        max_size,
    )
    return ContinuousSampleSize(
        (n,) * int(groups), achieved, previous, target_power, "ANOVA: exact F"
    )
