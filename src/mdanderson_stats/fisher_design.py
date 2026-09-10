"""Unconditional power of conditional Fisher tests under independent binomial sampling."""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import binom, hypergeom

from ._validation import FloatArray, count, finite, scalar
from .binary_sample_size import BinarySampleSize
from .continuous_sample_size import _level, _probability


def fisher_power(
    control_size: int,
    treatment_size: int,
    control_rate: ArrayLike,
    treatment_rate: ArrayLike,
    *,
    alpha: float = 0.05,
    alternative: Literal["greater", "less", "two-sided"] = "greater",
) -> FloatArray:
    """Exact enumeration over both response counts, broadcasting response-rate pairs.

    greater/less refers to the treatment response rate. The two-sided test uses
    probability ordering, including hypergeometric masses tied within relative
    tolerance 1e-12 (as in the package's Fisher analysis). Both arm sizes are fixed;
    the combined response total is random. Rates may include zero and one.
    """
    alpha = _level(alpha, 1)
    sizes = count([control_size, treatment_size], "group sizes")
    if sizes.shape != (2,) or np.any((sizes < 1) | (sizes > 2000)):
        raise ValueError("each group size must be a scalar integer in [1, 2000]")
    nc, nt = map(int, sizes)
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError("alternative must be greater, less or two-sided")
    pc, pt = np.broadcast_arrays(
        finite(control_rate, "control_rate"), finite(treatment_rate, "treatment_rate")
    )
    if np.any((pc < 0) | (pc > 1) | (pt < 0) | (pt > 1)):
        raise ValueError("response rates must lie in [0, 1]")
    control_mass = binom.pmf(np.arange(nc + 1), nc, pc[..., None])
    treatment_mass = binom.pmf(np.arange(nt + 1), nt, pt[..., None])
    result = np.zeros(pc.shape)
    for successes in range(nc + nt + 1):
        x = np.arange(max(0, successes - nc), min(nt, successes) + 1)
        if alternative == "greater":
            pvalue = hypergeom.sf(x - 1, nc + nt, successes, nt)
        elif alternative == "less":
            pvalue = hypergeom.cdf(x, nc + nt, successes, nt)
        else:
            mass = hypergeom.pmf(x, nc + nt, successes, nt)
            ordered = np.sort(mass)
            cumulative = np.cumsum(ordered)
            last = np.searchsorted(ordered, mass * (1 + 1e-12), side="right") - 1
            pvalue = cumulative[last]
        selected = x[pvalue <= alpha]
        if selected.size:
            result += np.sum(
                treatment_mass[..., selected] * control_mass[..., successes - selected], axis=-1
            )
    return _probability(result)


def fisher_sample_size(
    control_rate: float,
    treatment_rate: float,
    *,
    allocation_ratio: float = 1,
    alpha: float = 0.05,
    alternative: Literal["greater", "less", "two-sided"] = "greater",
    target_power: float = 0.8,
    max_size: int = 500,
) -> BinarySampleSize:
    """First qualifying control size; scan all integers because exact power oscillates.

    Treatment size is ceil(allocation_ratio*control_size). The search covers
    control sizes 1..max_size, limited to 2000 in either arm. No feasible design
    raises ValueError. This performs exact enumeration, not a normal approximation
    to a Fisher sample-size formula.
    """
    pc, pt = scalar(control_rate, "control_rate"), scalar(treatment_rate, "treatment_rate")
    ratio = scalar(allocation_ratio, "allocation_ratio")
    target, limit = scalar(target_power, "target_power"), scalar(max_size, "max_size")
    if not 0 <= pc <= 1 or not 0 <= pt <= 1 or pc == pt:
        raise ValueError("distinct response rates in [0, 1] are required")
    if (alternative == "greater" and pt < pc) or (alternative == "less" and pt > pc):
        raise ValueError("rates must satisfy the requested alternative")
    if not 0 < target < 1 or not 0 < ratio <= 2000:
        raise ValueError("require target_power in (0, 1) and allocation_ratio in (0, 2000]")
    if limit != int(limit) or not 1 <= limit <= 2000:
        raise ValueError("max_size must be an integer in [1, 2000]")
    previous = None
    for nc in range(1, int(limit) + 1):
        nt = int(np.ceil(nc * ratio))
        if nt > 2000:
            raise ValueError("no design found before treatment enrollment exceeded 2000")
        power = float(fisher_power(nc, nt, pc, pt, alpha=alpha, alternative=alternative))
        if power >= target:
            return BinarySampleSize(
                (nc, nt), power, previous, target_power, f"Fisher: exact {alternative}"
            )
        previous = power
    raise ValueError("target power is not attained within max_size")
