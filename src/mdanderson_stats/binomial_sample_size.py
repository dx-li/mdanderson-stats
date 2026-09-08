"""Minimum sample size for a nonrandomized one-sided binomial design."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite
from .binomial_design import BinomialPower, binomial_power


def binomial_sample_size(
    null_probability: ArrayLike,
    alternative_probability: ArrayLike,
    alpha: ArrayLike = 0.05,
    target_power: ArrayLike = 0.8,
    *,
    min_trials: int = 2,
    max_trials: int = 10**10,
    batch_size: int = 256,
) -> BinomialPower:
    """Find the first integer sample size satisfying significance and target power.

    Inputs broadcast. Search bounds are inclusive, with 1<=min_trials<=max_trials
    <=1e10. The returned BinomialPower includes the selected sample size, critical
    count, achieved significance/power and adjacent more permissive test.
    Randomized-test power locates a lower search bound; ordered batches then find
    the first nonrandomized design, preserving the discrete power oscillations.
    No design within the supplied bounds raises ValueError.
    """
    for value, name in (
        (min_trials, "min_trials"),
        (max_trials, "max_trials"),
        (batch_size, "batch_size"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if min_trials > max_trials or max_trials > 10**10:
        raise ValueError("Require min_trials<=max_trials<=1e10")
    p0, pa, level, target = np.broadcast_arrays(
        finite(null_probability, "null_probability"),
        finite(alternative_probability, "alternative_probability"),
        finite(alpha, "alpha"),
        finite(target_power, "target_power"),
    )
    # Reuse the forward design's domain validation before starting the search.
    binomial_power(min_trials, p0, pa, level)
    if np.any((target < level) | (target >= 1)):
        raise ValueError("Require alpha<=target_power<1")

    def relaxed(n: FloatArray) -> FloatArray:
        result = binomial_power(n, p0, pa, level)
        mass = result.next_significance - result.significance
        weight = np.ones(mass.shape)
        np.divide(level - result.significance, mass, out=weight, where=mass > 0)
        power = result.power + np.clip(weight, 0, 1) * (result.next_power - result.power)
        # If the boundary mass cannot be resolved, one is a safe upper bound.
        # A small upward cushion makes the lower-bound search conservative near ties.
        return np.where(mass > 0, np.minimum(1, power + 1e-12), 1)

    lo = np.full(p0.shape, min_trials - 1, dtype=float)
    hi = np.full(p0.shape, min_trials, dtype=float)
    while True:
        insufficient = relaxed(hi) < target
        if not np.any(insufficient):
            break
        if np.any(insufficient & (hi == max_trials)):
            raise ValueError("No qualifying design within the requested sample-size bounds")
        lo = np.where(insufficient, hi, lo)
        hi = np.where(insufficient, np.minimum(max_trials, hi * 2), hi)
    while np.any(hi - lo > 1):
        mid = lo + np.floor((hi - lo) / 2)
        # Completed entries can have mid=0 when min_trials=1.
        insufficient = relaxed(np.maximum(mid, 1)) < target
        active = hi - lo > 1
        lo = np.where(active & insufficient, mid, lo)
        hi = np.where(active & ~insufficient, mid, hi)
    start = np.maximum(min_trials, hi - 1)
    solution = np.zeros(p0.shape)
    pending = np.ones(p0.shape, dtype=bool)
    offsets = np.arange(min(batch_size, max_trials - min_trials + 1))
    while np.any(pending):
        trials = np.minimum(start[..., None] + offsets, max_trials)
        candidate = binomial_power(trials, p0[..., None], pa[..., None], level[..., None])
        passes = candidate.power >= target[..., None]
        found = np.any(passes, axis=-1) & pending
        first = np.argmax(passes, axis=-1)
        selected = np.take_along_axis(trials, first[..., None], axis=-1)[..., 0]
        solution = np.where(found, selected, solution)
        pending &= ~found
        if np.any(pending & (trials[..., -1] == max_trials)):
            raise ValueError("No qualifying design within the requested sample-size bounds")
        start = np.where(pending, start + len(offsets), start)
    return binomial_power(solution, p0, pa, level)
