"""Vectorized reproduction of CID2BP's native Peskun grid traversal."""

import numpy as np


def _peskun(n1: int, x1: int, n2: int, x2: int, z: float) -> tuple[float, float]:
    if (n1 + 1) * (n2 + 1) > 2000000:
        raise ValueError("peskun_native permits at most 2 million grid points")
    difference = x1 / n1 - x2 / n2
    # Preserve Fortran i-outer/j-inner ordering and its 1e-6 exclusion tolerance.
    values = ((np.arange(n1 + 1) / n1 - difference)[:, None] - np.arange(n2 + 1) / n2).ravel()
    magnitude = np.abs(values)
    valid = magnitude > 1e-6
    negative = np.where(valid & (values <= 0), magnitude, np.inf)
    cumulative = np.minimum.accumulate(negative)
    previous = np.concatenate(([np.inf], cumulative[:-1]))
    updated_negative = negative < previous
    low_gap = cumulative[-1] if np.isfinite(cumulative[-1]) else 0.0
    # The native ELSE branch also admits negative values that did not establish
    # a new minimum. Deliberately retain it for this explicitly native method.
    upper_candidates = magnitude[valid & ~updated_negative]
    high_gap = float(upper_candidates.min()) if upper_candidates.size else 1e30
    cl, cu = low_gap / 2, high_gap / 2
    total = n1 + n2
    a = (total + z * z) / (4 * n1 * n2)
    denominator = 1 + z * z / total
    with np.errstate(invalid="ignore"):
        lower = (difference - cl - z * np.sqrt(a - (difference - cl) ** 2 / total)) / denominator
        upper = (difference + cu + z * np.sqrt(a - (difference + cu) ** 2 / total)) / denominator
    # The caller applies the source's exact special-case adjustments next.
    return float(lower), float(upper)
