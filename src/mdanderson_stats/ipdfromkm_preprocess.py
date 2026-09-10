"""IPDfromKM's native digitized-coordinate cleaning rules."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, scalar
from .boin import _owned
from .ipdfromkm import _coordinates


@dataclass(frozen=True)
class PreparedKMCurve:
    time: FloatArray
    survival: FloatArray
    source_index: FloatArray
    omitted_missing: int
    omitted_outliers: int
    omitted_redundant: int
    omitted_out_of_range: int
    adjusted_survival: int
    baseline_added: bool


def prepare_km_coordinates(
    time: ArrayLike,
    survival: ArrayLike,
    *,
    scale: float = 100,
) -> PreparedKMCurve:
    """Clean coordinates using the native preprocess.R ordering and fences.

    scale is 100 for percentages (the native default) or 1 for probabilities.
    NaN rows are omitted; infinities are rejected. Outlier deletion precedes a
    cumulative-minimum survival correction, duplicate reduction and range
    filtering. source_index is zero-based, with -1 for an inserted (0,1).
    Inspect the result: the source's heuristic can remove real survival drops.
    """
    t, s = np.asarray(time, dtype=float), np.asarray(survival, dtype=float)
    if t.ndim != 1 or s.shape != t.shape or not 5 <= t.size <= 100_000:
        raise ValueError("time and survival must be equal-length vectors of 5..100000 points")
    divisor = scalar(scale, "scale")
    if divisor not in (1, 100):
        raise ValueError("scale must be 1 or 100")
    if np.any(np.isinf(t)) or np.any(np.isinf(s)):
        raise ValueError("coordinates must not contain infinities")
    valid = ~np.isnan(t) & ~np.isnan(s)
    omitted_missing = int(np.count_nonzero(~valid))
    if np.count_nonzero(valid) < 2:
        raise ValueError("too few finite coordinates remain")
    index = np.flatnonzero(valid)
    order = np.argsort(t[index], kind="stable")
    index = index[order]
    t, s = t[index], s[index] / divisor
    with np.errstate(over="ignore"):
        spans = np.abs(np.diff(s, prepend=s[0]))
    if not np.all(np.isfinite(spans)):
        raise ValueError("survival differences overflow; review coordinate scale")
    q1, q3 = np.quantile(spans, [0.25, 0.75])
    fence = (q3 - q1) / 2
    flagged = (spans <= q1 - fence) | (spans >= q3 + fence)
    remove = flagged & ~np.r_[False, flagged[:-1]] & np.r_[flagged[1:], True]
    omitted_outliers = int(remove.sum())
    t, s, index = t[~remove], s[~remove], index[~remove]
    if not t.size:
        raise ValueError("outlier filtering removed every coordinate")
    monotone = np.minimum.accumulate(s)
    adjusted = int(np.count_nonzero(monotone != s))
    s = monotone
    # Cumulative minima preserve descending order within each tied time. Keep
    # its first and last distinct survival values, equivalent to native slicing.
    first = np.r_[True, t[1:] != t[:-1]]
    last = np.r_[t[1:] != t[:-1], True]
    keep = first | last
    chosen = np.flatnonzero(keep)
    repeated = np.r_[False, (t[chosen][1:] == t[chosen][:-1]) & (s[chosen][1:] == s[chosen][:-1])]
    chosen = chosen[~repeated]
    omitted_redundant = int(t.size - chosen.size)
    t, s, index = t[chosen], s[chosen], index[chosen]
    in_range = (t >= 0) & (s >= 0) & (s <= 1)
    omitted_range = int(np.count_nonzero(~in_range))
    t, s, index = t[in_range], s[in_range], index[in_range]
    if not t.size:
        raise ValueError("no in-range coordinates remain after cleaning")
    baseline = bool(t[0] != 0 or s[0] != 1)
    if baseline:
        t, s, index = np.r_[0, t], np.r_[1, s], np.r_[-1, index]
    _coordinates(t, s)
    return PreparedKMCurve(
        _owned(t),
        _owned(s),
        _owned(index.astype(float)),
        omitted_missing,
        omitted_outliers,
        omitted_redundant,
        omitted_range,
        adjusted,
        baseline,
    )
