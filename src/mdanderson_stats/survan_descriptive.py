"""SURVAN frequency tables and numerically stable descriptive statistics."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray


def _data(value: ArrayLike) -> FloatArray:
    if np.iscomplexobj(value):
        raise ValueError("descriptive data must be real")
    x = np.asarray(value, dtype=float)
    if np.isinf(x).any():
        raise ValueError("infinite observations are invalid; use NaN for missing observations")
    return x


@dataclass(frozen=True)
class SurvanDescription:
    count: FloatArray
    missing: FloatArray
    mean: FloatArray
    variance: FloatArray
    standard_deviation: FloatArray
    standard_error: FloatArray
    legacy_standard_error: FloatArray
    minimum: FloatArray
    maximum: FloatArray
    range: FloatArray


def survan_describe(x: ArrayLike) -> SurvanDescription:
    """Column summaries; NaNs are omitted per column, variance uses n-1.

    Standard error is SD/sqrt(n). legacy_standard_error reproduces SURVAN's
    SD/(n-1), which is not the usual estimated standard error of a sample mean.
    Missing summaries and sample dispersion with fewer than two values are NaN.
    """
    a = _data(x)
    if a.ndim == 1:
        a = a[:, None]
    if (
        a.ndim != 2
        or not 1 <= a.shape[0] <= 1_000_000
        or not 1 <= a.shape[1] <= 200
        or a.size > 2_000_000
    ):
        raise ValueError("require 1..1000000 rows, 1..200 columns and <=2000000 entries")
    n = np.sum(~np.isnan(a), axis=0).astype(float)
    mean, var, sd, se, old_se, low, high, span = (np.full(a.shape[1], np.nan) for _ in range(8))
    for j in range(a.shape[1]):
        values = a[~np.isnan(a[:, j]), j]
        if not values.size:
            continue
        low[j], high[j] = values.min(), values.max()
        anchor = values[0]
        with np.errstate(over="ignore"):
            delta = values - anchor
            span[j] = high[j] - low[j]
        if not np.isfinite(delta).all():
            anchor, delta = 0.0, values
        scale = float(np.max(np.abs(delta)))
        normalized = delta / scale if scale else np.zeros_like(delta)
        average = float(normalized.mean())
        mean[j] = anchor + scale * average
        if values.size > 1:
            centered = normalized - average
            normalized_sd = np.sqrt(np.sum(centered * centered) / (values.size - 1))
            with np.errstate(over="ignore", under="ignore"):
                sd[j] = scale * normalized_sd
                var[j] = sd[j] * sd[j]
            se[j] = sd[j] / np.sqrt(values.size)
            old_se[j] = sd[j] / (values.size - 1)
            if not np.isfinite(var[j]) or (var[j] == 0 and sd[j] > 0):
                raise ArithmeticError("sample variance exceeds representable numerical range")
        if not np.isfinite(mean[j]) or not np.isfinite(span[j]):
            raise ArithmeticError("descriptive mean or range exceeds numerical range")
    return SurvanDescription(
        _freeze(n),
        _freeze(a.shape[0] - n),
        _freeze(mean),
        _freeze(var),
        _freeze(sd),
        _freeze(se),
        _freeze(old_se),
        _freeze(low),
        _freeze(high),
        _freeze(span),
    )


@dataclass(frozen=True)
class SurvanFrequencies:
    value: FloatArray
    count: FloatArray
    percent: FloatArray
    cumulative_percent: FloatArray
    missing: int
    observations: int
    legacy_grouping: bool


def survan_frequencies(x: ArrayLike, *, legacy_grouping: bool = False) -> SurvanFrequencies:
    """Sorted numeric frequencies, with optional native first-matching grouping.

    NaNs count as missing and are excluded from percentages. Exact equality is
    the default. Legacy groups retain their first-observed representative and
    match when |a-b|/max(|a|+|b|,1e-20) < 1e-5; results depend on input order.
    """
    a = _data(x)
    if a.ndim != 1 or not 1 <= a.size <= 1_000_000 or not isinstance(legacy_grouping, bool):
        raise ValueError("require a vector of 1..1000000 records and a boolean grouping flag")
    missing = int(np.isnan(a).sum())
    values = a[~np.isnan(a)]
    if not legacy_grouping:
        representatives, counts = np.unique(values, return_counts=True)
    else:
        # Preserve original insertion order until all matches have been assigned.
        representatives = np.empty(values.size)
        counts = np.zeros(values.size, dtype=int)
        used, work = 0, 0
        for value in values:
            work += used
            if work > 10_000_000:
                raise ValueError("native approximate grouping exceeds 10000000 comparisons")
            prior = representatives[:used]
            scale = np.maximum(np.maximum(np.abs(prior), abs(value)), 1e-20)
            distance = np.abs(prior / scale - value / scale)
            denominator = np.maximum(np.abs(prior / scale) + abs(value) / scale, 1e-20 / scale)
            matches = np.flatnonzero(distance < 1e-5 * denominator)
            if matches.size:
                counts[int(matches[0])] += 1
            else:
                representatives[used], counts[used] = value, 1
                used += 1
        order = np.argsort(representatives[:used], kind="stable")
        representatives, counts = representatives[:used][order], counts[:used][order]
    percentage = 100 * counts.astype(float) / values.size if values.size else np.empty(0)
    cumulative = 100 * np.cumsum(counts, dtype=float) / values.size if values.size else np.empty(0)
    return SurvanFrequencies(
        _freeze(representatives),
        _freeze(counts),
        _freeze(percentage),
        _freeze(cumulative),
        missing,
        int(values.size),
        legacy_grouping,
    )
