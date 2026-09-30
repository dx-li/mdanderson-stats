"""Reconstruction diagnostics reported by IPDfromKM's getIPD routine."""

from __future__ import annotations

from collections.abc import Sized
from dataclasses import dataclass
from math import comb, exp, pi, sqrt
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray
from .boin import _owned

_MAX_CURVE_POINTS = 100_000
_EXACT_KS_PRODUCT_LIMIT = 10_000
_MAX_EXACT_TRANSITIONS = 5_000_000


@dataclass(frozen=True)
class IPDReconstructionDiagnostics:
    """Rounded report metrics and a nominal independent-sample KS comparison.

    ``legacy_signed_max_error`` preserves the source's ``max(diff)`` value,
    despite its report label saying "max absolute error". The actual maximum
    absolute error is returned alongside it.
    """

    rounded_reconstructed_survival: FloatArray
    rounded_difference: FloatArray
    retained_indices: NDArray[np.int64]
    n_points: int
    omitted_points: int
    rmse: float
    mean_absolute_error: float
    legacy_signed_max_error: float
    max_absolute_error: float
    ks_statistic: float
    ks_pvalue: float
    ks_method: Literal["exact", "asymptotic"]


def _probability_vector(value: ArrayLike, name: str) -> FloatArray:
    if isinstance(value, Sized):
        raw_length = len(value)
        if raw_length > _MAX_CURVE_POINTS:
            raise ValueError(f"{name} may contain at most {_MAX_CURVE_POINTS} values")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric vector") from exc
    if raw.ndim != 1 or not 2 <= raw.size <= _MAX_CURVE_POINTS:
        raise ValueError(f"{name} must contain 2..{_MAX_CURVE_POINTS} values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric vector") from exc
    if np.any(np.isinf(result)) or np.any(np.isfinite(result) & ((result < 0) | (result > 1))):
        raise ValueError(f"{name} must contain probabilities in [0, 1] or paired NaNs")
    return result


def _r_round(value: FloatArray | float, digits: int = 3) -> FloatArray | float:
    """Match R 4.4.1 ``round``'s distance-based decimal rounding."""
    values = np.asarray(value, dtype=np.float64)
    sign = np.where(values < 0.0, -1.0, 1.0)
    magnitude = np.abs(values)
    scale = float(10**digits)
    scaled = magnitude * scale
    lower_index = np.floor(scaled)
    lower = lower_index / scale
    upper = np.ceil(scaled) / scale
    distance_up = upper - magnitude
    distance_down = magnitude - lower
    choose_up = (distance_up < distance_down) | (
        (distance_up == distance_down) & (np.fmod(lower_index, 2.0) == 1.0)
    )
    rounded = sign * np.where(choose_up, upper, lower)
    if np.ndim(value) == 0:
        return float(rounded)
    return np.asarray(rounded, dtype=np.float64)


def _ks_distance(first: FloatArray, second: FloatArray) -> tuple[int, float, np.ndarray, int, int]:
    """Return the exact rational D, pooled block counts, and sample sizes."""
    n_first, n_second = int(first.size), int(second.size)
    pooled = np.concatenate((first, second))
    labels = np.concatenate((np.zeros(n_first, dtype=np.int8), np.ones(n_second, dtype=np.int8)))
    order = np.argsort(pooled, kind="stable")
    ordered = pooled[order]
    boundaries = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    pooled_counts = np.diff(np.r_[boundaries, ordered.size])
    first_counts = np.add.reduceat(labels[order] == 0, boundaries).astype(np.int64)
    second_counts = pooled_counts - first_counts
    cum_first = np.cumsum(first_counts, dtype=np.int64)
    cum_second = np.cumsum(second_counts, dtype=np.int64)
    numerator = int(np.max(np.abs(cum_first * n_second - cum_second * n_first)))
    statistic = numerator / (n_first * n_second)
    return numerator, statistic, pooled_counts, n_first, n_second


def _exact_tied_ks_pvalue(
    pooled_counts: np.ndarray,
    n_first: int,
    n_second: int,
    observed_numerator: int,
) -> float:
    """Conditional permutation tail, with ties kept as indivisible value blocks."""
    if n_first != n_second:
        raise ValueError("internal KS reference samples must have equal lengths")
    pooled_sizes = np.asarray(pooled_counts, dtype=np.int64)
    n_small, n_large = n_first, n_second
    total = int(n_small + n_large)
    transitions = (n_small + 1) * total
    if transitions > _MAX_EXACT_TRANSITIONS:
        raise ValueError("exact KS permutation calculation exceeds its work limit")

    # ``below[x]`` counts assignments with x small-sample labels in the
    # processed pooled blocks whose boundary D remains strictly below Dobs.
    below: dict[int, int] = {0: 1}
    cumulative_total = 0
    for block_size_raw in pooled_sizes:
        block_size = int(block_size_raw)
        cumulative_total += block_size
        remaining_total = total - cumulative_total
        next_below: dict[int, int] = {}
        for used_small, paths in below.items():
            low = max(0, n_small - used_small - remaining_total)
            high = min(block_size, n_small - used_small)
            for in_block in range(low, high + 1):
                new_small = used_small + in_block
                large_so_far = cumulative_total - new_small
                distance_numerator = abs(new_small * n_large - large_so_far * n_small)
                if distance_numerator < observed_numerator:
                    next_below[new_small] = next_below.get(new_small, 0) + (
                        paths * comb(block_size, in_block)
                    )
        below = next_below

    total_assignments = comb(total, n_small)
    below_assignments = sum(below.values())
    return (total_assignments - below_assignments) / total_assignments


def _kolmogorov_two_sided_survival(value: float) -> float:
    """Stable two-sided limiting KS tail used by R's asymptotic branch."""
    if value <= 0:
        return 1.0
    if value < 0.82:
        # Theta-transformed CDF converges rapidly near zero.
        cdf_sum = 0.0
        index = 1
        while True:
            exponent = -((2 * index - 1) ** 2) * pi**2 / (8 * value**2)
            term = exp(exponent)
            if term == 0.0:
                break
            cdf_sum += term
            if term < 1e-17 * max(cdf_sum, 1e-300):
                break
            index += 1
        cdf = sqrt(2 * pi) * cdf_sum / value
        return min(1.0, max(0.0, 1.0 - cdf))

    tail = 0.0
    index = 1
    while True:
        term = 2.0 * exp(-2.0 * index * index * value * value)
        if term == 0.0:
            break
        tail += term if index % 2 else -term
        if term < 1e-17 * max(abs(tail), 1e-300):
            break
        index += 1
    return min(1.0, max(0.0, tail))


def _ks_pvalue(
    d_numerator: int,
    pooled_counts: np.ndarray,
    n_first: int,
    n_second: int,
) -> tuple[float, Literal["exact", "asymptotic"]]:
    product = n_first * n_second
    if product < _EXACT_KS_PRODUCT_LIMIT:
        return (
            _exact_tied_ks_pvalue(pooled_counts, n_first, n_second, d_numerator),
            "exact",
        )
    d = d_numerator / product
    effective_n = n_first * n_second / (n_first + n_second)
    return _kolmogorov_two_sided_survival(sqrt(effective_n) * d), "asymptotic"


def ipd_reconstruction_diagnostics(
    observed_survival: ArrayLike,
    reconstructed_survival: ArrayLike,
) -> IPDReconstructionDiagnostics:
    """Reproduce IPDfromKM's rounded reconstruction report for paired inputs.

    The vectors are corresponding observed and fitted probabilities at the
    same curve coordinates. Observed probabilities remain
    unrounded; fitted probabilities are rounded to three decimal places, then
    the fitted-minus-observed differences are rounded to three places. A point
    is omitted if either member of a pair is NaN, matching the source's
    row-wise ``na.omit``; infinities and out-of-range finite values are
    rejected. The result includes retained input indices and the omitted count.

    The native RMSE divides the sum of squared rounded differences by ``n-1``;
    its other precision summaries divide by ``n``. The native report's
    "max absolute error" is actually ``max(diff)`` and is returned as
    ``legacy_signed_max_error``. ``max_absolute_error`` provides the correctly
    named counterpart. The two-sided KS p-value follows R 4.4.1's
    ``ks.test(x, y)`` default: exact tied-label permutation for ``n_x*n_y <
    10000``, otherwise the asymptotic limiting distribution. Since these are
    fitted, paired curve coordinates, the independent-sample KS p-value is a
    nominal report diagnostic, not a valid independent-sample inference.
    """
    observed = _probability_vector(observed_survival, "observed_survival")
    reconstructed = _probability_vector(reconstructed_survival, "reconstructed_survival")
    if reconstructed.shape != observed.shape:
        raise ValueError("observed_survival and reconstructed_survival must have equal lengths")
    keep = ~(np.isnan(observed) | np.isnan(reconstructed))
    retained_indices = np.flatnonzero(keep).astype(np.int64, copy=False)
    observed = observed[keep]
    reconstructed = reconstructed[keep]
    if observed.size < 2:
        raise ValueError("at least two complete observed/reconstructed pairs are required")

    fitted_rounded = _r_round(reconstructed)
    assert isinstance(fitted_rounded, np.ndarray)
    difference = _r_round(fitted_rounded - observed)
    assert isinstance(difference, np.ndarray)
    n_points = int(observed.size)
    rmse = float(_r_round(np.sqrt(np.sum(difference * difference) / (n_points - 1))))
    mean_absolute = float(_r_round(np.mean(np.abs(difference))))
    legacy_signed_max = float(_r_round(np.max(difference)))
    max_absolute = float(_r_round(np.max(np.abs(difference))))

    d_numerator, statistic, pooled_counts, n_first, n_second = _ks_distance(
        fitted_rounded, observed
    )
    ks_pvalue, ks_method = _ks_pvalue(d_numerator, pooled_counts, n_first, n_second)
    return IPDReconstructionDiagnostics(
        rounded_reconstructed_survival=_owned(fitted_rounded),
        rounded_difference=_owned(difference),
        retained_indices=_owned(retained_indices),
        n_points=n_points,
        omitted_points=int(keep.size - observed.size),
        rmse=rmse,
        mean_absolute_error=mean_absolute,
        legacy_signed_max_error=legacy_signed_max,
        max_absolute_error=max_absolute,
        ks_statistic=statistic,
        ks_pvalue=ks_pvalue,
        ks_method=ks_method,
    )
