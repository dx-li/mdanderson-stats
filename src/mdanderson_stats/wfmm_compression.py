"""Native-verified WFMM energy compression for supplied coefficient matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray
from .wfmm_basis import WFMMBasis, _frozen_int
from .wfmm_selection import WFMMSelection, wfmm_select_coefficients

_MAX_CELLS = 2_000_000
TiePolicy = Literal["reject_split", "original_index"]


@dataclass(frozen=True)
class WFMMCompression:
    """Selection, original-column vote counts and actual per-curve energy.

    ``energy_fraction`` is zero for zero-energy curves. At ``alpha=1`` the
    native bypass retains every column, irrespective of ``t``; ``vote_counts``
    is then None because voting was not performed. Selection indices and
    metadata remain in the original basis order.
    """

    selection: WFMMSelection
    alpha: float
    t: int
    tie_policy: TiePolicy
    vote_counts: NDArray[np.int64] | None
    energy_fraction: FloatArray


def wfmm_compress_coefficients(
    coefficients: ArrayLike,
    basis: WFMMBasis,
    *,
    alpha: float = 1.0,
    t: int = 0,
    tie_policy: TiePolicy = "reject_split",
) -> WFMMCompression:
    """Apply WFMM 3.1's strict cumulative-energy / curve-vote rule.

    Within each curve, rank squared coefficients in descending order. A
    coefficient votes only when cumulative energy INCLUDING that coefficient
    is strictly below ``alpha`` times total energy. Keep a column when its
    vote count is strictly greater than ``t``. The crossing coefficient is
    excluded: this rule does not guarantee retention of ``alpha`` energy.

    ``alpha=1`` bypasses compression, matching the native executable. Empty
    selections raise ValueError. Zero-energy curves cast no votes. The default
    rejects equal squared energies split by the threshold: the native C++
    sort does not specify stable ties. ``original_index`` explicitly resolves
    such ties by the smaller original index, without claiming native parity.

    Accepts a finite real curves-by-coefficients matrix, at most 2,000,000
    cells. Rows are rescaled before squaring to avoid overflow/underflow of
    total energy. Floating-point boundary decisions need not be bit-identical
    to the legacy binary. Transform, boundary extension and pass filtering
    are separate operations; supplied coefficient order determines identity.
    """
    if not isinstance(basis, WFMMBasis):
        raise ValueError("basis must be a WFMMBasis")
    raw = np.asarray(coefficients)
    if (
        raw.ndim != 2
        or raw.shape[1] != basis.time_count
        or raw.dtype.kind not in "iuf"
        or raw.size == 0
        or raw.size > _MAX_CELLS
    ):
        raise ValueError("coefficients must be a real nonempty matrix within the cell limit")
    curves = raw.shape[0]
    threshold = np.asarray(alpha)
    if threshold.ndim != 0 or threshold.dtype.kind not in "iuf":
        raise ValueError("alpha must be a real scalar in (0,1]")
    fraction = float(threshold)
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("alpha must be a real scalar in (0,1]")
    if (
        isinstance(t, (bool, np.bool_))
        or not isinstance(t, (int, np.integer))
        or not 0 <= t <= curves
    ):
        raise ValueError("t must be an integer in [0, number of curves]")
    if tie_policy not in ("reject_split", "original_index"):
        raise ValueError("tie_policy must be 'reject_split' or 'original_index'")
    values = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("coefficients must contain only finite values")

    votes = np.zeros(basis.time_count, dtype=np.int64)
    if fraction == 1:
        indices = np.arange(basis.time_count)
        vote_counts = None
    else:
        # One curve of temporaries rather than full sorting/cumulative matrices.
        for row in values:
            maximum = float(np.max(np.abs(row)))
            if maximum == 0:
                continue
            with np.errstate(under="ignore"):
                energy = np.square(row / maximum)
            order = np.argsort(-energy, kind="stable")
            ranked = energy[order]
            cumulative = np.cumsum(ranked)
            eligible = cumulative < fraction * cumulative[-1]
            if tie_policy == "reject_split" and np.any(
                (ranked[1:] == ranked[:-1]) & (eligible[1:] != eligible[:-1])
            ):
                raise ValueError("energy threshold splits tied coefficients; choose a tie_policy")
            votes[order[eligible]] += 1
        indices = np.flatnonzero(votes > int(t))
        if indices.size == 0:
            raise ValueError("compression retained no coefficients; increase alpha or reduce t")
        vote_counts = _frozen_int(votes)

    actual = np.zeros(curves, dtype=np.float64)
    for i, row in enumerate(values):
        maximum = float(np.max(np.abs(row)))
        if maximum > 0:
            if fraction == 1:
                actual[i] = 1.0
            else:
                with np.errstate(under="ignore"):
                    energy = np.square(row / maximum)
                actual[i] = min(1.0, float(energy[indices].sum() / energy.sum()))
    selection = wfmm_select_coefficients(values, basis, indices=indices)
    return WFMMCompression(selection, fraction, int(t), tie_policy, vote_counts, _freeze(actual))
