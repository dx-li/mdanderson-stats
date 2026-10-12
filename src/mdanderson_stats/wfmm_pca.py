"""Uncentered WFMM principal-component and hybrid analysis/synthesis pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .wfmm_basis import WFMMBasis, _frozen_int, _setting, wfmm_transform


@dataclass(frozen=True)
class WFMMPCTransform:
    """Calibrated transform; PCA signs are canonical Python choices, not native RNG state.

    PC and wPC reconstruct the retained projection. PCw adds a wavelet analysis
    of the orthogonal residual and reconstructs all input directions. Its
    redundant coordinates are not independent observations in data space;
    fitting independent coefficient variances is an explicit working model.
    """

    basis: WFMMBasis
    kind: str
    retained_components: int
    relative_singular_values: FloatArray
    input_scale: float
    energy_fraction: float
    numerical_rank: int
    partition_sizes: tuple[int, ...]


def _partition_sizes(singular, scale, count, sizes, number, base):
    if sizes is not None and number is not None:
        raise ValueError("supply partition_sizes or npartitions, not both")
    if sizes is not None:
        raw = np.asarray(sizes)
        if (
            raw.ndim != 1
            or raw.dtype.kind not in "iu"
            or raw.size == 0
            or raw.size > count
            or np.any(raw <= 0)
            or np.any(raw > count)
        ):
            raise ValueError("partition_sizes must be positive integer counts")
        result = tuple(int(x) for x in raw)
        if sum(result) != count:
            raise ValueError("partition_sizes must sum to retained component count")
        return result
    if number is not None:
        groups = _setting(number, "npartitions", 1, count)
        width = count // groups
        return (width,) * (groups - 1) + (count - width * (groups - 1),)
    # Native GetPrincipalComponents bins log-base singular values in half units,
    # despite the guide's eigenvalue wording. Sparse bins avoid huge allocations
    # when the user chooses a base close to one. Merge singletons forward and
    # the final singleton backward. No source endpoint-allocation defect is copied.
    with np.errstate(over="ignore", under="ignore"):
        absolute = singular[:count] * scale
    if np.all(np.isfinite(absolute)) and np.all(absolute > 0):
        logs = np.log(absolute) / np.log(base)
    else:
        logs = (np.log(singular[:count]) + np.log(scale)) / np.log(base)
    top = np.ceil(2 * logs[0]) / 2
    bins = np.floor(2 * (top - logs))
    groups_out: list[int] = []
    pending = 0
    _, widths = np.unique(bins, return_counts=True)
    for width in widths:
        if pending > 1:
            groups_out.append(pending)
            pending = 0
        pending += int(width)
    if pending:
        if pending == 1 and groups_out:
            groups_out[-1] += pending
        else:
            groups_out.append(pending)
    return tuple(groups_out)


def wfmm_pca_basis(
    curves: ArrayLike,
    *,
    kind: str = "PC",
    alpha: float = 0.99,
    wavelet_basis: WFMMBasis | None = None,
    partition_sizes: ArrayLike | None = None,
    npartitions: int | None = None,
    partition_base: float = 10.0,
    max_work: int = 200_000_000,
) -> WFMMPCTransform:
    """Calibrate an uncentered PC, PCw or wPC transform from observed curve rows.

    For alpha<1 retain the first cumulative squared-singular-value sum strictly
    greater than alpha, as in WFMM1. At alpha=1 retain the numerical row space;
    the original program's zero-column failure is not reproduced. Zero-energy
    inputs and cuts splitting numerically tied singular values are rejected.
    No arbitrary direction is inferred from a tied cutoff.

    Hybrid transforms require an explicitly supplied periodic wavelet basis.
    PCw concatenates PC scores with wavelet scores of the residual. wPC applies
    wavelet analysis before PCA. No centering or scaling of individual variables
    is performed. New curves use the fixed returned basis without refitting.
    """
    if kind not in ("PC", "PCw", "wPC"):
        raise ValueError("kind must be PC, PCw or wPC")
    budget = _setting(max_work, "max_work", 1, 200_000_000)
    raw_alpha, raw_base = np.asarray(alpha), np.asarray(partition_base)
    if (
        raw_alpha.ndim != 0
        or raw_alpha.dtype.kind not in "iuf"
        or not np.isfinite(raw_alpha)
        or not 0 <= float(raw_alpha) <= 1
    ):
        raise ValueError("alpha must be a finite real scalar in [0,1]")
    if (
        raw_base.ndim != 0
        or raw_base.dtype.kind not in "iuf"
        or not np.isfinite(raw_base)
        or float(raw_base) <= 1
    ):
        raise ValueError("partition_base must be a finite scalar greater than one")
    raw = np.asarray(curves)
    if (
        raw.ndim != 2
        or raw.dtype.kind not in "iuf"
        or not 2 <= raw.shape[0] <= 500
        or not 2 <= raw.shape[1] <= 4096
        or raw.size > 2_000_000
    ):
        raise ValueError("curves must be a bounded real matrix with at least two rows/columns")
    rows, times = raw.shape
    maximum_k = min(rows, times) + (times if kind == "PCw" else 0)
    # Include dense hybrid projection, wavelet matrix and result ownership copies.
    work = 8 * rows * times * min(rows, times) + 4 * times**2 * maximum_k
    if work > budget or 8 * times * maximum_k + 3 * raw.size > 2_000_000:
        raise ValueError("PCA transform exceeds bounded work or live-cell limits")
    values = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("curves must be finite")
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        raise ValueError("zero-energy input has no identified principal directions")
    normalized = values / scale
    if kind == "PC":
        if wavelet_basis is not None:
            raise ValueError("PC does not accept a wavelet basis")
        wave = None
    else:
        if (
            not isinstance(wavelet_basis, WFMMBasis)
            or wavelet_basis.transform != "wavelet"
            or wavelet_basis.time_count != times
        ):
            raise ValueError("hybrid transforms require a matching wavelet basis")
        wave = wfmm_transform(np.eye(times), wavelet_basis).coefficients
    source = normalized
    if kind == "wPC":
        assert wave is not None
        source = normalized @ wave
    _, singular, vt = np.linalg.svd(source, full_matrices=False)
    tolerance = np.finfo(float).eps * max(rows, times) * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    energies = (singular / singular[0]) ** 2
    cumulative = np.cumsum(energies) / np.sum(energies)
    count = (
        rank
        if float(raw_alpha) == 1
        else int(np.searchsorted(cumulative, float(raw_alpha), side="right")) + 1
    )
    count = min(count, rank)
    if count < 1:
        raise ArithmeticError("no representable principal direction")
    if count < rank and singular[count - 1] - singular[count] <= 16 * tolerance:
        raise ValueError("energy cutoff splits tied principal directions")
    vectors = vt[:count].T.copy()
    # Sign conventions vary by LAPACK. Pin the largest-magnitude loading positive.
    signs = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(count)])
    vectors *= signs
    sizes = _partition_sizes(singular, scale, count, partition_sizes, npartitions, float(raw_base))
    partitions = np.repeat(np.arange(len(sizes)), sizes)
    if kind == "PC":
        analysis, synthesis = vectors, vectors.T
    elif kind == "wPC":
        assert wave is not None
        analysis = wave @ vectors
        synthesis = analysis.T
    else:
        assert wave is not None and wavelet_basis is not None
        residual_wave = wave - vectors @ (vectors.T @ wave)
        analysis = np.column_stack((vectors, residual_wave))
        synthesis = np.vstack((vectors.T, wave.T))
        partitions = np.r_[partitions, wavelet_basis.coefficient_partition + len(sizes)]
    scales = np.zeros(analysis.shape[1], dtype=np.int64)
    if kind == "PCw":
        assert wavelet_basis is not None
        scales[count:] = wavelet_basis.coefficient_scale
    empty = _freeze(np.empty(0))
    basis = WFMMBasis(
        "custom",
        times,
        0,
        None,
        empty,
        empty,
        _freeze(analysis),
        _frozen_int(scales),
        _frozen_int(partitions),
        budget,
        _freeze(synthesis),
    )
    return WFMMPCTransform(
        basis, kind, count, _freeze(singular), scale, float(cumulative[count - 1]), rank, sizes
    )
