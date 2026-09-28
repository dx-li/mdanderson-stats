"""Bounded orthogonal time-series transforms used by WFMM.

Wavelet transforms use periodic decimation with a documented Python packing
convention: ``[a_J, d_J, ..., d_1]``.  Coefficients are generated with the
package's minimum-phase Daubechies taps; this layer does not claim native
WFMM binary coefficient ordering or boundary-mode parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray
from .pinnacle_wavelet import pinnacle_daubechies_filter

_MAX_TIME_POINTS = 4096
_MAX_INPUT_CELLS = 2_000_000
_MAX_WORK = 200_000_000
_MAX_LEVELS = 12

TransformKind = Literal["identity", "wavelet", "custom"]


def _frozen_int(value: ArrayLike) -> NDArray[np.int64]:
    array = np.ascontiguousarray(value, dtype=np.int64)
    return np.frombuffer(array.tobytes(), dtype=np.int64).reshape(array.shape)


def _setting(value: object, name: str, minimum: int, maximum: int) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iu" or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    result = int(raw)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum},{maximum}]")
    return result


@dataclass(frozen=True)
class WFMMBasis:
    """Immutable transform metadata, coefficient groups and optional custom basis."""

    transform: TransformKind
    time_count: int
    levels: int
    filter_length: int | None
    scaling_filter: FloatArray
    wavelet_filter: FloatArray
    custom_matrix: FloatArray | None
    coefficient_scale: NDArray[np.int64]
    coefficient_partition: NDArray[np.int64]
    max_work: int


@dataclass(frozen=True)
class WFMMTransformed:
    """Transformed curve rows and the corresponding scale/partition labels."""

    coefficients: FloatArray
    coefficient_scale: NDArray[np.int64]
    coefficient_partition: NDArray[np.int64]


def wfmm_basis(
    time_count: int,
    *,
    transform: TransformKind = "wavelet",
    levels: int | None = None,
    filter_length: int = 8,
    custom_matrix: ArrayLike | None = None,
    max_work: int = _MAX_WORK,
) -> WFMMBasis:
    """Build identity, periodic orthogonal-wavelet, or full orthogonal basis metadata.

    Wavelet filter lengths are even numbers 2..20 (Daubechies db1..db10).
    ``levels`` must divide ``time_count`` by ``2**levels``; the default uses
    the largest supported level. A custom transform is a square orthogonal
    matrix whose columns are basis vectors. It cannot be combined with wavelet
    settings. Work and coefficient storage are checked before matrix copies.
    """
    size = _setting(time_count, "time_count", 2, _MAX_TIME_POINTS)
    budget = _setting(max_work, "max_work", 1, _MAX_WORK)
    if transform not in ("identity", "wavelet", "custom"):
        raise ValueError("transform must be 'identity', 'wavelet' or 'custom'")

    empty = _freeze(np.empty(0, dtype=np.float64))
    if transform == "identity":
        if levels is not None or custom_matrix is not None:
            raise ValueError("identity transform does not accept levels or custom_matrix")
        scale = np.zeros(size, dtype=np.int64)
        partition = np.zeros(size, dtype=np.int64)
        return WFMMBasis(
            "identity",
            size,
            0,
            None,
            empty,
            empty,
            None,
            _frozen_int(scale),
            _frozen_int(partition),
            budget,
        )

    if transform == "custom":
        if levels is not None:
            raise ValueError("custom transform does not accept levels")
        if custom_matrix is None:
            raise ValueError("custom transform requires custom_matrix")
        if size * size > _MAX_INPUT_CELLS or size**3 > budget:
            raise ValueError("custom orthogonal basis exceeds bounded matrix/work limits")
        raw = np.asarray(custom_matrix)
        if raw.shape != (size, size) or raw.dtype.kind not in "iuf":
            raise ValueError("custom_matrix must be a real square matrix matching time_count")
        matrix = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("custom_matrix must be finite")
        gram = matrix.T @ matrix
        if not np.all(np.isfinite(gram)) or not np.allclose(
            gram, np.eye(size), rtol=0.0, atol=2e-10
        ):
            raise ValueError("custom_matrix columns must be orthonormal")
        zeros = np.zeros(size, dtype=np.int64)
        return WFMMBasis(
            "custom",
            size,
            0,
            None,
            empty,
            empty,
            _freeze(matrix),
            _frozen_int(zeros),
            _frozen_int(zeros),
            budget,
        )

    if custom_matrix is not None:
        raise ValueError("wavelet transform does not accept custom_matrix")
    maximum_levels = min(_MAX_LEVELS, (size & -size).bit_length() - 1)
    level_count = maximum_levels if levels is None else _setting(levels, "levels", 1, _MAX_LEVELS)
    if level_count < 1 or level_count > maximum_levels or size % (1 << level_count):
        raise ValueError("2**levels must divide time_count")
    if isinstance(filter_length, (bool, np.bool_)) or not isinstance(
        filter_length, (int, np.integer)
    ):
        raise ValueError("filter_length must be an even integer in [2,20]")
    length = int(filter_length)
    if length < 2 or length > 20 or length % 2:
        raise ValueError("filter_length must be an even integer in [2,20]")
    if size * (2 * level_count + 4) > _MAX_INPUT_CELLS:
        raise ValueError("wavelet basis metadata exceeds bounded storage")
    scaling, wavelet = pinnacle_daubechies_filter(length)

    approximation_count = size >> level_count
    scales = [level_count] * approximation_count
    partitions = [0] * approximation_count
    next_partition = 1
    # Pack approximation, then details from coarsest to finest.
    for level in range(level_count, 0, -1):
        width = size >> level
        scales.extend([level] * width)
        partitions.extend([next_partition] * width)
        next_partition += 1
    return WFMMBasis(
        "wavelet",
        size,
        level_count,
        length,
        scaling,
        wavelet,
        None,
        _frozen_int(scales),
        _frozen_int(partitions),
        budget,
    )


def _validate_curves(curves: ArrayLike, basis: WFMMBasis) -> FloatArray:
    if not isinstance(basis, WFMMBasis):
        raise ValueError("basis must be a WFMMBasis")
    raw = np.asarray(curves)
    if raw.ndim != 2 or raw.shape[1] != basis.time_count or raw.shape[0] < 1:
        raise ValueError("curves must be a nonempty (curves,time_points) matrix matching basis")
    if raw.dtype.kind not in "iuf" or raw.size > _MAX_INPUT_CELLS:
        raise ValueError("curves must be real and within the bounded input-cell limit")
    if basis.transform == "wavelet":
        if basis.filter_length is None:
            raise ValueError("wavelet basis is missing its filter length")
        work = 2 * raw.shape[0] * basis.time_count * basis.filter_length * basis.levels
    elif basis.transform == "custom":
        work = raw.shape[0] * basis.time_count * basis.time_count
    else:
        work = raw.size
    if work > basis.max_work:
        raise ValueError("curve transform exceeds basis max_work")
    values = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("curves must be finite")
    return values


def _analysis_pair(
    values: FloatArray, low: FloatArray, high: FloatArray
) -> tuple[FloatArray, FloatArray]:
    n = values.shape[1]
    half = n // 2
    low_out = np.zeros((values.shape[0], half), dtype=np.float64)
    high_out = np.zeros_like(low_out)
    sample = 2 * np.arange(half)
    with np.errstate(over="ignore", invalid="ignore"):
        for tap, (lo, hi) in enumerate(zip(low, high, strict=True)):
            indices = (sample + tap) % n
            low_out += float(lo) * values[:, indices]
            high_out += float(hi) * values[:, indices]
    if not np.all(np.isfinite(low_out)) or not np.all(np.isfinite(high_out)):
        raise ArithmeticError("wavelet coefficients exceed floating-point range")
    return low_out, high_out


def _synthesis_pair(
    low_values: FloatArray, high_values: FloatArray, low: FloatArray, high: FloatArray
) -> FloatArray:
    n = 2 * low_values.shape[1]
    output = np.zeros((low_values.shape[0], n), dtype=np.float64)
    sample = 2 * np.arange(low_values.shape[1])
    with np.errstate(over="ignore", invalid="ignore"):
        for tap, (lo, hi) in enumerate(zip(low, high, strict=True)):
            indices = (sample + tap) % n
            output[:, indices] += float(lo) * low_values + float(hi) * high_values
    if not np.all(np.isfinite(output)):
        raise ArithmeticError("inverse wavelet transform is not finite")
    return output


def wfmm_transform(curves: ArrayLike, basis: WFMMBasis) -> WFMMTransformed:
    """Transform row-wise curves and attach immutable scale/partition labels."""
    values = _validate_curves(curves, basis)
    if basis.transform == "identity":
        result = values.copy()
    elif basis.transform == "custom":
        assert basis.custom_matrix is not None
        with np.errstate(over="ignore", invalid="ignore"):
            result = values @ basis.custom_matrix
    else:
        current = values
        detail_bands: list[FloatArray] = []
        for _ in range(basis.levels):
            current, detail = _analysis_pair(current, basis.scaling_filter, basis.wavelet_filter)
            detail_bands.append(detail)
        result = np.concatenate((current, *detail_bands[::-1]), axis=1)
    if result.shape != values.shape or not np.all(np.isfinite(result)):
        raise ArithmeticError("WFMM transform produced invalid coefficients")
    return WFMMTransformed(_freeze(result), basis.coefficient_scale, basis.coefficient_partition)


def wfmm_inverse(coefficients: ArrayLike, basis: WFMMBasis) -> FloatArray:
    """Invert row-wise transformed curves using the same packed convention."""
    values = _validate_curves(coefficients, basis)
    if basis.transform == "identity":
        result = values.copy()
    elif basis.transform == "custom":
        assert basis.custom_matrix is not None
        with np.errstate(over="ignore", invalid="ignore"):
            result = values @ basis.custom_matrix.T
    else:
        approximation_count = basis.time_count >> basis.levels
        current = values[:, :approximation_count]
        offset = approximation_count
        details: dict[int, FloatArray] = {}
        for level in range(basis.levels, 0, -1):
            width = basis.time_count >> level
            details[level] = values[:, offset : offset + width]
            offset += width
        for level in range(basis.levels, 0, -1):
            current = _synthesis_pair(
                current, details[level], basis.scaling_filter, basis.wavelet_filter
            )
        result = current
    if result.shape != values.shape or not np.all(np.isfinite(result)):
        raise ArithmeticError("WFMM inverse produced invalid curves")
    return _freeze(result)
