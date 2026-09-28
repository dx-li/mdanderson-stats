"""Select and reconstruct retained WFMM coefficient columns.

Selection is explicit by original coefficient index or the basis' existing
partition labels. This module does not infer native high/low-pass flags,
energy thresholds or principal components.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite
from .wfmm_basis import WFMMBasis

_MAX_INPUT_CELLS = 2_000_000
_MAX_RECONSTRUCTED_CELLS = 2_000_000


def _frozen_float(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _frozen_int(value: ArrayLike) -> NDArray[np.int64]:
    array = np.ascontiguousarray(value, dtype=np.int64)
    return np.frombuffer(array.tobytes(), dtype=np.int64).reshape(array.shape)


def _selector_values(value: ArrayLike, name: str, max_size: int) -> NDArray[np.int64]:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if raw.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if raw.size > max_size:
        raise ValueError(f"{name} cannot contain more than {max_size} unique values")
    if np.unique(raw).size != raw.size:
        raise ValueError(f"{name} must not contain duplicates")
    if np.any(raw > np.iinfo(np.int64).max):
        raise ValueError(f"{name} values exceed int64")
    return np.asarray(raw, dtype=np.int64)


@dataclass(frozen=True)
class WFMMSelection:
    """Readonly selected values and their original coefficient identities.

    ``retained_indices`` is zero-based and strictly increasing, so selected
    data always follow the original packed basis order. Scale and partition
    labels retain their original values and are not renumbered.
    """

    coefficients: FloatArray
    retained_indices: NDArray[np.int64]
    coefficient_scale: NDArray[np.int64]
    coefficient_partition: NDArray[np.int64]
    original_coefficient_count: int


def wfmm_select_coefficients(
    coefficients: ArrayLike,
    basis: WFMMBasis,
    *,
    indices: ArrayLike | None = None,
    partitions: ArrayLike | None = None,
) -> WFMMSelection:
    """Select coefficient columns by original indices or partition labels.

    ``coefficients`` may have any leading dimensions but must end in the
    basis coefficient count. Exactly one selector is required. Explicit
    indices are sorted into original basis order; partition selection retains
    every coefficient whose unchanged basis label is requested.
    """
    if not isinstance(basis, WFMMBasis):
        raise ValueError("basis must be a WFMMBasis")
    if (indices is None) == (partitions is None):
        raise ValueError("supply exactly one of indices or partitions")
    raw = np.asarray(coefficients)
    if raw.ndim < 1 or raw.shape[-1] != basis.time_count or raw.dtype.kind not in "iuf":
        raise ValueError("coefficients must be a real array ending in the basis coefficient count")
    if raw.size == 0 or raw.size > _MAX_INPUT_CELLS:
        raise ValueError("coefficient array must be nonempty and within the input-cell limit")
    values = finite(raw, "coefficients")
    if indices is not None:
        requested = _selector_values(indices, "indices", basis.time_count)
        if np.any(requested < 0) or np.any(requested >= basis.time_count):
            raise ValueError("coefficient indices must lie within the original basis")
        selected = np.sort(requested)
    else:
        assert partitions is not None
        available = np.unique(basis.coefficient_partition)
        requested = _selector_values(partitions, "partitions", int(available.size))
        if not np.all(np.isin(requested, available)):
            raise ValueError("requested partition labels are not present in the basis")
        selected = np.flatnonzero(np.isin(basis.coefficient_partition, requested))
    retained_cells = prod(values.shape[:-1]) * int(selected.size)
    if selected.size == 0 or retained_cells > _MAX_INPUT_CELLS:
        raise ValueError("selected coefficient array exceeds the input-cell limit")
    return WFMMSelection(
        _frozen_float(np.take(values, selected, axis=-1)),
        _frozen_int(selected),
        _frozen_int(basis.coefficient_scale[selected]),
        _frozen_int(basis.coefficient_partition[selected]),
        basis.time_count,
    )


def wfmm_restore_coefficients(coefficients: ArrayLike, selection: WFMMSelection) -> FloatArray:
    """Zero-fill retained coefficient arrays to the original final-axis size.

    The leading dimensions are preserved, allowing posterior coefficient or
    variance arrays to be restored before passing coefficients to
    :func:`wfmm_inverse` or :func:`wfmm_summarize`.
    """
    if not isinstance(selection, WFMMSelection):
        raise ValueError("selection must be a WFMMSelection")
    raw = np.asarray(coefficients)
    indices = np.asarray(selection.retained_indices)
    scales = np.asarray(selection.coefficient_scale)
    partitions = np.asarray(selection.coefficient_partition)
    original_count = selection.original_coefficient_count
    if (
        raw.ndim < 1
        or raw.shape[-1] != indices.size
        or raw.dtype.kind not in "iuf"
        or not isinstance(original_count, (int, np.integer))
        or isinstance(original_count, (bool, np.bool_))
        or original_count < 1
    ):
        raise ValueError("coefficients and selection have incompatible dimensions")
    if (
        indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or indices.size == 0
        or np.any(indices < 0)
        or np.any(indices >= original_count)
        or np.any(indices[1:] <= indices[:-1])
        or scales.shape != indices.shape
        or scales.dtype.kind not in "iu"
        or np.any(scales < 0)
        or partitions.shape != indices.shape
        or partitions.dtype.kind not in "iu"
        or np.any(partitions < 0)
    ):
        raise ValueError("selection metadata is inconsistent")
    leading_cells = prod(raw.shape[:-1])
    if (
        raw.size == 0
        or raw.size > _MAX_INPUT_CELLS
        or leading_cells * int(original_count) > _MAX_RECONSTRUCTED_CELLS
    ):
        raise ValueError("restored coefficient array exceeds the bounded cell limit")
    values = finite(raw, "coefficients")
    restored = np.zeros((*raw.shape[:-1], int(original_count)), dtype=np.float64)
    restored[..., indices] = values
    if not np.all(np.isfinite(restored)):
        raise ArithmeticError("zero-filled coefficients are nonfinite")
    restored.flags.writeable = False
    return restored
