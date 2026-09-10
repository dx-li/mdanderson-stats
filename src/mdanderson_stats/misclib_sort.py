"""MISCLIB matrix-column sorting and direct/reversed gather permutations."""

from collections.abc import Callable
from functools import cmp_to_key

import numpy as np
from numpy.typing import ArrayLike, NDArray

type MatrixComparison = Callable[[NDArray, NDArray, int], bool]


def _freeze(array: NDArray) -> NDArray:
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _matrix(values: ArrayLike) -> NDArray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] < 1 or array.size > 2_000_000:
        raise ValueError("values must be a matrix with at least one row and at most 2M elements")
    if array.dtype.kind == "U":
        if not isinstance(values, np.ndarray) and not all(
            isinstance(v, str) for v in np.asarray(values, dtype=object).flat
        ):
            raise ValueError("Do not mix strings and numbers")
    elif array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        raise ValueError("values must contain finite real numbers or Unicode strings")
    return array


def _count(value: int | None, default: int, name: str) -> int:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not 0 <= value <= default:
        raise ValueError(f"{name} must be between zero and {default}")
    return int(value)


def _order(array: NDArray, count: int, irow: int, a_gt_b: MatrixComparison | None) -> NDArray:
    if isinstance(irow, (bool, np.bool_)) or not isinstance(irow, (int, np.integer)):
        raise ValueError("irow must be an integer")
    if a_gt_b is None:
        if not 0 <= irow < array.shape[0]:
            raise ValueError("irow must identify a zero-based row")
        if array.dtype.kind == "U":
            width = array.dtype.itemsize // 4
            order = sorted(range(count), key=lambda j: str(array[irow, j]).ljust(width))
            return np.asarray(order, dtype=np.int64)
        return np.argsort(array[irow, :count], kind="stable")
    if not callable(a_gt_b):
        raise ValueError("a_gt_b must be callable")
    # Callbacks receive immutable columns, padded to a common Fortran character
    # width. Source input and sort keys cannot be mutated during comparison.
    if array.dtype.kind == "U":
        width = array.dtype.itemsize // 4
        data = np.asarray([[str(v).ljust(width) for v in row] for row in array])
    else:
        data = array
    data = _freeze(data)

    def compare(i: int, j: int) -> int:
        after = a_gt_b(data[:, i], data[:, j], int(irow))
        before = a_gt_b(data[:, j], data[:, i], int(irow))
        if not isinstance(after, (bool, np.bool_)) or not isinstance(before, (bool, np.bool_)):
            raise ValueError("a_gt_b must return a scalar boolean")
        return int(after) - int(before)

    return np.asarray(sorted(range(count), key=cmp_to_key(compare)), dtype=np.int64)


def permutation_sort_matrix(
    values: ArrayLike,
    *,
    ncol: int | None = None,
    irow: int = 0,
    a_gt_b: MatrixComparison | None = None,
) -> NDArray[np.int64]:
    """Return stable zero-based gather indices for the first ncol columns.

    Default order uses row irow. A custom comparator receives full columns and
    irow unchanged, and says whether the first column belongs after the second.
    """
    array = _matrix(values)
    count = _count(ncol, array.shape[1], "ncol")
    return _freeze(_order(array, count, irow, a_gt_b).astype(np.int64))


def permute_matrix(
    values: ArrayLike,
    index: ArrayLike | None = None,
    *,
    opt: int = 1,
    ncol: int | None = None,
    nrowus: int | None = None,
) -> NDArray:
    """Gather selected columns, reverse the gather for opt<0, or copy for opt=0.

    The result owns immutable storage. Only the leading nrowus-by-ncol block
    changes; the rest is copied unchanged. Negative opt is NOT inverse scatter.
    """
    array = _matrix(values)
    count = _count(ncol, array.shape[1], "ncol")
    rows = _count(nrowus, array.shape[0], "nrowus")
    if isinstance(opt, (bool, np.bool_)) or not isinstance(opt, (int, np.integer)):
        raise ValueError("opt must be an integer")
    result = array.copy()
    if opt:
        order = np.asarray(index)
        if order.ndim != 1 or order.size != count or order.dtype.kind not in "iu":
            # An empty Python list has floating dtype but is the unique empty permutation.
            if not (order.ndim == 1 and order.size == count == 0):
                raise ValueError("index must be an integer permutation of 0..ncol-1")
        if not np.array_equal(np.sort(order), np.arange(count)):
            raise ValueError("index must be a permutation of 0..ncol-1")
        order = order.astype(np.int64)
        if opt < 0:
            order = order[::-1]
        result[:rows, :count] = array[:rows, order]
    return _freeze(result)


def sort_matrix(
    values: ArrayLike,
    *,
    ncol: int | None = None,
    nrowus: int | None = None,
    irow: int = 0,
    a_gt_b: MatrixComparison | None = None,
) -> NDArray:
    """Sort the leading block's columns; preserve inactive rows and columns.

    Ties remain in input order. Custom comparators see the original full columns;
    sorting a partial row block requires keys within that block for native parity.
    """
    array = _matrix(values)
    rows = _count(nrowus, array.shape[0], "nrowus")
    count = _count(ncol, array.shape[1], "ncol")
    if a_gt_b is None and not 0 <= irow < rows:
        raise ValueError("Default irow must lie within the rows being sorted")
    order = _order(array, count, irow, a_gt_b)
    result = array.copy()
    result[:rows, :count] = array[:rows, order]
    return _freeze(result)
