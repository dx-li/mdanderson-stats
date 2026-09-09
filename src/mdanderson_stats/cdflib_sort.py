"""CDFLIB's generic list sorting with stable ties and no string truncation."""

from collections.abc import Callable, Sequence
from functools import cmp_to_key

import numpy as np
from numpy.typing import NDArray

type SortValue = int | float | str
type SortComparison = Callable[[SortValue, SortValue], bool]
type SortResult = NDArray[np.generic] | tuple[str, ...]


def sort_list(
    values: Sequence[SortValue] | NDArray[np.generic],
    ncol: int | None = None,
    a_gt_b: SortComparison | None = None,
) -> SortResult:
    """Return an owned, immutable copy with its first ncol elements sorted.

    Numbers retain the NumPy dtype; strings return a tuple. Default string
    ordering pads with blanks, as Fortran character comparisons do. A custom
    a_gt_b(a,b) says a belongs after b, and must define a consistent ordering.
    Both strict and inclusive comparisons are accepted; ties remain stable.
    """
    if isinstance(values, (str, bytes)):
        raise ValueError("values must be a one-dimensional sequence, not a string")
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    # Inspect the original sequence to avoid silently converting mixed values
    # such as [1, '2'] into a character array through NumPy type promotion.
    strings = array.dtype.kind in "UO" and all(isinstance(v, str) for v in values)
    if not strings:
        if array.dtype.kind not in "iuf" or (
            not isinstance(values, np.ndarray)
            and any(isinstance(v, (bool, np.bool_)) for v in values)
        ):
            raise ValueError("values must be real numbers or exclusively strings")
        if not np.all(np.isfinite(array)):
            raise ValueError("numeric values must be finite")
    size = len(array)
    count = size if ncol is None else ncol
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
        raise ValueError("ncol must be an integer")
    if not 0 <= count <= size:
        raise ValueError("ncol must lie between zero and the input length")
    if a_gt_b is not None and not callable(a_gt_b):
        raise ValueError("a_gt_b must be callable")
    if strings:
        text = tuple(str(v) for v in values)
        width = max((len(v) for v in text), default=0)
        # A callback receives the fixed-width values of the Fortran interface.
        items: list[SortValue] = list(text[:count])
    else:
        items = []
    if a_gt_b is not None:
        if not strings:
            items = array[:count].tolist()

        def compare(a: SortValue, b: SortValue) -> int:
            if strings:
                a, b = str(a).ljust(width), str(b).ljust(width)
            after, before = a_gt_b(a, b), a_gt_b(b, a)
            if not isinstance(after, (bool, np.bool_)) or not isinstance(before, (bool, np.bool_)):
                raise ValueError("a_gt_b must return a scalar boolean")
            return int(after) - int(before)

        ordered = sorted(items, key=cmp_to_key(compare))
        if strings:
            return tuple(str(v) for v in ordered) + text[count:]
        result = array.copy()
        result[:count] = ordered
    elif strings:
        return tuple(sorted(text[:count], key=lambda v: v.ljust(width))) + text[count:]
    else:
        result = array.copy()
        result[:count] = np.sort(array[:count], kind="stable")
    return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(result.shape)
