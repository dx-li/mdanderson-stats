"""Shared validation at numerical API boundaries."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def finite(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def count(value: ArrayLike, name: str) -> FloatArray:
    array = finite(value, name)
    if np.any((array < 0) | (array != np.floor(array)) | (array >= 2**53)):
        raise ValueError(f"{name} must contain nonnegative integers smaller than 2**53")
    return array


def scalar(value: float, name: str) -> float:
    array = finite(value, name)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    return float(array)
