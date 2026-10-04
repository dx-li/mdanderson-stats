"""Per-array intensity quantiles used by the PerfectMatch normalization QC view."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray
from .beta_binomial import _owned

_PROBABILITIES = (0.02, 0.25, 0.5, 0.75, 0.98)
_MAX_MATRIX_CELLS = 2_000_000
_MAX_SAMPLES = 5_000


@dataclass(frozen=True)
class PDNNArrayQuantiles:
    """Sorted-intensity quantiles by sample, in 2%, 25%, 50%, 75%, 98% order."""

    sample_names: tuple[str, ...]
    probabilities: tuple[float, ...]
    values: FloatArray


def _preflight_matrix(value: ArrayLike) -> tuple[int, int]:
    if isinstance(value, np.ndarray):
        if value.ndim != 2 or value.dtype.kind not in "biuf":
            raise ValueError("intensities must be a real two-dimensional numeric matrix")
        rows, columns = value.shape
    elif isinstance(value, (list, tuple)):
        rows = len(value)
        if rows == 0 or rows > _MAX_MATRIX_CELLS:
            raise ValueError("intensities must contain at least one probe")
        first = value[0]
        if not isinstance(first, (list, tuple, np.ndarray)) or getattr(first, "ndim", 1) != 1:
            raise ValueError("intensities must be a rectangular probes-by-samples matrix")
        columns = len(first)
        if columns == 0:
            raise ValueError("intensities must contain at least one sample")
        if columns > _MAX_SAMPLES or rows * columns > _MAX_MATRIX_CELLS:
            raise ValueError(f"intensities exceed the {_MAX_MATRIX_CELLS}-cell workspace limit")
        if any(
            not isinstance(row, (list, tuple, np.ndarray))
            or getattr(row, "ndim", 1) != 1
            or len(row) != columns
            for row in value
        ):
            raise ValueError("intensities must be a rectangular probes-by-samples matrix")
        if any(
            isinstance(item, complex)
            or not isinstance(item, (int, float, np.integer, np.floating, bool, np.bool_))
            for row in value
            for item in row
        ):
            raise ValueError("intensities must contain real scalar values")
    else:
        raise TypeError("intensities must be a NumPy matrix or nested list/tuple")
    if rows < 1 or columns < 1 or columns > _MAX_SAMPLES:
        raise ValueError(f"require at least one probe and 1..{_MAX_SAMPLES} samples")
    if rows * columns > _MAX_MATRIX_CELLS:
        raise ValueError(f"intensities exceed the {_MAX_MATRIX_CELLS}-cell workspace limit")
    return int(rows), int(columns)


def _sample_names(value: tuple[str, ...] | list[str] | None, count: int) -> tuple[str, ...]:
    if value is None:
        return tuple(f"sample_{index + 1}" for index in range(count))
    if not isinstance(value, (tuple, list)) or len(value) != count:
        raise ValueError("sample_names must contain one label per sample")
    if any(not isinstance(name, str) or not name or len(name) > 256 for name in value):
        raise ValueError("sample_names must be nonempty strings of at most 256 characters")
    if len(set(value)) != count:
        raise ValueError("sample_names must be unique")
    return tuple(value)


def pdnn_array_quantiles(
    intensities: ArrayLike,
    *,
    sample_names: tuple[str, ...] | list[str] | None = None,
) -> PDNNArrayQuantiles:
    """Summarize five sorted-intensity quantiles for each decoded array.

    ``intensities`` has probes as rows and samples as columns, matching
    :func:`quantile_normalize`. Intensities must be finite and nonnegative.
    The quantiles use NumPy's linear interpolation convention at positions
    ``(n_probes - 1) * p``. Columns are processed independently, so no sorted
    copy of the whole input matrix is retained. Larger collections can be
    processed in separate sample-column batches because profiles are independent.
    This reports descriptive values only; it does not assign chip-quality flags
    or parse CEL files.
    """
    rows, columns = _preflight_matrix(intensities)
    names = _sample_names(sample_names, columns)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            matrix = np.asarray(intensities, dtype=np.float64)
    except OverflowError as error:
        raise ValueError("intensities must be representable as finite float64 values") from error
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("intensities must be finite and nonnegative")

    probabilities = np.asarray(_PROBABILITIES, dtype=np.float64)
    positions = (rows - 1) * probabilities
    lower = np.floor(positions).astype(np.intp)
    upper = np.ceil(positions).astype(np.intp)
    kth = tuple(sorted(set(lower.tolist() + upper.tolist())))
    fraction = positions - lower
    values = np.empty((columns, len(_PROBABILITIES)), dtype=np.float64)
    for sample in range(columns):
        selected = matrix[:, sample].copy()
        selected.partition(kth)
        low_values = selected[lower]
        high_values = selected[upper]
        values[sample] = low_values + fraction * (high_values - low_values)
    if not np.all(np.isfinite(values)):
        raise ArithmeticError("intensity quantiles cannot be represented")
    return PDNNArrayQuantiles(names, _PROBABILITIES, _owned(values))
