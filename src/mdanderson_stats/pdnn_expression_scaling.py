"""Paper-defined array-average rescaling for PDNN expression estimates."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._validation import FloatArray, count, finite
from .beta_binomial import _owned
from .pdnn import PDNNExpression

_MAX_PDNN_VALUES = 500_000


@dataclass(frozen=True)
class PDNNScaledExpression:
    """Scaled natural-log expression, retaining the source probeset ordering."""

    probeset_ids: NDArray[np.int64]
    log_expression: FloatArray
    log_scale_factor: float
    target_mean: float


def _preflight_vector(value: object, name: str) -> int:
    if not isinstance(value, np.ndarray) or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional NumPy vector")
    size = value.size
    if not 1 <= size <= _MAX_PDNN_VALUES:
        raise ValueError(f"{name} must contain 1..{_MAX_PDNN_VALUES} values")
    return size


def pdnn_scale_expression(expression: PDNNExpression) -> PDNNScaledExpression:
    """Scale all supplied expression estimates to arithmetic mean 500 on the array.

    This is the array-level rescaling stated after equation (5) in Zhang, Miles
    and Aldape (2003, p. 4). It multiplies the supplied per-probeset expression
    estimates by a common factor, represented by ``log_scale_factor``. The
    transform stays in the natural-log domain, so it does not exponentiate large
    expression estimates. It preserves the probeset IDs and never mutates the
    input or its probe-level fitted signals, which precede this array-wide step.

    ``expression`` defines the gene set to scale: no additional filtering or
    missing-gene policy is inferred. The mean is over every entry in its
    ``log_expression`` vector.
    """
    if not isinstance(expression, PDNNExpression):
        raise TypeError("expression must be a PDNNExpression")
    size = _preflight_vector(expression.log_expression, "log_expression")
    if _preflight_vector(expression.probeset_ids, "probeset_ids") != size:
        raise ValueError("expression must have matching nonempty probeset IDs and log values")
    log_values = finite(expression.log_expression, "log_expression")
    identifiers = count(expression.probeset_ids, "probeset_ids")
    if (
        log_values.ndim != 1
        or identifiers.shape != log_values.shape
        or np.any(np.diff(identifiers) <= 0)
    ):
        raise ValueError("expression must have matching nonempty probeset IDs and log values")

    maximum = float(np.max(log_values))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        centered = log_values - maximum
        log_mean_centered = float(np.log(np.mean(np.exp(centered))))
        log_target = float(np.log(500.0))
        log_factor = (log_target - maximum) - log_mean_centered
    scaled = centered + log_target - log_mean_centered
    if not np.isfinite(log_factor) or np.any(~np.isfinite(scaled)):
        raise ArithmeticError("scaled PDNN expression cannot be represented in log space")
    scaled_ids = identifiers.astype(np.int64, copy=True)
    scaled_ids.flags.writeable = False
    return PDNNScaledExpression(scaled_ids, _owned(scaled), log_factor, 500.0)
