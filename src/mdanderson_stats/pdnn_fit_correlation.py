"""Observed-versus-fitted log-intensity correlations by probeset."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import count, finite
from .beta_binomial import _owned

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
_MAX_PROBES = 10_000_000


@dataclass(frozen=True)
class PDNNFitCorrelations:
    """Sorted probeset labels and within-probeset Pearson correlations.

    Correlations are NaN for singleton or constant-log-signal groups, where
    Pearson correlation is mathematically undefined.
    """

    probeset_ids: IntArray
    correlation: FloatArray


def pdnn_fit_correlations(
    observed_signal: ArrayLike,
    fitted_signal: ArrayLike,
    probeset_ids: ArrayLike,
) -> PDNNFitCorrelations:
    """Correlate observed and fitted ln(PM) values within each probeset.

    ``probeset_ids`` must map every original probe row to its probeset. The
    unique IDs stored by :class:`PDNNFit` do not provide that mapping. Signals
    must be finite and strictly positive. This implements Pearson correlation;
    the manual names a correlation coefficient but does not specify edge-case
    behavior, so undefined groups are returned as NaN.
    """
    observed = finite(observed_signal, "observed_signal")
    fitted = finite(fitted_signal, "fitted_signal")
    ids = count(probeset_ids, "probeset_ids")
    if observed.ndim != 1 or fitted.shape != observed.shape or ids.shape != observed.shape:
        raise ValueError("signals and probeset_ids must be equal-length one-dimensional arrays")
    n = observed.size
    if n == 0 or n > _MAX_PROBES:
        raise ValueError(f"probe count must be in 1..{_MAX_PROBES}")
    if np.any(observed <= 0) or np.any(fitted <= 0):
        raise ValueError("signals must be strictly positive")

    labels, groups = np.unique(ids.astype(np.int64), return_inverse=True)
    x, y = np.log(observed), np.log(fitted)
    k = labels.size
    sizes = np.bincount(groups, minlength=k)
    mean_x = np.bincount(groups, weights=x, minlength=k) / sizes
    mean_y = np.bincount(groups, weights=y, minlength=k) / sizes
    dx, dy = x - mean_x[groups], y - mean_y[groups]
    scale_x, scale_y = np.zeros(k), np.zeros(k)
    np.maximum.at(scale_x, groups, np.abs(dx))
    np.maximum.at(scale_y, groups, np.abs(dy))
    valid = (sizes > 1) & (scale_x > 0) & (scale_y > 0)
    corr = np.full(k, np.nan)
    if np.any(valid):
        sx = np.zeros(n)
        sy = np.zeros(n)
        np.divide(dx, scale_x[groups], out=sx, where=scale_x[groups] > 0)
        np.divide(dy, scale_y[groups], out=sy, where=scale_y[groups] > 0)
        cross = np.bincount(groups, weights=sx * sy, minlength=k)
        xx = np.bincount(groups, weights=sx * sx, minlength=k)
        yy = np.bincount(groups, weights=sy * sy, minlength=k)
        corr[valid] = np.clip(cross[valid] / np.sqrt(xx[valid] * yy[valid]), -1.0, 1.0)
    owned_labels = np.array(labels, dtype=np.int64, copy=True)
    owned_labels.flags.writeable = False
    return PDNNFitCorrelations(owned_labels, _owned(corr))
