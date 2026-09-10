"""PerfectMatch's reference and average-rank quantile normalization."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite
from .beta_binomial import _owned


def quantile_normalize(intensities: ArrayLike, *, reference: ArrayLike | None = None) -> FloatArray:
    """Normalize a (probes, samples) matrix to a reference or mean sorted profile.

    Input ties receive the mean target over their occupied ranks. No missing-value
    imputation or CEL parsing is performed. Intensities must be nonnegative.
    """
    x = finite(intensities, "intensities")
    if x.ndim != 2 or min(x.shape) == 0 or np.any(x < 0):
        raise ValueError("intensities must be a nonempty nonnegative (probes,samples) matrix")
    order = np.argsort(x, axis=0, kind="stable")
    sorted_x = np.take_along_axis(x, order, axis=0)
    if reference is None:
        target = np.sum(sorted_x / x.shape[1], axis=1)
    else:
        r = finite(reference, "reference")
        if r.shape != (len(x),) or np.any(r < 0):
            raise ValueError("reference must contain one nonnegative intensity per probe")
        target = np.sort(r)
    result = np.empty_like(x)
    result[order, np.arange(x.shape[1])] = target[:, None]
    for column in range(x.shape[1]):
        values = sorted_x[:, column]
        starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
        ends = np.r_[starts[1:], len(values)]
        tied = ends - starts > 1
        for start, end in zip(starts[tied], ends[tied], strict=True):
            result[order[start:end, column], column] = np.sum(target[start:end] / (end - start))
    if np.any(~np.isfinite(result)):
        raise ArithmeticError("normalized intensities cannot be represented")
    return _owned(result)
