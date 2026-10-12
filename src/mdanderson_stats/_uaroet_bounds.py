"""Analytic probability bounds for rejecting numerically extreme proposals."""

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtri_exp

from ._validation import FloatArray


def _log_boundaries(log_probabilities: FloatArray) -> FloatArray:
    values = [-np.inf]
    for i in range(1, len(log_probabilities)):
        lower = float(logsumexp(log_probabilities[:i]))
        upper = float(logsumexp(log_probabilities[i:]))
        values.append(float(ndtri_exp(lower) if lower <= -np.log(2) else -ndtri_exp(upper)))
    return np.asarray([*values, np.inf])


def _likelihood_upper_bound(
    log_e: FloatArray,
    log_t: FloatArray,
    observed: np.ndarray,
    partial: np.ndarray,
    rho: float | None = None,
) -> float:
    """Upper-bound joint likelihood using marginals and optional Gaussian halfspaces.

    A rectangle is contained in each halfspace +/-X +/-Y >= its corresponding
    corner sum. Those sums are univariate normal with variance2*(1 +/- rho).
    These are probability bounds, not substitutes for an accepted likelihood.
    The positive numerical margin makes threshold rejection conservative with
    respect to ordinary floating-point accumulation error, not interval-certified.
    """
    bounds = np.minimum(log_e[:, :, None], log_t[:, None, :])
    if rho is not None:
        for dose in range(log_e.shape[0]):
            e = _log_boundaries(log_e[dose])
            t = _log_boundaries(log_t[dose])
            for i, j in np.argwhere(observed[dose] > 0):
                for a, b, covariance in (
                    (e[i], t[j], rho),
                    (e[i], -t[j + 1], -rho),
                    (-e[i + 1], t[j], -rho),
                    (-e[i + 1], -t[j + 1], rho),
                ):
                    if np.isfinite(a) and np.isfinite(b):
                        bound = log_ndtr(-(a + b) / np.sqrt(2 * (1 + covariance)))
                        bounds[dose, i, j] = min(bounds[dose, i, j], bound)
    mask = observed > 0
    partial_mask = partial > 0
    with np.errstate(over="ignore"):
        result = float(np.sum(observed[mask] * bounds[mask]))
        result += float(np.sum(partial[partial_mask] * log_t[partial_mask]))
    if np.isfinite(result):
        result += 1e-10 * max(1.0, abs(result))
    return result
