"""Convert SINGLE's marginal prior inputs to latent normal parameters."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite


def single_prior_parameters(
    mean: ArrayLike,
    variance: ArrayLike,
    correlation: ArrayLike,
    *,
    lognormal: ArrayLike | None = None,
    conversion: str = "exact",
) -> tuple[FloatArray, FloatArray]:
    """Return latent mean/covariance from raw marginal means and variances.

    Correlations are between latent normal coordinates, as in SINGLE's input
    workflow; they are not correlations between exponentiated parameters.
    Exact conversion matches each log-normal marginal's raw moments. The
    'legacy' conversion uses ALNTON's first-order approximation instead.
    The returned arrays can be passed to single_normal_criterion with the same
    lognormal mask. Two or three parameters are supported.
    """
    mu, var, corr = (
        finite(x, name)
        for x, name in [(mean, "mean"), (variance, "variance"), (correlation, "correlation")]
    )
    if mu.ndim != 1 or mu.size not in (2, 3) or var.shape != mu.shape or np.any(var < 0):
        raise ValueError("Require two or three means and matching nonnegative variances")
    if (
        corr.shape != (mu.size, mu.size)
        or not np.array_equal(corr, corr.T)
        or np.any(np.diag(corr) != 1)
        or np.any(np.abs(corr) > 1)
    ):
        raise ValueError("correlation must be symmetric with unit diagonal and entries in [-1,1]")
    if np.min(np.linalg.eigvalsh(corr)) < -32 * np.finfo(float).eps:
        raise ValueError("correlation must be positive semidefinite")
    mask = np.zeros(mu.size, dtype=bool) if lognormal is None else np.asarray(lognormal)
    if mask.shape != mu.shape or mask.dtype != np.bool_:
        raise ValueError("lognormal must be a boolean vector matching the mean")
    if np.any(mu[mask] <= 0):
        raise ValueError("Log-normal marginal means must be positive")
    if conversion not in ("exact", "legacy"):
        raise ValueError("conversion must be exact or legacy")
    latent_mean, latent_variance = mu.copy(), var.copy()
    # Work in log space: variance / mean**2 may overflow even when its log is finite.
    with np.errstate(divide="ignore", over="ignore", under="ignore"):
        log_ratio = np.log(var[mask]) - 2 * np.log(mu[mask])
        if conversion == "exact":
            latent_variance[mask] = np.logaddexp(0, log_ratio)
            latent_mean[mask] = np.log(mu[mask]) - latent_variance[mask] / 2
        else:
            latent_variance[mask] = np.exp(log_ratio)
            latent_mean[mask] = np.log(mu[mask])
    if not np.all(np.isfinite(latent_variance)):
        raise ValueError("Legacy log-normal variance conversion overflowed")
    sd = np.sqrt(latent_variance)
    covariance = corr * (sd[:, None] * sd[None, :])
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Latent covariance overflowed")
    return latent_mean, covariance
