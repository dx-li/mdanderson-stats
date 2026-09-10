"""Johnson's posterior chi-square diagnostic for complete continuous observations."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammainc, gammaincc, gammaincinv, logsumexp

from ._validation import FloatArray, finite, scalar
from .boin import _owned


@dataclass(frozen=True)
class BayesianChiSquare:
    bin_counts: NDArray[np.int64]
    statistic: FloatArray
    reference_tail_probability: FloatArray
    degrees_of_freedom: int
    mean_statistic: float
    area_against_reference: float
    mean_reference_tail: float
    critical_value: float
    critical_exceedance_fraction: float


def bayesian_chi_square_cdf(
    cdf_samples: ArrayLike,
    *,
    bins: int | None = None,
    critical_probability: float = 0.95,
) -> BayesianChiSquare:
    """Evaluate Johnson (2004) equations (2)-(3) for posterior CDF draws.

    Input shape is (posterior samples, observations); each row evaluates the
    *observed* data under one jointly sampled parameter vector. Do not substitute
    posterior-predictive observations or repeated MLEs. Equal-probability bins
    include the upper boundary. Numerical CDF zero belongs to the first bin.
    Reference tails and their posterior average are not calibrated p-values for
    the averaged statistic. Degrees of freedom are bins-1, with no parameter deduction.
    """
    cdf = finite(cdf_samples, "cdf_samples")
    if cdf.ndim != 2 or cdf.shape[0] < 1 or cdf.shape[1] < 2 or cdf.size > 20000000:
        raise ValueError(
            "require a sample-by-observation matrix, >=2 observations and <=20 million cells"
        )
    if np.any((cdf < 0) | (cdf > 1)):
        raise ValueError("CDF samples must be in [0,1]")
    n = cdf.shape[1]
    k = max(2, int(np.floor(n**0.4 + 0.5))) if bins is None else scalar(bins, "bins")
    level = scalar(critical_probability, "critical_probability")
    if k != int(k) or not 2 <= k <= 1000 or not 0 < level < 1 or cdf.shape[0] * k > 20000000:
        raise ValueError(
            "require 2..1000 bins, critical probability in (0,1), <=20 million count cells"
        )
    k = int(k)
    counts = np.empty((cdf.shape[0], k), dtype=np.int64)
    edges = np.arange(1, k) / k
    for start in range(0, cdf.shape[0], 256):
        batch = cdf[start : start + 256]
        ids = np.searchsorted(edges, batch, side="left")
        counts[start : start + len(batch)] = np.bincount(
            (ids + np.arange(len(batch))[:, None] * k).ravel(),
            minlength=len(batch) * k,
        ).reshape(len(batch), k)
    statistic = np.sum((counts - n / k) ** 2 / (n / k), axis=1)
    tails = gammaincc((k - 1) / 2, statistic / 2)
    area = float(np.mean(gammainc((k - 1) / 2, statistic / 2)))
    critical = float(2 * gammaincinv((k - 1) / 2, level))
    return BayesianChiSquare(
        _owned(counts),
        _owned(statistic),
        _owned(tails),
        k - 1,
        float(statistic.mean()),
        area,
        float(tails.mean()),
        critical,
        float(np.mean(statistic > critical)),
    )


@dataclass(frozen=True)
class ExponentialBayesianGOF:
    log_rate_samples: FloatArray
    posterior_shape: float
    log_posterior_rate: float
    diagnostic: BayesianChiSquare


def exponential_bayesian_gof(
    times: ArrayLike,
    *,
    prior_shape: float,
    prior_rate: float,
    samples: int = 1000,
    bins: int | None = None,
    critical_probability: float = 0.95,
    rng: int | np.random.Generator | None = None,
) -> ExponentialBayesianGOF:
    """Exact Gamma-prior posterior for exponential rates, with uncensored data only.

    prior_rate has time units; Gamma density is proportional to
    rate**(prior_shape-1)*exp(-prior_rate*rate). Both hyperparameters are explicit.
    Shape=rate=0 denotes the improper 1/rate prior, proper after these observations.
    This is an independent prior convention, not a recovered BCSTTE default.
    """
    x = finite(times, "times")
    a, b = scalar(prior_shape, "prior_shape"), scalar(prior_rate, "prior_rate")
    size = scalar(samples, "samples")
    if x.ndim != 1 or x.size < 2 or np.any(x <= 0) or a < 0 or b < 0:
        raise ValueError("require >=2 positive complete times and nonnegative prior parameters")
    if size != int(size) or not 1 <= size <= 100000 or size * x.size > 20000000:
        raise ValueError(
            "require 1..100000 posterior samples and <=20 million sample-observation cells"
        )
    shape = a + x.size
    log_rate = float(np.logaddexp(logsumexp(np.log(x)), np.log(b) if b > 0 else -np.inf))
    draws = np.random.default_rng(rng).gamma(shape, size=int(size))
    if np.any(~np.isfinite(draws)) or np.any(draws <= 0):
        raise ArithmeticError("Gamma posterior draws cannot be represented")
    log_draws = np.log(draws) - log_rate
    with np.errstate(over="ignore", under="ignore"):
        cdf = -np.expm1(-np.exp(log_draws[:, None] + np.log(x)))
    diagnostic = bayesian_chi_square_cdf(cdf, bins=bins, critical_probability=critical_probability)
    return ExponentialBayesianGOF(_owned(log_draws), shape, log_rate, diagnostic)
