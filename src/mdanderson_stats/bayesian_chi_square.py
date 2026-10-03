"""Johnson's posterior chi-square diagnostic for complete continuous observations."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammainc, gammaincc, gammaincinv, logsumexp

from ._validation import FloatArray, finite, scalar
from .boin import _owned
from .chi_square_order_bounds import ChiSquareOrderBounds, chi_square_order_bounds


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

    def order_bounds(self, *, upper_trim: float = 0.005) -> ChiSquareOrderBounds:
        """Order-statistic diagnostic and fixed-rank bounds; see returned conventions."""
        return chi_square_order_bounds(
            self.statistic, self.degrees_of_freedom, upper_trim=upper_trim
        )


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
    diagnostic: BayesianChiSquare | None


def _event_indicator(event: ArrayLike | None, n: int) -> NDArray[np.bool_]:
    """Validate an optional actual-Boolean event vector."""
    if event is None:
        return np.ones(n, dtype=bool)
    shape = getattr(event, "shape", None)
    if shape is not None:
        if tuple(shape) != (n,):
            raise ValueError("event must be a one-dimensional Boolean vector matching times")
    else:
        sequence = cast(Sequence[object], event)
        try:
            event_length = len(sequence)
        except TypeError as exc:
            raise ValueError(
                "event must be a one-dimensional Boolean vector matching times"
            ) from exc
        if event_length != n:
            raise ValueError("event must be a one-dimensional Boolean vector matching times")
        if any(not isinstance(value, (bool, np.bool_)) for value in sequence):
            raise ValueError("event must contain actual Boolean values")
    raw = np.asarray(event)
    if raw.shape != (n,) or raw.dtype.kind != "b":
        raise ValueError("event must be a one-dimensional Boolean vector matching times")
    return raw.astype(bool, copy=True)


def exponential_bayesian_gof(
    times: ArrayLike,
    *,
    prior_shape: float,
    prior_rate: float,
    event: ArrayLike | None = None,
    samples: int = 1000,
    bins: int | None = None,
    critical_probability: float = 0.95,
    rng: int | np.random.Generator | None = None,
) -> ExponentialBayesianGOF:
    """Exact Gamma-prior posterior for exponential rates with optional censoring.

    prior_rate has time units; Gamma density is proportional to
    rate**(prior_shape-1)*exp(-prior_rate*rate). Both hyperparameters are explicit.
    ``event=True`` marks an exact event; ``False`` marks a right-censored time.
    The noninformative-censoring likelihood updates shape by the event count and
    rate by the sum of all observed times. Shape=rate=0 denotes the improper
    1/rate prior; it is accepted only when the posterior is proper. This is an
    independent prior convention, not a recovered BCSTTE default. A censored
    fit has no Johnson diagnostic because the source does not define a censored
    CDF transform.
    """
    if np.iscomplexobj(times):
        raise ValueError("times must be real")
    x = finite(times, "times")
    a, b = scalar(prior_shape, "prior_shape"), scalar(prior_rate, "prior_rate")
    size = scalar(samples, "samples")
    if x.ndim != 1 or x.size < 2 or a < 0 or b < 0:
        raise ValueError("require >=2 observations and nonnegative prior parameters")
    events = _event_indicator(event, x.size)
    if np.any(x < 0) or np.any(x[events] <= 0):
        raise ValueError("event times must be positive and all times nonnegative")
    if size != int(size) or not 1 <= size <= 100000 or size * x.size > 20000000:
        raise ValueError(
            "require 1..100000 posterior samples and <=20 million sample-observation cells"
        )
    bin_count = max(2, int(np.floor(x.size**0.4 + 0.5))) if bins is None else scalar(bins, "bins")
    level = scalar(critical_probability, "critical_probability")
    if (
        not np.isfinite(bin_count)
        or bin_count != int(bin_count)
        or not 2 <= bin_count <= 1000
        or not 0 < level < 1
        or (np.all(events) and size * bin_count > 20000000)
    ):
        raise ValueError("bins must be 2..1000 and critical_probability must lie in (0,1)")
    shape = a + int(np.count_nonzero(events))
    log_exposure = float(logsumexp(np.log(x[x > 0]))) if np.any(x > 0) else -np.inf
    log_rate = float(np.logaddexp(log_exposure, np.log(b) if b > 0 else -np.inf))
    if not np.isfinite(shape) or shape <= 0 or not np.isfinite(log_rate):
        raise ValueError("the Gamma posterior must have positive shape and rate")
    draws = np.random.default_rng(rng).gamma(shape, size=int(size))
    if np.any(~np.isfinite(draws)) or np.any(draws <= 0):
        raise ArithmeticError("Gamma posterior draws cannot be represented")
    log_draws = np.log(draws) - log_rate
    diagnostic = None
    if np.all(events):
        with np.errstate(over="ignore", under="ignore"):
            cdf = -np.expm1(-np.exp(log_draws[:, None] + np.log(x)))
        diagnostic = bayesian_chi_square_cdf(cdf, bins=int(bin_count), critical_probability=level)
    return ExponentialBayesianGOF(_owned(log_draws), shape, log_rate, diagnostic)
