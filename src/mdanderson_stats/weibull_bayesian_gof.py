"""Fixed-shape Weibull Bayesian chi-square diagnostics.

This is a complete-data Python model with caller-specified priors, not a
reconstruction of BCSTTE's unavailable fitter or defaults. For Weibull shape
``beta`` and scale ``eta``, use the transformed rate ``lambda = eta**(-beta)``.
A Gamma(shape, rate) prior on lambda is conjugate when beta is fixed. Posterior
CDF draws are passed to the existing Johnson (2004) statistic.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite, scalar
from .bayesian_chi_square import BayesianChiSquare, bayesian_chi_square_cdf
from .boin import _owned

_MAX_CDF_AND_SUMMARY_CELLS = 20_000_000
_MAX_SAMPLES = 100_000


@dataclass(frozen=True)
class WeibullBayesianGOF:
    """Fixed-shape posterior diagnostics with a centered rate encoding.

    ``log_rate = centered_log_rate_samples + log_rate_offset``. Retain the
    centered values and offset separately: their sum can lose Gamma variation
    when the absolute log-rate is very large. ``posterior_rate_log_scale`` and
    ``posterior_rate_scaled_sum`` encode the denominator as
    ``exp(scale) * scaled_sum``; ``log_posterior_rate`` is a convenience scalar
    that may lose low-order precision when the scale is extreme.
    """

    centered_log_rate_samples: FloatArray
    log_rate_offset: float
    posterior_shape: float
    log_posterior_rate: float
    posterior_rate_log_scale: float
    posterior_rate_scaled_sum: float
    weibull_shape: float
    diagnostic: BayesianChiSquare


def weibull_fixed_shape_bayesian_gof(
    times: ArrayLike,
    *,
    weibull_shape: float,
    prior_shape: float,
    prior_rate: float,
    samples: int = 1000,
    bins: int | None = None,
    critical_probability: float = 0.95,
    rng: int | np.random.Generator | None = None,
) -> WeibullBayesianGOF:
    """Fit a complete-data Weibull model with fixed shape and explicit Gamma prior.

    The Weibull CDF is ``1 - exp(-lambda * t**weibull_shape)`` with
    ``lambda = scale**(-weibull_shape)``. ``prior_shape`` and ``prior_rate``
    specify a Gamma prior on lambda, parameterized by shape and rate. The
    prior rate has units ``time**weibull_shape``. Zero prior parameters are
    permitted when the resulting posterior is proper (the observations here
    ensure positive posterior shape and rate).

    Only complete positive event times are accepted. Censoring, rounded-time
    likelihoods, and unknown-shape fitting are outside this API. CDFs are
    evaluated in log space and generated in row chunks; the joint CDF and
    Johnson diagnostic workspaces have a combined 20-million-cell cap checked
    before random draws.
    """
    for value, name in (
        (weibull_shape, "weibull_shape"),
        (prior_shape, "prior_shape"),
        (prior_rate, "prior_rate"),
        (samples, "samples"),
        (critical_probability, "critical_probability"),
    ):
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real")
    if bins is not None and np.iscomplexobj(bins):
        raise ValueError("bins must be real")
    beta = scalar(weibull_shape, "weibull_shape")
    a, b = scalar(prior_shape, "prior_shape"), scalar(prior_rate, "prior_rate")
    size = scalar(samples, "samples")
    time_shape = getattr(times, "shape", None)
    if time_shape is None:
        sequence = cast(Sequence[object], times)
        try:
            n_times = len(sequence)
        except TypeError as exc:
            raise ValueError("times must be a one-dimensional vector") from exc
        if not 2 <= n_times <= _MAX_CDF_AND_SUMMARY_CELLS:
            raise ValueError(
                "require a one-dimensional vector of at least two complete event times"
            )
        for item in sequence:
            if not np.isscalar(item):
                raise ValueError("times must be a one-dimensional vector")
            if np.iscomplexobj(item):
                raise ValueError("times must be real")
    else:
        if len(time_shape) != 1:
            raise ValueError("times must be a one-dimensional vector")
        n_times = int(time_shape[0])
        if np.iscomplexobj(times):
            raise ValueError("times must be real")
        if not 2 <= n_times <= _MAX_CDF_AND_SUMMARY_CELLS:
            raise ValueError(
                "require a one-dimensional vector of at least two complete event times"
            )
    if beta <= 0 or a < 0 or b < 0:
        raise ValueError("weibull_shape must be positive; prior_shape/rate must be nonnegative")
    if not np.isfinite(size) or size != int(size) or not 1 <= size <= _MAX_SAMPLES:
        raise ValueError(f"samples must be an integer in [1,{_MAX_SAMPLES}]")

    bin_count = max(2, int(np.floor(n_times**0.4 + 0.5))) if bins is None else scalar(bins, "bins")
    level = scalar(critical_probability, "critical_probability")
    if (
        not np.isfinite(bin_count)
        or bin_count != int(bin_count)
        or not 2 <= bin_count <= 1000
        or not 0 < level < 1
    ):
        raise ValueError("bins must be 2..1000 and critical_probability must lie in (0,1)")
    cdf_cells = int(size) * n_times
    summary_cells = int(size) * int(bin_count)
    # Account for CDF output, count/freeze copies, draws and centered samples.
    chunk_cells = min(int(size), 256) * n_times
    if (
        cdf_cells + 2 * summary_cells + 3 * int(size) + 6 * n_times + 4 * chunk_cells
        > _MAX_CDF_AND_SUMMARY_CELLS
    ):
        raise ValueError("Weibull CDF and diagnostic workspaces exceed 20 million cells")

    x = finite(times, "times")
    if x.ndim != 1 or np.any(x <= 0):
        raise ValueError("require at least two positive complete event times")
    posterior_shape = a + n_times
    if not np.isfinite(posterior_shape) or posterior_shape <= 0:
        raise ArithmeticError("Weibull posterior Gamma shape is not representable")
    xmax = float(np.max(x))
    relative = (x - xmax) / xmax
    centered_log_powers = np.empty_like(x)
    near_max = relative > -0.5
    with np.errstate(over="ignore", invalid="ignore"):
        centered_log_powers[near_max] = beta * np.log1p(relative[near_max])
        centered_log_powers[~near_max] = beta * (np.log(x[~near_max]) - np.log(xmax))
    with np.errstate(over="ignore", invalid="ignore"):
        log_power_scale = beta * np.log(xmax)
    if not np.isfinite(log_power_scale) or np.any(np.isnan(centered_log_powers)):
        raise ArithmeticError("weibull_shape * log(times) exceeds floating-point range")
    log_prior_rate = np.log(b) if b > 0 else -np.inf
    # Scale likelihood powers relative to the maximum observed time before
    # multiplying by shape; this preserves close-time differences even when
    # beta*log(t) itself is enormous. Scale the prior rate by the same factor.
    rate_log_scale = float(max(log_power_scale, log_prior_rate))
    power_log_offset = log_power_scale - rate_log_scale
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled_power = np.exp(centered_log_powers + power_log_offset)
        scaled_prior_rate = float(np.exp(log_prior_rate - rate_log_scale))
    scaled_rate_sum = float(scaled_power.sum() + scaled_prior_rate)
    if not np.isfinite(scaled_rate_sum) or scaled_rate_sum <= 0:
        raise ArithmeticError("Weibull scaled posterior Gamma rate is not representable")
    log_posterior_rate = rate_log_scale + float(np.log(scaled_rate_sum))
    if not np.isfinite(log_posterior_rate):
        # The pair (rate_log_scale, scaled_rate_sum) remains a valid exact-scale
        # representation; this scalar is only a convenience summary.
        log_posterior_rate = rate_log_scale

    generator = np.random.default_rng(rng)
    try:
        rate_draws = generator.gamma(posterior_shape, size=int(size))
    except (ValueError, OverflowError, FloatingPointError) as exc:
        raise ArithmeticError("Weibull posterior Gamma draws cannot be represented") from exc
    if np.any(~np.isfinite(rate_draws)) or np.any(rate_draws <= 0):
        raise ArithmeticError("Weibull posterior Gamma draws cannot be represented")
    centered_log_rates = np.log(rate_draws) - np.log(scaled_rate_sum)
    log_rate_offset = -rate_log_scale

    cdf = np.empty((int(size), n_times), dtype=float)
    for start in range(0, int(size), 256):
        stop = min(start + 256, int(size))
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            log_hazard = (
                centered_log_rates[start:stop, None]
                + centered_log_powers[None, :]
                + power_log_offset
            )
            hazard = np.exp(log_hazard)
            cdf[start:stop] = -np.expm1(-hazard)
        if np.any(np.isnan(cdf[start:stop])):
            raise ArithmeticError("Weibull CDF evaluation produced indeterminate values")

    diagnostic = bayesian_chi_square_cdf(cdf, bins=int(bin_count), critical_probability=level)
    return WeibullBayesianGOF(
        _owned(centered_log_rates),
        log_rate_offset,
        float(posterior_shape),
        log_posterior_rate,
        rate_log_scale,
        scaled_rate_sum,
        beta,
        diagnostic,
    )
