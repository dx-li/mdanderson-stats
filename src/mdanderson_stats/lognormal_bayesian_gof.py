"""Complete-data log-normal Bayesian chi-square diagnostics.

The model uses a proper, explicit Normal-Inverse-Gamma prior on the log-time
location and variance. It is a conjugate Python extension, not a recovery of
the BCSTTE executable's unavailable prior or fitting defaults.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

from ._validation import FloatArray, finite, scalar
from .bayesian_chi_square import BayesianChiSquare, bayesian_chi_square_cdf
from .boin import _owned

_MAX_CDF_AND_SUMMARY_CELLS = 20_000_000
_MAX_SAMPLES = 100_000


@dataclass(frozen=True)
class LognormalBayesianGOF:
    """Joint posterior diagnostics for log-location and log-variance draws.

    ``centered_location_samples`` plus ``location_offset`` encode the paired
    log-location draws used for every observation's posterior CDF. Centered
    coordinates retain variation under extreme time-unit shifts. The posterior
    inverse-gamma scale is kept in log form to avoid exponentiating extremes.
    """

    centered_location_samples: FloatArray
    location_offset: float
    log_variance_samples: FloatArray
    posterior_centered_location: float
    posterior_location: float
    posterior_location_precision: float
    posterior_variance_shape: float
    posterior_log_variance_scale: float
    diagnostic: BayesianChiSquare


def _log_abs_difference(first: float, second: float) -> float:
    """Return log(abs(first-second)) without overflowing opposite signs."""
    difference = first - second
    if np.isfinite(difference):
        return float(np.log(abs(difference))) if difference != 0 else -np.inf
    scale = max(abs(first), abs(second))
    relative = first / scale - second / scale
    return float(np.log(scale) + np.log(abs(relative)))


def _convex_location(
    first: float, second: float, first_weight: float, second_weight: float
) -> float:
    """Compute a weighted mean without losing a tiny, material weight."""
    return float(first_weight * first + second_weight * second)


def lognormal_complete_data_bayesian_gof(
    times: ArrayLike,
    *,
    prior_location: float,
    prior_location_precision: float,
    prior_variance_shape: float,
    prior_variance_scale: float,
    samples: int = 1000,
    bins: int | None = None,
    critical_probability: float = 0.95,
    rng: int | np.random.Generator | None = None,
) -> LognormalBayesianGOF:
    """Fit a complete-data log-normal model under a proper N-InvGamma prior.

    If ``Y=log(T)``, then ``Y | mu,sigma² ~ Normal(mu,sigma²)``. The prior is
    ``sigma² ~ InvGamma(a0,b0)`` in the shape/scale convention and
    ``mu | sigma² ~ Normal(m0,sigma²/kappa0)``. All four prior parameters must
    be supplied, with ``kappa0,a0,b0 > 0``. Censoring, rounded-time likelihoods,
    unknown-prior defaults and BCSTTE executable parity are outside this API.

    Posterior draws are paired across observations. CDF values are evaluated
    from log times and log variances in bounded chunks and sent to Johnson's
    existing complete-data diagnostic. Workspaces have a combined 20-million
    cell cap checked before converting observations or consuming randomness.
    """
    for value, name in (
        (prior_location, "prior_location"),
        (prior_location_precision, "prior_location_precision"),
        (prior_variance_shape, "prior_variance_shape"),
        (prior_variance_scale, "prior_variance_scale"),
        (samples, "samples"),
        (critical_probability, "critical_probability"),
    ):
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real")
    if bins is not None and np.iscomplexobj(bins):
        raise ValueError("bins must be real")
    m0 = scalar(prior_location, "prior_location")
    k0 = scalar(prior_location_precision, "prior_location_precision")
    a0 = scalar(prior_variance_shape, "prior_variance_shape")
    b0 = scalar(prior_variance_scale, "prior_variance_scale")
    size = scalar(samples, "samples")
    time_shape = getattr(times, "shape", None)
    if time_shape is None:
        sequence = cast(Sequence[object], times)
        try:
            n = len(sequence)
        except TypeError as exc:
            raise ValueError("times must be a one-dimensional vector") from exc
        if not 2 <= n <= _MAX_CDF_AND_SUMMARY_CELLS:
            raise ValueError("require at least two complete positive event times")
        for item in sequence:
            if not np.isscalar(item):
                raise ValueError("times must be a one-dimensional vector")
            if np.iscomplexobj(item):
                raise ValueError("times must be real")
    else:
        if len(time_shape) != 1:
            raise ValueError("times must be a one-dimensional vector")
        n = int(time_shape[0])
        if np.iscomplexobj(times):
            raise ValueError("times must be real")
        if not 2 <= n <= _MAX_CDF_AND_SUMMARY_CELLS:
            raise ValueError("require at least two complete positive event times")
    if k0 <= 0 or a0 <= 0 or b0 <= 0:
        raise ValueError(
            "prior location precision, variance shape, and variance scale must be positive"
        )
    if not np.isfinite(size) or size != int(size) or not 1 <= size <= _MAX_SAMPLES:
        raise ValueError(f"samples must be an integer in [1,{_MAX_SAMPLES}]")
    bins_count = max(2, int(np.floor(n**0.4 + 0.5))) if bins is None else scalar(bins, "bins")
    level = scalar(critical_probability, "critical_probability")
    if (
        not np.isfinite(bins_count)
        or bins_count != int(bins_count)
        or not 2 <= bins_count <= 1000
        or not 0 < level < 1
    ):
        raise ValueError("bins must be 2..1000 and critical_probability must lie in (0,1)")
    cdf_cells = int(size) * n
    summary_cells = int(size) * int(bins_count)
    chunk_cells = min(int(size), 256) * n
    if (
        cdf_cells + 3 * summary_cells + 6 * int(size) + 8 * n + 4 * chunk_cells
        > _MAX_CDF_AND_SUMMARY_CELLS
    ):
        raise ValueError("log-normal CDF and diagnostic workspaces exceed 20 million cells")

    x = finite(times, "times")
    if x.ndim != 1 or np.any(x <= 0):
        raise ValueError("require at least two complete positive event times")
    # Center log times before averaging so small relative differences survive
    # large common time-unit offsets. Near-anchor differences use log1p.
    time_reference = float(x[0])
    location_offset = float(np.log(time_reference))
    relative_log_times = np.empty_like(x)
    with np.errstate(over="ignore", under="ignore"):
        near_reference = (x >= time_reference * 0.5) & (x <= time_reference * 1.5)
    relative_log_times[near_reference] = np.log1p(
        (x[near_reference] - time_reference) / time_reference
    )
    relative_log_times[~near_reference] = np.log(x[~near_reference]) - location_offset
    y = relative_log_times
    ybar = float(y[0] + np.mean(y - y[0]))
    deviations = y - ybar
    deviation_scale = float(np.max(np.abs(deviations)))
    if deviation_scale == 0:
        log_sse_half = -np.inf
    else:
        normalized = deviations / deviation_scale
        log_sse_half = (
            2 * np.log(deviation_scale)
            + np.log(float(np.dot(normalized, normalized)))
            - np.log(2.0)
        )
    posterior_precision = k0 + n
    if not np.isfinite(posterior_precision) or posterior_precision <= 0:
        raise ArithmeticError("posterior location precision is not representable")
    centered_prior_location = m0 - location_offset
    posterior_centered_location = _convex_location(
        centered_prior_location,
        ybar,
        k0 / posterior_precision,
        n / posterior_precision,
    )
    posterior_location = location_offset + posterior_centered_location
    if not np.isfinite(posterior_location):
        raise ArithmeticError("posterior log-location mean is not representable")
    posterior_shape = a0 + n / 2.0
    if not np.isfinite(posterior_shape) or posterior_shape <= 0:
        raise ArithmeticError("posterior inverse-gamma shape is not representable")
    log_between = (
        np.log(k0)
        + np.log(float(n))
        - np.log(posterior_precision)
        - np.log(2.0)
        + 2.0 * _log_abs_difference(centered_prior_location, ybar)
    )
    log_posterior_scale = float(np.logaddexp(np.log(b0), np.logaddexp(log_sse_half, log_between)))
    if not np.isfinite(log_posterior_scale):
        raise ArithmeticError("posterior inverse-gamma scale is not representable")

    generator = np.random.default_rng(rng)
    try:
        unit_gamma = generator.gamma(posterior_shape, size=int(size))
        normal = generator.standard_normal(int(size))
    except (ValueError, OverflowError, FloatingPointError) as exc:
        raise ArithmeticError("log-normal posterior draws cannot be represented") from exc
    if np.any(~np.isfinite(unit_gamma)) or np.any(unit_gamma <= 0) or np.any(~np.isfinite(normal)):
        raise ArithmeticError("log-normal posterior draws cannot be represented")
    log_variance = log_posterior_scale - np.log(unit_gamma)
    log_location_sd = 0.5 * (log_variance - np.log(posterior_precision))
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        log_abs_noise = np.log(np.abs(normal)) + log_location_sd
        noise = np.sign(normal) * np.exp(log_abs_noise)
        centered_location = posterior_centered_location + noise
    if np.any(~np.isfinite(centered_location)):
        raise ArithmeticError("posterior log-location draws exceed floating-point range")

    cdf = np.empty((int(size), n), dtype=float)
    for start in range(0, int(size), 256):
        stop = min(start + 256, int(size))
        mu = centered_location[start:stop, None]
        log_sigma = 0.5 * log_variance[start:stop, None]
        delta = y[None, :] - mu
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            log_abs_z = np.log(np.abs(delta)) - log_sigma
            z = np.sign(delta) * np.exp(log_abs_z)
            cdf[start:stop] = ndtr(z)
        if np.any(np.isnan(cdf[start:stop])):
            raise ArithmeticError("log-normal CDF evaluation produced indeterminate values")

    diagnostic = bayesian_chi_square_cdf(cdf, bins=int(bins_count), critical_probability=level)
    return LognormalBayesianGOF(
        _owned(centered_location),
        location_offset,
        _owned(log_variance),
        posterior_centered_location,
        posterior_location,
        float(posterior_precision),
        float(posterior_shape),
        log_posterior_scale,
        diagnostic,
    )
