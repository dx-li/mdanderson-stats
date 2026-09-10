"""Subject-level Metropolis-within-Gibbs and variance conditionals for ANOVA DDP."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite, scalar
from .anovaddp import anovaddp_amplitude_posterior, anovaddp_curve, anovaddp_loglikelihood


@dataclass(frozen=True)
class AnovaDDPSubjectUpdate:
    parameters: FloatArray
    accepted: NDArray[np.bool_]
    log_acceptance: FloatArray
    normal_draws: FloatArray
    uniform_draws: FloatArray


def anovaddp_subject_update(
    parameters: ArrayLike,
    time: ArrayLike,
    observations: ArrayLike,
    *,
    prior_mean: ArrayLike,
    prior_covariance: ArrayLike,
    variance: float,
    seed: int | None = None,
) -> AnovaDDPSubjectUpdate:
    """One native simtheta sweep conditional on the subject's Gaussian prior.

    Draw amplitudes jointly; propose each knot from its conditional prior;
    propose slope by a symmetric normal random walk with native scale 0.45.
    Acceptance arrays index tau1,tau2,b1. No ordering truncation is imposed.
    This conditional transition is not the full dependent-DP sampler.
    """
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    amplitude = anovaddp_amplitude_posterior(
        parameters,
        time,
        observations,
        prior_mean=prior_mean,
        prior_covariance=prior_covariance,
        variance=variance,
    )
    theta = finite(parameters, "parameters").copy()
    m = finite(prior_mean, "prior_mean")
    c = finite(prior_covariance, "prior_covariance")
    c = c / 2 + c.T / 2
    rng = np.random.default_rng(seed)
    normal = rng.standard_normal(6)
    uniform = rng.random(3)
    try:
        theta[:3] = amplitude.mean + np.linalg.cholesky(amplitude.covariance) @ normal[:3]
    except np.linalg.LinAlgError as exc:
        raise ArithmeticError(
            "amplitude posterior covariance is not numerically positive definite"
        ) from exc
    current = float(anovaddp_loglikelihood(theta, time, observations, variance))
    accepted = np.zeros(3, dtype=bool)
    log_acceptance = np.empty(3)
    for j, index in enumerate((3, 4, 5)):
        other = np.arange(6) != index
        cross = np.linalg.solve(c[np.ix_(other, other)], c[other, index])
        mean = float(m[index] + cross @ (theta[other] - m[other]))
        conditional_variance = float(c[index, index] - cross @ c[other, index])
        if not np.isfinite(conditional_variance) or conditional_variance <= 0:
            raise ArithmeticError("conditional Gaussian variance is not numerically positive")
        sd = np.sqrt(conditional_variance)
        proposal = theta.copy()
        proposal[index] = (mean if index < 5 else theta[index]) + normal[index] * sd * (
            1 if index < 5 else 0.45
        )
        proposed = float(anovaddp_loglikelihood(proposal, time, observations, variance))
        ratio = proposed - current
        if index == 5:
            ratio -= (
                0.5 * ((proposal[index] - mean) / sd) ** 2 - 0.5 * ((theta[index] - mean) / sd) ** 2
            )
        if not np.isfinite(ratio):
            raise ArithmeticError("Metropolis log acceptance exceeds floating-point range")
        log_acceptance[j] = min(0.0, ratio)
        # Comparing logarithms avoids exp(ratio) overflow. A zero uniform draw
        # is handled explicitly, without producing a logarithm warning.
        if uniform[j] == 0 or np.log(uniform[j]) < log_acceptance[j]:
            theta = proposal
            current = proposed
            accepted[j] = True
    return AnovaDDPSubjectUpdate(
        _freeze(theta),
        np.frombuffer(accepted.tobytes(), dtype=np.bool_),
        _freeze(log_acceptance),
        _freeze(normal),
        _freeze(uniform),
    )


@dataclass(frozen=True)
class AnovaDDPVariancePosterior:
    shape: float
    scale: float
    residual_sum_squares: float
    observations: int
    mode: str


def anovaddp_variance_posterior(
    parameters: ArrayLike,
    time: ArrayLike,
    observations: ArrayLike,
    subject: ArrayLike,
    *,
    alpha0: float,
    beta0: float,
    mode: Literal["documented", "source"] = "documented",
) -> AnovaDDPVariancePosterior:
    """Inverse-gamma residual variance conditional, with zero-based subject IDs.

    Documented prior is IG(alpha0/2,beta0/2). Source mode explicitly reproduces
    vardati's omission of beta0. It rejects the resulting zero-scale distribution
    if residuals are all zero. Rows of parameters index subjects.
    """
    if any(np.iscomplexobj(a) for a in (parameters, time, observations, subject)):
        raise ValueError("data and parameters must be real")
    theta, t, y = (
        finite(parameters, "parameters"),
        finite(time, "time"),
        finite(observations, "observations"),
    )
    ids = count(subject, "subject")
    if (
        theta.ndim != 2
        or theta.shape[1] != 6
        or not theta.shape[0]
        or t.ndim != 1
        or not t.size
        or y.shape != t.shape
        or ids.shape != t.shape
        or np.any(ids >= theta.shape[0])
    ):
        raise ValueError("require (subjects,6) parameters, aligned vectors and valid subject IDs")
    a, b = scalar(alpha0, "alpha0"), scalar(beta0, "beta0")
    if a <= 0 or b <= 0 or mode not in ("documented", "source"):
        raise ValueError("positive alpha0/beta0 and mode documented or source required")
    order = np.argsort(ids, kind="stable")
    sorted_time, sorted_y = t[order], y[order]
    subjects, starts = np.unique(ids[order], return_index=True)
    ends = np.r_[starts[1:], t.size]
    residual = np.empty(t.size)
    for i, start, end in zip(subjects, starts, ends, strict=True):
        residual[start:end] = sorted_y[start:end] - anovaddp_curve(
            theta[int(i)], sorted_time[start:end]
        )
    maximum = float(np.max(np.abs(residual)))
    with np.errstate(over="ignore"):
        sse = (
            0.0 if maximum == 0 else float(maximum * (maximum * np.sum((residual / maximum) ** 2)))
        )
    shape = a / 2 + t.size / 2
    scale = sse / 2 + (b / 2 if mode == "documented" else 0.0)
    if not np.isfinite(sse) or not np.isfinite(scale) or scale <= 0:
        raise ArithmeticError(
            "inverse-gamma residual scale is zero or exceeds floating-point range"
        )
    return AnovaDDPVariancePosterior(shape, scale, sse, t.size, mode)
