"""Nonlinear response and Gaussian conditional kernels for ANOVA DDP."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar


def anovaddp_curve(
    parameters: ArrayLike, time: ArrayLike, *, repair_order: bool = False
) -> FloatArray:
    """Archived regressione curve; parameters (...,6), time a nonempty vector.

    Order: z1,z2,z3,tau1,tau2,b1; logistic intercept is fixed at -2.
    repair_order=True reproduces fittare's tau2=tau1+1 for reversed knots.
    The source's 1e-6 transition denominator floor is retained.
    """
    if np.iscomplexobj(parameters) or np.iscomplexobj(time):
        raise ValueError("parameters and time must be real")
    theta, t = finite(parameters, "parameters"), finite(time, "time")
    if theta.ndim < 1 or theta.shape[-1] != 6 or t.ndim != 1 or not t.size or not theta.size:
        raise ValueError("require parameters with final axis six and a nonempty time vector")
    if theta.size // 6 * t.size > 10_000_000:
        raise ValueError("curve evaluation exceeds 10 million values")
    if not isinstance(repair_order, bool):
        raise ValueError("repair_order must be boolean")
    z1, z2, z3, tau1, tau2, b1 = (theta[..., i, None] for i in range(6))
    if repair_order:
        reversed_knots = tau2 < tau1
        tau2 = np.where(reversed_knots, tau1 + 1, tau2)
        if np.any(~np.isfinite(tau2)) or np.any(reversed_knots & (tau2 <= tau1)):
            raise ArithmeticError("repaired knot is not representable")
    with np.errstate(over="ignore", invalid="ignore"):
        r = (tau2 - t) / np.maximum(tau2 - tau1, 1e-6)
        eta = -2 + b1 * (t - tau2)
        # Infinite products have a well-defined sigmoid limit, except 0*inf.
        eta = np.where(b1 == 0, -2, eta)
        logistic = z2 + z3 * expit(eta)
        middle = r * z1 + (1 - r) * (z2 + z3 * expit(-2.0))
        result = np.where(t < tau1, z1, np.where(t < tau2, middle, logistic))
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("response curve exceeds floating-point range")
    return _freeze(result)


def anovaddp_loglikelihood(
    parameters: ArrayLike,
    time: ArrayLike,
    observations: ArrayLike,
    variance: float,
    *,
    normalized: bool = False,
) -> FloatArray:
    """Independent Gaussian observation likelihood, summed over supplied times.

    Default omits the normalizing constant, matching native loglik. Parameter
    batches are supported; variance is a single strictly positive scalar.
    """
    if np.iscomplexobj(observations):
        raise ValueError("observations must be real")
    y = finite(observations, "observations")
    fitted = anovaddp_curve(parameters, time)
    v = scalar(variance, "variance")
    if y.ndim != 1 or y.size != fitted.shape[-1] or v <= 0 or not isinstance(normalized, bool):
        raise ValueError("require aligned observations, positive variance and boolean normalized")
    with np.errstate(over="ignore"):
        residual = (y - fitted) / np.sqrt(v)
        value = -0.5 * np.sum(residual * residual, axis=-1)
    if normalized:
        value -= 0.5 * y.size * (np.log(2 * np.pi) + np.log(v))
    if not np.all(np.isfinite(value)):
        raise ArithmeticError("Gaussian log likelihood exceeds floating-point range")
    return _freeze(np.asarray(value))


@dataclass(frozen=True)
class AnovaDDPAmplitudePosterior:
    mean: FloatArray
    covariance: FloatArray
    design: FloatArray


def anovaddp_amplitude_posterior(
    parameters: ArrayLike,
    time: ArrayLike,
    observations: ArrayLike,
    *,
    prior_mean: ArrayLike,
    prior_covariance: ArrayLike,
    variance: float,
) -> AnovaDDPAmplitudePosterior:
    """Exact Gaussian update of z1,z2,z3 conditional on tau1,tau2,b1.

    This is the first block in native simtheta, not a fitted ANOVA DDP model.
    The six-dimensional Gaussian prior may correlate amplitudes and timing.
    """
    if any(np.iscomplexobj(a) for a in (parameters, observations, prior_mean, prior_covariance)):
        raise ValueError("parameters, observations and Gaussian prior must be real")
    theta, y, m, c = (
        finite(a, n)
        for a, n in zip(
            (parameters, observations, prior_mean, prior_covariance),
            ("parameters", "observations", "prior_mean", "prior_covariance"),
        )
    )
    v = scalar(variance, "variance")
    if theta.shape != (6,) or m.shape != (6,) or c.shape != (6, 6) or v <= 0:
        raise ValueError("require six parameters, six prior means, 6x6 covariance and variance>0")
    if not np.allclose(c, c.T, rtol=1e-12, atol=0):
        raise ValueError("prior covariance must be symmetric")
    c = c / 2 + c.T / 2
    try:
        np.linalg.cholesky(c)
        cross = np.linalg.solve(c[3:, 3:], c[3:, :3])
        conditional_cov = c[:3, :3] - c[:3, 3:] @ cross
        conditional_mean = m[:3] + cross.T @ (theta[3:] - m[3:])
        np.linalg.cholesky(conditional_cov)
        precision = np.linalg.solve(conditional_cov, np.eye(3))
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "prior covariance must be positive definite and numerically resolvable"
        ) from exc
    unit = np.column_stack((np.eye(3), np.tile(theta[3:], (3, 1))))
    design = anovaddp_curve(unit, time).T
    if y.ndim != 1 or y.size != design.shape[0]:
        raise ValueError("observations and time must be aligned vectors")
    scale = max(1.0, v)
    prior_weight = v / scale
    posterior_precision = prior_weight * precision + (design.T @ design) / scale
    try:
        covariance = prior_weight * np.linalg.solve(posterior_precision, np.eye(3))
        mean = np.linalg.solve(
            posterior_precision,
            prior_weight * (precision @ conditional_mean) + (design.T @ y) / scale,
        )
    except np.linalg.LinAlgError as exc:
        raise ArithmeticError("Gaussian amplitude update is numerically singular") from exc
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise ArithmeticError("Gaussian amplitude posterior exceeds floating-point range")
    return AnovaDDPAmplitudePosterior(
        _freeze(mean), _freeze((covariance + covariance.T) / 2), _freeze(design)
    )
