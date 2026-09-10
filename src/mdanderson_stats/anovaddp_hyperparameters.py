"""Gaussian, inverse-Wishart and concentration updates for ANOVA DDP."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite, scalar
from .anovaddp_clusters import AnovaDDPAtomPosterior, _spd


@dataclass(frozen=True)
class AnovaDDPHyperparameters:
    base_mean: FloatArray
    base_covariance: FloatArray
    residual_covariance: FloatArray
    concentration: float
    base_mean_posterior: AnovaDDPAtomPosterior
    base_intercept_df: int
    base_intercept_scale: FloatArray
    residual_df: int
    residual_scale: FloatArray
    concentration_auxiliary: float
    concentration_mixture_probability: float
    concentration_gamma_shape: float
    concentration_gamma_rate: float


def _inverse_wishart(df: int, scale: FloatArray, rng: np.random.Generator) -> FloatArray:
    """Bartlett draw through solves; no sampled precision matrix is inverted."""
    p = scale.shape[0]
    lower = np.zeros((p, p))
    lower[np.diag_indices(p)] = np.sqrt(rng.chisquare(df - np.arange(p)))
    indices = np.tril_indices(p, -1)
    lower[indices] = rng.standard_normal(len(indices[0]))
    if np.any(np.diag(lower) <= 0):
        raise ArithmeticError("Bartlett diagonal underflow")
    factor = np.linalg.solve(lower, np.linalg.cholesky(scale).T)
    covariance = factor.T @ factor
    if not np.all(np.isfinite(covariance)):
        raise ArithmeticError("inverse-Wishart covariance exceeds floating-point range")
    np.linalg.cholesky(covariance)
    return covariance


def anovaddp_hyperparameter_update(
    parameters: ArrayLike,
    design: ArrayLike,
    labels: ArrayLike,
    atoms: ArrayLike,
    *,
    base_covariance: ArrayLike,
    base_prior: ArrayLike,
    covariance_prior: ArrayLike,
    covariance_df: int,
    concentration: float,
    concentration_shape: float,
    concentration_rate: float,
    seed: int | None = None,
) -> AnovaDDPHyperparameters:
    """Update base mean, base intercept covariance, random-effect covariance and M.

    Six curve parameters per subject, five modeled random effects per design
    column. Prior first-coordinate mean is fixed at 2. The base-mean hyperprior
    is N(base_prior,1000 I); base intercept covariance prior is IW(10,10 I_5).
    Remaining base-covariance blocks stay fixed and must be independent of the
    intercept block. All returned covariance draws are retained.
    """
    if any(np.iscomplexobj(a) for a in (parameters, design, labels, atoms, base_prior)):
        raise ValueError("parameters, design, labels, atoms and means must be real")
    theta, x, a, b = (
        finite(v, n)
        for v, n in zip(
            (parameters, design, atoms, base_prior),
            ("parameters", "design", "atoms", "base_prior"),
        )
    )
    lab = count(labels, "labels")
    if (
        theta.ndim != 2
        or theta.shape[1] != 6
        or not 1 <= theta.shape[0] <= 10000
        or x.ndim != 2
        or x.shape[0] != theta.shape[0]
        or not 1 <= x.shape[1] <= 20
        or np.any(x[:, 0] != 1)
    ):
        raise ValueError(
            "require (N,6) parameters, (N,q) design with intercept one; N<=10000,q<=20"
        )
    dimension = 5 * x.shape[1]
    if (
        a.ndim != 2
        or not a.shape[0]
        or a.shape[1] != dimension
        or b.shape != (dimension,)
        or lab.shape != (theta.shape[0],)
        or np.any(lab >= a.shape[0])
    ):
        raise ValueError("atoms, means and labels must match subjects and q*5 coefficients")
    ids = lab.astype(np.int64)
    if np.any(np.bincount(ids, minlength=a.shape[0]) == 0):
        raise ValueError("every supplied atom must be occupied")
    if (
        isinstance(covariance_df, bool)
        or not isinstance(covariance_df, int)
        or not 6 <= covariance_df <= 1000000
    ):
        raise ValueError("covariance_df must be an integer in [6,1000000]")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    mass, shape, rate = (
        scalar(v, n)
        for v, n in zip(
            (concentration, concentration_shape, concentration_rate),
            ("concentration", "concentration_shape", "concentration_rate"),
        )
    )
    if min(mass, shape, rate) <= 0:
        raise ValueError("concentration and its gamma prior parameters must be positive")
    c, _ = _spd(base_covariance, dimension, "base_covariance")
    s0, _ = _spd(covariance_prior, 6, "covariance_prior")
    if np.any(c[:5, 5:] != 0):
        raise ValueError("base intercept block must be independent of other coefficient blocks")
    rng = np.random.default_rng(seed)
    # Basesim: Gaussian atom observations with the base covariance as their
    # sampling covariance, and a fixed variance-1000 mean hyperprior.
    # All atoms have the same observation covariance: aggregate their sum
    # instead of repeating a matrix solve for every occupied cluster.
    system = a.shape[0] * np.eye(dimension) + 0.001 * c
    posterior_covariance = np.linalg.solve(system, c)
    posterior_mean = np.linalg.solve(system, a.sum(axis=0) + 0.001 * (c @ b))
    posterior = AnovaDDPAtomPosterior(
        _freeze(posterior_mean),
        _freeze(posterior_covariance / 2 + posterior_covariance.T / 2),
    )
    updated_mean = posterior.mean + np.linalg.cholesky(posterior.covariance) @ rng.standard_normal(
        dimension
    )
    centered = a[:, :5] - updated_mean[:5]
    base_scale = 10 * np.eye(5) + centered.T @ centered
    updated_covariance = c.copy()
    updated_covariance[:5, :5] = _inverse_wishart(10 + a.shape[0], base_scale, rng)
    subject_mean = np.empty_like(theta)
    subject_mean[:, 0] = 2
    subject_mean[:, 1:] = np.einsum("nq,nqd->nd", x, a[ids].reshape(theta.shape[0], x.shape[1], 5))
    residual = theta - subject_mean
    residual_scale = covariance_df * s0 + residual.T @ residual
    residual_covariance = _inverse_wishart(covariance_df + theta.shape[0], residual_scale, rng)
    # Escobar-West beta augmentation and two-gamma mixture from M_sample.
    auxiliary = float(rng.beta(mass + 1, theta.shape[0]))
    if not 0 < auxiliary <= 1:
        raise ArithmeticError("concentration beta auxiliary underflow")
    gamma_rate = rate - np.log(auxiliary)
    lower_shape = shape + (a.shape[0] - 1)
    log_odds = np.log(lower_shape) - np.log(theta.shape[0]) - np.log(gamma_rate)
    probability = float(np.exp(-np.logaddexp(0, -log_odds)))
    gamma_shape = lower_shape + int(rng.random() <= probability)
    updated_mass = float(rng.gamma(gamma_shape) / gamma_rate)
    if not np.isfinite(updated_mass) or updated_mass <= 0 or not np.all(np.isfinite(updated_mean)):
        raise ArithmeticError("hyperparameter draw exceeds floating-point range")
    return AnovaDDPHyperparameters(
        _freeze(updated_mean),
        _freeze(updated_covariance),
        _freeze(residual_covariance),
        updated_mass,
        posterior,
        10 + a.shape[0],
        _freeze(base_scale),
        covariance_df + theta.shape[0],
        _freeze(residual_scale),
        auxiliary,
        probability,
        gamma_shape,
        float(gamma_rate),
    )
