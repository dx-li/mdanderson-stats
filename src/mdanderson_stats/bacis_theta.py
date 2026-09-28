"""Analytical latent-sign posterior for the BaCIS first-stage model.

The native model assigns each subgroup a latent ``theta ~ Normal(0, 1/tau)``
and selects the low or high response-rate component according to its sign.
After integrating the response-rate likelihood, the posterior for theta is a
two-sided half-normal mixture with weights supplied by ``bacis_classify``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import erf, ndtr

from ._validation import FloatArray, scalar
from .bacis import BaCISClassification, _integer, _readonly


@dataclass(frozen=True)
class BaCISThetaPosterior:
    """Latent-theta density and distribution on a caller-supplied grid.

    Grid-valued arrays have shape ``(len(theta), groups)``. ``mean`` and
    ``variance`` are analytical posterior moments. The density at zero uses
    the positive-side convention, but the CDF is continuous there.
    """

    theta: FloatArray
    density: FloatArray
    cdf: FloatArray
    survival: FloatArray
    mean: FloatArray
    variance: FloatArray
    high_probability: FloatArray
    latent_precision: float


def _classification_weights(
    classification: BaCISClassification,
) -> tuple[FloatArray, FloatArray]:
    if not isinstance(classification, BaCISClassification):
        raise TypeError("classification must be a BaCISClassification")
    low_shape = np.shape(classification.low_probability)
    high_shape = np.shape(classification.high_probability)
    if low_shape != high_shape or len(low_shape) != 1 or not 1 <= low_shape[0] <= 100:
        raise ValueError("classification probabilities must have matching shape (groups,), 1..100")
    low = np.asarray(classification.low_probability, dtype=float)
    high = np.asarray(classification.high_probability, dtype=float)
    if (
        not np.isfinite(low).all()
        or not np.isfinite(high).all()
        or np.any((low < 0) | (low > 1))
        or np.any((high < 0) | (high > 1))
        or not np.allclose(low + high, 1.0, rtol=0.0, atol=8 * np.finfo(float).eps)
    ):
        raise ValueError("classification probabilities must be finite complementary values")
    return low, high


def _precision(value: float) -> float:
    result = scalar(value, "latent_precision")
    if result <= 0:
        raise ValueError("latent_precision must be positive")
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        sd = 1.0 / np.sqrt(result)
        variance = 1.0 / result
    if not np.isfinite(sd) or sd <= 0 or not np.isfinite(variance) or variance <= 0:
        raise ValueError("latent_precision must yield representable normal scale and variance")
    return result


def bacis_theta_posterior(
    classification: BaCISClassification,
    theta: ArrayLike,
    *,
    latent_precision: float = 0.001,
) -> BaCISThetaPosterior:
    """Evaluate the latent-theta density, CDF, and survival on a grid.

    The prior is ``Normal(0, 1 / latent_precision)``. On each side of zero,
    the corresponding half-normal is weighted by twice its BaCIS component
    probability. Grid outputs are capped at 200,000 group-grid cells.
    """
    low, high = _classification_weights(classification)
    precision = _precision(latent_precision)
    if np.iscomplexobj(theta) or len(np.shape(theta)) > 1:
        raise ValueError("theta must be a real scalar or one-dimensional grid")
    raw = np.asarray(theta)
    if raw.size == 0 or raw.size * low.size > 200_000:
        raise ValueError("theta grid must produce 1..200,000 group-grid cells")
    if raw.dtype.kind not in "iuf" or np.isnan(raw).any():
        raise ValueError("theta values must be real numbers and must not be NaN")
    grid = np.asarray(raw, dtype=float).reshape(-1)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        z = grid * np.sqrt(precision)
    if np.isnan(z).any():
        raise ValueError("theta grid is not representable at latent_precision")

    negative = grid < 0
    density = np.empty((grid.size, low.size), dtype=float)
    cdf = np.empty_like(density)
    survival = np.empty_like(density)
    with np.errstate(over="ignore"):
        standard_density = (np.sqrt(precision) / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z**2)
    if np.any(negative):
        density[negative] = 2.0 * standard_density[negative, None] * low[None, :]
        cdf[negative] = 2.0 * ndtr(z[negative, None]) * low[None, :]
        survival[negative] = high[None, :] + low[None, :] * (-erf(z[negative, None] / np.sqrt(2.0)))
    if np.any(~negative):
        density[~negative] = 2.0 * standard_density[~negative, None] * high[None, :]
        middle_mass = erf(z[~negative, None] / np.sqrt(2.0))
        cdf[~negative] = low[None, :] + high[None, :] * middle_mass
        survival[~negative] = 2.0 * ndtr(-z[~negative, None]) * high[None, :]

    difference = high - low
    sd = 1.0 / np.sqrt(precision)
    mean = difference * (np.sqrt(2.0 / np.pi) * sd)
    variance = (1.0 - (2.0 / np.pi) * difference**2) / precision
    if (
        not np.isfinite(density).all()
        or not np.isfinite(mean).all()
        or not np.isfinite(variance).all()
    ):
        raise ArithmeticError("latent-theta posterior is not representable")
    return BaCISThetaPosterior(
        _readonly(grid),
        _readonly(density),
        _readonly(cdf),
        _readonly(survival),
        _readonly(mean),
        _readonly(variance),
        _readonly(high),
        precision,
    )


def sample_bacis_theta(
    classification: BaCISClassification,
    *,
    draws: int = 1000,
    latent_precision: float = 0.001,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Draw independent latent thetas from the analytical BaCIS posterior.

    The result has shape ``(draws, groups)`` and uses a NumPy Generator rather
    than attempting to reproduce JAGS's RNG stream.
    """
    low, high = _classification_weights(classification)
    precision = _precision(latent_precision)
    repetitions = _integer(draws, "draws", 1, 100_000)
    if repetitions * low.size > 500_000:
        raise ValueError("draws * groups must be at most 500,000")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None")
    generator = np.random.default_rng() if rng is None else rng
    positive = generator.random((repetitions, low.size)) < high[None, :]
    magnitude = np.abs(generator.standard_normal((repetitions, low.size))) / np.sqrt(precision)
    result = np.where(positive, magnitude, -magnitude)
    if not np.isfinite(result).all():
        raise ArithmeticError("latent-theta draws are not representable")
    return _readonly(result)
