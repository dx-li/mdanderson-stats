"""Conjugate prior information and explicit BayesESS native conventions."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite
from .boin import _owned


def conjugate_prior_ess(
    model: str, parameters: ArrayLike, *, convention: str = "information"
) -> FloatArray:
    """Calculate prior ESS, vectorized over leading parameter-array dimensions.

    Models/last-axis parameters:
      beta_binomial: alpha, beta
      gamma_exponential / gamma_poisson: shape, rate
      dirichlet_multinomial: alpha_1, ..., alpha_K (K>=2)
      normal_normal: prior mean, known sampling variance, prior mean variance
      inverse_chi_squared_normal: degrees of freedom, scale variance (mean known)
      inverse_gamma_normal: shape, scale (mean known)

    Native convention differs only for gamma_exponential: it returns rate
    (prior exposure), whereas information returns shape (equivalent count).
    This is prior information, not effective MCMC sample size. Returns read-only
    arrays, including a zero-dimensional array for one parameter vector.
    """
    dimensions = {
        "beta_binomial": 2,
        "gamma_exponential": 2,
        "gamma_poisson": 2,
        "dirichlet_multinomial": None,
        "normal_normal": 3,
        "inverse_chi_squared_normal": 2,
        "inverse_gamma_normal": 2,
    }
    if model not in dimensions or convention not in {"information", "native"}:
        raise ValueError("unknown model or convention; use information or native")
    p = finite(parameters, "parameters")
    width = dimensions[model]
    if (
        p.ndim == 0
        or p.size > 2000000
        or p.size == 0
        or (p.shape[-1] < 2 if width is None else p.shape[-1] != width)
    ):
        raise ValueError("parameters require the model's final-axis width and 1..2 million values")
    positive = p[..., 1:] if model == "normal_normal" else p
    if np.any(positive <= 0):
        raise ValueError("all parameters except the normal prior mean must be positive")
    # Avoid evaluating unused ratios/products that might overflow.
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        if model in {"beta_binomial", "dirichlet_multinomial"}:
            result = p.sum(axis=-1)
        elif model == "normal_normal":
            result = p[..., 1] / p[..., 2]
        elif model == "inverse_gamma_normal":
            result = 2 * p[..., 0]
        elif model == "gamma_poisson" or (model == "gamma_exponential" and convention == "native"):
            result = p[..., 1]
        else:
            result = p[..., 0]
    if np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise ArithmeticError("ESS is outside positive float64 range")
    return _owned(result)
