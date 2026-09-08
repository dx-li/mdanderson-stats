"""Stukel's two-shape generalized logistic link and model prediction."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit

from ._validation import FloatArray, finite


def stukel_log_odds(
    eta: ArrayLike,
    alpha1: ArrayLike = 0,
    alpha2: ArrayLike = 0,
    *,
    legacy_prediction: bool = False,
) -> FloatArray:
    """Transform a linear predictor to log odds, with broadcast shape parameters.

    Each half-line has magnitude expm1(a*abs(eta))/a for a>0,
    log1p(-a*abs(eta))/(-a) for a<0, and abs(eta) for a=0.
    alpha1 applies above zero and alpha2 below. Unrepresentably large log odds
    return signed infinity, giving the correct limiting probability.
    legacy_prediction reproduces predict.glr's skipped alpha1<0 branch.
    """
    eta, a1, a2 = np.broadcast_arrays(
        finite(eta, "eta"), finite(alpha1, "alpha1"), finite(alpha2, "alpha2")
    )
    shape = np.where(eta >= 0, a1, a2)
    if legacy_prediction:
        shape = np.where((eta > 0) & (shape < 0), 0, shape)
    magnitude = np.abs(eta)
    out = np.array(magnitude, copy=True)
    active = (shape != 0) & (magnitude != 0)
    a, m = np.abs(shape[active]), magnitude[active]
    with np.errstate(over="ignore", under="ignore"):
        z = a * m
    positive = shape[active] > 0
    values = np.empty_like(z)
    small = z <= 1
    ratio = np.ones_like(z[small])
    numerator = np.where(positive[small], np.expm1(z[small]), np.log1p(z[small]))
    np.divide(numerator, z[small], out=ratio, where=z[small] != 0)
    with np.errstate(over="ignore"):
        values[small] = m[small] * ratio
        upper = ~small & positive
        values[upper] = np.exp(z[upper] + np.log1p(-np.exp(-z[upper])) - np.log(a[upper]))
    lower = ~small & ~positive
    # logaddexp avoids overflowing the product a*m in the logarithmic branch.
    values[lower] = np.logaddexp(0, np.log(a[lower]) + np.log(m[lower])) / a[lower]
    out[active] = values
    return np.asarray(np.copysign(out, eta))


def stukel_probability(
    eta: ArrayLike,
    alpha1: ArrayLike = 0,
    alpha2: ArrayLike = 0,
    *,
    legacy_prediction: bool = False,
) -> FloatArray:
    """Generalized inverse link; zero shapes recover ordinary logistic probabilities."""
    return np.asarray(
        expit(stukel_log_odds(eta, alpha1, alpha2, legacy_prediction=legacy_prediction))
    )


def predict_stukel(
    x: ArrayLike,
    beta: ArrayLike,
    alpha1: float = 0,
    alpha2: float = 0,
    *,
    intercept: bool = True,
    legacy_prediction: bool = False,
) -> FloatArray:
    """Predict from supplied regression coefficients; no model fitting is performed.

    x has observations by covariates in its final two axes, with optional batches.
    beta is one vector, with its first element the intercept when intercept=True.
    Both shape parameters must be scalars. Uses the corrected Fortran link unless
    the archived S prediction bug is explicitly requested.
    """
    x, beta = finite(x, "x"), finite(beta, "beta")
    if x.ndim < 2 or beta.ndim != 1:
        raise ValueError("x must be a matrix or batch of matrices; beta must be a vector")
    if not isinstance(intercept, bool):
        raise ValueError("intercept must be boolean")
    if beta.size != x.shape[-1] + int(intercept):
        raise ValueError("Coefficient count does not match covariates and intercept")
    if np.ndim(alpha1) or np.ndim(alpha2):
        raise ValueError("Prediction shape parameters must be scalars")
    eta = x @ beta[1:] + beta[0] if intercept else x @ beta
    return stukel_probability(eta, alpha1, alpha2, legacy_prediction=legacy_prediction)
