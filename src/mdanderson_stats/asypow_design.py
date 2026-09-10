"""General design-matrix logistic and multiplicative binomial ASYPOW models."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from ._validation import FloatArray, finite
from .asypow_regression import _log_information
from .boin import _owned


def asypow_design_information(
    coefficients: ArrayLike,
    design: ArrayLike,
    *,
    model: str = "logistic",
    observations: ArrayLike = 1,
) -> FloatArray:
    """Per-observation information for native mvlogistic or mvloglin models.

    logistic: P(event) = logistic(design @ coefficients).
    loglinear: P(event) = exp(design @ log(coefficients)), with positive
    coefficients and probabilities strictly below one at used design points.
    The latter is a multiplicative BINOMIAL model, not Poisson regression.
    Information is with respect to the supplied coefficients in both cases.
    """
    theta, x = finite(coefficients, "coefficients"), finite(design, "design")
    if theta.ndim != 1 or not 1 <= theta.size <= 500:
        raise ValueError("coefficients must be a vector of length 1..500")
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != theta.size or not 1 <= x.size <= 1_000_000:
        raise ValueError("design must have one column per coefficient and 1..1000000 entries")
    count = np.broadcast_to(finite(observations, "observations"), (len(x),))
    if np.any(count < 0) or not np.any(count > 0):
        raise ValueError("observations must be nonnegative with positive total allocation")
    if model not in ("logistic", "loglinear"):
        raise ValueError("model must be logistic or loglinear")
    if model == "loglinear" and np.any(theta <= 0):
        raise ValueError("loglinear coefficients must be strictly positive")
    used = count > 0
    x = x[used]
    log_allocation = np.log(count[used]) - logsumexp(np.log(count[used]))
    with np.errstate(over="ignore", invalid="ignore"):
        eta = x @ (theta if model == "logistic" else np.log(theta))
    if not np.all(np.isfinite(eta)):
        raise ArithmeticError("design predictor overflow; rescale the design")
    if model == "logistic":
        log_weight = log_allocation + _log_information(eta, "logistic", None)
        log_scale: float | FloatArray = 0.0
    else:
        if np.any(eta >= 0):
            raise ValueError("loglinear event probabilities must be strictly below one")
        log_weight = log_allocation + eta - np.log(-np.expm1(eta))
        log_scale = np.log(theta)
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        features = np.sign(x) * np.exp(np.log(np.abs(x)) - log_scale + log_weight[:, None] / 2)
        information = features.T @ features
    if not np.all(np.isfinite(information)):
        raise ArithmeticError("design information exceeds floating-point range")
    return _owned(information)


def asypow_reparameterize(information: ArrayLike, jacobian: ArrayLike) -> FloatArray:
    """Delta-method information for new parameters with Jacobian d(new)/d(old).

    The Jacobian may have fewer rows than columns, but must have full row rank.
    Return inverse(J @ inverse(information) @ J.T), using factorization/solves.
    """
    info, j = finite(information, "information"), finite(jacobian, "jacobian")
    if info.ndim != 2 or info.shape[0] != info.shape[1] or not 1 <= len(info) <= 500:
        raise ValueError("information must be square with dimension 1..500")
    if j.ndim == 1:
        j = j[None, :]
    if j.ndim != 2 or j.shape[1] != len(info) or not 1 <= len(j) <= len(info):
        raise ValueError("jacobian must have p columns and 1..p rows")
    if not np.allclose(info, info.T, rtol=1e-12, atol=0):
        raise ValueError("information must be symmetric")
    try:
        chol = np.linalg.cholesky(info / 2 + info.T / 2)
    except np.linalg.LinAlgError as error:
        raise ValueError("information must be positive definite") from error
    # Normalize rows before rank assessment so changes in parameter units do
    # not turn an independent transformation into a falsely singular one.
    scale = np.max(np.abs(j), axis=1)
    if np.any(scale == 0):
        raise ValueError("jacobian must have full row rank")
    transformed = np.linalg.solve(chol, (j / scale[:, None]).T).T
    u, singular, _ = np.linalg.svd(transformed, full_matrices=False)
    if singular[-1] <= np.finfo(float).eps * max(j.shape) * singular[0]:
        raise ValueError("jacobian is dependent or numerically unresolved")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        factors = (u / singular) / scale[:, None]
        result = factors @ factors.T
    if not np.all(np.isfinite(result)) or np.any(np.diag(result) <= 0):
        raise ArithmeticError("transformed information exceeds floating-point range")
    return _owned(result)
