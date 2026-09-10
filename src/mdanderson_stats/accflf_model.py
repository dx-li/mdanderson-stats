"""Right-censored accelerated log-F likelihood and fixed-shape fitting."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.special import digamma, polygamma

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar
from .accflf import AccflfShape, accflf_logf, accflf_shape


def _data(
    time: ArrayLike, event: ArrayLike, covariates: ArrayLike | None, weights: ArrayLike | None
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    if any(np.iscomplexobj(v) for v in (time, event, covariates, weights) if v is not None):
        raise ValueError("data must be real")
    t, e = finite(time, "time"), finite(event, "event")
    if t.ndim != 1 or not 1 <= t.size <= 20000 or e.shape != t.shape or np.any(t <= 0):
        raise ValueError(
            "time/event require aligned nonempty vectors, positive times, <=20000 rows"
        )
    if np.any((e != 0) & (e != 1)):
        raise ValueError("event must be 1 for a failure or 0 for right censoring")
    x = np.empty((t.size, 0)) if covariates is None else finite(covariates, "covariates")
    if x.ndim != 2 or x.shape[0] != t.size or x.shape[1] > 16:
        raise ValueError("covariates require one row per observation and <=16 columns")
    weight = np.ones(t.size) if weights is None else finite(weights, "weights")
    if weight.shape != t.shape or np.any(weight <= 0):
        raise ValueError("weights must be aligned positive frequency weights")
    return np.log(t), e, np.column_stack((np.ones(t.size), x)), weight


def _likelihood(
    parameters: FloatArray,
    y: FloatArray,
    event: FloatArray,
    design: FloatArray,
    weight: FloatArray,
    shape: AccflfShape,
) -> tuple[float, FloatArray, FloatArray]:
    sigma = np.exp(parameters[0])
    w = (y - design @ parameters[1:]) / sigma
    result = accflf_logf(w, shape.numerator_df, shape.denominator_df)
    value = np.where(event == 1, result.log_density - parameters[0], result.log_survival)
    d1 = np.where(event == 1, result.density_score, result.survival_score)
    d2 = np.where(event == 1, result.density_curvature, result.survival_curvature)
    loss = float(-np.dot(weight, value))
    gradient = np.r_[np.dot(weight, d1 * w + event), design.T @ (weight * d1) / sigma]
    hessian = np.empty((parameters.size, parameters.size))
    hessian[0, 0] = -np.dot(weight, d2 * w * w + d1 * w)
    hessian[0, 1:] = design.T @ (-weight * (d2 * w + d1)) / sigma
    hessian[1:, 0] = hessian[0, 1:]
    hessian[1:, 1:] = (design.T * (-weight * d2)) @ design / sigma**2
    if not np.isfinite(loss) or not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
        raise ArithmeticError("accelerated log-F likelihood exceeds numerical range")
    return loss, gradient, hessian


def accflf_loglikelihood(
    time: ArrayLike,
    event: ArrayLike,
    *,
    p: float,
    q: float,
    sigma: float,
    coefficients: ArrayLike,
    covariates: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    time_density: bool = False,
) -> float:
    """ACCFLF likelihood for log times; optionally include event time Jacobians.

    coefficients contains intercept then covariate coefficients. The source's
    printed likelihood omits -log(time) for events (time_density=False).
    """
    y, e, x, weight = _data(time, event, covariates, weights)
    if np.iscomplexobj(coefficients):
        raise ValueError("coefficients must be real")
    beta = finite(coefficients, "coefficients")
    scale = scalar(sigma, "sigma")
    if beta.shape != (x.shape[1],) or scale <= 0 or not isinstance(time_density, bool):
        raise ValueError("positive sigma, aligned coefficients and boolean time_density required")
    loss, _, _ = _likelihood(np.r_[np.log(scale), beta], y, e, x, weight, accflf_shape(p, q))
    return -loss - (float(np.dot(weight * e, y)) if time_density else 0)


@dataclass(frozen=True)
class AccflfFit:
    p: float
    q: float
    shape: AccflfShape
    sigma: float
    coefficients: FloatArray
    covariance: FloatArray
    log_likelihood: float
    time_log_likelihood: float
    score_error: float
    iterations: int
    sigma_fixed: bool


def fit_accflf(
    time: ArrayLike,
    event: ArrayLike,
    *,
    p: float,
    q: float,
    covariates: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    fixed_sigma: float | None = None,
) -> AccflfFit:
    """Fit an ACCFLF submodel at fixed p,q using analytic likelihood derivatives.

    Fit sigma unless fixed_sigma is supplied (use 1 for the exponential model).
    Covariance coordinates are [log(sigma), intercept, covariate coefficients];
    the fixed log-sigma row/column are zero. This does not estimate p or q.
    """
    y, e, x, weight = _data(time, event, covariates, weights)
    shape = accflf_shape(p, q)
    if not np.any(e == 1) or np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("at least one failure and a full-rank design are required")
    center, scale = float(np.mean(y)), float(np.std(y))
    if scale <= 0:
        raise ValueError("varying log times are required to identify a fit")
    yn = (y - center) / scale
    # Normalize covariates as well; map estimates and covariance back below.
    xm, xs = x[:, 1:].mean(axis=0), x[:, 1:].std(axis=0)
    xn = np.column_stack((np.ones(y.size), (x[:, 1:] - xm) / xs))
    a, b = shape.numerator_df / 2, shape.denominator_df / 2
    variance = float(polygamma(1, a) + polygamma(1, b))
    mean = float((digamma(a) - np.log(a)) - (digamma(b) - np.log(b)))
    beta = np.linalg.lstsq(xn, yn, rcond=None)[0]
    residual_sd = float(np.std(yn - xn @ beta))
    initial_sigma = max(residual_sd, 0.1) / np.sqrt(variance)
    if fixed_sigma is not None:
        fixed_sigma = scalar(fixed_sigma, "fixed_sigma")
        if fixed_sigma <= 0:
            raise ValueError("fixed_sigma must be positive")
        initial_sigma = fixed_sigma / scale
    beta[0] -= initial_sigma * mean
    initial = np.r_[np.log(initial_sigma), beta]
    free = np.arange(0 if fixed_sigma is None else 1, initial.size)

    def evaluate(v: FloatArray) -> tuple[float, FloatArray]:
        parameters = initial.copy()
        parameters[free] = v
        value, gradient, _ = _likelihood(parameters, yn, e, xn, weight, shape)
        return value / weight.sum(), gradient[free] / weight.sum()

    result = minimize(
        evaluate, initial[free], jac=True, method="BFGS", options={"gtol": 1e-8, "maxiter": 500}
    )
    error = float(np.max(np.abs(result.jac)))
    if not np.isfinite(error) or error > 1e-6:
        raise ArithmeticError(f"fixed-shape log-F fit did not converge: {result.message}")
    parameters = initial.copy()
    parameters[free] = result.x
    _, _, information = _likelihood(parameters, yn, e, xn, weight, shape)
    sub = information[np.ix_(free, free)]
    try:
        np.linalg.cholesky(sub)
        covariance = np.zeros(information.shape)
        covariance[np.ix_(free, free)] = np.linalg.solve(sub, np.eye(free.size))
    except np.linalg.LinAlgError as exc:
        raise ArithmeticError("fit has nonpositive/singular observed information") from exc
    transform = np.eye(parameters.size)
    transform[1, 1] = scale
    transform[1, 2:] = -scale * xm / xs
    transform[2:, 2:] = np.diag(scale / xs)
    beta = transform[1:, 1:] @ parameters[1:]
    beta[0] += center
    sigma = float(np.exp(parameters[0]) * scale)
    covariance = transform @ covariance @ transform.T
    likelihood = accflf_loglikelihood(
        time,
        event,
        p=p,
        q=q,
        sigma=sigma,
        coefficients=beta,
        covariates=covariates,
        weights=weights,
    )
    return AccflfFit(
        float(p),
        float(q),
        shape,
        sigma,
        _freeze(beta),
        _freeze(covariance),
        likelihood,
        likelihood - float(np.dot(weight * e, y)),
        error,
        int(result.nit),
        fixed_sigma is not None,
    )


def accflf_survival(
    time: ArrayLike,
    *,
    p: float,
    q: float,
    sigma: float,
    coefficients: ArrayLike,
    covariates: ArrayLike | None = None,
    log: bool = False,
) -> FloatArray:
    """Predict survival at positive times, one covariate row per time.

    Pass a fitted result's p, q, sigma and coefficients. log=True retains tiny
    survival probabilities in log space rather than underflowing to zero.
    """
    y, _, x, _ = _data(time, np.zeros(np.shape(time)), covariates, None)
    if np.iscomplexobj(coefficients):
        raise ValueError("coefficients must be real")
    beta = finite(coefficients, "coefficients")
    scale = scalar(sigma, "sigma")
    if beta.shape != (x.shape[1],) or scale <= 0 or not isinstance(log, bool):
        raise ValueError("positive sigma, aligned coefficients and boolean log required")
    shape = accflf_shape(p, q)
    result = accflf_logf((y - x @ beta) / scale, shape.numerator_df, shape.denominator_df)
    return result.log_survival if log else _freeze(np.exp(result.log_survival))
