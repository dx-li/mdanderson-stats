"""Exact right-censored parametric AFT models for common survival distributions."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtri

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar

_DISTRIBUTIONS = ("weibull", "lognormal", "loglogistic")


@dataclass(frozen=True)
class ParametricSurvivalFit:
    """Fitted AFT model; covariance order is coefficients then log(sigma)."""

    distribution: str
    sigma: float
    coefficients: FloatArray
    covariance: FloatArray
    information: FloatArray
    log_likelihood: float
    score_error: float
    iterations: int
    scaled_parameters: FloatArray
    scaled_covariance: FloatArray
    covariate_mean: FloatArray
    covariate_scale: FloatArray
    log_time_center: float
    log_time_scale: float


@dataclass(frozen=True)
class ParametricSurvivalPrediction:
    """Survival surface with pointwise delta-method intervals on log cumulative hazard."""

    profile: FloatArray
    times: FloatArray
    log_survival: FloatArray
    survival: FloatArray
    cumulative_hazard: FloatArray
    lower: FloatArray
    upper: FloatArray
    se_log_cumulative_hazard: FloatArray
    confidence: float


def _data(
    time: ArrayLike, event: ArrayLike, covariates: ArrayLike | None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if any(np.iscomplexobj(value) for value in (time, event, covariates) if value is not None):
        raise ValueError("data must be real")
    t, e = finite(time, "time"), finite(event, "event")
    if t.ndim != 1 or not 1 <= t.size <= 20_000 or e.shape != t.shape or np.any(t < 0):
        raise ValueError("time/event require aligned nonempty vectors, nonnegative times, <=20000 rows")
    if np.any((e != 0) & (e != 1)) or not np.any(e == 1):
        raise ValueError("event must be binary and include at least one exact failure")
    if np.any((e == 1) & (t == 0)):
        raise ValueError("exact failure times must be positive")
    x = np.empty((t.size, 0)) if covariates is None else finite(covariates, "covariates")
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] != t.size or x.shape[1] > 16:
        raise ValueError("covariates require one row per observation and at most 16 columns")
    return t, e, np.column_stack((np.ones(t.size), x))


def _terms(z: FloatArray, distribution: str) -> tuple[FloatArray, ...]:
    """Return log density/survival and their first two derivatives in z."""
    if distribution == "weibull":
        with np.errstate(over="ignore", invalid="ignore"):
            ez = np.exp(z)
            logpdf, logsf = z - ez, -ez
        dlogpdf, dlogsf = 1.0 - ez, -ez
        d2logpdf = d2logsf = -ez
    elif distribution == "lognormal":
        logpdf = -0.5 * z * z - 0.5 * log(2 * pi)
        logsf = log_ndtr(-z)
        dlogpdf = -z
        d2logpdf = np.full(z.shape, -1.0)
        with np.errstate(over="ignore", invalid="ignore"):
            ratio = np.exp(logpdf - logsf)
            dlogsf = -ratio
            d2logsf = -ratio * (ratio - z)
    else:
        p = np.exp(-np.logaddexp(0.0, -z))
        logpdf = -np.logaddexp(0.0, z) - np.logaddexp(0.0, -z)
        logsf = -np.logaddexp(0.0, z)
        dlogpdf = 1.0 - 2.0 * p
        dlogsf = -p
        d2logpdf = -2.0 * p * (1.0 - p)
        d2logsf = -p * (1.0 - p)
    return logpdf, dlogpdf, d2logpdf, logsf, dlogsf, d2logsf


def _likelihood(
    parameters: FloatArray,
    y: FloatArray,
    event: FloatArray,
    design: FloatArray,
    distribution: str,
) -> tuple[float, FloatArray, FloatArray]:
    beta, log_sigma = parameters[:-1], parameters[-1]
    sigma = float(np.exp(log_sigma))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ArithmeticError("parametric survival scale exceeds numerical range")
    z = (y - design @ beta) / sigma
    logpdf, dpdf, d2pdf, logsf, dsf, d2sf = _terms(z, distribution)
    selected_log = np.where(event == 1, logpdf - log_sigma, logsf)
    d1 = np.where(event == 1, dpdf, dsf)
    d2 = np.where(event == 1, d2pdf, d2sf)
    loss = -float(np.sum(selected_log))
    gradient = np.r_[design.T @ (d1 / sigma), np.dot(d1, z) + event.sum()]
    hessian = np.empty((parameters.size, parameters.size))
    hessian[:-1, :-1] = (design.T * (-d2 / sigma**2)) @ design
    cross = -design.T @ ((d2 * z + d1) / sigma)
    hessian[:-1, -1] = hessian[-1, :-1] = cross
    hessian[-1, -1] = -np.dot(d2, z * z) - np.dot(d1, z)
    if not np.isfinite(loss) or not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
        raise ArithmeticError("parametric survival likelihood exceeds numerical range")
    return loss, gradient, hessian


def fit_parametric_survival(
    time: ArrayLike,
    event: ArrayLike,
    covariates: ArrayLike | None = None,
    *,
    distribution: str = "weibull",
    initial: ArrayLike | None = None,
    tolerance: float = 1e-7,
    max_iterations: int = 500,
) -> ParametricSurvivalFit:
    """Fit a Weibull, lognormal, or loglogistic right-censored AFT model.

    The implicit intercept and all slopes model log time. The parameter
    covariance is for ``[intercept, slopes..., log(sigma)]``. For Weibull,
    shape is ``1 / sigma`` and scale is ``exp(linear predictor)``.
    """
    if distribution not in _DISTRIBUTIONS:
        raise ValueError(f"distribution must be one of {_DISTRIBUTIONS}")
    tol = scalar(tolerance, "tolerance")
    if not 1e-10 <= tol <= 1e-2:
        raise ValueError("tolerance must be in [1e-10, 1e-2]")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, (int, np.integer)):
        raise ValueError("max_iterations must be an integer in [1, 10000]")
    if not 1 <= max_iterations <= 10_000:
        raise ValueError("max_iterations must be an integer in [1, 10000]")
    t, event_values, raw_design = _data(time, event, covariates)
    observed = t > 0
    # A censoring time of zero contributes log S(0)=0 and has no score.
    y_all = np.zeros(t.size)
    y_all[observed] = np.log(t[observed])
    design = raw_design[observed]
    y, e = y_all[observed], event_values[observed]
    if design.shape[0] <= design.shape[1]:
        raise ValueError("more informative observations than fitted parameters are required")
    center = float(np.mean(y))
    time_scale = float(np.std(y))
    if not np.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("varying positive log times are required to identify a fit")
    yn = (y - center) / time_scale
    x = raw_design[:, 1:][observed]
    xm = np.mean(x, axis=0) if x.shape[1] else np.empty(0)
    xs = np.std(x, axis=0) if x.shape[1] else np.empty(0)
    if np.any(xs <= 0) or not np.isfinite(xs).all():
        raise ValueError("all covariates must vary among informative observations")
    xn = np.column_stack((np.ones(y.size), (x - xm) / xs))
    if np.linalg.matrix_rank(xn) != xn.shape[1]:
        raise ValueError("informative observations require a full-rank design")
    ols = np.linalg.lstsq(xn, yn, rcond=None)[0]
    resid_sd = float(np.std(yn - xn @ ols))
    if distribution == "weibull":
        z_mean, z_sd = -0.5772156649015329, pi / np.sqrt(6)
    elif distribution == "lognormal":
        z_mean, z_sd = 0.0, 1.0
    else:
        z_mean, z_sd = 0.0, pi / np.sqrt(3)
    start = ols.copy()
    start[0] -= max(resid_sd, 0.1) * z_mean / z_sd
    start = np.r_[start, np.log(max(resid_sd / z_sd, 0.1))]
    if initial is not None:
        if np.iscomplexobj(initial):
            raise ValueError("initial must be real")
        supplied = finite(initial, "initial")
        if supplied.shape != (raw_design.shape[1] + 1,):
            raise ValueError("initial must contain coefficients followed by log(sigma)")
        # Convert original coordinates to normalized coordinates.
        betas = supplied[:-1]
        scaled_beta = np.empty_like(betas)
        scaled_beta[1:] = betas[1:] * xs / time_scale if xs.size else np.empty(0)
        scaled_beta[0] = (betas[0] + (np.dot(betas[1:], xm) if xs.size else 0) - center) / time_scale
        start = np.r_[scaled_beta, supplied[-1] - np.log(time_scale)]

    def objective(value: FloatArray) -> tuple[float, FloatArray]:
        try:
            loss, gradient, _ = _likelihood(value, yn, e, xn, distribution)
            return loss / y.size, gradient / y.size
        except (ArithmeticError, FloatingPointError):
            # Give the line search a finite rejection point; no estimate is accepted here.
            return 1e100, np.zeros_like(value)

    result = minimize(
        objective,
        start,
        jac=True,
        method="BFGS",
        options={"gtol": tol / 100, "maxiter": int(max_iterations)},
    )
    scaled = np.asarray(result.x, dtype=np.float64)
    try:
        final_loss, final_gradient, information_scaled = _likelihood(
            scaled, yn, e, xn, distribution
        )
    except (ArithmeticError, FloatingPointError) as exc:
        raise ArithmeticError("parametric survival fit ended outside the finite likelihood region") from exc
    score_error = float(np.max(np.abs(final_gradient)) / y.size)
    if not np.isfinite(score_error) or score_error > tol or not np.isfinite(final_loss):
        raise ArithmeticError(f"parametric survival fit did not converge: {result.message}")
    try:
        np.linalg.cholesky(information_scaled)
        covariance_scaled = np.linalg.solve(information_scaled, np.eye(scaled.size))
    except np.linalg.LinAlgError as exc:
        raise ArithmeticError("fit has nonpositive or singular observed information") from exc

    # Map normalized AFT coefficients and log scale to original units.
    transform = np.zeros((scaled.size, scaled.size))
    transform[-1, -1] = 1.0
    if xs.size:
        transform[0, 0] = time_scale
        transform[0, 1:-1] = -time_scale * xm / xs
        transform[1:-1, 1:-1] = np.diag(time_scale / xs)
    else:
        transform[0, 0] = time_scale
    beta = transform[:-1, :-1] @ scaled[:-1]
    beta[0] += center
    original_sigma = float(np.exp(scaled[-1]) * time_scale)
    if not np.isfinite(original_sigma) or original_sigma <= 0 or not np.isfinite(beta).all():
        raise ArithmeticError("fitted sigma is not representable")
    covariance = transform @ covariance_scaled @ transform.T
    inverse_transform = np.linalg.solve(transform, np.eye(transform.shape[0]))
    information = inverse_transform.T @ information_scaled @ inverse_transform
    if not np.isfinite(covariance).all() or not np.isfinite(information).all():
        raise ArithmeticError("fitted covariance or information is not representable")
    log_likelihood = (
        -final_loss
        - float(np.sum(e * y))
        - float(e.sum()) * np.log(time_scale)
    )
    # Include the omitted zero-time censoring rows (their contribution is exactly zero).
    return ParametricSurvivalFit(
        distribution,
        original_sigma,
        _freeze(beta),
        _freeze(covariance),
        _freeze(information),
        log_likelihood,
        score_error,
        int(result.nit),
        _freeze(scaled),
        _freeze(covariance_scaled),
        _freeze(xm),
        _freeze(xs),
        center,
        time_scale,
    )


def predict_parametric_survival(
    fit: ParametricSurvivalFit,
    times: ArrayLike,
    profiles: ArrayLike | None = None,
    *,
    confidence: float = 0.95,
) -> ParametricSurvivalPrediction:
    """Predict profiles by times with delta-method bands on log cumulative hazard."""
    conf = scalar(confidence, "confidence")
    if not 0 < conf < 1:
        raise ValueError("confidence must be in (0, 1)")
    if np.iscomplexobj(times):
        raise ValueError("times must be real")
    t = finite(times, "times")
    if t.ndim != 1 or not 1 <= t.size <= 100_000 or np.any(t < 0):
        raise ValueError("times must be a nonempty vector of nonnegative values")
    n_cov = fit.coefficients.size - 1
    if profiles is None:
        profile_values = np.zeros((1, n_cov))
    else:
        if np.iscomplexobj(profiles):
            raise ValueError("profiles must be real")
        profile_values = finite(profiles, "profiles")
        if profile_values.ndim == 1:
            profile_values = profile_values[None, :]
        if profile_values.ndim != 2 or profile_values.shape[1] != n_cov:
            raise ValueError("profiles must have one column per fitted covariate")
    if (
        profile_values.shape[0] > 100_000
        or profile_values.size > 2_000_000
        or 8 * profile_values.shape[0] * t.size + profile_values.shape[0] * fit.coefficients.size
        > 2_000_000
    ):
        raise ValueError("prediction surface exceeds 2,000,000 cells")
    if fit.distribution not in _DISTRIBUTIONS:
        raise ValueError("fit has an unsupported distribution")
    zcrit = float(-ndtri((1 - conf) / 2))
    profiles_scaled = (
        (profile_values - fit.covariate_mean) / fit.covariate_scale
        if n_cov
        else np.empty((profile_values.shape[0], 0))
    )
    if not np.isfinite(profiles_scaled).all():
        raise ArithmeticError("normalized prediction profiles exceed numerical range")
    x = np.column_stack((np.ones(profile_values.shape[0]), profiles_scaled))
    positive = t > 0
    yn = np.zeros(t.shape)
    yn[positive] = (np.log(t[positive]) - fit.log_time_center) / fit.log_time_scale
    sigma = float(np.exp(fit.scaled_parameters[-1]))
    z = np.zeros((x.shape[0], t.size))
    z[:, positive] = (
        yn[None, positive] - x @ fit.scaled_parameters[:-1, None]
    ) / sigma
    if not np.isfinite(z[:, positive]).all():
        raise ArithmeticError("prediction linear predictor exceeds numerical range")
    if fit.distribution == "weibull":
        log_hazard = z.copy()
        dlogh_dz = np.ones(z.shape)
    elif fit.distribution == "lognormal":
        log_hazard = np.zeros(z.shape)
        dlogh_dz = np.zeros(z.shape)
        zp = z[:, positive]
        logpdf, _, _, logsf, _, _ = _terms(zp, fit.distribution)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            hazard_log = np.log(-logsf)
            tail = zp < -8
            hazard_log[tail] = log_ndtr(zp[tail])
            derivative = np.exp(logpdf - logsf - hazard_log)
            derivative[tail] = np.exp(logpdf[tail] - hazard_log[tail])
        log_hazard[:, positive] = hazard_log
        dlogh_dz[:, positive] = derivative
    else:
        log_hazard = np.zeros(z.shape)
        dlogh_dz = np.zeros(z.shape)
        zp = z[:, positive]
        # H=log(1+exp(z)); keep log(H) for the delta method.
        h = np.logaddexp(0.0, zp)
        tail = zp < -35
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            hazard_log = np.log(h)
            derivative = np.exp(-np.logaddexp(0.0, -zp) - hazard_log)
        hazard_log[tail] = zp[tail]
        derivative[tail] = 1.0
        log_hazard[:, positive] = hazard_log
        dlogh_dz[:, positive] = derivative
    log_hazard[:, ~positive] = -np.inf
    dlogh_dz[:, ~positive] = 0.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        hazard = np.exp(log_hazard)
        log_survival = -hazard
        survival = np.exp(log_survival)
    # Contract the full joint coefficient/log-scale covariance without allocating
    # a profiles x times x parameters gradient tensor.
    covariance = fit.scaled_covariance
    v_location = np.einsum("pi,ij,pj->p", x, covariance[:-1, :-1], x)
    cov_location_scale = x @ covariance[:-1, -1]
    v_scale = covariance[-1, -1]
    with np.errstate(over="ignore", invalid="ignore"):
        variance = dlogh_dz**2 * (
            v_location[:, None] / sigma**2
            + 2 * z * cov_location_scale[:, None] / sigma
            + z**2 * v_scale
        )
        variance_scale = dlogh_dz**2 * (
            v_location[:, None] / sigma**2
            + 2 * np.abs(z * cov_location_scale[:, None]) / sigma
            + z**2 * v_scale
        )
    if not np.isfinite(variance).all():
        raise ArithmeticError("prediction interval variance exceeds numerical range")
    if np.any(variance < -1e-12 * np.maximum(variance_scale, 1.0)):
        raise ArithmeticError("prediction interval variance is negative")
    variance = np.where(variance < 0, 0.0, variance)
    se = np.sqrt(variance)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        lower = np.exp(-np.exp(log_hazard + zcrit * se))
        upper = np.exp(-np.exp(log_hazard - zcrit * se))
    return ParametricSurvivalPrediction(
        _freeze(profile_values),
        _freeze(t),
        _freeze(log_survival),
        _freeze(survival),
        _freeze(hazard),
        _freeze(lower),
        _freeze(upper),
        _freeze(se),
        conf,
    )
