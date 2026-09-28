"""Exact Prentice and Stacy generalized-gamma survival AFT models."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult, minimize
from scipy.special import gammainc, gammaincc, gammaln, log_ndtr, ndtri

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar
from .cdflib_gamma_factor import _large_log_factor
from .cdflib_gamma_ratios import _delta
from .cdflib_incomplete_gamma import _lower_ratio, _upper_ratio
from .parametric_survival import _check_location_separation, _data

_LOG_2PI = log(2 * pi)
_LOG_SQRT_2PI = 0.5 * _LOG_2PI
_MAX_CELLS = 2_000_000


@dataclass(frozen=True)
class GeneralizedGammaFit:
    """Generalized-gamma fit with a full joint observed-information covariance.

    Prentice coordinates are ``[coefficients..., log(sigma), Q]``. Stacy
    coordinates are ``[log(scale coefficients)..., log(shape), log(k)]``.
    ``coefficients`` model mu for Prentice and log(scale) for Stacy.
    """

    parameterization: Literal["prentice", "stacy"]
    coefficients: FloatArray
    sigma: float | None
    q: float | None
    shape: float | None
    k: float | None
    parameter_names: tuple[str, ...]
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
class GeneralizedGammaPrediction:
    """Survival prediction surfaces and delta-method confidence limits."""

    profile: FloatArray
    times: FloatArray
    log_survival: FloatArray
    survival: FloatArray
    cumulative_hazard: FloatArray
    lower: FloatArray
    upper: FloatArray
    se_log_cumulative_hazard: FloatArray
    confidence: float


def _expm1_minus_x(x: FloatArray) -> FloatArray:
    """Evaluate expm1(x)-x without cancellation near zero."""
    result = np.empty(x.shape)
    small = np.abs(x) < 1e-3
    xs = x[small]
    # x^2/2! + x^3/3! + ...; Horner evaluation avoids subtracting x.
    value = np.full(xs.shape, 1 / 3_628_800)
    for coefficient in (1 / 362_880, 1 / 40_320, 1 / 5_040, 1 / 720, 1 / 120, 1 / 24, 1 / 6, 1 / 2):
        value = coefficient + xs * value
    result[small] = xs * xs * value
    result[~small] = np.expm1(x[~small]) - x[~small]
    return result


def _near_zero_mask(q: float, w: FloatArray) -> np.ndarray:
    if q == 0:
        return np.ones(w.shape, dtype=bool)
    # The omitted O(Q^3) Edgeworth term remains small over this verified domain.
    bounded = np.log(abs(q)) + 6 * np.log1p(np.abs(w)) <= log(1e-3)
    return (abs(q) <= 1e-3) & bounded


def _prentice_log_density(w: FloatArray, q: float) -> FloatArray:
    """Log density of standardized log time W (without the log-sigma term)."""
    if q == 0:
        with np.errstate(over="ignore"):
            return -0.5 * w * w - _LOG_SQRT_2PI
    near = _near_zero_mask(q, w)
    result = np.empty(w.shape)
    if np.any(near):
        wn = w[near]
        result[near] = -0.5 * wn * wn - _LOG_SQRT_2PI - q * wn**3 / 6 - q**2 * (wn**4 / 24 + 1 / 12)
    regular = ~near
    if np.any(regular):
        wr = w[regular]
        k = 1 / (q * q)
        if k >= 8:
            constant = -_LOG_SQRT_2PI - float(_delta(np.asarray(q * q)))
        else:
            constant = log(abs(q)) * (1 - 2 * k) - float(gammaln(k)) - k
        r = q * wr
        with np.errstate(over="ignore", invalid="ignore"):
            result[regular] = constant - k * _expm1_minus_x(r)
    return result


def _near_zero_log_tails(w: FloatArray, q: float) -> tuple[FloatArray, FloatArray]:
    """Two-term Q expansion for log CDF and log survival near Prentice Q=0."""
    if q == 0:
        return log_ndtr(w), log_ndtr(-w)
    logcdf0, logsf0 = log_ndtr(w), log_ndtr(-w)
    logphi = -0.5 * w * w - _LOG_SQRT_2PI
    poly1 = w * w + 2
    poly2 = w**5 + 2 * w**3 + 6 * w
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        rel1_cdf = np.exp(logphi + np.log(poly1) - log(6) - logcdf0)
        rel1_sf = np.exp(logphi + np.log(poly1) - log(6) - logsf0)
        sign2 = np.sign(poly2)
        log_abs_poly2 = np.log(np.abs(poly2), where=poly2 != 0, out=np.full(w.shape, -np.inf))
        log_rel2 = logphi + log_abs_poly2 - log(72)
        rel2 = sign2 * np.exp(log_rel2 - logcdf0)
        rel2_sf = sign2 * np.exp(log_rel2 - logsf0)
        cdf = logcdf0 + np.log1p(q * rel1_cdf - q * q * rel2)
        sf = logsf0 + np.log1p(-q * rel1_sf + q * q * rel2_sf)
    return cdf, sf


def _log_regularized_gamma(a: float, log_x: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Log P(a,x), log Q(a,x), preserving the directly evaluated small tail."""
    log_p = np.empty(log_x.shape)
    log_q = np.empty(log_x.shape)
    log_max, log_tiny = log(np.finfo(float).max), log(np.nextafter(0.0, 1.0))
    huge = log_x >= log_max
    tiny = (log_x <= log_tiny) & ~huge
    regular = ~(huge | tiny)
    log_p[huge], log_q[huge] = 0.0, -np.inf
    if np.any(tiny):
        lx = log_x[tiny]
        lp = a * lx - gammaln(a + 1)
        log_p[tiny] = lp
        log_q[tiny] = _log1mexp(lp)
    if np.any(regular):
        lx = log_x[regular]
        x = np.exp(lx)
        p, q = gammainc(a, x), gammaincc(a, x)
        lp, lq = np.full(x.shape, -np.inf), np.full(x.shape, -np.inf)
        ppositive, qpositive = p > 0, q > 0
        lp[ppositive], lq[qpositive] = np.log(p[ppositive]), np.log(q[qpositive])
        lower_recover = ~ppositive & (x < a)
        if np.any(lower_recover):
            xx = x[lower_recover]
            aa = np.full(xx.shape, a)
            if a >= 8:
                log_r = _large_log_factor(aa, xx)
                ratio = _lower_ratio(aa, xx)
                lp[lower_recover] = log_r + np.log(ratio)
            else:
                # P(a,x)=x^a exp(-x)/Gamma(a+1) times a positive series.
                term = np.ones(xx.shape)
                series = np.ones(xx.shape)
                for n in range(1, 1000):
                    term *= xx / (a + n)
                    series += term
                    if np.all(term <= 2e-16 * series):
                        break
                else:
                    raise ArithmeticError("lower gamma tail series did not converge")
                lp[lower_recover] = a * lx[lower_recover] - xx - gammaln(a + 1) + np.log(series)
        upper_recover = ~qpositive & (x > a)
        if np.any(upper_recover):
            xx = x[upper_recover]
            aa = np.full(xx.shape, a)
            log_r = aa * lx[upper_recover] - xx - gammaln(a)
            if a >= 8:
                log_r = _large_log_factor(aa, xx)
            ratio = _upper_ratio(aa, xx, np.full(xx.shape, 1e-14))
            lq[upper_recover] = log_r + np.log(ratio)
        # Reconstruct the larger tail from the directly computed smaller one.
        lower_small = lp <= lq
        lq[lower_small] = _log1mexp(lp[lower_small])
        lp[~lower_small] = _log1mexp(lq[~lower_small])
        if np.any(~np.isfinite(lp) & (lp != -np.inf)) or np.any(~np.isfinite(lq) & (lq != -np.inf)):
            raise ArithmeticError("gamma log-tail evaluation failed")
        log_p[regular], log_q[regular] = lp, lq
    return log_p, log_q


def _log1mexp(value: FloatArray) -> FloatArray:
    """Stable log(1-exp(value)) for value<=0."""
    result = np.empty(value.shape)
    split = value < -log(2)
    result[split] = np.log1p(-np.exp(value[split]))
    result[~split] = np.log(-np.expm1(value[~split]))
    return result


def _prentice_logtails(w: FloatArray, q: float) -> tuple[FloatArray, FloatArray]:
    """Log CDF and log survival for standardized Prentice generalized gamma."""
    if q == 0:
        return log_ndtr(w), log_ndtr(-w)
    near = _near_zero_mask(q, w)
    logcdf, logsf = np.empty(w.shape), np.empty(w.shape)
    if np.any(near):
        logcdf[near], logsf[near] = _near_zero_log_tails(w[near], q)
    regular = ~near
    if np.any(regular):
        k = 1 / (q * q)
        wr = w[regular]
        log_x = -2 * log(abs(q)) + q * wr
        logp, logq = _log_regularized_gamma(k, log_x)
        if q > 0:
            logcdf[regular], logsf[regular] = logp, logq
        else:
            logcdf[regular], logsf[regular] = logq, logp
    return logcdf, logsf


def _stacy_log_density_y(v: FloatArray, log_shape: float, log_k: float) -> FloatArray:
    # Algebraically map Stacy to positive-Q Prentice, avoiding b*k*v-lgamma(k)
    # cancellation as k grows. w=sqrt(k)*(b*v-log(k)) retains the centered term.
    b = np.exp(log_shape)
    sqrt_k = np.exp(0.5 * log_k)
    q = np.exp(-0.5 * log_k)
    with np.errstate(over="ignore", invalid="ignore"):
        w = sqrt_k * (b * v - log_k)
        return _prentice_log_density(w, float(q)) + log_shape + 0.5 * log_k


def _stacy_log_survival(v: FloatArray, log_shape: float, log_k: float) -> FloatArray:
    b, k = np.exp(log_shape), np.exp(log_k)
    log_x = b * v
    _, logq = _log_regularized_gamma(float(k), log_x)
    return logq


def _prepare(
    time: ArrayLike, event: ArrayLike, covariates: ArrayLike | None
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, float, float]:
    t, e, raw_design = _data(time, event, covariates)
    observed = t > 0
    y = np.log(t[observed])
    event_values, design = e[observed], raw_design[observed]
    if design.shape[0] <= design.shape[1]:
        raise ValueError("more informative observations than fitted parameters are required")
    center = float(np.mean(y))
    time_scale = float(np.std(y))
    if not np.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("varying positive log times are required to identify a fit")
    yn = (y - center) / time_scale
    x = design[:, 1:]
    xm = np.mean(x, axis=0) if x.shape[1] else np.empty(0)
    xs = np.std(x, axis=0) if x.shape[1] else np.empty(0)
    if np.any(xs <= 0) or not np.isfinite(xs).all():
        raise ValueError("all covariates must vary among informative observations")
    xn = np.column_stack((np.ones(y.size), (x - xm) / xs))
    if np.linalg.matrix_rank(xn) != xn.shape[1]:
        raise ValueError("informative observations require a full-rank design")
    _check_location_separation(xn, event_values)
    return t[observed], event_values, yn, xn, xm, xs, center, time_scale


def _prentice_loss(
    theta: FloatArray, y: FloatArray, event: FloatArray, design: FloatArray
) -> float:
    p = design.shape[1]
    beta, log_sigma, q = theta[:p], float(theta[p]), float(theta[p + 1])
    sigma = float(np.exp(log_sigma))
    if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(q):
        raise ArithmeticError("Prentice scale or Q is outside the finite parameter range")
    w = (y - design @ beta) / sigma
    log_density_w = _prentice_log_density(w, q)
    _, log_sf = _prentice_logtails(w, q)
    selected = np.where(event == 1, log_density_w - log_sigma, log_sf)
    if not np.isfinite(selected).all():
        raise ArithmeticError("Prentice likelihood is outside the finite numerical range")
    return -float(np.sum(selected))


def _prentice_gradient(
    theta: FloatArray, y: FloatArray, event: FloatArray, design: FloatArray
) -> FloatArray:
    p = design.shape[1]
    beta, log_sigma, q = theta[:p], float(theta[p]), float(theta[p + 1])
    sigma = float(np.exp(log_sigma))
    w = (y - design @ beta) / sigma
    log_density_w = _prentice_log_density(w, q)
    _, log_sf = _prentice_logtails(w, q)
    near = _near_zero_mask(q, w)
    d_density = np.empty(w.shape)
    if q == 0:
        d_density = -w
    else:
        if np.any(near):
            wn = w[near]
            d_density[near] = -wn - q * wn**2 / 2 - q**2 * wn**3 / 6
        regular = ~near
        if np.any(regular):
            d_density[regular] = -np.expm1(q * w[regular]) / q
    d_survival = -np.exp(log_density_w - log_sf)
    d_w = np.where(event == 1, d_density, d_survival)
    gradient = np.empty(theta.shape)
    gradient[:p] = design.T @ (d_w / sigma)
    gradient[p] = np.dot(d_w, w) + event.sum()

    if q == 0 or np.all(near):
        logphi = -0.5 * w * w - _LOG_SQRT_2PI
        poly1 = w * w + 2
        poly2 = w**5 + 2 * w**3 + 6 * w
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            a = np.exp(logphi + np.log(poly1) - log(6) - log_sf)
            b = np.sign(poly2) * np.exp(
                logphi
                + np.log(np.abs(poly2), where=poly2 != 0, out=np.full(w.shape, -np.inf))
                - log(72)
                - log_sf
            )
        d_log_density_q = -(w**3) / 6 - q * (w**4 / 12 + 1 / 6)
        d_log_survival_q = -a + 2 * q * b
        gradient[p + 1] = -float(np.sum(np.where(event == 1, d_log_density_q, d_log_survival_q)))
    else:
        step = min(1e-4, max(1e-6, abs(q) * 1e-3))
        values = []
        for offset in (-2, -1, 1, 2):
            shifted = theta.copy()
            shifted[p + 1] += offset * step
            values.append(_prentice_loss(shifted, y, event, design))
        gradient[p + 1] = (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step)
    if not np.isfinite(gradient).all():
        raise ArithmeticError("Prentice score exceeds numerical range")
    return gradient


def _stacy_loss(theta: FloatArray, y: FloatArray, event: FloatArray, design: FloatArray) -> float:
    p = design.shape[1]
    beta = theta[:p]
    log_shape, log_k = float(theta[p]), float(theta[p + 1])
    b, k = float(np.exp(log_shape)), float(np.exp(log_k))
    if not np.isfinite(b) or not np.isfinite(k) or b <= 0 or k <= 0:
        raise ArithmeticError("Stacy shape parameters exceed numerical range")
    v = y - design @ beta
    q = float(np.exp(-0.5 * log_k))
    sigma = float(np.exp(-log_shape - 0.5 * log_k))
    if not np.isfinite(q) or not np.isfinite(sigma) or sigma <= 0:
        raise ArithmeticError("Stacy mapped Prentice parameters exceed numerical range")
    w = (v - log_k / b) / sigma
    log_density_w = _prentice_log_density(w, q)
    _, log_sf = _prentice_logtails(w, q)
    # Stacy's exact density equals the mapped Prentice density; this form avoids
    # cancellation in b*k*v - lgamma(k) for large k.
    log_density_y = log_density_w - np.log(sigma)
    selected = np.where(event == 1, log_density_y, log_sf)
    if not np.isfinite(selected).all():
        raise ArithmeticError("Stacy likelihood is outside the finite numerical range")
    return -float(np.sum(selected))


def _five_point_gradient(
    function: object, parameters: FloatArray, step_scale: float = 1e-4
) -> FloatArray:
    evaluate = function
    result = np.empty(parameters.shape)
    for j in range(parameters.size):
        step = step_scale * max(1.0, abs(float(parameters[j])))
        values = []
        for offset in (-2, -1, 1, 2):
            shifted = parameters.copy()
            shifted[j] += offset * step
            values.append(evaluate(shifted))  # type: ignore[operator]
        result[j] = (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step)
    return result


def _observed_information(gradient: object, parameters: FloatArray, count: int) -> FloatArray:
    dimension = parameters.size
    information = np.empty((dimension, dimension))
    for j in range(dimension):
        step = 1e-4 * max(1.0, abs(float(parameters[j])))
        plus, minus = parameters.copy(), parameters.copy()
        plus[j] += step
        minus[j] -= step
        gp = gradient(plus)  # type: ignore[operator]
        gm = gradient(minus)  # type: ignore[operator]
        information[:, j] = count * (gp - gm) / (2 * step)
    return (information + information.T) / 2


def fit_generalized_gamma(
    time: ArrayLike,
    event: ArrayLike,
    covariates: ArrayLike | None = None,
    *,
    parameterization: Literal["prentice", "stacy"] = "prentice",
    initial: ArrayLike | None = None,
    tolerance: float = 1e-7,
    max_iterations: int = 1000,
) -> GeneralizedGammaFit:
    """Fit exact right-censored generalized-gamma AFT models.

    Prentice permits positive and negative Q and includes the exact Q=0
    log-normal member. Stacy is the original positive shape/scale/k model;
    its covariance is reported in log-shape/log-k coordinates.
    """
    if parameterization not in ("prentice", "stacy"):
        raise ValueError("parameterization must be 'prentice' or 'stacy'")
    if np.iscomplexobj(tolerance):
        raise ValueError("tolerance must be real")
    tol = scalar(tolerance, "tolerance")
    if not 1e-10 <= tol <= 1e-2:
        raise ValueError("tolerance must be in [1e-10, 1e-2]")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or not 1 <= max_iterations <= 10_000
    ):
        raise ValueError("max_iterations must be an integer in [1, 10000]")
    observed_t, e, y, design, xm, xs, center, time_scale = _prepare(time, event, covariates)
    p, n = design.shape[1], y.size
    beta_ols = np.linalg.lstsq(design, y, rcond=None)[0]
    residual_scale = max(float(np.std(y - design @ beta_ols)), 0.1)

    if parameterization == "prentice":
        initial_theta = np.r_[beta_ols, np.log(residual_scale), 0.0]
        seeds = [0.0, -1.0, -0.5, 0.5, 1.0]
        if initial is not None:
            if np.iscomplexobj(initial):
                raise ValueError("initial must be real")
            supplied = finite(initial, "initial")
            if supplied.shape != (p + 2,):
                raise ValueError("initial must be [coefficients, log(sigma), Q]")
            transformed = supplied.copy()
            transformed[1:p] = supplied[1:p] * xs / time_scale if xs.size else np.empty(0)
            transformed[0] = (
                supplied[0] + (np.dot(supplied[1:p], xm) if xs.size else 0) - center
            ) / time_scale
            transformed[p] -= np.log(time_scale)
            starts = [transformed]
        else:
            starts = [initial_theta.copy() for _ in seeds]
            for start, q0 in zip(starts, seeds, strict=True):
                start[p + 1] = q0

        def loss(theta: FloatArray) -> float:
            return _prentice_loss(theta, y, e, design) / n

        def gradient(theta: FloatArray) -> FloatArray:
            return _prentice_gradient(theta, y, e, design) / n

        iterations_limit = int(max_iterations)
    else:
        initial_theta = np.r_[beta_ols, 0.0, 0.0]
        if initial is not None:
            if np.iscomplexobj(initial):
                raise ValueError("initial must be real")
            supplied = finite(initial, "initial")
            if supplied.shape != (p + 2,):
                raise ValueError("initial must be [log-scale coefficients, log(shape), log(k)]")
            transformed = supplied.copy()
            transformed[1:p] = supplied[1:p] * xs / time_scale if xs.size else np.empty(0)
            transformed[0] = (
                supplied[0] + (np.dot(supplied[1:p], xm) if xs.size else 0) - center
            ) / time_scale
            transformed[p] += np.log(time_scale)
            starts = [transformed]
        else:
            starts = [initial_theta.copy() for _ in range(3)]
            starts[1][p], starts[1][p + 1] = -0.7, 1.0
            starts[2][p], starts[2][p + 1] = 0.7, 2.0

        def loss(theta: FloatArray) -> float:
            return _stacy_loss(theta, y, e, design) / n

        def gradient(theta: FloatArray) -> FloatArray:
            return _five_point_gradient(loss, theta)

        iterations_limit = int(max_iterations)

    accepted: list[tuple[float, OptimizeResult, FloatArray, FloatArray]] = []
    failures: list[str] = []
    for start in starts:
        try:
            result = minimize(
                loss,
                start,
                jac=gradient,
                method="BFGS",
                options={"gtol": tol / 10, "maxiter": iterations_limit},
            )
            theta = np.asarray(result.x, dtype=np.float64)
            final_loss = loss(theta)
            final_gradient = gradient(theta)
            score_error = float(np.max(np.abs(final_gradient)))
            if not np.isfinite(final_loss) or not np.isfinite(score_error) or score_error > tol:
                failures.append(str(result.message))
                continue
            info_scaled = _observed_information(gradient, theta, n)
            np.linalg.cholesky(info_scaled)
            accepted.append((final_loss, result, theta, info_scaled))
        except (ArithmeticError, FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            failures.append(str(exc))
    if not accepted:
        reason = failures[-1] if failures else "no finite candidate"
        raise ArithmeticError(f"generalized-gamma fit has no finite identified optimum: {reason}")
    _, result, theta, info_scaled = min(accepted, key=lambda item: item[0])
    covariance_scaled = np.linalg.solve(info_scaled, np.eye(info_scaled.shape[0]))

    # A maps normalized covariate/log-time coefficients to original input units.
    transform = np.zeros((p + 2, p + 2))
    if xs.size:
        transform[0, 0] = time_scale
        transform[0, 1:p] = -time_scale * xm / xs
        transform[1:p, 1:p] = np.diag(time_scale / xs)
    else:
        transform[0, 0] = time_scale
    beta = transform[:p, :p] @ theta[:p]
    beta[0] += center
    sigma: float | None
    q: float | None
    shape: float | None
    kval: float | None
    if parameterization == "prentice":
        transform[p, p] = 1
        transform[p + 1, p + 1] = 1
        sigma = float(np.exp(theta[p]) * time_scale)
        q = float(theta[p + 1])
        shape = kval = None
        names = ("intercept", *(f"x{j + 1}" for j in range(p - 1)), "log_sigma", "Q")
    else:
        transform[p, p] = 1
        transform[p + 1, p + 1] = 1
        # The log-shape coordinate shifts by -log(time_scale), so its
        # Jacobian remains one; shape itself is converted below.
        sigma = q = None
        shape = float(np.exp(theta[p]) / time_scale)
        kval = float(np.exp(theta[p + 1]))
        names = (
            "log_scale_intercept",
            *(f"log_scale_x{j + 1}" for j in range(p - 1)),
            "log_shape",
            "log_k",
        )
    covariance = transform @ covariance_scaled @ transform.T
    inverse_transform = np.linalg.solve(transform, np.eye(transform.shape[0]))
    information = inverse_transform.T @ info_scaled @ inverse_transform
    if (
        not np.isfinite(beta).all()
        or not np.isfinite(covariance).all()
        or not np.isfinite(information).all()
    ):
        raise ArithmeticError("generalized-gamma fit or covariance is not representable")
    log_likelihood = (
        -float(
            _prentice_loss(theta, y, e, design)
            if parameterization == "prentice"
            else _stacy_loss(theta, y, e, design)
        )
        - float(e.sum()) * np.log(time_scale)
        - float(np.dot(e, np.log(observed_t)))
    )
    score_error = float(np.max(np.abs(gradient(theta))))
    return GeneralizedGammaFit(
        parameterization,
        _freeze(beta),
        sigma,
        q,
        shape,
        kval,
        tuple(names),
        _freeze(covariance),
        _freeze(information),
        log_likelihood,
        score_error,
        int(result.nit),
        _freeze(theta),
        _freeze(covariance_scaled),
        _freeze(xm),
        _freeze(xs),
        center,
        time_scale,
    )


def _log_hazard_from_parameters(
    parameters: FloatArray,
    parameterization: Literal["prentice", "stacy"],
    normalized_time: FloatArray,
    design: FloatArray,
) -> FloatArray:
    p = design.shape[1]
    residual = normalized_time[None, :] - (design @ parameters[:p])[:, None]
    if parameterization == "prentice":
        sigma = float(np.exp(parameters[p]))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ArithmeticError("prediction scale exceeds numerical range")
        w = residual / sigma
        log_cdf, log_sf = _prentice_logtails(w, float(parameters[p + 1]))
    else:
        log_shape, log_k = float(parameters[p]), float(parameters[p + 1])
        b, sqrt_k, q = np.exp(log_shape), np.exp(0.5 * log_k), np.exp(-0.5 * log_k)
        if not np.isfinite((b, sqrt_k, q)).all():
            raise ArithmeticError("prediction shape exceeds numerical range")
        w = sqrt_k * (b * residual - log_k)
        log_cdf, log_sf = _prentice_logtails(w, float(q))
    log_hazard = np.empty(log_sf.shape)
    # For a tiny CDF, -log(S)=F to relative accuracy O(F). Else retain the
    # exact cumulative hazard from the log survival probability.
    small_hazard = log_cdf < -35
    log_hazard[small_hazard] = log_cdf[small_hazard]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_hazard[~small_hazard] = np.log(-log_sf[~small_hazard])
    if not np.isfinite(w).all() or np.any(np.isnan(log_hazard)):
        raise ArithmeticError("prediction values exceed numerical range")
    return log_hazard


def predict_generalized_gamma(
    fit: GeneralizedGammaFit,
    times: ArrayLike,
    profiles: ArrayLike | None = None,
    *,
    confidence: float = 0.95,
) -> GeneralizedGammaPrediction:
    """Predict survival and pointwise delta-method intervals by profile and time.

    Confidence limits use the full joint covariance and are formed on log
    cumulative hazard. They are deterministic delta-method intervals.
    """
    if np.iscomplexobj(confidence):
        raise ValueError("confidence must be real")
    conf = scalar(confidence, "confidence")
    if not 0 < conf < 1:
        raise ValueError("confidence must be in (0, 1)")
    if np.iscomplexobj(times):
        raise ValueError("times must be real")
    t = finite(times, "times")
    if t.ndim != 1 or not 1 <= t.size <= 100_000 or np.any(t < 0):
        raise ValueError("times must be a nonempty vector of nonnegative values")
    n_covariates = fit.coefficients.size - 1
    if profiles is None:
        profile_values = np.zeros((1, n_covariates))
    else:
        if np.iscomplexobj(profiles):
            raise ValueError("profiles must be real")
        profile_values = finite(profiles, "profiles")
        if profile_values.ndim == 1:
            profile_values = profile_values[None, :]
        if profile_values.ndim != 2 or profile_values.shape[1] != n_covariates:
            raise ValueError("profiles must have one column per fitted covariate")
    if (
        not 1 <= profile_values.shape[0] <= 100_000
        or profile_values.size > _MAX_CELLS
        or 8 * profile_values.shape[0] * t.size + profile_values.shape[0] * fit.coefficients.size
        > _MAX_CELLS
    ):
        raise ValueError("prediction surface exceeds 2,000,000 cells")
    if fit.parameterization not in ("prentice", "stacy"):
        raise ValueError("fit has an unsupported generalized-gamma parameterization")
    standardized_profiles = (
        (profile_values - fit.covariate_mean) / fit.covariate_scale
        if n_covariates
        else np.empty((profile_values.shape[0], 0))
    )
    if not np.isfinite(standardized_profiles).all():
        raise ArithmeticError("normalized prediction profiles exceed numerical range")
    design = np.column_stack((np.ones(profile_values.shape[0]), standardized_profiles))
    positive = t > 0
    normalized_time = np.zeros(t.shape)
    normalized_time[positive] = (np.log(t[positive]) - fit.log_time_center) / fit.log_time_scale
    if not np.isfinite(normalized_time[positive]).all():
        raise ArithmeticError("normalized prediction times exceed numerical range")
    parameters = np.asarray(fit.scaled_parameters)
    log_hazard = np.zeros((design.shape[0], t.size))
    if np.any(positive):
        log_hazard[:, positive] = _log_hazard_from_parameters(
            parameters, fit.parameterization, normalized_time[positive], design
        )
    variance = np.zeros(log_hazard.shape)
    # Process each profile separately to bound temporary gradients to
    # times-by-parameters instead of a profiles-by-times-by-parameters array.
    for row in range(design.shape[0]):
        if not np.any(positive):
            continue
        row_design = design[row : row + 1]
        gradient = np.empty((int(np.sum(positive)), parameters.size))
        for j in range(parameters.size):
            step = 1e-4 * max(1.0, abs(float(parameters[j])))
            values = []
            for offset in (-2, -1, 1, 2):
                shifted = parameters.copy()
                shifted[j] += offset * step
                values.append(
                    _log_hazard_from_parameters(
                        shifted, fit.parameterization, normalized_time[positive], row_design
                    )[0]
                )
            gradient[:, j] = (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step)
        variance[row, positive] = np.einsum(
            "ti,ij,tj->t", gradient, fit.scaled_covariance, gradient
        )
    if not np.isfinite(variance).all():
        raise ArithmeticError("prediction interval variance exceeds numerical range")
    if np.any(variance < -1e-12):
        raise ArithmeticError("prediction interval variance is negative")
    se = np.sqrt(np.maximum(variance, 0.0))
    zcrit = float(-ndtri((1 - conf) / 2))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        hazard = np.exp(log_hazard)
        log_survival = -hazard
        survival = np.exp(log_survival)
        lower = np.exp(-np.exp(log_hazard + zcrit * se))
        upper = np.exp(-np.exp(log_hazard - zcrit * se))
    zero = ~positive
    log_survival[:, zero] = 0.0
    survival[:, zero] = 1.0
    hazard[:, zero] = 0.0
    lower[:, zero] = 1.0
    upper[:, zero] = 1.0
    se[:, zero] = 0.0
    return GeneralizedGammaPrediction(
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
