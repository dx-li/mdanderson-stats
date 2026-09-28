"""Exact Royston--Parmar spline survival models with right censoring."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult, minimize
from scipy.special import erfcx, expit, log_ndtr, ndtri

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar
from .parametric_survival import _check_location_separation, _data

_SCALES = ("hazard", "odds", "normal")
_LOG_2PI = log(2 * pi)
_MAX_CELLS = 2_000_000
_MAX_INTERNAL_KNOTS = 32
_LOG_SQRT_2_OVER_PI = 0.5 * log(2 / pi)


def _normal_upper_mills(value: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return phi(x)/Phi(-x) and its stable difference from x for x>0."""
    x = np.asarray(value, dtype=np.float64)
    ratio = np.empty(x.shape)
    excess = np.empty(x.shape)
    asymptotic = x > 40
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        ratio[:] = np.exp(_LOG_SQRT_2_OVER_PI - np.log(erfcx(x / np.sqrt(2))))
        inverse = 1 / x[asymptotic]
        inverse2 = inverse * inverse
        excess[asymptotic] = inverse * (
            1 + inverse2 * (-2 + inverse2 * (10 + inverse2 * (-74 + inverse2 * 706)))
        )
        regular = ~asymptotic
        excess[regular] = ratio[regular] - x[regular]
    return ratio, excess


@dataclass(frozen=True)
class SurvivalSplineFit:
    """Fitted Royston--Parmar model and joint observed-information covariance.

    ``coefficients`` and covariance use native raw-log-time basis coefficients,
    followed by slopes for the original covariate units. ``scaled_parameters``
    and ``scaled_covariance`` are the equivalent centered numerical coordinates
    used for stable prediction.
    """

    scale: Literal["hazard", "odds", "normal"]
    k: int
    knots: FloatArray
    coefficients: FloatArray
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
    scaled_knots: FloatArray


@dataclass(frozen=True)
class SurvivalSplinePrediction:
    """Survival surface with delta-method intervals on log cumulative hazard."""

    profile: FloatArray
    times: FloatArray
    log_survival: FloatArray
    survival: FloatArray
    cumulative_hazard: FloatArray
    lower: FloatArray
    upper: FloatArray
    se_log_cumulative_hazard: FloatArray
    confidence: float


def _basis(knots: FloatArray, values: FloatArray, derivative: int = 0) -> FloatArray:
    """Native RP natural-cubic basis, with exact linear extrapolation tails."""
    if derivative not in (0, 1):
        raise ValueError("basis derivative must be zero or one")
    z = np.asarray(values, dtype=np.float64)
    a, b = float(knots[0]), float(knots[-1])
    internal = knots[1:-1]
    result = np.zeros((z.size, knots.size), dtype=np.float64)
    if derivative == 0:
        result[:, 0] = 1.0
        result[:, 1] = z
    else:
        result[:, 1] = 1.0
    if internal.size == 0:
        return result

    left = z <= a
    right = z >= b
    middle = ~(left | right)
    lam = (b - internal) / (b - a)
    if np.any(middle):
        zm = z[middle, None]
        pos_internal = np.maximum(zm - internal[None, :], 0.0)
        pos_left = np.maximum(zm - a, 0.0)
        pos_right = np.maximum(zm - b, 0.0)
        if derivative == 0:
            cubic = pos_internal**3 - lam[None, :] * pos_left**3 - (1 - lam)[None, :] * pos_right**3
        else:
            cubic = 3 * (
                pos_internal**2 - lam[None, :] * pos_left**2 - (1 - lam)[None, :] * pos_right**2
            )
        result[middle, 2:] = cubic
    if np.any(right):
        distance = b - internal
        slope_at_b = 3 * distance * (a - internal)
        if derivative == 0:
            value_at_b = distance * (a - internal) * (2 * b - a - internal)
            result[right, 2:] = value_at_b + (z[right, None] - b) * slope_at_b
        else:
            result[right, 2:] = slope_at_b
    return result


def _minimum_slope(gamma: FloatArray, knots: FloatArray) -> float:
    """Exact global minimum of the piecewise-quadratic eta derivative."""
    if knots.size == 2:
        return float(gamma[1])
    minimum = float(np.min(_basis(knots, knots, derivative=1) @ gamma))
    for left, right in zip(knots[:-1], knots[1:], strict=True):
        points = np.array([left, (left + right) / 2, right])
        values = _basis(knots, points, derivative=1) @ gamma
        # Express the segment derivative as A*v^2 + B*v + C, v in [0,1].
        a = 2 * (values[2] - 2 * values[1] + values[0])
        b = values[2] - values[0] - a
        if a > 0:
            vertex = -b / (2 * a)
            if 0 < vertex < 1:
                minimum = min(minimum, float(a * vertex**2 + b * vertex + values[0]))
    return minimum


def _link_terms(
    eta: FloatArray, scale: Literal["hazard", "odds", "normal"]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return log survival, log density factor, and first/second eta derivatives."""
    if scale == "hazard":
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            hazard = np.exp(eta)
            log_survival = -hazard
            log_density_factor = eta - hazard
            first_event = 1 - hazard
            second_event = -hazard
        first_survival = second_survival = -hazard
    elif scale == "odds":
        softplus = np.logaddexp(0.0, eta)
        probability = expit(eta)
        log_survival = -softplus
        log_density_factor = eta - 2 * softplus
        first_event = 1 - 2 * probability
        second_event = -2 * probability * (1 - probability)
        first_survival = -probability
        second_survival = -probability * (1 - probability)
    else:
        log_survival = log_ndtr(-eta)
        with np.errstate(over="ignore", invalid="ignore"):
            log_density_factor = -0.5 * eta**2 - 0.5 * _LOG_2PI
        first_event = -eta
        second_event = np.full(eta.shape, -1.0)
        first_survival = np.zeros(eta.shape)
        second_survival = np.zeros(eta.shape)
        positive_tail = eta > 10
        ordinary = (~positive_tail) & (eta > -8)
        negative_tail = eta <= -8
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            log_ratio = -0.5 * eta[ordinary] ** 2 - 0.5 * _LOG_2PI - log_survival[ordinary]
            ratio = np.exp(log_ratio)
            first_survival[ordinary] = -ratio
            second_survival[ordinary] = -ratio * (ratio - eta[ordinary])
        if np.any(positive_tail):
            ratio, excess = _normal_upper_mills(eta[positive_tail])
            first_survival[positive_tail] = -ratio
            second_survival[positive_tail] = -ratio * excess
        if np.any(negative_tail):
            # The lower-tail ratio underflows harmlessly for very negative eta.
            ratio = np.exp(
                -0.5 * eta[negative_tail] ** 2 - 0.5 * _LOG_2PI - log_survival[negative_tail]
            )
            first_survival[negative_tail] = -ratio
            second_survival[negative_tail] = -ratio * (ratio - eta[negative_tail])
    return (
        log_survival,
        log_density_factor,
        first_event,
        second_event,
        first_survival,
        second_survival,
    )


def _loglikelihood(
    parameters: FloatArray,
    event: FloatArray,
    basis: FloatArray,
    derivative_basis: FloatArray,
    covariate_design: FloatArray,
    scale: Literal["hazard", "odds", "normal"],
    *,
    information: bool = False,
) -> tuple[float, FloatArray, FloatArray | None]:
    m = basis.shape[1]
    design = np.column_stack((basis, covariate_design))
    derivative = derivative_basis @ parameters[:m]
    eta = design @ parameters
    if not np.isfinite(eta).all() or not np.isfinite(derivative).all():
        raise ArithmeticError("spline linear predictor exceeds numerical range")
    if np.any((event == 1) & (derivative <= 0)):
        raise ArithmeticError("spline derivative is nonpositive at an event time")
    (
        log_survival,
        log_density_factor,
        first_event,
        second_event,
        first_survival,
        second_survival,
    ) = _link_terms(eta, scale)
    selected = np.where(
        event == 1,
        np.log(np.maximum(derivative, np.finfo(float).tiny)) + log_density_factor,
        log_survival,
    )
    if not np.isfinite(selected).all():
        raise ArithmeticError("spline likelihood exceeds numerical range")
    nll = -float(np.sum(selected))
    eta_score = np.where(event == 1, first_event, first_survival)
    score = -(design.T @ eta_score)
    inverse_derivative = np.zeros(derivative.shape)
    np.divide(1.0, derivative, out=inverse_derivative, where=event == 1)
    score[:m] -= derivative_basis.T @ inverse_derivative
    if not np.isfinite(nll) or not np.isfinite(score).all():
        raise ArithmeticError("spline score exceeds numerical range")
    if not information:
        return nll, score, None
    eta_second = np.where(event == 1, second_event, second_survival)
    observed = (design.T * (-eta_second)) @ design
    inverse_derivative_squared = np.zeros(derivative.shape)
    np.divide(1.0, derivative**2, out=inverse_derivative_squared, where=event == 1)
    observed[:m, :m] += (derivative_basis.T * inverse_derivative_squared) @ derivative_basis
    observed = (observed + observed.T) / 2
    if not np.isfinite(observed).all():
        raise ArithmeticError("spline observed information exceeds numerical range")
    return nll, score, observed


def _scaled_to_original_matrix(
    time_center: float,
    time_scale: float,
    covariate_mean: FloatArray,
    covariate_scale: FloatArray,
    basis_count: int,
) -> FloatArray:
    cov_count = covariate_mean.size
    transform = np.zeros((basis_count + cov_count, basis_count + cov_count))
    # scaled parameters = transform @ raw parameters. Cubic RP columns scale
    # exactly by time_scale**3 under affine log-time normalization.
    transform[0, 0] = 1
    transform[0, 1] = time_center
    if cov_count:
        transform[0, basis_count:] = covariate_mean
    transform[1, 1] = time_scale
    if basis_count > 2:
        transform[2:basis_count, 2:basis_count] = np.eye(basis_count - 2) * time_scale**3
    if cov_count:
        transform[basis_count:, basis_count:] = np.diag(covariate_scale)
    return transform


def fit_survival_spline(
    time: ArrayLike,
    event: ArrayLike,
    covariates: ArrayLike | None = None,
    *,
    scale: Literal["hazard", "odds", "normal"] = "hazard",
    k: int | None = None,
    internal_knots: ArrayLike | None = None,
    tolerance: float = 1e-7,
    max_iterations: int = 1000,
) -> SurvivalSplineFit:
    """Fit an exact right-censored Royston--Parmar model.

    With no knot arguments, four internal knots are placed at type-7 quantiles
    of event log times. Boundary knots are the minimum and maximum event log
    times. Explicit internal knots are expressed in raw log-time units.
    """
    if scale not in _SCALES:
        raise ValueError(f"scale must be one of {_SCALES}")
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
    t, event_values, raw_design = _data(time, event, covariates)
    positive = t > 0
    log_time = np.log(t[positive])
    e = event_values[positive]
    x = raw_design[positive, 1:]
    event_times = log_time[e == 1]
    boundary = np.array([np.min(event_times), np.max(event_times)])
    if not np.isfinite(boundary).all():
        raise ValueError("positive event times are required to identify spline boundaries")

    if internal_knots is None:
        knot_count = 4 if k is None else k
        if (
            isinstance(knot_count, (bool, np.bool_))
            or not isinstance(knot_count, (int, np.integer))
            or not 0 <= knot_count <= _MAX_INTERNAL_KNOTS
        ):
            raise ValueError(f"k must be an integer in [0, {_MAX_INTERNAL_KNOTS}]")
        if knot_count > 0 and knot_count >= event_times.size:
            raise ValueError("k must be smaller than the number of exact event times")
        probabilities = np.arange(1, int(knot_count) + 1) / (int(knot_count) + 1)
        inner = (
            np.quantile(event_times, probabilities, method="linear") if knot_count else np.empty(0)
        )
    else:
        if np.iscomplexobj(internal_knots):
            raise ValueError("internal_knots must be real")
        inner = finite(internal_knots, "internal_knots")
        if inner.ndim != 1 or inner.size > _MAX_INTERNAL_KNOTS:
            raise ValueError(
                f"internal_knots must be a vector of at most {_MAX_INTERNAL_KNOTS} values"
            )
        if k is not None:
            if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
                raise ValueError("k must be an integer knot count")
            if not 0 <= k <= _MAX_INTERNAL_KNOTS:
                raise ValueError(f"k must be an integer in [0, {_MAX_INTERNAL_KNOTS}]")
            if k != inner.size:
                raise ValueError("k must equal the number of explicit internal_knots")
        knot_count = int(inner.size)
    if knot_count > 0 and boundary[0] >= boundary[1]:
        raise ValueError("distinct positive event times are required to identify spline knots")
    if inner.size and (
        np.any(np.diff(inner) <= 0) or inner[0] <= boundary[0] or inner[-1] >= boundary[-1]
    ):
        raise ValueError("internal knots must be distinct and strictly inside event boundaries")
    raw_knots = np.r_[boundary[0], inner, boundary[1]]

    time_center = float(np.mean(log_time))
    time_scale = float(np.std(log_time))
    if not np.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("varying positive log times are required to identify a fit")
    scaled_knots = (raw_knots - time_center) / time_scale
    normalized_time = (log_time - time_center) / time_scale
    basis = _basis(scaled_knots, normalized_time)
    derivative_basis = _basis(scaled_knots, normalized_time, derivative=1)

    covariate_mean = np.mean(x, axis=0) if x.shape[1] else np.empty(0)
    covariate_scale = np.std(x, axis=0) if x.shape[1] else np.empty(0)
    if np.any(covariate_scale <= 0) or not np.isfinite(covariate_scale).all():
        raise ValueError("all covariates must vary among positive-time observations")
    covariate_design = (
        (x - covariate_mean) / covariate_scale if x.shape[1] else np.empty((x.shape[0], 0))
    )
    full_design = np.column_stack((basis, covariate_design))
    if np.linalg.matrix_rank(full_design) != full_design.shape[1]:
        raise ValueError("positive-time observations require a full-rank spline design")
    separation_design = np.column_stack((np.ones(log_time.size), covariate_design))
    _check_location_separation(separation_design, e)

    m = scaled_knots.size
    dimension = m + covariate_design.shape[1]
    theta0 = np.zeros(dimension)
    theta0[1] = 1.0
    starts = [theta0.copy(), theta0.copy(), theta0.copy()]
    starts[1][0], starts[1][1] = -0.5, 0.8
    starts[2][0], starts[2][1] = 0.5, 1.3
    distribution_scale: Literal["hazard", "odds", "normal"] = scale

    def actual_loss(theta: FloatArray) -> tuple[float, FloatArray]:
        loss, score, _ = _loglikelihood(
            theta,
            e,
            basis,
            derivative_basis,
            covariate_design,
            distribution_scale,
        )
        return loss / log_time.size, score / log_time.size

    accepted: list[tuple[float, OptimizeResult, FloatArray, FloatArray]] = []
    failures: list[str] = []
    for start in starts:
        try:

            def loss(theta: FloatArray) -> float:
                try:
                    return actual_loss(theta)[0]
                except (ArithmeticError, FloatingPointError, ValueError):
                    return 1e100 + float(np.dot(theta, theta))

            def gradient(theta: FloatArray) -> FloatArray:
                try:
                    return actual_loss(theta)[1]
                except (ArithmeticError, FloatingPointError, ValueError):
                    return 1e6 * np.sign(theta + 1e-12)

            result = minimize(
                loss,
                start,
                jac=gradient,
                method="BFGS",
                options={"gtol": tol / 10, "maxiter": int(max_iterations)},
            )
            theta = np.asarray(result.x, dtype=np.float64)
            final_loss, final_score = actual_loss(theta)
            score_error = float(np.max(np.abs(final_score)))
            min_slope = _minimum_slope(theta[:m], scaled_knots)
            if (
                not np.isfinite(final_loss)
                or not np.isfinite(score_error)
                or score_error > tol
                or not np.isfinite(min_slope)
                or min_slope <= 0
            ):
                failures.append(
                    f"{result.message}; score={score_error:.3g}; minimum slope={min_slope:.3g}"
                )
                continue
            _, _, observed_information = _loglikelihood(
                theta,
                e,
                basis,
                derivative_basis,
                covariate_design,
                distribution_scale,
                information=True,
            )
            assert observed_information is not None
            np.linalg.cholesky(observed_information)
            accepted.append((final_loss, result, theta, observed_information))
        except (ArithmeticError, FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            failures.append(str(exc))
    if not accepted:
        reason = failures[-1] if failures else "no finite candidate"
        raise ArithmeticError(f"spline fit has no finite monotone identified optimum: {reason}")
    final_loss, result, theta, info_scaled = min(accepted, key=lambda item: item[0])
    covariance_scaled = np.linalg.solve(info_scaled, np.eye(dimension))
    covariance_scaled = (covariance_scaled + covariance_scaled.T) / 2
    np.linalg.cholesky(covariance_scaled)
    transform = _scaled_to_original_matrix(
        time_center, time_scale, covariate_mean, covariate_scale, m
    )
    inverse_transform = np.linalg.solve(transform, np.eye(dimension))
    coefficients = inverse_transform @ theta
    covariance = inverse_transform @ covariance_scaled @ inverse_transform.T
    covariance = (covariance + covariance.T) / 2
    np.linalg.cholesky(covariance)
    information = transform.T @ info_scaled @ transform
    if not all(
        np.isfinite(value).all()
        for value in (coefficients, covariance, information, covariance_scaled, info_scaled)
    ):
        raise ArithmeticError("spline coefficients or covariance exceed numerical range")
    parameter_names = (*[f"gamma{j}" for j in range(m)], *(f"x{j + 1}" for j in range(x.shape[1])))
    log_likelihood = (
        -final_loss * log_time.size
        - float(e.sum()) * np.log(time_scale)
        - float(np.dot(e, log_time))
    )
    score_error = float(np.max(np.abs(actual_loss(theta)[1])))
    return SurvivalSplineFit(
        scale,
        int(knot_count),
        _freeze(raw_knots),
        _freeze(coefficients),
        tuple(parameter_names),
        _freeze(covariance),
        _freeze(information),
        log_likelihood,
        score_error,
        int(result.nit),
        _freeze(theta),
        _freeze(covariance_scaled),
        _freeze(covariate_mean),
        _freeze(covariate_scale),
        time_center,
        time_scale,
        _freeze(scaled_knots),
    )


def _log_survival_and_log_hazard(
    eta: FloatArray, scale: Literal["hazard", "odds", "normal"]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    if scale == "hazard":
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            log_hazard = eta
            cumulative_hazard = np.exp(eta)
            log_survival = -cumulative_hazard
        derivative = np.ones(eta.shape)
    elif scale == "odds":
        softplus = np.logaddexp(0.0, eta)
        log_survival = -softplus
        cumulative_hazard = softplus
        with np.errstate(divide="ignore", invalid="ignore"):
            log_hazard = np.log(softplus)
            derivative = expit(eta) / softplus
        early = eta < -35
        log_hazard[early] = eta[early]
        derivative[early] = 1.0
    else:
        log_survival = log_ndtr(-eta)
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            cumulative_hazard = -log_survival
            log_hazard = np.log(cumulative_hazard)
            log_density = -0.5 * eta**2 - 0.5 * _LOG_2PI
            derivative = np.exp(log_density - log_survival - log_hazard)
        early = eta < -8
        large_early = eta < -10
        ordinary_early = early & ~large_early
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            log_hazard[early] = log_ndtr(eta[early])
            derivative[ordinary_early] = np.exp(
                log_density[ordinary_early] - log_ndtr(eta[ordinary_early])
            )
        if np.any(large_early):
            derivative[large_early] = _normal_upper_mills(-eta[large_early])[0]
        large_late = eta > 10
        if np.any(large_late):
            mills = _normal_upper_mills(eta[large_late])[0]
            derivative[large_late] = np.exp(np.log(mills) - log_hazard[large_late])
    return log_survival, cumulative_hazard, log_hazard, derivative


def predict_survival_spline(
    fit: SurvivalSplineFit,
    times: ArrayLike,
    profiles: ArrayLike | None = None,
    *,
    confidence: float = 0.95,
) -> SurvivalSplinePrediction:
    """Predict survival with pointwise delta-method intervals on log hazard."""
    if np.iscomplexobj(confidence):
        raise ValueError("confidence must be real")
    conf = scalar(confidence, "confidence")
    if not 0 < conf < 1:
        raise ValueError("confidence must be in (0, 1)")
    if np.iscomplexobj(times):
        raise ValueError("times must be real")
    time_values = finite(times, "times")
    if time_values.ndim != 1 or not 1 <= time_values.size <= 100_000 or np.any(time_values < 0):
        raise ValueError("times must be a nonempty vector of nonnegative values")
    covariate_count = fit.covariate_mean.size
    if profiles is None:
        profile_values = np.zeros((1, covariate_count))
    else:
        if np.iscomplexobj(profiles):
            raise ValueError("profiles must be real")
        profile_values = finite(profiles, "profiles")
        if profile_values.ndim == 1:
            profile_values = profile_values[None, :]
        if profile_values.ndim != 2 or profile_values.shape[1] != covariate_count:
            raise ValueError("profiles must have one column per fitted covariate")
    parameter_count = fit.scaled_parameters.size
    if (
        not 1 <= profile_values.shape[0] <= 100_000
        or profile_values.size > _MAX_CELLS
        or 8 * profile_values.shape[0] * time_values.size
        + profile_values.shape[0] * parameter_count
        > _MAX_CELLS
    ):
        raise ValueError("prediction surface exceeds 2,000,000 cells")
    standardized_profiles = (
        (profile_values - fit.covariate_mean) / fit.covariate_scale
        if covariate_count
        else np.empty((profile_values.shape[0], 0))
    )
    if not np.isfinite(standardized_profiles).all():
        raise ArithmeticError("normalized spline profiles exceed numerical range")
    positive = time_values > 0
    normalized_time = np.zeros(time_values.shape)
    normalized_time[positive] = (
        np.log(time_values[positive]) - fit.log_time_center
    ) / fit.log_time_scale
    if not np.isfinite(normalized_time[positive]).all():
        raise ArithmeticError("normalized spline times exceed numerical range")
    positive_indices = np.flatnonzero(positive)
    profile_design = (
        standardized_profiles if covariate_count else np.empty((profile_values.shape[0], 0))
    )
    surface_shape = (profile_values.shape[0], time_values.size)
    log_survival = np.zeros(surface_shape)
    log_hazard = np.full(surface_shape, -np.inf)
    cumulative_hazard = np.zeros(surface_shape)
    standard_error = np.zeros(surface_shape)
    lower = np.ones(surface_shape)
    upper = np.ones(surface_shape)
    zcrit = float(-ndtri((1 - conf) / 2))
    covariance = np.asarray(fit.scaled_covariance)

    # Process profile and time blocks separately; temporary design arrays are
    # bounded independently of the full prediction surface dimensions.
    max_design_cells = 1_000_000
    block_width = max(1, max_design_cells // parameter_count)
    for profile_index in range(profile_values.shape[0]):
        for start in range(0, positive_indices.size, block_width):
            stop = min(start + block_width, positive_indices.size)
            destination = positive_indices[start:stop]
            basis_block = _basis(fit.scaled_knots, normalized_time[destination])
            xrow = profile_design[profile_index]
            design = np.column_stack(
                (basis_block, np.broadcast_to(xrow, (basis_block.shape[0], xrow.size)))
            )
            eta_block = design @ fit.scaled_parameters
            if not np.isfinite(eta_block).all():
                raise ArithmeticError("spline prediction linear predictor exceeds numerical range")
            log_s_block, hazard_block, log_h_block, dlogh = _log_survival_and_log_hazard(
                eta_block, fit.scale
            )
            log_survival[profile_index, destination] = log_s_block
            log_hazard[profile_index, destination] = log_h_block
            cumulative_hazard[profile_index, destination] = hazard_block
            variance = dlogh**2 * np.einsum("ti,ij,tj->t", design, covariance, design)
            if not np.isfinite(variance).all() or np.any(variance < -1e-12):
                raise ArithmeticError("spline prediction variance is nonfinite or negative")
            se_block = np.sqrt(np.maximum(variance, 0.0))
            standard_error[profile_index, destination] = se_block
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                lower[profile_index, destination] = np.exp(-np.exp(log_h_block + zcrit * se_block))
                upper[profile_index, destination] = np.exp(-np.exp(log_h_block - zcrit * se_block))
    log_survival[:, ~positive] = 0.0
    cumulative_hazard[:, ~positive] = 0.0
    lower[:, ~positive] = 1.0
    upper[:, ~positive] = 1.0
    standard_error[:, ~positive] = 0.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        survival = np.exp(log_survival)
    return SurvivalSplinePrediction(
        _freeze(profile_values),
        _freeze(time_values),
        _freeze(log_survival),
        _freeze(survival),
        _freeze(cumulative_hazard),
        _freeze(lower),
        _freeze(upper),
        _freeze(standard_error),
        conf,
    )
