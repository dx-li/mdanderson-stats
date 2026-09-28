"""Semiparametric two-drug response surface from Kong and Lee (2008).

Responses are supplied on the caller's transformed scale, ``Y = g(E)``. The
module fits either the raw-dose or log-dose additive marginal baseline and a
natural bivariate thin-plate spline to combination residuals. It does not
implement response transformation or wild-bootstrap intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve, null_space
from scipy.optimize import minimize_scalar

from ._validation import FloatArray, finite, scalar

_MAX_OBSERVATIONS = 500
_MAX_SURFACE_CELLS = 2_000_000
_MAX_WORK = 250_000_000
_LOG_SMOOTHING_BOUNDS = (-30.0, 30.0)
_REML_SCREEN_POINTS = 61
_TPS_CONSTANT = 16.0 * np.pi


def _owned(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _owned_bool(value: ArrayLike) -> NDArray[np.bool_]:
    result = np.array(value, dtype=np.bool_, copy=True)
    result.flags.writeable = False
    return result


def _real_finite(value: ArrayLike, name: str) -> FloatArray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    return finite(raw, name)


def _dose_vectors(
    dose1: ArrayLike, dose2: ArrayLike, response: ArrayLike | None = None
) -> tuple[FloatArray, FloatArray, FloatArray | None]:
    d1_raw, d2_raw = np.asarray(dose1), np.asarray(dose2)
    if d1_raw.ndim != 1 or d2_raw.shape != d1_raw.shape:
        raise ValueError("dose1 and dose2 must be matching one-dimensional arrays")
    if d1_raw.size < 4 or d1_raw.size > _MAX_OBSERVATIONS:
        raise ValueError(f"require 4..{_MAX_OBSERVATIONS} dose observations")
    d1, d2 = _real_finite(d1_raw, "dose1"), _real_finite(d2_raw, "dose2")
    if np.any(d1 < 0.0) or np.any(d2 < 0.0):
        raise ValueError("doses must be nonnegative")
    if max(float(np.max(d1)), float(np.max(d2))) > 1e50:
        raise ValueError("dose magnitudes above 1e50 are outside the stable kernel range")
    y: FloatArray | None = None
    if response is not None:
        raw_y = np.asarray(response)
        if raw_y.ndim != 1 or raw_y.shape != d1.shape:
            raise ValueError("response must have one value per dose observation")
        y = _real_finite(raw_y, "response")
    return d1, d2, y


def _dose_scale(d1: FloatArray, d2: FloatArray) -> FloatArray:
    scale = np.array([np.max(d1), np.max(d2)], dtype=np.float64)
    scale[scale == 0.0] = 1.0
    return scale


def _affine_design(d1: FloatArray, d2: FloatArray, scale: FloatArray) -> FloatArray:
    return np.column_stack((np.ones(d1.size), d1 / scale[0], d2 / scale[1]))


def _fit_line(x: FloatArray, y: FloatArray, name: str) -> tuple[float, float]:
    center = float(np.mean(x))
    spread = float(np.max(np.abs(x - center)))
    if not np.isfinite(spread) or spread <= 0.0:
        raise ValueError(f"{name} marginal doses must contain at least two distinct values")
    normalized = (x - center) / spread
    design = np.column_stack((np.ones(x.size), normalized))
    if np.linalg.matrix_rank(design) != 2:
        raise ValueError(f"{name} marginal linear fit is rank deficient")
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != 2:
        raise ValueError(f"{name} marginal linear fit is rank deficient")
    slope = float(coefficients[1] / spread)
    intercept = float(coefficients[0] - slope * center)
    if not np.isfinite(intercept) or not np.isfinite(slope):
        raise ArithmeticError(f"{name} marginal coefficients are not representable")
    return intercept, slope


def _log_dose_baseline(
    d1: FloatArray,
    d2: FloatArray,
    dose1_fit: tuple[float, float],
    dose2_fit: tuple[float, float],
    *,
    allow_both_zero: bool,
) -> FloatArray:
    beta0, beta1 = dose1_fit
    alpha0, alpha1 = dose2_fit
    if beta1 == 0.0 or alpha1 == 0.0 or np.sign(beta1) != np.sign(alpha1):
        raise ValueError("log-dose marginal slopes must be nonzero and have the same sign")
    ratio = alpha1 / beta1
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("log-dose marginal slope ratio is not representable and positive")
    gamma1 = (alpha0 - beta0) / beta1
    gamma2 = ratio - 1.0
    derivative_floor = min(1.0, ratio)
    if not np.isfinite(gamma1) or not np.isfinite(gamma2) or derivative_floor <= 0.0:
        raise ArithmeticError("log-dose relative-potency parameters are not representable")

    result = np.empty(d1.shape, dtype=np.float64)
    both_positive = (d1 > 0.0) & (d2 > 0.0)
    first_only = (d1 > 0.0) & (d2 == 0.0)
    second_only = (d1 == 0.0) & (d2 > 0.0)
    both_zero = (d1 == 0.0) & (d2 == 0.0)
    result[first_only] = beta0 + beta1 * np.log(d1[first_only])
    result[second_only] = alpha0 + alpha1 * np.log(d2[second_only])
    if np.any(both_zero):
        if not allow_both_zero:
            raise ValueError("the log-dose baseline is undefined at the both-zero dose")
        result[both_zero] = np.nan
    if np.any(both_positive):
        log_d1 = np.log(d1[both_positive])
        log_d2 = np.log(d2[both_positive])

        def equation(u: FloatArray) -> FloatArray:
            return u - gamma1 - gamma2 * np.logaddexp(log_d1 - u, log_d2)

        at_zero = equation(np.zeros_like(log_d1))
        radius = np.abs(at_zero) / derivative_floor + 1.0
        if not np.all(np.isfinite(at_zero)) or not np.all(np.isfinite(radius)):
            raise ArithmeticError("relative-potency root is outside floating-point range")
        lower, upper = -radius, radius
        low_value, high_value = equation(lower), equation(upper)
        if np.any(np.isnan(low_value)) or np.any(np.isnan(high_value)):
            raise ArithmeticError("relative-potency root bracket is not representable")
        for _ in range(4):
            need_lower = low_value > 0.0
            need_upper = high_value < 0.0
            if not np.any(need_lower | need_upper):
                break
            lower[need_lower] *= 2.0
            upper[need_upper] *= 2.0
            low_value, high_value = equation(lower), equation(upper)
            if np.any(np.isnan(low_value)) or np.any(np.isnan(high_value)):
                raise ArithmeticError("relative-potency root bracket is not representable")
        if np.any(low_value > 0.0) or np.any(high_value < 0.0):
            raise ArithmeticError("could not bracket the log-dose relative-potency root")
        for _ in range(80):
            middle = lower * 0.5 + upper * 0.5
            value = equation(middle)
            if np.any(np.isnan(value)):
                raise ArithmeticError("relative-potency root iteration is not representable")
            move_upper = value > 0.0
            upper = np.where(move_upper, middle, upper)
            lower = np.where(move_upper, lower, middle)
        u = lower * 0.5 + upper * 0.5
        if np.any(upper - lower > 16.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(u))):
            raise ArithmeticError("log-dose relative-potency root did not converge")
        log_combined = np.logaddexp(log_d1, u + log_d2)
        result[both_positive] = beta0 + beta1 * log_combined
    if np.any(~np.isfinite(result[~both_zero])):
        raise ArithmeticError("log-dose baseline prediction is not representable")
    return result


def _thin_plate_kernel(left: FloatArray, right: FloatArray) -> FloatArray:
    delta1 = left[:, None, 0] - right[None, :, 0]
    delta2 = left[:, None, 1] - right[None, :, 1]
    radius = np.hypot(delta1, delta2)
    result = np.zeros(radius.shape, dtype=np.float64)
    positive = radius > 0.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        radius_squared = np.square(radius[positive])
        result[positive] = radius_squared * (2.0 * np.log(radius[positive])) / _TPS_CONSTANT
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("thin-plate kernel is outside floating-point range")
    return result


def _thin_plate_system(knots: FloatArray, dose_scale: FloatArray) -> tuple[FloatArray, FloatArray]:
    scaled_knots = knots / dose_scale
    affine = np.column_stack((np.ones(knots.shape[0]), scaled_knots))
    if knots.shape[0] < 4 or np.linalg.matrix_rank(affine) != 3:
        raise ValueError("thin-plate spline requires at least four noncollinear dose knots")
    constraint_null = null_space(affine.T, rcond=1e-12)
    if constraint_null.shape != (knots.shape[0], knots.shape[0] - 3):
        raise ValueError("thin-plate affine nullspace is rank deficient")
    omega = _thin_plate_kernel(knots, knots)
    penalty = constraint_null.T @ omega @ constraint_null
    penalty = (penalty + penalty.T) * 0.5
    try:
        np.linalg.cholesky(penalty)
    except np.linalg.LinAlgError as exc:
        raise ValueError("thin-plate penalty is rank deficient for these dose knots") from exc
    return constraint_null, penalty


def _reml_precompute(
    response: FloatArray,
    affine: FloatArray,
    random_basis: FloatArray,
    penalty: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    rows = response.size
    if rows <= affine.shape[1]:
        raise ValueError("REML requires more observations than affine surface terms")
    try:
        factor = cho_factor(penalty, lower=True, check_finite=False)
        solved_basis = cho_solve(factor, random_basis.T, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("thin-plate penalty is not positive definite") from exc
    covariance = random_basis @ solved_basis
    covariance = (covariance + covariance.T) * 0.5
    if not np.all(np.isfinite(covariance)):
        raise ArithmeticError("thin-plate mixed-model covariance is nonfinite")
    orthogonal, _ = np.linalg.qr(affine, mode="complete")
    error_space = orthogonal[:, affine.shape[1] :]
    restricted = error_space.T @ covariance @ error_space
    restricted = (restricted + restricted.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(restricted)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if np.min(eigenvalues) < -1e-10 * scale:
        raise ArithmeticError("restricted spline covariance is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = eigenvectors.T @ (error_space.T @ response)
    return covariance, eigenvalues, projected


def _profile_reml(
    log_smoothing: float, eigenvalues: FloatArray, projected_response: FloatArray
) -> tuple[float, float]:
    smoothing = float(np.exp(log_smoothing))
    ratio = eigenvalues / smoothing
    denominator = 1.0 + ratio
    quadratic = float(np.sum(np.square(projected_response) / denominator))
    degrees = projected_response.size
    if not np.isfinite(quadratic) or quadratic <= 0.0:
        return np.inf, 0.0
    variance = quadratic / degrees
    logdet = float(np.log1p(ratio).sum())
    value = 0.5 * (degrees * (np.log(2.0 * np.pi) + 1.0 + np.log(variance)) + logdet)
    if not np.isfinite(value):
        return np.inf, variance
    return value, variance


@dataclass(frozen=True)
class SynergySurfaceFit:
    """Fitted surface with coefficients in scaled dose/response coordinates.

    Baseline coefficients and spline coefficients use response divided by
    ``response_scale``. Raw-dose baseline slopes use doses divided by
    ``dose_scale``; log-dose baseline coefficients use natural-log doses.
    ``residual_variance_scaled`` is in squared response-scaled units.
    """

    baseline: Literal["raw", "log"]
    baseline_coefficients: FloatArray
    dose1_marginal: FloatArray
    dose2_marginal: FloatArray
    dose_scale: FloatArray
    response_scale: float
    knots: FloatArray
    affine_coefficients: FloatArray
    radial_weights: FloatArray
    constraint_nullspace: FloatArray
    penalty_matrix: FloatArray
    smoothing_parameter: float
    residual_variance_scaled: float
    reml_objective: float
    optimizer_success: bool
    optimizer_message: str
    smoothing_at_boundary: bool
    optimizer_evaluations: int
    baseline_rank: int
    spline_rank: int
    marginal_mask: NDArray[np.bool_]
    combination_mask: NDArray[np.bool_]
    fitted_baseline: FloatArray
    fitted_surface: FloatArray
    fitted_response: FloatArray


@dataclass(frozen=True)
class SynergySurfacePrediction:
    """Baseline, fitted departure and total response on the input Y scale."""

    baseline: FloatArray
    surface: FloatArray
    response: FloatArray


def _fit_baseline(
    d1: FloatArray,
    d2: FloatArray,
    y: FloatArray,
    mode: Literal["raw", "log"],
    dose_scale: FloatArray,
    response_scale: float,
) -> tuple[FloatArray, FloatArray, int]:
    marginal = (d1 == 0.0) | (d2 == 0.0)
    if mode == "raw":
        design = _affine_design(d1[marginal], d2[marginal], dose_scale)
        if design.shape[0] < 3 or np.linalg.matrix_rank(design) != 3:
            raise ValueError("raw-dose marginal baseline is rank deficient")
        coefficient, _, rank, _ = np.linalg.lstsq(design, y[marginal] / response_scale, rcond=None)
        if rank != 3:
            raise ValueError("raw-dose marginal baseline is rank deficient")
        baseline = _affine_design(d1, d2, dose_scale) @ coefficient
        return coefficient, baseline, int(rank)

    first = (d1 > 0.0) & (d2 == 0.0)
    second = (d1 == 0.0) & (d2 > 0.0)
    if np.count_nonzero(first) < 2 or np.count_nonzero(second) < 2:
        raise ValueError("log-dose baseline needs two positive marginal doses for each drug")
    fit1 = _fit_line(np.log(d1[first]), y[first] / response_scale, "dose1")
    fit2 = _fit_line(np.log(d2[second]), y[second] / response_scale, "dose2")
    baseline = _log_dose_baseline(d1, d2, fit1, fit2, allow_both_zero=True)
    return np.array((*fit1, *fit2)), baseline, 4


def fit_synergy_surface(
    dose1: ArrayLike,
    dose2: ArrayLike,
    response: ArrayLike,
    *,
    baseline: Literal["raw", "log"] = "raw",
    smoothing_parameter: float | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> SynergySurfaceFit:
    """Fit marginal baseline plus natural bivariate thin-plate spline.

    ``response`` is already transformed to ``Y=g(E)`` by the caller. For the
    raw baseline, marginal records fit a common-intercept linear model in
    ``dose1`` and ``dose2``. For ``baseline='log'``, separate log-dose lines
    use the source relative-potency root. The log-dose baseline is undefined
    at the both-zero dose; such training rows are masked from the spline and
    receive NaN as their reported baseline, and prediction there raises.

    A positive explicit smoothing parameter skips REML optimization. Otherwise
    restricted likelihood profiles residual variance and optimizes log lambda
    over [-30,30]. Boundary status is returned rather than hidden. No
    wild-bootstrap interval is provided.
    """
    d1, d2, y_optional = _dose_vectors(dose1, dose2, response)
    assert y_optional is not None
    y = y_optional
    if baseline not in ("raw", "log"):
        raise ValueError("baseline must be 'raw' or 'log'")
    tol = scalar(tolerance, "tolerance")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(
        max_iterations, (int, np.integer)
    ):
        raise ValueError("max_iterations must be an integer in [20,1000]")
    if not 20 <= int(max_iterations) <= 1000:
        raise ValueError("max_iterations must be an integer in [20,1000]")
    if smoothing_parameter is not None:
        smoothing = scalar(smoothing_parameter, "smoothing_parameter")
        if smoothing <= 0.0:
            raise ValueError("smoothing_parameter must be strictly positive")
    else:
        smoothing = np.nan

    knots = np.unique(np.column_stack((d1, d2)), axis=0)
    knot_count = knots.shape[0]
    if d1.size * knot_count + knot_count**2 > _MAX_SURFACE_CELLS:
        raise ValueError("thin-plate basis and knot matrices exceed the supported cell budget")
    reml_profile_budget = _REML_SCREEN_POINTS + (_REML_SCREEN_POINTS - 2) * int(max_iterations)
    estimated_work = (
        knot_count**3
        + d1.size**3
        + d1.size * knot_count**2
        + knot_count * d1.size**2
        + d1.size * reml_profile_budget
    )
    if estimated_work > _MAX_WORK:
        raise ValueError("thin-plate fit exceeds the supported work budget")

    dose_scale = _dose_scale(d1, d2)
    response_scale = float(np.max(np.abs(y)))
    if response_scale == 0.0:
        response_scale = 1.0
    y_scaled = y / response_scale
    baseline_coefficients, baseline_scaled, baseline_rank = _fit_baseline(
        d1, d2, y_scaled, baseline, dose_scale, 1.0
    )
    combination = (d1 > 0.0) & (d2 > 0.0)
    spline_response = np.zeros(y.size, dtype=np.float64)
    spline_response[combination] = y_scaled[combination] - baseline_scaled[combination]
    if not np.all(np.isfinite(spline_response)):
        raise ArithmeticError("marginal residuals are not representable")

    null, penalty = _thin_plate_system(knots, dose_scale)
    basis = _thin_plate_kernel(np.column_stack((d1, d2)), knots)
    random_basis = basis @ null
    affine = _affine_design(d1, d2, dose_scale)
    if np.linalg.matrix_rank(affine) != 3:
        raise ValueError("observed dose pairs do not identify the affine surface terms")
    covariance, eigenvalues, projected = _reml_precompute(
        spline_response, affine, random_basis, penalty
    )
    projected_energy = float(projected @ projected)
    response_energy = float(spline_response @ spline_response)
    exact_fit_tolerance = (256.0 * np.finfo(float).eps) ** 2 * max(1.0, response_energy)
    if projected_energy <= exact_fit_tolerance:
        raise ValueError(
            "surface residual is affine to numerical precision; REML variance is unidentified"
        )

    if smoothing_parameter is None:

        def profile(log_value: float) -> float:
            return _profile_reml(log_value, eigenvalues, projected)[0]

        grid = np.linspace(*_LOG_SMOOTHING_BOUNDS, _REML_SCREEN_POINTS)
        grid_values = np.array([profile(float(value)) for value in grid])
        profile_evaluations = int(grid.size)
        if not np.any(np.isfinite(grid_values)):
            raise ValueError("REML residual variance is unidentified over the supported range")
        candidates = [
            (float(value), float(point))
            for point, value in zip(grid, grid_values, strict=True)
            if np.isfinite(value)
        ]
        refinements = 0
        refinement_success = True
        for index in range(1, grid.size - 1):
            if (
                np.isfinite(grid_values[index])
                and grid_values[index] <= grid_values[index - 1]
                and grid_values[index] <= grid_values[index + 1]
            ):
                result = minimize_scalar(
                    profile,
                    method="bounded",
                    bounds=(float(grid[index - 1]), float(grid[index + 1])),
                    options={"xatol": tol, "maxiter": int(max_iterations)},
                )
                refinements += 1
                profile_evaluations += int(result.nfev)
                refinement_success = refinement_success and bool(result.success)
                if np.isfinite(result.fun):
                    candidates.append((float(result.fun), float(result.x)))
        best_value, log_smoothing = min(candidates, key=lambda item: item[0])
        if not np.isfinite(best_value):
            raise ArithmeticError("REML smoothing optimization failed to find a finite fit")
        smoothing = float(np.exp(log_smoothing))
        optimizer_success = refinement_success
        optimizer_message = (
            f"log-lambda grid screened; refined {refinements} local minima"
            if refinement_success
            else "one or more local REML refinements did not converge"
        )
        optimizer_evaluations = profile_evaluations
        at_boundary = (
            min(
                log_smoothing - _LOG_SMOOTHING_BOUNDS[0],
                _LOG_SMOOTHING_BOUNDS[1] - log_smoothing,
            )
            < 1e-3
        )
    else:
        log_smoothing = float(np.log(smoothing))
        optimizer_success = True
        optimizer_message = "explicit smoothing parameter"
        optimizer_evaluations = 0
        at_boundary = False
    reml_objective, residual_variance = _profile_reml(log_smoothing, eigenvalues, projected)
    if not np.isfinite(reml_objective) or residual_variance <= 0.0:
        raise ValueError("surface has an exact fit or unidentified residual variance")

    marginal_covariance = np.eye(y.size) + covariance / smoothing
    try:
        v_factor = cho_factor(marginal_covariance, lower=True, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise ArithmeticError("thin-plate prediction covariance is not positive definite") from exc
    solved_affine = cho_solve(v_factor, affine, check_finite=False)
    solved_response = cho_solve(v_factor, spline_response, check_finite=False)
    fixed_information = affine.T @ solved_affine
    if np.linalg.matrix_rank(fixed_information) != 3:
        raise ValueError("REML affine information is rank deficient")
    affine_coefficients = np.linalg.solve(fixed_information, affine.T @ solved_response)
    residual = spline_response - affine @ affine_coefficients
    solved_residual = cho_solve(v_factor, residual, check_finite=False)
    random_coefficients = (
        cho_solve(
            cho_factor(penalty, lower=True, check_finite=False),
            random_basis.T @ solved_residual,
            check_finite=False,
        )
        / smoothing
    )
    radial_weights = null @ random_coefficients
    fitted_surface_scaled = affine @ affine_coefficients + random_basis @ random_coefficients
    marginal = (d1 == 0.0) | (d2 == 0.0)
    fitted_baseline = baseline_scaled * response_scale
    fitted_surface = fitted_surface_scaled * response_scale
    fitted_response = np.where(
        np.isfinite(fitted_baseline), fitted_baseline + fitted_surface, np.nan
    )
    allowed_undefined = (baseline == "log") & (d1 == 0.0) & (d2 == 0.0)
    baseline_valid = np.isfinite(fitted_baseline)
    response_valid = np.isfinite(fitted_response)
    if (
        not np.all(np.isfinite(affine_coefficients))
        or not np.all(np.isfinite(radial_weights))
        or np.any(np.isinf(fitted_baseline))
        or np.any(np.isnan(fitted_baseline) & ~allowed_undefined)
        or not np.all(np.isfinite(fitted_surface))
        or np.any(np.isinf(fitted_response))
        or np.any(np.isnan(fitted_response) & ~allowed_undefined)
        or not np.all(np.isfinite(fitted_baseline[baseline_valid]))
        or not np.all(np.isfinite(fitted_response[response_valid]))
    ):
        raise ArithmeticError("fitted response surface is not representable")
    return SynergySurfaceFit(
        baseline,
        _owned(baseline_coefficients),
        _owned(baseline_coefficients[:2] if baseline == "log" else np.empty(0)),
        _owned(baseline_coefficients[2:] if baseline == "log" else np.empty(0)),
        _owned(dose_scale),
        response_scale,
        _owned(knots),
        _owned(affine_coefficients),
        _owned(radial_weights),
        _owned(null),
        _owned(penalty),
        smoothing,
        residual_variance,
        reml_objective,
        optimizer_success,
        optimizer_message,
        at_boundary,
        optimizer_evaluations,
        baseline_rank,
        int(np.linalg.matrix_rank(penalty)),
        _owned_bool(marginal),
        _owned_bool(combination),
        _owned(fitted_baseline),
        _owned(fitted_surface),
        _owned(fitted_response),
    )


def predict_synergy_surface(
    fit: SynergySurfaceFit, dose1: ArrayLike, dose2: ArrayLike
) -> SynergySurfacePrediction:
    """Predict baseline, spline departure and total transformed response."""
    if not isinstance(fit, SynergySurfaceFit):
        raise ValueError("fit must be a SynergySurfaceFit")
    raw1, raw2 = np.asarray(dose1), np.asarray(dose2)
    try:
        shape = np.broadcast_shapes(raw1.shape, raw2.shape)
    except ValueError as exc:
        raise ValueError("dose1 and dose2 are not broadcast-compatible") from exc
    cells = prod(shape)
    if cells == 0:
        raise ValueError("prediction doses must contain at least one value")
    knot_count = fit.knots.shape[0]
    if cells > 250_000 or 8 * cells + knot_count > _MAX_SURFACE_CELLS:
        raise ValueError("prediction surface exceeds the supported cell budget")
    if cells * knot_count > 50_000_000:
        raise ValueError("prediction kernel work exceeds the supported budget")
    if np.iscomplexobj(raw1) or np.iscomplexobj(raw2):
        raise ValueError("prediction doses must be real-valued")
    d1 = np.broadcast_to(finite(raw1, "dose1"), shape).reshape(-1)
    d2 = np.broadcast_to(finite(raw2, "dose2"), shape).reshape(-1)
    if np.any(d1 < 0.0) or np.any(d2 < 0.0):
        raise ValueError("doses must be nonnegative")
    if max(float(np.max(d1)), float(np.max(d2))) > 1e50:
        raise ValueError("dose magnitudes above 1e50 are outside the stable kernel range")
    if fit.baseline == "raw":
        baseline_scaled = _affine_design(d1, d2, fit.dose_scale) @ fit.baseline_coefficients
    else:
        if d1.size == 0:
            baseline_scaled = np.empty(0, dtype=np.float64)
        else:
            baseline_scaled = _log_dose_baseline(
                d1,
                d2,
                (float(fit.baseline_coefficients[0]), float(fit.baseline_coefficients[1])),
                (float(fit.baseline_coefficients[2]), float(fit.baseline_coefficients[3])),
                allow_both_zero=False,
            )
    affine = _affine_design(d1, d2, fit.dose_scale)
    surface_scaled = affine @ fit.affine_coefficients
    chunk_rows = max(1, _MAX_SURFACE_CELLS // (4 * knot_count))
    for start in range(0, cells, chunk_rows):
        stop = min(cells, start + chunk_rows)
        surface_scaled[start:stop] += (
            _thin_plate_kernel(np.column_stack((d1[start:stop], d2[start:stop])), fit.knots)
            @ fit.radial_weights
        )
    baseline = baseline_scaled * fit.response_scale
    surface = surface_scaled * fit.response_scale
    total = baseline + surface
    if not all(np.all(np.isfinite(value)) for value in (baseline, surface, total)):
        raise ArithmeticError("predicted response surface is not representable")
    return SynergySurfacePrediction(
        _owned(baseline.reshape(shape)),
        _owned(surface.reshape(shape)),
        _owned(total.reshape(shape)),
    )
