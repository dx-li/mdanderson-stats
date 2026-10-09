"""Four source-defined SYNERGY response surfaces for fraction surviving.

Equations are implemented independently; original S-PLUS listings are used
only as external numerical references. See docs/synergy-parametric.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares
from scipy.special import expit
from scipy.stats import t as student_t

from ._validation import FloatArray
from .wfmm_model import _finite_real

_NAMES = {
    "greco": ("m1", "m2", "dm1", "dm2", "alpha"),
    "machado": ("m1", "m2", "dm1", "dm2", "eta"),
    "plummer": ("beta0", "beta1", "beta2", "beta3", "beta4"),
    "carter": ("beta0", "beta1", "beta2", "beta12"),
}
_MAX_POINTS = 100_000


def _frozen(value: ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


def _model(model: str) -> str:
    if not isinstance(model, str) or model not in _NAMES:
        raise ValueError("model must be greco, machado, plummer or carter")
    return model


def _pair(dose1: ArrayLike, dose2: ArrayLike) -> tuple[FloatArray, FloatArray]:
    left, right = _finite_real(dose1, "dose1"), _finite_real(dose2, "dose2")
    shape = np.broadcast_shapes(left.shape, right.shape)
    if math.prod(shape) > _MAX_POINTS or left.size > _MAX_POINTS or right.size > _MAX_POINTS:
        raise ValueError("dose grid exceeds 100000 points")
    if np.any(left < 0) or np.any(right < 0):
        raise ValueError("doses must be nonnegative")
    broadcast_left, broadcast_right = np.broadcast_arrays(left, right)
    return broadcast_left, broadcast_right


def _bounds(model: str) -> tuple[FloatArray, FloatArray]:
    if model in {"greco", "machado"}:
        low = [-1000.0, -1000.0, 1e-12, 1e-12, -1000.0 if model == "greco" else 1e-6]
        high = [-1e-6, -1e-6, 1e12, 1e12, 1000.0]
    elif model == "plummer":
        low, high = [-1e4, -1000.0, -100.0, -0.999, -1.999999], [1e4, -1e-6, 100.0, 1000.0, 1000.0]
    else:
        low, high = [-1e4] * 4, [1e4] * 4
    return np.array(low), np.array(high)


def _parameters(model: str, parameters: ArrayLike) -> FloatArray:
    theta = _finite_real(parameters, "parameters")
    if theta.shape != (len(_NAMES[model]),):
        raise ValueError(f"parameters must be ordered as {_NAMES[model]}")
    lower, upper = _bounds(model)
    if np.any(theta < lower) or np.any(theta > upper):
        raise ValueError("parameters are outside the documented supported bounds")
    return theta


def _greco_limit(theta: FloatArray, left: FloatArray, right: FloatArray) -> float:
    combined = (left > 0) & (right > 0)
    if not np.any(combined):
        return -1000.0
    c1, c2 = -1 / theta[0], -1 / theta[1]
    log_product = float(np.max(np.log(left[combined]) + np.log(right[combined])))
    log_magnitude = np.log(4 * np.sqrt(c1 * c2) / (c1 + c2)) + 0.5 * (
        np.log(theta[2]) + np.log(theta[3]) - log_product
    )
    return -float(np.exp(min(log_magnitude, np.log(1000.0))))


def _greco_physical(coordinate: FloatArray, left: FloatArray, right: FloatArray) -> FloatArray:
    result = coordinate.copy()
    result[4] = _greco_limit(coordinate, left, right) + np.exp(coordinate[4])
    return result


def _greco_conversion(coordinate: FloatArray, left: FloatArray, right: FloatArray) -> FloatArray:
    limit = _greco_limit(coordinate, left, right)
    conversion = np.eye(5)
    if limit > -1000.0:
        m1, m2, dm1, dm2 = coordinate[:4]
        c1, c2 = -1 / m1, -1 / m2
        conversion[4, :4] = [
            limit * (0.5 / c1 - 1 / (c1 + c2)) / m1**2,
            limit * (0.5 / c2 - 1 / (c1 + c2)) / m2**2,
            limit / (2 * dm1),
            limit / (2 * dm2),
        ]
    conversion[4, 4] = np.exp(coordinate[4])
    return conversion


def _root(function, shape: tuple[int, ...]) -> FloatArray:
    lower, upper = np.full(shape, -1024.0), np.full(shape, 1024.0)
    if np.any(function(lower) > 0) or np.any(function(upper) < 0):
        raise ArithmeticError("implicit response root exceeds the supported log range")
    for _ in range(80):
        middle = (lower + upper) / 2
        positive = function(middle) > 0
        upper = np.where(positive, middle, upper)
        lower = np.where(positive, lower, middle)
    answer = (lower + upper) / 2
    if np.any(np.abs(function(answer)) > 1e-7):
        raise ArithmeticError("implicit response root did not satisfy its defining equation")
    return answer


def _logits(model: str, theta: FloatArray, left: FloatArray, right: FloatArray) -> FloatArray:
    if model == "carter":
        with np.errstate(over="ignore", invalid="ignore"):
            result = theta[0] + theta[1] * left + theta[2] * right + theta[3] * left * right
        if not np.isfinite(result).all():
            raise ArithmeticError("Carter linear predictor is not representable")
        return result
    with np.errstate(divide="ignore"):
        log_left, log_right = np.log(left), np.log(right)
    origin = (left == 0) & (right == 0)
    # The decreasing-response origin is one; mask it during implicit root work.
    log_left = np.where(origin, 0.0, log_left)
    if model in {"greco", "machado"}:
        m1, m2, dm1, dm2, interaction = theta
        a0, b0 = log_left - np.log(dm1), log_right - np.log(dm2)
        if model == "greco" and interaction < 0 and interaction <= _greco_limit(theta, left, right):
            raise ValueError(
                "Greco parameters do not ensure a unique monotone response on this dose grid"
            )

        def equation(z):
            a, b = a0 - z / m1, b0 - z / m2
            if model == "machado":
                return np.logaddexp(interaction * a, interaction * b)
            total = np.logaddexp(a, b)
            cross = a0 + b0 - z * (0.5 / m1 + 0.5 / m2)
            if interaction > 0:
                return np.logaddexp(total, np.log(interaction) + cross)
            if interaction == 0:
                return total
            return total + np.log1p(interaction * np.exp(cross - total))

        result = _root(equation, left.shape)
    else:
        beta0, beta1, beta2, beta3, beta4 = theta

        def equation(u):
            return u - np.logaddexp(log_right, log_left - beta2 - beta3 * u)

        u = _root(equation, left.shape)
        log_p = beta2 + beta3 * u
        a, b = log_left, log_p + log_right
        total = np.logaddexp(a, b)
        log_effective = total + np.log1p(beta4 * np.exp((a + b) / 2 - total))
        result = beta0 + beta1 * log_effective
    return np.where(origin, np.inf, result)


def synergy_parametric_response(
    model: str, parameters: ArrayLike, dose1: ArrayLike, dose2: ArrayLike
) -> FloatArray:
    """Predict fraction surviving for the original normalized-response models.

    Parameter order and supported domains are listed in the guide. Greco and
    Machado require decreasing marginal slopes; Greco additionally checks a
    sufficient condition for a unique implicit response. Log-dose models have
    theoretical control response one. Carter predicts its fitted intercept.
    """
    model = _model(model)
    theta = _parameters(model, parameters)
    left, right = _pair(dose1, dose2)
    return _frozen(expit(_logits(model, theta, left, right)))


@dataclass(frozen=True)
class SynergyParametricFit:
    model: str
    parameter_names: tuple[str, ...]
    initial_parameters: FloatArray
    parameters: FloatArray
    covariance: FloatArray
    standard_errors: FloatArray
    t_statistics: FloatArray
    p_values: FloatArray
    parameter_correlation: FloatArray
    residual_standard_error: float
    residual_sum_squares: float
    residual_degrees_freedom: int
    included_rows: tuple[bool, ...]
    dose1: FloatArray
    dose2: FloatArray
    response: FloatArray
    fitted_response: FloatArray
    observed_control_response: float | None
    evaluations: int


def _initial(model: str, d1: FloatArray, d2: FloatArray, logits: FloatArray) -> FloatArray:
    marginals = []
    for dose, other in [(d1, d2), (d2, d1)]:
        selected = (dose > 0) & (other == 0)
        if np.count_nonzero(selected) < 2 or np.unique(dose[selected]).size < 2:
            raise ValueError(
                "initialization needs at least two distinct positive doses per single drug"
            )
        covariate = dose[selected] if model == "carter" else np.log(dose[selected])
        design = np.column_stack((np.ones(len(covariate)), covariate))
        marginals.append(np.linalg.lstsq(design, logits[selected], rcond=None)[0])
    (a, m1), (b, m2) = marginals
    if model == "carter":
        return np.array([(a + b) / 2, m1, m2, 1.0])
    if m1 >= 0 or m2 >= 0:
        raise ValueError("log-dose models require decreasing single-drug responses")
    if model == "plummer":
        return np.array([a, m1, (b - a) / m1, m2 / m1 - 1, 2.5])
    with np.errstate(over="ignore", under="ignore"):
        dm1, dm2 = np.exp(-a / m1), np.exp(-b / m2)
    return np.array([m1, m2, dm1, dm2, 0.0 if model == "greco" else 0.7])


def fit_synergy_parametric(
    dose1: ArrayLike,
    dose2: ArrayLike,
    response: ArrayLike,
    *,
    model: str,
    initial_parameters: ArrayLike | None = None,
    max_evaluations: int = 1000,
) -> SynergyParametricFit:
    """Fit unweighted least squares to logit response, as in the author scripts.

    Carter omits responses exactly zero/one; the other three omit both-zero
    dose controls and require interior responses for every remaining record.
    Returned covariance and t inference use the local full-rank least-squares
    Jacobian, not a binomial likelihood. Fits on numerical/domain boundaries
    or with singular information are rejected. Native nls optimizer parity
    is not claimed. Limits: 500 records and 2000 residual evaluations.
    """
    model = _model(model)
    left, right = _pair(dose1, dose2)
    observed = _finite_real(response, "response")
    if left.ndim != 1 or observed.shape != left.shape or not 1 <= left.size <= 500:
        raise ValueError("training doses/responses must be vectors of equal length, at most 500")
    if np.any((observed < 0) | (observed > 1)):
        raise ValueError("response must lie in [0,1]")
    if isinstance(max_evaluations, (bool, np.bool_)) or not isinstance(
        max_evaluations, (int, np.integer)
    ):
        raise ValueError("max_evaluations must be an integer from 1 to 2000")
    if not 1 <= max_evaluations <= 2000:
        raise ValueError("max_evaluations must be an integer from 1 to 2000")
    origin = (left == 0) & (right == 0)
    included = ((observed > 0) & (observed < 1)) if model == "carter" else ~origin
    if np.any((observed[included] <= 0) | (observed[included] >= 1)):
        raise ValueError("noncontrol responses must be strictly between zero and one")
    d1, d2, y = left[included], right[included], observed[included]
    count = len(_NAMES[model])
    df = len(y) - count
    if df <= 0:
        raise ValueError("positive residual degrees of freedom are required")
    logits = np.log(y) - np.log1p(-y)
    initial = _parameters(
        model, _initial(model, d1, d2, logits) if initial_parameters is None else initial_parameters
    )
    evaluations = 0

    coordinate_initial = initial.copy()
    optimizer_bounds = _bounds(model)
    if model == "greco":
        limit = _greco_limit(initial, d1, d2)
        if initial[4] <= limit:
            raise ValueError("initial Greco parameters do not ensure a unique monotone response")
        coordinate_initial[4] = np.log(initial[4] - limit)
        optimizer_bounds[0][4], optimizer_bounds[1][4] = -30.0, np.log(2000.0)

    def residual(coordinate):
        nonlocal evaluations
        evaluations += 1
        if evaluations > max_evaluations:
            raise ArithmeticError("least-squares residual evaluation budget exhausted")
        theta = _greco_physical(coordinate, d1, d2) if model == "greco" else coordinate
        return _logits(model, theta, d1, d2) - logits

    if model == "carter":
        jacobian = np.column_stack((np.ones(len(y)), d1, d2, d1 * d2))
        parameters = np.linalg.lstsq(jacobian, logits, rcond=None)[0]
        errors = jacobian @ parameters - logits
        evaluations = 1
    else:
        result = least_squares(
            residual,
            coordinate_initial,
            bounds=optimizer_bounds,
            jac="3-point",
            max_nfev=max_evaluations,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        if not result.success or np.any(result.active_mask):
            raise ArithmeticError("least-squares fit did not converge to an interior solution")
        parameters, errors, jacobian = result.x, result.fun, result.jac
        if model == "greco":
            jacobian = np.linalg.solve(_greco_conversion(result.x, d1, d2).T, jacobian.T).T
            parameters = _greco_physical(result.x, d1, d2)
    parameters = _parameters(model, parameters)
    norms = np.linalg.norm(jacobian, axis=0)
    if np.any(norms == 0) or np.linalg.cond(jacobian / norms) > 1e8:
        raise ArithmeticError("least-squares parameter information is singular or ill-conditioned")
    sse = float(errors @ errors)
    if sse <= np.finfo(float).eps * float(logits @ logits):
        raise ArithmeticError("zero residual variance does not support the reported t inference")
    normalized = jacobian / norms
    covariance = (
        np.linalg.inv(normalized.T @ normalized) / norms[:, None] / norms[None, :] * (sse / df)
    )
    standard_errors = np.sqrt(np.diag(covariance))
    statistics = parameters / standard_errors
    correlation = covariance / standard_errors[:, None] / standard_errors[None, :]
    fitted = np.asarray(synergy_parametric_response(model, parameters, left, right)).copy()
    control = float(observed[origin].mean()) if np.any(origin) else None
    if model != "carter" and control is not None:
        fitted[origin] = control
    return SynergyParametricFit(
        model,
        _NAMES[model],
        _frozen(initial),
        _frozen(parameters),
        _frozen(covariance),
        _frozen(standard_errors),
        _frozen(statistics),
        _frozen(2 * student_t.sf(np.abs(statistics), df)),
        _frozen(correlation),
        float(np.sqrt(sse / df)),
        sse,
        df,
        tuple(bool(x) for x in included),
        _frozen(left),
        _frozen(right),
        _frozen(observed),
        _frozen(fitted),
        control,
        evaluations,
    )


def predict_synergy_parametric(
    fit: SynergyParametricFit, dose1: ArrayLike, dose2: ArrayLike
) -> FloatArray:
    """Predict from a fit, preserving native plots' observed log-dose control mean."""
    if not isinstance(fit, SynergyParametricFit):
        raise ValueError("fit must be a SynergyParametricFit")
    left, right = _pair(dose1, dose2)
    result = np.asarray(synergy_parametric_response(fit.model, fit.parameters, left, right)).copy()
    if fit.model != "carter" and fit.observed_control_response is not None:
        result[(left == 0) & (right == 0)] = fit.observed_control_response
    return _frozen(result)
