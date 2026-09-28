"""B-spline generalized-odds-rate regression for interval competing risks.

This implements the two-cause interval likelihood from ``intccr::bssmle``.
The fitted CIFs are constrained to be nondecreasing and to sum to at most one
on the observed covariate range. The lower-boundary CIF is a finite-tolerance
approximation because the source model reaches zero only as its linear
predictor tends to minus infinity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from math import floor
from typing import cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import BSpline
from scipy.optimize import minimize, nnls
from scipy.special import expit

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar

_MAX_ROWS = 20_000
_MAX_COVARIATES = 10
_MAX_CORNERS = 1_024
_MAX_WORK = 20_000_000
_MAX_OUTPUT_CELLS = 2_000_000
_MAX_ITERATIONS = 2_000
_ALPHA_MAX = 20.0


@dataclass(frozen=True)
class IntervalCompetingRiskFit:
    """Fitted two-cause GOR model and source-style regression covariance.

    ``coefficients`` contains the two regression coefficient vectors, first
    cause 1 and then cause 2. ``covariance`` is the residualized-score
    least-squares covariance for those regression coefficients only. It is
    not a joint covariance for the spline baselines. ``score_error`` is the
    largest absolute component of the summed score; ``kkt_error`` is the
    largest projected score component per observation.
    """

    coefficients: FloatArray
    covariance: FloatArray
    alpha: FloatArray
    baseline_coefficients: FloatArray
    knots: FloatArray
    boundary_knots: FloatArray
    covariate_mean: FloatArray
    covariate_scale: FloatArray
    covariate_unit_scale: FloatArray
    covariate_center_scaled: FloatArray
    covariate_scale_scaled: FloatArray
    covariate_min: FloatArray
    covariate_max: FloatArray
    log_likelihood: float
    iterations: int
    score_error: float
    kkt_error: float
    lower_boundary_cif: FloatArray
    max_joint_cif: float
    boundary_cif_tolerance: float
    converged: bool


@dataclass(frozen=True)
class IntervalCompetingRiskPrediction:
    """Predicted cumulative incidences; these are not confidence intervals."""

    profile: FloatArray
    times: FloatArray
    cif1: FloatArray
    cif2: FloatArray


def _readonly_int(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _gor_log_survival(eta: FloatArray, alpha: float) -> tuple[FloatArray, FloatArray]:
    """Return log survival and its eta derivative for a GOR CIF."""
    with np.errstate(over="ignore", invalid="ignore"):
        if alpha == 0.0:
            cumulative = np.exp(eta)
            log_survival = -cumulative
            derivative = -cumulative
        else:
            scaled = np.log(alpha) + eta
            log_survival = -np.logaddexp(0.0, scaled) / alpha
            derivative = -expit(scaled) / alpha
    return log_survival, derivative


def _gor_log_cif(eta: FloatArray, alpha: float) -> FloatArray:
    if alpha == 0.0:
        result = np.empty(eta.shape, dtype=np.float64)
        early = eta < -35.0
        result[early] = eta[early]
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            result[~early] = np.log(-np.expm1(-np.exp(eta[~early])))
        return result
    result = np.empty(eta.shape, dtype=np.float64)
    early = eta < -35.0
    result[early] = eta[early]
    log_survival, _ = _gor_log_survival(eta[~early], alpha)
    result[~early] = _log1mexp(log_survival)
    return result


def _gor_log_density_eta(eta: FloatArray, alpha: float) -> FloatArray:
    log_survival, dlog_survival = _gor_log_survival(eta, alpha)
    if alpha == 0.0:
        return log_survival + eta
    del dlog_survival
    scaled = np.log(alpha) + eta
    log_derivative = -np.logaddexp(0.0, -scaled) - np.log(alpha)
    return log_survival + log_derivative


def _log1mexp(log_value: FloatArray) -> FloatArray:
    """Stable log(1-exp(x)) for x <= 0."""
    result = np.empty(log_value.shape, dtype=np.float64)
    split = -np.log(2.0)
    low = log_value < split
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        result[low] = np.log1p(-np.exp(log_value[low]))
        result[~low] = np.log(-np.expm1(log_value[~low]))
    return result


def _gor_cif(eta: FloatArray, alpha: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    log_survival, dlog_survival = _gor_log_survival(eta, alpha)
    log_cif = _gor_log_cif(eta, alpha)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        cif = np.exp(log_cif)
        density_eta = np.exp(_gor_log_density_eta(eta, alpha))
    return cif, density_eta, log_cif


def _inverse_gor_cif(cif: float, alpha: float) -> float:
    if alpha == 0.0:
        return float(np.log(-np.log1p(-cif)))
    log_survival = np.log1p(-cif)
    log_argument = -alpha * log_survival
    return float(np.log(np.expm1(log_argument) / alpha))


def _basis(time: FloatArray, interior: FloatArray, boundaries: FloatArray) -> FloatArray:
    knots = np.r_[np.repeat(boundaries[0], 4), interior, np.repeat(boundaries[1], 4)]
    n_basis = knots.size - 4
    return np.asarray(BSpline.design_matrix(time, knots, 3, extrapolate=False).toarray())[
        :, :n_basis
    ]


def _choose_knots(endpoint: FloatArray, k: float) -> tuple[FloatArray, FloatArray]:
    boundaries = np.array([float(np.min(endpoint)), float(np.max(endpoint))])
    if boundaries[0] == boundaries[1]:
        raise ValueError("at least two distinct event/censor endpoints are required")
    n_interior = floor(k * endpoint.size ** (1.0 / 3.0))
    if n_interior <= 0:
        return np.empty(0, dtype=np.float64), boundaries
    interior_values = endpoint[(endpoint > boundaries[0]) & (endpoint < boundaries[1])]
    if interior_values.size == 0:
        return np.empty(0, dtype=np.float64), boundaries
    probabilities = np.arange(1, n_interior + 1, dtype=np.float64) / (n_interior + 1)
    candidates = np.quantile(interior_values, probabilities, method="linear")
    interior = np.unique(candidates[(candidates > boundaries[0]) & (candidates < boundaries[1])])
    return np.asarray(interior, dtype=np.float64), boundaries


def _corner_profiles(x_min: FloatArray, x_max: FloatArray) -> FloatArray:
    p = x_min.size
    if p == 0:
        return np.empty((1, 0), dtype=np.float64)
    patterns = np.asarray(list(product((0, 1), repeat=p)), dtype=np.int8)
    return np.where(patterns, x_max[None, :], x_min[None, :])


def _input_data(
    lower: ArrayLike,
    upper: ArrayLike,
    event: ArrayLike,
    covariates: ArrayLike | None,
) -> tuple[FloatArray, FloatArray, np.ndarray, FloatArray]:
    if any(
        np.iscomplexobj(value) for value in (lower, upper, event, covariates) if value is not None
    ):
        raise ValueError("interval competing-risk data must be real")
    lo = finite(lower, "lower")
    if np.iscomplexobj(upper):
        raise ValueError("upper must be real")
    hi = np.asarray(upper, dtype=np.float64)
    code = finite(event, "event")
    if (
        lo.ndim != 1
        or not 1 <= lo.size <= _MAX_ROWS
        or hi.shape != lo.shape
        or code.shape != lo.shape
    ):
        raise ValueError(f"lower, upper and event must be aligned vectors with 1..{_MAX_ROWS} rows")
    if np.any(lo < 0) or np.any(~np.isfinite(lo)):
        raise ValueError("lower endpoints must be finite and nonnegative")
    if np.any(~np.isin(code, (0.0, 1.0, 2.0))):
        raise ValueError("event must contain only 0 (censor), 1, or 2")
    is_event = code > 0
    if np.any(np.isnan(hi[is_event])) or np.any(~np.isfinite(hi[is_event])):
        raise ValueError("event upper endpoints must be finite")
    if np.any(hi[is_event] <= lo[is_event]):
        raise ValueError("event intervals must have upper > lower")
    # Censoring uses only its lower endpoint; its upper field is deliberately ignored.
    x = (
        np.empty((lo.size, 0), dtype=np.float64)
        if covariates is None
        else finite(covariates, "covariates")
    )
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] != lo.size or x.shape[1] > _MAX_COVARIATES:
        raise ValueError(
            f"covariates must have one row per observation and at most {_MAX_COVARIATES} columns"
        )
    if x.size > 2_000_000:
        raise ValueError("covariate matrix exceeds the 2000000-cell limit")
    if np.any(np.bincount(code[is_event].astype(np.int64), minlength=3)[1:] == 0):
        raise ValueError("both causes must have at least one observed event")
    return lo, hi, code.astype(np.int64), x


def _score_rows(
    parameters: FloatArray,
    basis_lower: FloatArray,
    basis_upper: FloatArray,
    x: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    event: np.ndarray,
    alpha: FloatArray,
    *,
    need_rows: bool = False,
) -> tuple[float, FloatArray, FloatArray | None]:
    n, n_basis = basis_lower.shape
    p = x.shape[1]
    phi = parameters[: 2 * n_basis].reshape(2, n_basis)
    beta = parameters[2 * n_basis :].reshape(2, p)
    eta_lower = np.vstack(
        (basis_lower @ phi[0] + x @ beta[0], basis_lower @ phi[1] + x @ beta[1])
    ).T
    eta_upper = np.vstack(
        (basis_upper @ phi[0] + x @ beta[0], basis_upper @ phi[1] + x @ beta[1])
    ).T
    log_s_lower = np.empty((n, 2), dtype=np.float64)
    dlog_s_lower = np.empty((n, 2), dtype=np.float64)
    log_s_upper = np.empty((n, 2), dtype=np.float64)
    dlog_s_upper = np.empty((n, 2), dtype=np.float64)
    for cause in range(2):
        log_s_lower[:, cause], dlog_s_lower[:, cause] = _gor_log_survival(
            eta_lower[:, cause], float(alpha[cause])
        )
        log_s_upper[:, cause], dlog_s_upper[:, cause] = _gor_log_survival(
            eta_upper[:, cause], float(alpha[cause])
        )

    score_lower = np.zeros((n, 2), dtype=np.float64)
    score_upper = np.zeros((n, 2), dtype=np.float64)
    loglik = np.zeros(n, dtype=np.float64)
    for cause in range(2):
        selected = event == cause + 1
        left_censored = selected & (lower == 0.0)
        interval = selected & (lower > 0.0)
        if np.any(left_censored):
            log_cif_u = _gor_log_cif(eta_upper[left_censored, cause], float(alpha[cause]))
            loglik[left_censored] = log_cif_u
            # d log(F) / d eta = f_eta / F, evaluated in log space.
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                score_upper[left_censored, cause] = np.exp(
                    _gor_log_density_eta(eta_upper[left_censored, cause], float(alpha[cause]))
                    - log_cif_u
                )
        if np.any(interval):
            lv = log_s_lower[interval, cause]
            lu = log_s_upper[interval, cause]
            difference = lu - lv
            if np.any(difference >= 0.0):
                return np.inf, np.full(parameters.shape, np.nan), None
            loglik[interval] = lv + _log1mexp(difference)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                ratio = 1.0 / np.expm1(-difference)
                score_lower[interval, cause] = (1.0 + ratio) * dlog_s_lower[interval, cause]
                score_upper[interval, cause] = -ratio * dlog_s_upper[interval, cause]

    censored = event == 0
    if np.any(censored):
        # log(1-F1-F2), retaining tiny failure probabilities in log space.
        log_f = np.column_stack((_log1mexp(log_s_lower[:, 0]), _log1mexp(log_s_lower[:, 1])))
        log_failure_sum = np.logaddexp(log_f[:, 0], log_f[:, 1])
        invalid = censored & (log_failure_sum >= 0.0)
        if np.any(invalid):
            return np.inf, np.full(parameters.shape, np.nan), None
        log_censor = _log1mexp(log_failure_sum[censored])
        loglik[censored] = log_censor
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            probability = np.exp(log_censor)
            for cause in range(2):
                score_lower[censored, cause] = (
                    np.exp(log_s_lower[censored, cause])
                    * dlog_s_lower[censored, cause]
                    / probability
                )

    contributions = np.empty((n, 2 * n_basis + 2 * p), dtype=np.float64) if need_rows else None
    gradient = np.zeros(parameters.size, dtype=np.float64)
    for cause in range(2):
        score = score_lower[:, cause] + score_upper[:, cause]
        gradient[cause * n_basis : (cause + 1) * n_basis] = (
            basis_lower.T @ score_lower[:, cause] + basis_upper.T @ score_upper[:, cause]
        )
        if p:
            beta_offset = 2 * n_basis + cause * p
            gradient[beta_offset : beta_offset + p] = x.T @ score
        if need_rows:
            assert contributions is not None
            contributions[:, cause * n_basis : (cause + 1) * n_basis] = (
                basis_lower * score_lower[:, cause, None]
                + basis_upper * score_upper[:, cause, None]
            )
            if p:
                beta_offset = 2 * n_basis + cause * p
                contributions[:, beta_offset : beta_offset + p] = x * score[:, None]
    value = float(np.sum(loglik))
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        return np.inf, np.full(parameters.shape, np.nan), None
    return value, gradient, contributions


def _constraint_setup(
    alpha: FloatArray,
    n_basis: int,
    p: int,
    upper_basis: FloatArray,
    corners: FloatArray,
    observed_x: FloatArray,
    boundary_eta: FloatArray,
    probability_slack: float,
) -> tuple[dict[str, object], ...]:
    n_parameters = 2 * n_basis + 2 * p
    monotone_matrix = np.zeros((2 * (n_basis - 1), n_parameters), dtype=np.float64)
    for cause in range(2):
        start = cause * n_basis
        for j in range(n_basis - 1):
            monotone_matrix[cause * (n_basis - 1) + j, start + j] = -1.0
            monotone_matrix[cause * (n_basis - 1) + j, start + j + 1] = 1.0

    x_lower = corners
    lower_count = corners.shape[0] * 2

    check_x = np.unique(np.vstack((corners, observed_x)), axis=0)
    if check_x.size > 2_000_000:
        raise ValueError("joint probability constraint design exceeds the work budget")
    upper_design = np.column_stack(
        (np.broadcast_to(upper_basis, (check_x.shape[0], n_basis)), check_x)
    )

    def monotone_fun(theta: FloatArray) -> FloatArray:
        phi = theta[: 2 * n_basis].reshape(2, n_basis)
        return np.diff(phi, axis=1).ravel()

    def lower_fun(theta: FloatArray) -> FloatArray:
        phi = theta[: 2 * n_basis].reshape(2, n_basis)
        beta = theta[2 * n_basis :].reshape(2, p)
        return np.concatenate(
            [boundary_eta[cause] - phi[cause, 0] - x_lower @ beta[cause] for cause in range(2)]
        )

    def lower_jac(theta: FloatArray) -> FloatArray:
        jac = np.zeros((lower_count, n_parameters), dtype=np.float64)
        for cause in range(2):
            rows = slice(cause * corners.shape[0], (cause + 1) * corners.shape[0])
            jac[rows, cause * n_basis] = -1.0
            if p:
                jac[rows, 2 * n_basis + cause * p : 2 * n_basis + (cause + 1) * p] = -corners
        return jac

    def joint_fun(theta: FloatArray) -> FloatArray:
        phi = theta[: 2 * n_basis].reshape(2, n_basis)
        beta = theta[2 * n_basis :].reshape(2, p)
        eta1 = upper_design @ np.r_[phi[0], beta[0]]
        eta2 = upper_design @ np.r_[phi[1], beta[1]]
        f1, _, _ = _gor_cif(eta1, float(alpha[0]))
        f2, _, _ = _gor_cif(eta2, float(alpha[1]))
        return 1.0 - probability_slack - f1 - f2

    def joint_jac(theta: FloatArray) -> FloatArray:
        phi = theta[: 2 * n_basis].reshape(2, n_basis)
        beta = theta[2 * n_basis :].reshape(2, p)
        jac = np.zeros((upper_design.shape[0], n_parameters), dtype=np.float64)
        eta1 = upper_design @ np.r_[phi[0], beta[0]]
        eta2 = upper_design @ np.r_[phi[1], beta[1]]
        _, d1, _ = _gor_cif(eta1, float(alpha[0]))
        _, d2, _ = _gor_cif(eta2, float(alpha[1]))
        jac[:, :n_basis] = -d1[:, None] * upper_basis
        jac[:, n_basis : 2 * n_basis] = -d2[:, None] * upper_basis
        if p:
            jac[:, 2 * n_basis : 2 * n_basis + p] = -d1[:, None] * check_x
            jac[:, 2 * n_basis + p :] = -d2[:, None] * check_x
        return jac

    return (
        {"type": "ineq", "fun": monotone_fun, "jac": lambda theta: monotone_matrix},
        {"type": "ineq", "fun": lower_fun, "jac": lower_jac},
        {"type": "ineq", "fun": joint_fun, "jac": joint_jac},
    )


def _score_covariance(rows: FloatArray, n_basis: int, p: int) -> FloatArray:
    if p == 0:
        return np.empty((0, 0), dtype=np.float64)
    phi_score = rows[:, : 2 * n_basis]
    beta_score = rows[:, 2 * n_basis :]
    projection, _, _, _ = np.linalg.lstsq(phi_score, beta_score, rcond=None)
    residual = beta_score - phi_score @ projection
    meat = residual.T @ residual / rows.shape[0]
    if np.linalg.matrix_rank(meat) < meat.shape[0]:
        raise ArithmeticError("residualized regression scores are rank deficient")
    return np.linalg.solve(meat, np.eye(meat.shape[0])) / rows.shape[0]


def _kkt_error(
    theta: FloatArray,
    objective_gradient: FloatArray,
    constraints: tuple[dict[str, object], ...],
) -> float:
    active_jacobians: list[FloatArray] = []
    for constraint in constraints:
        fun = cast(Callable[[FloatArray], FloatArray], constraint["fun"])
        jac = cast(Callable[[FloatArray], FloatArray], constraint["jac"])
        margin = np.asarray(fun(theta), dtype=np.float64)
        jacobian = np.asarray(jac(theta), dtype=np.float64)
        active = margin <= 1e-6
        if np.any(active):
            active_jacobians.append(jacobian[active])
    if active_jacobians:
        active_jacobian = np.vstack(active_jacobians)
        multipliers, _ = nnls(active_jacobian.T, objective_gradient)
        residual = objective_gradient - active_jacobian.T @ multipliers
    else:
        residual = objective_gradient
    return float(np.max(np.abs(residual)))


def fit_interval_competing_risk(
    lower: ArrayLike,
    upper: ArrayLike,
    event: ArrayLike,
    covariates: ArrayLike | None = None,
    *,
    alpha: ArrayLike = (0.0, 0.0),
    k: float = 1.0,
    boundary_cif_tolerance: float = 1e-7,
    tolerance: float = 1e-7,
    max_iterations: int = 1_000,
) -> IntervalCompetingRiskFit:
    """Fit two interval-censored generalized-odds-rate CIF regressions.

    The two ``alpha`` values fix each cause's link: zero is Fine--Gray and one
    is proportional odds. Each cause has a cubic B-spline baseline and its own
    numeric covariate slopes. The lower-boundary constraint is the finite
    approximation ``CIF_j(t_min) <= boundary_cif_tolerance``.
    """
    lo, hi, code, x = _input_data(lower, upper, event, covariates)
    if np.iscomplexobj(alpha):
        raise ValueError("alpha must be real")
    a = finite(alpha, "alpha")
    if a.shape != (2,) or np.any(a < 0.0) or np.any(a > _ALPHA_MAX):
        raise ValueError(f"alpha must contain two values in [0, {_ALPHA_MAX:g}]")
    knot_rate = scalar(k, "k")
    if not 0.5 <= knot_rate <= 1.0:
        raise ValueError("k must be between 0.5 and 1")
    boundary_tol = scalar(boundary_cif_tolerance, "boundary_cif_tolerance")
    if not 1e-12 <= boundary_tol <= 1e-3:
        raise ValueError("boundary_cif_tolerance must be in [1e-12, 1e-3]")
    tol = scalar(tolerance, "tolerance")
    if not 1e-9 <= tol <= 1e-3:
        raise ValueError("tolerance must be in [1e-9, 1e-3]")
    if isinstance(max_iterations, (bool, np.bool_)) or int(max_iterations) != max_iterations:
        raise ValueError("max_iterations must be an integer")
    if not 20 <= int(max_iterations) <= _MAX_ITERATIONS:
        raise ValueError(f"max_iterations must be in [20, {_MAX_ITERATIONS}]")

    event_upper = hi[code > 0]
    endpoints = np.r_[lo, event_upper]
    interior, boundaries = _choose_knots(endpoints, knot_rate)
    if not np.isfinite(boundaries).all() or boundaries[0] < 0.0:
        raise ValueError("finite nonnegative endpoint range required")
    x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
    corners = _corner_profiles(x_min, x_max)
    if corners.shape[0] > _MAX_CORNERS:
        raise ValueError(f"covariate range creates more than {_MAX_CORNERS} corners")
    if x.shape[1]:
        unit_scale = np.max(np.abs(x), axis=0)
        unit_scale = np.where(unit_scale > 0.0, unit_scale, 1.0)
        x_unit = x / unit_scale
        center_scaled = np.mean(x_unit, axis=0)
        scale_scaled = np.std(x_unit, axis=0)
        scale_scaled = np.where(scale_scaled > 0.0, scale_scaled, 1.0)
        xs = (x_unit - center_scaled) / scale_scaled
        cs = (corners / unit_scale - center_scaled) / scale_scaled
        mean = center_scaled * unit_scale
        scale = scale_scaled * unit_scale
        if not np.isfinite(scale).all():
            raise ValueError("covariate scale exceeds the supported numerical range")
    else:
        unit_scale = center_scaled = scale_scaled = mean = scale = np.empty(0, dtype=np.float64)
        xs, cs = x.copy(), corners.copy()
    # Normalize time to [0, 1]; knots and predictions are transformed back for metadata.
    span = boundaries[1] - boundaries[0]
    t_scale = span

    def norm(values: FloatArray) -> FloatArray:
        return (values - boundaries[0]) / t_scale

    internal_norm = norm(interior)
    boundary_norm = np.array([0.0, 1.0])
    n_basis = interior.size + 4
    p = xs.shape[1]
    work = lo.size * (2 * n_basis + 2 * p) + (corners.shape[0] + lo.size) * (2 * n_basis + 2 * p)
    if work > _MAX_WORK:
        raise ValueError(f"fit exceeds the {_MAX_WORK}-operation work budget")
    basis_lower = _basis(norm(lo), internal_norm, boundary_norm)
    upper_used = np.where(code > 0, hi, boundaries[1])
    basis_upper = _basis(norm(upper_used), internal_norm, boundary_norm)

    upper_basis = _basis(np.array([1.0]), internal_norm, boundary_norm)[0]
    lower_basis = _basis(np.array([0.0]), internal_norm, boundary_norm)[0]
    probability_slack = 1e-9
    inv_lower = np.array([_inverse_gor_cif(boundary_tol, float(z)) for z in a])
    constraints = _constraint_setup(
        a,
        n_basis,
        p,
        upper_basis,
        cs,
        xs,
        inv_lower,
        probability_slack,
    )
    initial = np.zeros(2 * n_basis + 2 * p, dtype=np.float64)
    phi0 = np.linspace(min(inv_lower) - 1.0, -3.0, n_basis)
    initial[:n_basis] = phi0
    initial[n_basis : 2 * n_basis] = phi0

    def objective(theta: FloatArray) -> tuple[float, FloatArray]:
        value, gradient, _ = _score_rows(
            theta, basis_lower, basis_upper, xs, lo, upper_used, code, a
        )
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            # SLSQP probes infeasible steps; a large finite loss lets its line
            # search return to the explicitly constrained feasible region.
            return 1e100, np.zeros(theta.shape, dtype=np.float64)
        return -value, -gradient

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=True,
        constraints=constraints,
        options={"maxiter": int(max_iterations), "ftol": tol, "disp": False},
    )
    theta = np.asarray(result.x, dtype=np.float64)
    loglik, score, row_scores = _score_rows(
        theta, basis_lower, basis_upper, xs, lo, upper_used, code, a, need_rows=True
    )
    constraint_functions = [
        cast(Callable[[FloatArray], FloatArray], constraint["fun"]) for constraint in constraints
    ]
    margins = np.concatenate(
        [np.asarray(fun(theta), dtype=np.float64) for fun in constraint_functions]
    )
    score_error = float(np.max(np.abs(score)))
    kkt_error = _kkt_error(theta, -score, constraints) / lo.size
    if not np.isfinite(theta).all() or not np.isfinite(loglik) or not np.isfinite(score_error):
        raise ArithmeticError("interval competing-risk optimization did not reach a finite fit")
    if np.min(margins) < -max(1e-7, tol * 10):
        raise ArithmeticError("interval competing-risk optimizer returned an infeasible fit")
    if not result.success and kkt_error > max(2e-5, tol * 100):
        raise ArithmeticError(
            "interval competing-risk optimization failed: "
            f"{result.message}; KKT error={kkt_error:.3g}"
        )
    if kkt_error > max(2e-5, tol * 100):
        raise ArithmeticError(
            f"interval competing-risk fit failed the constrained score check ({kkt_error:.3g})"
        )
    if row_scores is None:
        raise ArithmeticError("could not form individual likelihood scores")
    covariance = _score_covariance(row_scores, n_basis, p)
    phi = theta[: 2 * n_basis].reshape(2, n_basis)
    beta_scaled = theta[2 * n_basis :].reshape(2, p)
    beta = beta_scaled / scale[None, :] if p else beta_scaled
    if p:
        covariance = covariance / np.outer(np.tile(scale, 2), np.tile(scale, 2))
    lower_cif = np.empty(2, dtype=np.float64)
    for cause in range(2):
        eta = lower_basis[0] * phi[cause, 0] + cs @ beta_scaled[cause]
        lower_cif[cause] = np.max(_gor_cif(eta, float(a[cause]))[0])
    eta1 = upper_basis @ phi[0] + cs @ beta_scaled[0]
    eta2 = upper_basis @ phi[1] + cs @ beta_scaled[1]
    f1, _, _ = _gor_cif(eta1, float(a[0]))
    f2, _, _ = _gor_cif(eta2, float(a[1]))
    max_joint = float(np.max(f1 + f2))
    return IntervalCompetingRiskFit(
        coefficients=_freeze(beta.ravel()),
        covariance=_freeze(covariance),
        alpha=_freeze(a),
        baseline_coefficients=_freeze(phi),
        knots=_freeze(interior),
        boundary_knots=_freeze(boundaries),
        covariate_mean=_freeze(mean),
        covariate_scale=_freeze(scale),
        covariate_unit_scale=_freeze(unit_scale),
        covariate_center_scaled=_freeze(center_scaled),
        covariate_scale_scaled=_freeze(scale_scaled),
        covariate_min=_freeze(x_min),
        covariate_max=_freeze(x_max),
        log_likelihood=float(loglik),
        iterations=int(result.nit),
        score_error=score_error,
        kkt_error=kkt_error,
        lower_boundary_cif=_freeze(lower_cif),
        max_joint_cif=max_joint,
        boundary_cif_tolerance=float(boundary_tol),
        converged=bool(kkt_error <= max(2e-5, tol * 100)),
    )


def predict_interval_competing_risk(
    fit: IntervalCompetingRiskFit,
    times: ArrayLike,
    profiles: ArrayLike | None = None,
) -> IntervalCompetingRiskPrediction:
    """Predict both CIFs within the fitted endpoint range."""
    t = finite(times, "times")
    if t.ndim != 1 or not 1 <= t.size <= 100_000:
        raise ValueError("times must be a nonempty vector with at most 100000 values")
    if np.any(t < fit.boundary_knots[0]) or np.any(t > fit.boundary_knots[1]):
        raise ValueError("prediction times must lie within the fitted endpoint range")
    p = fit.covariate_mean.size
    if profiles is None:
        x = fit.covariate_mean[None, :]
    else:
        x = finite(profiles, "profiles")
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != p or x.shape[0] == 0 or x.shape[0] > 100_000:
            raise ValueError("profiles must have one column per covariate and 1..100000 rows")
    n_basis = fit.knots.size + 4
    output_work = (
        8 * x.shape[0] * t.size + x.size + t.size + t.size * n_basis + x.shape[0] * n_basis
    )
    if output_work > _MAX_OUTPUT_CELLS:
        raise ValueError("prediction exceeds the 2000000-cell combined memory budget")
    span = fit.boundary_knots[1] - fit.boundary_knots[0]
    basis = _basis(
        (t - fit.boundary_knots[0]) / span,
        (fit.knots - fit.boundary_knots[0]) / span,
        np.array([0.0, 1.0]),
    )
    xs = (
        (x / fit.covariate_unit_scale - fit.covariate_center_scaled) / fit.covariate_scale_scaled
        if p
        else x.copy()
    )
    beta_scaled = (
        fit.coefficients.reshape(2, p) * fit.covariate_scale[None, :]
        if p
        else fit.coefficients.reshape(2, 0)
    )
    baseline = fit.baseline_coefficients
    cif = np.empty((x.shape[0], t.size, 2), dtype=np.float64)
    for cause in range(2):
        eta = basis @ baseline[cause]
        eta = eta[None, :] + (xs @ beta_scaled[cause])[:, None]
        cif[:, :, cause] = _gor_cif(eta, float(fit.alpha[cause]))[0]
    terminal_basis = _basis(
        np.array([1.0]),
        (fit.knots - fit.boundary_knots[0]) / span,
        np.array([0.0, 1.0]),
    )[0]
    terminal_eta = np.column_stack(
        (
            terminal_basis @ baseline[0] + xs @ beta_scaled[0],
            terminal_basis @ baseline[1] + xs @ beta_scaled[1],
        )
    )
    terminal_cif = np.column_stack(
        [_gor_cif(terminal_eta[:, cause], float(fit.alpha[cause]))[0] for cause in range(2)]
    )
    if not np.isfinite(cif).all() or not np.isfinite(terminal_cif).all():
        raise ArithmeticError("prediction exceeded the finite probability range")
    if np.any(terminal_cif.sum(axis=1) > 1.0 + 1e-8):
        raise ValueError("requested profile violates the fitted joint-probability constraint")
    return IntervalCompetingRiskPrediction(
        profile=_freeze(x),
        times=_freeze(t),
        cif1=_freeze(cif[:, :, 0]),
        cif2=_freeze(cif[:, :, 1]),
    )
