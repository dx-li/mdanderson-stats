"""Explicit Python REML initialization for WFMM variance components.

This is an opt-in likelihood-based initializer for the covariance model used
by :mod:`wfmm_model`. It does not infer the native ``delta_omega`` inverse-
gamma prior or claim parity with the unavailable Windows initialization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import OptimizeResult, minimize

from ._validation import FloatArray
from .wfmm_model import (
    _MAX_COEFFICIENTS,
    _MAX_DESIGN_CELLS,
    _MAX_FIXED_EFFECTS,
    _MAX_RANDOM_EFFECTS,
    _MAX_ROWS,
    _finite_real,
    _marginal_covariance,
    _strata,
)

_MAX_BASIS_CELLS = 8_000_000
_MAX_RANK_WORK = 50_000_000
_MAX_INITIALIZATION_WORK = 250_000_000
_LOG_RATIO_BOUNDS = (-18.420680743952367, 18.420680743952367)  # log(1e-8), log(1e8)


def _frozen(value: ArrayLike) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    array.flags.writeable = False
    return array


def _frozen_bool(value: ArrayLike) -> NDArray[np.bool_]:
    array = np.array(value, dtype=np.bool_, copy=True)
    array.flags.writeable = False
    return array


def _frozen_int(value: ArrayLike) -> NDArray[np.int64]:
    array = np.array(value, dtype=np.int64, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class WFMMVarianceInitialization:
    """Raw REML estimates and positive sampler starts, indexed by component/K.

    ``random_variance`` and ``residual_variance`` are the positive starts for
    ``fit_wfmm_coefficients`` when ``starts_usable`` is true. Their raw REML
    estimates are retained separately; zero lower-bound estimates are lifted
    by the documented scale-relative floor. The floor is a Python numerical
    policy and does not define an inverse-gamma prior.
    """

    random_variance: FloatArray
    residual_variance: FloatArray
    raw_random_variance: FloatArray
    raw_residual_variance: FloatArray
    fixed_effect_estimates: FloatArray
    data_scale: FloatArray
    normalized_residual_norm: FloatArray
    floor_applied_random: NDArray[np.bool_]
    floor_applied_residual: NDArray[np.bool_]
    lower_bound_random: NDArray[np.bool_]
    lower_bound_residual: NDArray[np.bool_]
    upper_bound_random: NDArray[np.bool_]
    upper_bound_residual: NDArray[np.bool_]
    restricted_log_likelihood: FloatArray
    optimizer_success: NDArray[np.bool_]
    optimizer_message: tuple[str, ...]
    iterations: NDArray[np.int64]
    evaluations: NDArray[np.int64]
    status: tuple[str, ...]
    floor_ratio: float
    starts_usable: bool


def _component_basis_rank(
    x: FloatArray,
    z: FloatArray,
    random_groups: NDArray[np.int64],
    residual_groups: NDArray[np.int64],
) -> None:
    """Reject covariance components aliased after projecting out fixed effects."""
    rows, fixed_count = x.shape
    residual_df = rows - fixed_count
    component_count = (
        (int(random_groups.max()) + 1 if random_groups.size else 0) + int(residual_groups.max()) + 1
    )
    cells = residual_df * residual_df * component_count
    if cells > _MAX_BASIS_CELLS:
        raise ValueError("projected variance-component rank check exceeds the 8000000-cell limit")
    q_full, _ = np.linalg.qr(x, mode="complete")
    q = q_full[:, fixed_count:]
    bases = np.empty((residual_df * residual_df, component_count), dtype=np.float64)
    column = 0
    if z.shape[1]:
        for group in range(int(random_groups.max()) + 1):
            columns = random_groups == group
            projected = q.T @ z[:, columns]
            basis = projected @ projected.T
            norm = float(np.linalg.norm(basis))
            design_norm = float(np.linalg.norm(z[:, columns]))
            if norm <= 1e-10 * design_norm * design_norm or not np.isfinite(norm):
                raise ValueError("a random variance component is not identifiable under REML")
            bases[:, column] = (basis / norm).ravel()
            column += 1
    for group in range(int(residual_groups.max()) + 1):
        selected = residual_groups == group
        projected = q[selected, :]
        basis = projected.T @ projected
        norm = float(np.linalg.norm(basis))
        if norm <= 1e-10 * np.sqrt(float(np.count_nonzero(selected))) or not np.isfinite(norm):
            raise ValueError("a residual variance component is not identifiable under REML")
        bases[:, column] = (basis / norm).ravel()
        column += 1
    singular_values = np.linalg.svd(bases, compute_uv=False)
    if (
        singular_values.size != component_count
        or singular_values[0] == 0.0
        or singular_values[-1] <= singular_values[0] * 1e-10
    ):
        raise ValueError(
            "variance components are nonidentifiable in the fixed-effect residual space"
        )


def _reml_fit(
    y: FloatArray,
    x: FloatArray,
    z: FloatArray,
    random_groups: NDArray[np.int64],
    residual_groups: NDArray[np.int64],
    component_average_diagonal: FloatArray,
    *,
    max_iterations: int,
    max_evaluations: int,
    tolerance: float,
) -> tuple[OptimizeResult, FloatArray, FloatArray, float, int]:
    rows, fixed_count = x.shape
    random_count = int(random_groups.max()) + 1 if random_groups.size else 0
    residual_count = int(residual_groups.max()) + 1
    component_count = random_count + residual_count

    def evaluate(log_ratio: FloatArray) -> tuple[float, FloatArray, float]:
        relative = np.exp(log_ratio) / (component_count * component_average_diagonal)
        random_variance = relative[:random_count, None]
        residual_variance = relative[random_count:, None]
        covariance = _marginal_covariance(
            z,
            random_groups,
            random_variance,
            residual_groups,
            residual_variance,
            0,
        )
        try:
            factor = cho_factor(covariance, lower=True, check_finite=False)
            inv_y = cho_solve(factor, y, check_finite=False)
            inv_x = cho_solve(factor, x, check_finite=False)
            information = x.T @ inv_x
            info_factor = cho_factor(information, lower=True, check_finite=False)
            beta = cho_solve(info_factor, x.T @ inv_y, check_finite=False)
        except np.linalg.LinAlgError as exc:
            raise ArithmeticError("REML covariance or information matrix is singular") from exc
        residual = y - x @ beta
        inv_residual = cho_solve(factor, residual, check_finite=False)
        logdet_covariance = 2.0 * float(np.log(np.diag(factor[0])).sum())
        logdet_information = 2.0 * float(np.log(np.diag(info_factor[0])).sum())
        quadratic = float(residual @ inv_residual)
        objective = logdet_covariance + logdet_information + quadratic
        if not np.isfinite(objective) or not np.isfinite(beta).all():
            raise ArithmeticError("REML objective or fixed-effect estimate is nonfinite")
        return objective, beta, quadratic

    evaluation_count = 0
    best_value = np.inf
    best_ratio = np.zeros(component_count, dtype=np.float64)
    best_beta = np.zeros(fixed_count, dtype=np.float64)

    def objective(log_ratio: FloatArray) -> float:
        nonlocal evaluation_count, best_value, best_ratio, best_beta
        if evaluation_count >= max_evaluations:
            return best_value if np.isfinite(best_value) else 1e100
        evaluation_count += 1
        try:
            value, beta, _ = evaluate(log_ratio)
        except ArithmeticError:
            return 1e100
        if value < best_value:
            best_value = value
            best_ratio = np.array(log_ratio, copy=True)
            best_beta = beta
        return value

    result = minimize(
        objective,
        np.zeros(component_count, dtype=np.float64),
        method="L-BFGS-B",
        bounds=[_LOG_RATIO_BOUNDS] * component_count,
        options={
            "maxiter": max_iterations,
            "maxfun": max_evaluations,
            "ftol": tolerance,
            "gtol": tolerance,
            "maxls": 30,
        },
    )
    if not np.isfinite(best_value):
        raise ArithmeticError("REML optimizer did not evaluate a valid covariance")
    if evaluation_count >= max_evaluations and result.nfev >= max_evaluations:
        result.success = False
        result.message = "maximum REML likelihood-evaluation budget reached"
    df = rows - fixed_count
    restricted_log_likelihood = -0.5 * (df * np.log(2.0 * np.pi) + best_value)
    return result, best_ratio, best_beta, restricted_log_likelihood, evaluation_count


def initialize_wfmm_variances(
    coefficients: ArrayLike,
    fixed_design: ArrayLike,
    random_design: ArrayLike | None = None,
    *,
    random_strata: ArrayLike | None = None,
    residual_strata: ArrayLike | None = None,
    positive_floor_ratio: float = 1e-8,
    zero_data_floor: float | None = None,
    max_iterations: int = 500,
    max_evaluations: int = 2_000,
    tolerance: float = 1e-9,
) -> WFMMVarianceInitialization:
    """Initialize WFMM random/residual variances by bounded per-coefficient REML.

    REML is applied to the implemented covariance
    ``sum_g(q_g Z_g Z_g.T) + diag(s[residual_strata])``. Variance parameters
    are optimized independently for each coefficient, in sequence. This is a
    Python opt-in initialization policy, not the unspecified native
    ``delta_omega`` prior/default. No inverse-gamma hyperparameters are created.

    Nonzero data use a relative positive floor ``positive_floor_ratio *
    max(abs(y))**2`` only in sampler starts; raw REML values and floor flags
    remain available. All-zero data have no intrinsic variance scale, so they
    yield zero raw/start values and ``starts_usable=False`` unless an explicit
    positive absolute ``zero_data_floor`` is supplied.
    """
    raw_y, raw_x = np.asarray(coefficients), np.asarray(fixed_design)
    if (
        raw_y.ndim != 2
        or not 1 <= raw_y.shape[0] <= _MAX_ROWS
        or not 1 <= raw_y.shape[1] <= _MAX_COEFFICIENTS
    ):
        raise ValueError("coefficients must be an N-by-K matrix within supported limits")
    rows, coefficient_count = raw_y.shape
    if raw_x.ndim != 2 or raw_x.shape[0] != rows or not 1 <= raw_x.shape[1] <= _MAX_FIXED_EFFECTS:
        raise ValueError("fixed_design must be an N-by-P matrix within supported limits")
    fixed_count = raw_x.shape[1]
    if rows <= fixed_count:
        raise ValueError("REML initialization requires positive residual degrees of freedom N-P")
    if random_design is None:
        z = np.empty((rows, 0), dtype=np.float64)
    else:
        raw_z = np.asarray(random_design)
        if raw_z.ndim != 2 or raw_z.shape[0] != rows or raw_z.shape[1] > _MAX_RANDOM_EFFECTS:
            raise ValueError("random_design must be N-by-M with at most 500 columns")
        z = _finite_real(raw_z, "random_design")
    if rows * (fixed_count + z.shape[1] + coefficient_count) > _MAX_DESIGN_CELLS:
        raise ValueError("WFMM initializer design exceeds the 2000000-cell limit")
    y, x = _finite_real(raw_y, "coefficients"), _finite_real(raw_x, "fixed_design")
    random_groups = (
        _strata(random_strata, z.shape[1], "random_strata")
        if z.shape[1]
        else np.empty(0, dtype=np.int64)
    )
    if z.shape[1] == 0 and random_strata is not None:
        raise ValueError("random_strata requires a nonempty random_design")
    residual_groups = _strata(residual_strata, rows, "residual_strata")
    random_count = int(random_groups.max()) + 1 if random_groups.size else 0
    residual_count = int(residual_groups.max()) + 1
    component_count = random_count + residual_count
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or not 1 <= int(max_iterations) <= 10_000
    ):
        raise ValueError("max_iterations must be an integer in [1,10000]")
    if (
        isinstance(max_evaluations, (bool, np.bool_))
        or not isinstance(max_evaluations, (int, np.integer))
        or not 1 <= int(max_evaluations) <= 10_000
    ):
        raise ValueError("max_evaluations must be an integer in [1,10000]")
    floor_ratio = float(positive_floor_ratio)
    if not np.isfinite(floor_ratio) or not 0.0 < floor_ratio < 1.0:
        raise ValueError("positive_floor_ratio must be finite and lie in (0,1)")
    zero_floor = None if zero_data_floor is None else float(zero_data_floor)
    if zero_floor is not None and (not np.isfinite(zero_floor) or zero_floor <= 0.0):
        raise ValueError("zero_data_floor must be finite and positive")
    tol = float(tolerance)
    if not np.isfinite(tol) or not 0.0 < tol < 1.0:
        raise ValueError("tolerance must be finite and lie in (0,1)")

    rank_cells = (rows - fixed_count) ** 2 * component_count
    rank_work = rank_cells * component_count
    analytic_residual_only = random_count == 0 and residual_count == 1
    work = (
        0
        if analytic_residual_only
        else coefficient_count
        * int(max_evaluations)
        * (rows**3 + fixed_count**3 + component_count * rows**2)
    )
    if rank_cells > _MAX_BASIS_CELLS:
        raise ValueError("projected covariance-component rank check exceeds the 8000000-cell limit")
    if rank_work > _MAX_RANK_WORK:
        raise ValueError("projected covariance-component rank check exceeds the bounded work limit")
    if work > _MAX_INITIALIZATION_WORK:
        raise ValueError("WFMM REML initialization exceeds the bounded work limit")

    x_scale = np.max(np.abs(x), axis=0)
    if np.any(x_scale == 0.0):
        raise ValueError("fixed_design must have full column rank")
    scaled_x = x / x_scale
    column_norm = np.sqrt(np.sum(scaled_x**2, axis=0))
    log_normalizer = np.log(x_scale) + np.log(column_norm)
    scaled_x /= column_norm
    if np.linalg.matrix_rank(scaled_x) != fixed_count:
        raise ValueError("fixed_design must have full column rank")
    if np.any(y != 0.0):
        _component_basis_rank(scaled_x, z, random_groups, residual_groups)

    raw_random = np.zeros((random_count, coefficient_count), dtype=np.float64)
    raw_residual = np.zeros((residual_count, coefficient_count), dtype=np.float64)
    start_random = np.zeros_like(raw_random)
    start_residual = np.zeros_like(raw_residual)
    fixed = np.empty((fixed_count, coefficient_count), dtype=np.float64)
    scales = np.empty(coefficient_count, dtype=np.float64)
    residual_norms = np.zeros(coefficient_count, dtype=np.float64)
    ll = np.full(coefficient_count, np.inf, dtype=np.float64)
    success = np.zeros(coefficient_count, dtype=np.bool_)
    iterations = np.zeros(coefficient_count, dtype=np.int64)
    evaluations = np.zeros(coefficient_count, dtype=np.int64)
    status: list[str] = []
    messages: list[str] = []
    lower = np.zeros((component_count, coefficient_count), dtype=np.bool_)
    upper = np.zeros_like(lower)
    floored = np.zeros_like(lower)
    for coefficient in range(coefficient_count):
        values = y[:, coefficient]
        observed_scale = float(np.max(np.abs(values)))
        scales[coefficient] = observed_scale
        if observed_scale == 0.0:
            fixed[:, coefficient] = 0.0
            ll[coefficient] = np.nan
            if zero_floor is not None:
                start_random[:, coefficient] = zero_floor
                start_residual[:, coefficient] = zero_floor
                floored[:, coefficient] = True
                lower[:, coefficient] = True
                status.append("zero_data_absolute_floor")
            else:
                status.append("zero_data_no_scale")
            messages.append("all observations are zero; covariance scale is unidentified")
            continue
        variance_scale = observed_scale * observed_scale
        if not np.isfinite(variance_scale) or variance_scale == 0.0:
            raise ArithmeticError("observed coefficient scale cannot be squared representably")
        normalized_y = values / observed_scale
        beta_ols, *_ = np.linalg.lstsq(scaled_x, normalized_y, rcond=None)
        ols_residual = normalized_y - scaled_x @ beta_ols
        residual_norms[coefficient] = float(np.linalg.norm(ols_residual))
        numerical_zero_tolerance = (
            8.0
            * np.finfo(np.float64).eps
            * max(rows, fixed_count)
            * max(1.0, float(np.linalg.norm(normalized_y)))
        )
        if residual_norms[coefficient] <= numerical_zero_tolerance:
            fixed[:, coefficient] = (observed_scale / x_scale) * (beta_ols / column_norm)
            floored[:, coefficient] = True
            floor = floor_ratio * variance_scale
            if floor == 0.0 or not np.isfinite(floor):
                raise ArithmeticError("relative sampler-start floor is not representable")
            start_random[:, coefficient] = floor
            start_residual[:, coefficient] = floor
            lower[:, coefficient] = True
            status.append("numerical_zero_residual_boundary")
            ll[coefficient] = np.nan
            messages.append(
                "residual norm is at linear-algebra roundoff scale; variance boundary is numerical"
            )
            continue

        if random_count == 0 and residual_count == 1:
            residual_sum_squares = float(ols_residual @ ols_residual)
            normalized_variance = residual_sum_squares / (rows - fixed_count)
            variance = normalized_variance * variance_scale
            floor = floor_ratio * variance_scale
            if (
                not np.isfinite(variance)
                or variance <= 0.0
                or floor <= 0.0
                or not np.isfinite(floor)
            ):
                raise ArithmeticError("analytic residual-only REML variance is not representable")
            raw_residual[0, coefficient] = variance
            start_residual[0, coefficient] = max(variance, floor)
            floored[0, coefficient] = variance < floor
            fixed[:, coefficient] = (observed_scale / x_scale) * (beta_ols / column_norm)
            sign, logdet_information = np.linalg.slogdet(scaled_x.T @ scaled_x)
            if sign <= 0.0 or not np.isfinite(logdet_information):
                raise ArithmeticError("fixed-effect information determinant is invalid")
            df = rows - fixed_count
            ll[coefficient] = -0.5 * (
                df * np.log(2.0 * np.pi)
                + logdet_information
                + df * (np.log(normalized_variance) + np.log(variance_scale) + 1.0)
            ) - float(log_normalizer.sum())
            success[coefficient] = True
            status.append("analytic_residual_only_reml")
            messages.append("closed-form restricted maximum-likelihood estimate")
            continue

        component_average = np.empty(component_count, dtype=np.float64)
        position = 0
        if z.shape[1]:
            for group in range(random_count):
                block = z[:, random_groups == group]
                component_average[position] = float(np.sum(block * block) / rows)
                position += 1
        for group in range(residual_count):
            component_average[position] = float(np.mean(residual_groups == group))
            position += 1
        result, log_ratios, beta_scaled, restricted, evaluation_count = _reml_fit(
            normalized_y,
            scaled_x,
            z,
            random_groups,
            residual_groups,
            component_average,
            max_iterations=int(max_iterations),
            max_evaluations=int(max_evaluations),
            tolerance=tol,
        )
        relative_variances = (
            np.exp(log_ratios) * variance_scale / (component_count * component_average)
        )
        if not np.isfinite(relative_variances).all():
            raise ArithmeticError("estimated variance components are not representable")
        at_lower = log_ratios <= _LOG_RATIO_BOUNDS[0] + 1e-5
        at_upper = log_ratios >= _LOG_RATIO_BOUNDS[1] - 1e-5
        raw = relative_variances.copy()
        start = np.maximum(raw, floor_ratio * variance_scale)
        if not np.isfinite(start).all() or np.any(start <= 0.0):
            raise ArithmeticError("positive variance starts are not representable")
        raw_random[:, coefficient] = raw[:random_count]
        raw_residual[:, coefficient] = raw[random_count:]
        start_random[:, coefficient] = start[:random_count]
        start_residual[:, coefficient] = start[random_count:]
        lower[:, coefficient], upper[:, coefficient] = at_lower, at_upper
        floored[:, coefficient] = raw < start
        fixed[:, coefficient] = (observed_scale / x_scale) * (beta_scaled / column_norm)
        ll[coefficient] = (
            restricted
            - 0.5 * (rows - fixed_count) * np.log(variance_scale)
            - float(log_normalizer.sum())
        )
        success[coefficient] = bool(result.success)
        iterations[coefficient] = int(result.nit)
        evaluations[coefficient] = evaluation_count
        status.append("optimized" if result.success else "optimizer_limit_or_failure")
        messages.append(str(result.message))

    floor_random = floored[:random_count]
    floor_residual = floored[random_count:]
    starts_usable = bool(np.all(start_random > 0.0) and np.all(start_residual > 0.0))
    return WFMMVarianceInitialization(
        _frozen(start_random),
        _frozen(start_residual),
        _frozen(raw_random),
        _frozen(raw_residual),
        _frozen(fixed),
        _frozen(scales),
        _frozen(residual_norms),
        _frozen_bool(floor_random),
        _frozen_bool(floor_residual),
        _frozen_bool(lower[:random_count]),
        _frozen_bool(lower[random_count:]),
        _frozen_bool(upper[:random_count]),
        _frozen_bool(upper[random_count:]),
        _frozen(ll),
        _frozen_bool(success),
        tuple(messages),
        _frozen_int(iterations),
        _frozen_int(evaluations),
        tuple(status),
        floor_ratio,
        starts_usable,
    )
