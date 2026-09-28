"""Caret-style radial epsilon-SVR refinement for CondiS-X imputations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite, scalar
from .boin import _owned
from .condis import CondiSImputation
from .condis_regularized import _make_folds


@dataclass(frozen=True)
class CondiSSVMRefinement:
    """Tuned radial SVR result and cross-validation diagnostics."""

    fitted_time: FloatArray
    refined_time: FloatArray
    below_censoring: NDArray[np.bool_]
    above_horizon: NDArray[np.bool_]
    sigma: float
    sigma_pair_indices: NDArray[np.int64]
    cost_grid: FloatArray
    fold_rmse: FloatArray
    mean_rmse: FloatArray
    sd_rmse: FloatArray
    fold_kkt_error: FloatArray
    fold_duality_gap: FloatArray
    fold_standardized: NDArray[np.bool_]
    best_index: int
    best_cost: float
    final_kkt_error: float
    final_duality_gap: float
    final_primal_objective: float
    final_dual_objective: float
    final_iterations: int
    support_indices: NDArray[np.int64]
    dual_coefficients: FloatArray
    intercept: float
    response_center: float
    response_scale: float
    predictor_magnitude: FloatArray
    predictor_center: FloatArray
    predictor_scale: FloatArray
    standardized: bool
    fold_ids: NDArray[np.int64]
    random_state: int | None
    enforce_censoring: bool


@dataclass(frozen=True)
class _SVRSolution:
    prediction: FloatArray
    support_indices: NDArray[np.int64]
    dual_coefficients: FloatArray
    intercept: float
    kkt_error: float
    duality_gap: float
    iterations: int
    primal_objective: float
    dual_objective: float
    response_center: float
    response_scale: float
    predictor_magnitude: FloatArray
    predictor_center: FloatArray
    predictor_scale: FloatArray
    standardized: bool


def _readonly_int(values: NDArray[np.int64]) -> NDArray[np.int64]:
    result = np.array(values, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _design(imputation: CondiSImputation, covariates: ArrayLike) -> tuple[FloatArray, FloatArray]:
    if not isinstance(imputation, CondiSImputation):
        raise TypeError("imputation must be CondiSImputation")
    x = finite(covariates, "covariates")
    n = imputation.imputed_time.size
    if x.ndim != 2 or x.shape[0] != n or not 1 <= x.shape[1] <= 500 or x.size > 2_000_000:
        raise ValueError("covariates must have n rows, 1..500 columns and <=2 million cells")
    # The original formula is pred_time ~ ., so status is an input feature.
    design = np.column_stack((imputation.status.astype(np.float64), x))
    if not np.isfinite(design).all():
        raise ValueError("predictor matrix must contain only finite real values")
    return design, imputation.imputed_time


def _stable_center_scale(values: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Scale columns with sample SD while avoiding overflow in raw units."""
    magnitude = np.max(np.abs(values), axis=0)
    safe_magnitude = np.where(magnitude > 0.0, magnitude, 1.0)
    normalized = values / safe_magnitude
    center = normalized.mean(axis=0)
    centered = normalized - center
    sd_normalized = np.sqrt(np.sum(centered * centered, axis=0) / (values.shape[0] - 1))
    variable = sd_normalized > 0.0
    return normalized, center, np.where(variable, sd_normalized, 0.0)


def _standardize_for_fit(
    x_train: FloatArray, y_train: FloatArray, x_test: FloatArray
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    float,
    float,
    FloatArray,
    FloatArray,
    FloatArray,
    bool,
]:
    magnitude = np.max(np.abs(x_train), axis=0)
    safe_magnitude = np.where(magnitude > 0.0, magnitude, 1.0)
    xnorm, xcenter, xsd = _stable_center_scale(x_train)
    variable = xsd > 0.0
    # kernlab disables predictor AND response scaling if any column is constant.
    if not np.all(variable):
        return (
            x_train.copy(),
            y_train.copy(),
            x_test.copy(),
            0.0,
            1.0,
            np.ones(x_train.shape[1]),
            np.zeros(x_train.shape[1]),
            np.ones(x_train.shape[1]),
            False,
        )
    xs = (xnorm - xcenter) / xsd
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        xtest_norm = x_test / np.max(np.abs(x_train), axis=0)
        xte = (xtest_norm - xcenter) / xsd
    if not np.isfinite(xte).all():
        raise ArithmeticError("standardized test predictors are not finite")
    ymagnitude = float(np.max(np.abs(y_train)))
    if ymagnitude == 0.0:
        return xs, np.zeros_like(y_train), xte, 0.0, 1.0, safe_magnitude, xcenter, xsd, True
    ynorm = y_train / ymagnitude
    ycenter_norm = float(ynorm.mean())
    ysd_norm = float(np.sqrt(np.sum((ynorm - ycenter_norm) ** 2) / (y_train.size - 1)))
    if ysd_norm == 0.0:
        return (
            xs,
            np.zeros_like(y_train),
            xte,
            ycenter_norm * ymagnitude,
            ymagnitude,
            safe_magnitude,
            xcenter,
            xsd,
            True,
        )
    yscale = ymagnitude * ysd_norm
    return (
        xs,
        (ynorm - ycenter_norm) / ysd_norm,
        xte,
        ycenter_norm * ymagnitude,
        yscale,
        safe_magnitude,
        xcenter,
        xsd,
        True,
    )


def _scaled_for_sigma(x: FloatArray) -> FloatArray:
    normalized, center, scale = _stable_center_scale(x)
    centered = normalized - center
    if np.any(scale == 0.0):
        # Native sigest abandons scaling for every column if any is constant.
        return x.copy()
    return centered / scale


def _pairs(
    n: int,
    pair_indices: ArrayLike | None,
    random_state: int | None,
) -> NDArray[np.int64]:
    pair_count = n // 2
    if pair_indices is None:
        rng = np.random.default_rng(random_state)
        return rng.integers(0, n, size=(pair_count, 2), dtype=np.int64)
    raw = np.asarray(pair_indices)
    if (
        np.iscomplexobj(raw)
        or raw.dtype.kind not in "iu"
        or raw.dtype.kind == "b"
        or raw.shape != (pair_count, 2)
    ):
        raise ValueError("sigma_pair_indices must be integer row pairs with shape (n//2, 2)")
    result = raw.astype(np.int64, copy=True)
    if np.any(result < 0) or np.any(result >= n):
        raise ValueError("sigma_pair_indices contains an out-of-range row")
    return result


def _estimate_sigma(x: FloatArray, pair_indices: NDArray[np.int64]) -> float:
    paired = _scaled_for_sigma(x)
    left = paired[pair_indices[:, 0]]
    right = paired[pair_indices[:, 1]]
    with np.errstate(over="ignore", invalid="ignore"):
        delta = left - right
    if np.isfinite(delta).all() and np.max(np.abs(delta), initial=0.0) < np.sqrt(
        np.finfo(float).max / max(1, x.shape[1])
    ):
        distances = np.einsum("ij,ij->i", delta, delta)
        positive = distances > 0.0
        if not np.any(positive):
            raise ValueError(
                "sigma estimation found no positive sampled pair distance; supply sigma"
            )
        upper = float(np.quantile(distances[positive], 0.9))
        lower = float(np.quantile(distances[positive], 0.1))
        if lower <= 0.0 or upper <= 0.0:
            raise ValueError("sigma estimation has a zero distance quantile; supply sigma")
        estimate = 0.5 / upper + 0.5 / lower
    else:
        log_distances = np.full(delta.shape[0], -np.inf, dtype=np.float64)
        for row in range(delta.shape[0]):
            if np.isfinite(delta[row]).all():
                distance_scale = float(np.max(np.abs(delta[row]), initial=0.0))
                if distance_scale == 0.0:
                    continue
                normalized_delta = delta[row] / distance_scale
            else:
                distance_scale = float(
                    max(
                        np.max(np.abs(left[row]), initial=0.0),
                        np.max(np.abs(right[row]), initial=0.0),
                    )
                )
                if distance_scale == 0.0 or not np.isfinite(distance_scale):
                    continue
                normalized_delta = left[row] / distance_scale - right[row] / distance_scale
            normalized_squared = float(normalized_delta @ normalized_delta)
            if normalized_squared > 0.0:
                log_distances[row] = 2.0 * np.log(distance_scale) + np.log(normalized_squared)
        positive = np.isfinite(log_distances)
        if not np.any(positive):
            raise ValueError(
                "sigma estimation found no positive sampled pair distance; supply sigma"
            )
        sorted_logs = np.sort(log_distances[positive])
        log_upper = _log_quantile_type7(sorted_logs, 0.9)
        log_lower = _log_quantile_type7(sorted_logs, 0.1)
        log_sigma = float(np.logaddexp(-log_upper, -log_lower) - np.log(2.0))
        if log_sigma < np.log(np.nextafter(0.0, 1.0)) or log_sigma > np.log(np.finfo(float).max):
            raise ValueError("sigma estimate is outside float64 range; supply sigma")
        estimate = float(np.exp(log_sigma))
    if not np.isfinite(estimate) or estimate <= 0.0:
        raise ValueError("sigma estimate is not finite and positive; supply sigma")
    return estimate


def _log_quantile_type7(sorted_log_values: FloatArray, probability: float) -> float:
    """R type-7 quantile of positive values represented by their logarithms."""
    if sorted_log_values.size == 1:
        return float(sorted_log_values[0])
    position = (sorted_log_values.size - 1) * probability
    lower_index = int(np.floor(position))
    fraction = position - lower_index
    lower = float(sorted_log_values[lower_index])
    if fraction == 0.0:
        return lower
    upper = float(sorted_log_values[lower_index + 1])
    return float(np.logaddexp(lower + np.log1p(-fraction), upper + np.log(fraction)))


def _rbf_kernel(x1: FloatArray, x2: FloatArray, sigma: float) -> FloatArray:
    # Featurewise accumulation retains close differences under large common
    # offsets without allocating an n×m×p tensor.
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("RBF predictor widths must match")
    distance = np.zeros((x1.shape[0], x2.shape[0]), dtype=np.float64)
    root_sigma = float(np.sqrt(sigma))
    for column in range(x1.shape[1]):
        left = x1[:, column, None]
        right = x2[None, :, column]
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            delta = left - right
            scaled = delta * root_sigma
            # If subtraction overflowed, scaling before subtracting can still
            # recover a finite kernel distance for very small sigma.
            overflow = ~np.isfinite(scaled) & np.isfinite(left) & np.isfinite(right)
            if np.any(overflow):
                scaled[overflow] = (left * root_sigma - right * root_sigma)[overflow]
            distance += scaled * scaled
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.exp(-distance)
    if not np.isfinite(result).all():
        raise ArithmeticError("RBF kernel produced non-finite values")
    return result


def _working_pair(
    beta_gradient: FloatArray,
    alpha: FloatArray,
    signs: FloatArray,
    cost: float,
) -> tuple[int, int, float]:
    n = signs.size // 2
    h = -signs * beta_gradient
    increase = np.r_[(alpha[:n] < cost), (alpha[n:] > 0.0)]
    decrease = np.r_[(alpha[:n] > 0.0), (alpha[n:] < cost)]
    eligible_i = np.flatnonzero(increase)
    eligible_j = np.flatnonzero(decrease)
    if eligible_i.size == 0 or eligible_j.size == 0:
        return -1, -1, 0.0
    i = int(eligible_i[np.argmax(h[eligible_i])])
    j = int(eligible_j[np.argmin(h[eligible_j])])
    if i == j:
        return i, j, 0.0
    return i, j, float(h[i] - h[j])


def _intercept_from_solution(
    gradient: FloatArray,
    signs: FloatArray,
    alpha: FloatArray,
    cost: float,
) -> float:
    y_gradient = signs * gradient
    bound_tol = max(np.finfo(float).eps * cost * 32.0, np.nextafter(0.0, 1.0))
    free = (alpha > bound_tol) & (alpha < cost - bound_tol)
    if np.any(free):
        rho = float(np.mean(y_gradient[free]))
        return -rho
    positive = signs > 0
    at_upper = alpha >= cost - bound_tol
    at_lower = alpha <= bound_tol
    lower_rho = max(
        np.max(y_gradient[at_upper & positive], initial=-np.inf),
        np.max(y_gradient[at_lower & ~positive], initial=-np.inf),
    )
    upper_rho = np.min(y_gradient[at_upper & ~positive], initial=np.inf)
    upper_rho = min(upper_rho, np.min(y_gradient[at_lower & positive], initial=np.inf))
    rho = 0.5 * lower_rho + 0.5 * upper_rho
    if not np.isfinite(rho):
        raise ArithmeticError("SVR solution does not identify a finite intercept")
    return -float(rho)


def _fit_scaled(
    x_train: FloatArray,
    y_train: FloatArray,
    x_test: FloatArray,
    cost: float,
    sigma: float,
    epsilon: float,
    tolerance: float,
    max_iterations: int,
) -> _SVRSolution:
    n = y_train.size
    if n * n > 1_000_000:
        raise ValueError("SVR fit exceeds the one-million-cell kernel budget")
    kernel = _rbf_kernel(x_train, x_train, sigma)
    # Dual variables have signs +1 for alpha+ and -1 for alpha-.  A pair
    # update preserves sum(sign * alpha)=0 and changes beta at two rows.
    signs = np.r_[np.ones(n), -np.ones(n)]
    alpha = np.zeros(2 * n, dtype=np.float64)
    beta = np.zeros(n, dtype=np.float64)
    kernel_beta = np.zeros(n, dtype=np.float64)
    iterations = 0
    while iterations < max_iterations:
        gradient = signs * np.r_[kernel_beta - y_train, kernel_beta - y_train] + epsilon
        i, j, violation = _working_pair(gradient, alpha, signs, cost)
        if violation <= tolerance or i < 0 or j < 0:
            break
        if i == j:
            raise ArithmeticError("epsilon-SVR selected a degenerate same-variable pair")
        row_i, row_j = i % n, j % n
        step_i = cost - alpha[i] if signs[i] > 0 else alpha[i]
        step_j = alpha[j] if signs[j] > 0 else cost - alpha[j]
        maximum_step = min(float(step_i), float(step_j))
        curvature = float(kernel[row_i, row_i] + kernel[row_j, row_j] - 2 * kernel[row_i, row_j])
        if curvature > np.finfo(float).tiny:
            step = min(maximum_step, violation / curvature)
        else:
            step = maximum_step
        if step <= 0.0:
            raise ArithmeticError("epsilon-SVR pair update has no feasible positive step")
        alpha[i] += signs[i] * step
        alpha[j] -= signs[j] * step
        beta[row_i] += step
        beta[row_j] -= step
        kernel_beta += step * (kernel[:, row_i] - kernel[:, row_j])
        iterations += 1
        if iterations % max(50, 4 * n) == 0:
            kernel_beta = kernel @ beta
    # Recompute from the kernel before computing stopping and optimality
    # certificates, so accumulated update roundoff cannot mask a bad fit.
    kernel_beta = kernel @ beta
    gradient = signs * np.r_[kernel_beta - y_train, kernel_beta - y_train] + epsilon
    i, j, working_gap = _working_pair(gradient, alpha, signs, cost)
    if working_gap > tolerance:
        raise ArithmeticError(f"epsilon-SVR KKT working-set gap is too large ({working_gap:g})")
    bias = _intercept_from_solution(gradient, signs, alpha, cost)
    stationarity = np.r_[gradient[:n] + bias, gradient[n:] - bias]
    bound_tolerance = max(np.finfo(float).eps * cost * 32, np.nextafter(0.0, 1.0))
    at_lower = alpha <= bound_tolerance
    at_upper = alpha >= cost - bound_tolerance
    interior = ~(at_lower | at_upper)
    kkt = float(
        max(
            np.max(np.maximum(-stationarity[at_lower], 0.0), initial=0.0),
            np.max(np.maximum(stationarity[at_upper], 0.0), initial=0.0),
            np.max(np.abs(stationarity[interior]), initial=0.0),
            abs(float(np.sum(beta))),
        )
    )
    primal_residual = kernel_beta + bias - y_train
    primal = 0.5 * float(beta @ kernel_beta) + cost * float(
        np.sum(np.maximum(np.abs(primal_residual) - epsilon, 0.0))
    )
    dual = float(y_train @ beta - 0.5 * beta @ kernel_beta - epsilon * np.sum(alpha))
    gap = primal - dual
    if gap < -max(1e-10, tolerance * 10) or not np.isfinite(gap):
        raise ArithmeticError("epsilon-SVR primal/dual check failed")
    if kkt > max(1e-7, tolerance * 10):
        raise ArithmeticError(f"epsilon-SVR KKT residual is too large ({kkt:g})")
    test_kernel = _rbf_kernel(x_test, x_train, sigma)
    prediction = test_kernel @ beta + bias
    support = np.flatnonzero(beta != 0.0).astype(np.int64)
    if support.size == 0:
        raise ArithmeticError("epsilon-SVR has no support vectors for this fit")
    if not np.isfinite(prediction).all():
        raise ArithmeticError("epsilon-SVR prediction is non-finite")
    return _SVRSolution(
        prediction,
        support,
        beta,
        bias,
        kkt,
        max(0.0, gap),
        iterations,
        primal,
        dual,
        0.0,
        1.0,
        np.ones(x_train.shape[1]),
        np.zeros(x_train.shape[1]),
        np.ones(x_train.shape[1]),
        True,
    )


def _fit_predict(
    x_train: FloatArray,
    y_train: FloatArray,
    x_test: FloatArray,
    cost: float,
    sigma: float,
    epsilon: float,
    tolerance: float,
    max_iterations: int,
) -> _SVRSolution:
    xs, ys, xt, y_center, y_scale, x_magnitude, x_center, x_scale, scaled = _standardize_for_fit(
        x_train, y_train, x_test
    )
    if scaled and np.ptp(ys) == 0.0:
        raise ArithmeticError("epsilon-SVR has a constant response and no support vectors")
    fit = _fit_scaled(xs, ys, xt, cost, sigma, epsilon, tolerance, max_iterations)
    if scaled:
        prediction = y_scale * fit.prediction + y_center
    else:
        prediction = fit.prediction
    if not np.isfinite(prediction).all():
        raise ArithmeticError("epsilon-SVR output cannot be represented in original units")
    return _SVRSolution(
        prediction,
        fit.support_indices,
        fit.dual_coefficients,
        fit.intercept,
        fit.kkt_error,
        fit.duality_gap,
        fit.iterations,
        fit.primal_objective,
        fit.dual_objective,
        y_center,
        y_scale,
        x_magnitude,
        x_center,
        x_scale,
        scaled,
    )


def condis_svm_refine(
    imputation: CondiSImputation,
    covariates: ArrayLike,
    *,
    folds: int = 10,
    repeats: int = 1,
    random_state: int | None = 0,
    fold_ids: ArrayLike | None = None,
    sigma: float | None = None,
    sigma_pair_indices: ArrayLike | None = None,
    cost_grid: ArrayLike = (0.25, 0.5, 1.0),
    epsilon: float = 0.1,
    solver_tolerance: float = 1e-8,
    max_iterations: int = 2_000,
    enforce_censoring: bool = False,
) -> CondiSSVMRefinement:
    """Tune and refit source-faithful RBF epsilon-SVR on imputed time.

    The status indicator is included as a predictor.  By default sigma is
    estimated with kernlab's sampled-distance rule, while the random stream
    and fold assignment are NumPy-specific.  Supplying ``sigma_pair_indices``
    (shape ``(n // 2, 2)``) or ``sigma`` makes that part directly repeatable.
    """
    if not isinstance(enforce_censoring, bool):
        raise ValueError("enforce_censoring must be boolean")
    if (sigma is not None) and (sigma_pair_indices is not None):
        raise ValueError("provide sigma or sigma_pair_indices, not both")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(
        max_iterations, (int, np.integer)
    ):
        raise ValueError("max_iterations must be an integer")
    if not 1 <= max_iterations <= 20_000:
        raise ValueError("max_iterations must be between 1 and 20,000")
    solver_tolerance = scalar(solver_tolerance, "solver_tolerance")
    epsilon = scalar(epsilon, "epsilon")
    if not 1e-12 <= solver_tolerance <= 1e-2:
        raise ValueError("solver_tolerance must lie between 1e-12 and 1e-2")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    x, y = _design(imputation, covariates)
    n = y.size
    if n < 3:
        raise ValueError("epsilon-SVR refinement requires at least three observations")
    if isinstance(folds, (bool, np.bool_)) or not isinstance(folds, (int, np.integer)):
        raise ValueError("folds must be an integer")
    if not 2 <= folds <= 100:
        raise ValueError("folds must be between 2 and 100")
    fold_count = min(int(folds), n)
    assignments = _make_folds(n, y, fold_count, repeats, random_state, fold_ids)
    training_sizes = np.asarray(
        [
            np.count_nonzero(assignments[rep] != fold)
            for rep in range(repeats)
            for fold in range(fold_count)
        ],
        dtype=np.int64,
    )
    if np.any(training_sizes < 2):
        raise ValueError("each epsilon-SVR CV training fold must contain at least two rows")
    grid_raw = finite(cost_grid, "cost_grid")
    if grid_raw.ndim != 1 or grid_raw.size == 0 or grid_raw.size > 20:
        raise ValueError("cost_grid must be a nonempty vector of at most 20 positive values")
    if np.any(grid_raw <= 0.0):
        raise ValueError("cost_grid values must be positive")
    costs = np.unique(grid_raw)
    if sigma is None:
        pairs = _pairs(n, sigma_pair_indices, random_state)
        sigma_value = _estimate_sigma(x, pairs)
    else:
        sigma_value = scalar(sigma, "sigma")
        pairs = np.empty((0, 2), dtype=np.int64)
        if sigma_value <= 0.0:
            raise ValueError("sigma must be positive")
    if not np.isfinite(sigma_value):
        raise ValueError("sigma must be finite")
    if n * n > 1_000_000 or np.any(training_sizes * training_sizes > 1_000_000):
        raise ValueError("SVR data exceed the one-million-cell per-fit kernel budget")
    work = int(n * n + costs.size * np.sum(training_sizes * training_sizes))
    if work > 30_000_000:
        raise ValueError("SVR cross-validation exceeds the 30 million kernel-cell work budget")
    update_budget = int((n + costs.size * np.sum(training_sizes)) * int(max_iterations))
    if update_budget > 100_000_000:
        raise ValueError("SVR pair-update budget exceeds 100 million row updates")
    scores = np.empty((repeats, fold_count, costs.size), dtype=np.float64)
    kkt = np.empty_like(scores)
    gaps = np.empty_like(scores)
    standardized = np.empty((repeats, fold_count), dtype=np.bool_)
    for rep in range(repeats):
        for fold in range(fold_count):
            test_mask = assignments[rep] == fold
            train_mask = ~test_mask
            x_train, y_train = x[train_mask], y[train_mask]
            x_test, y_test = x[test_mask], y[test_mask]
            for j, cost in enumerate(costs):
                fit = _fit_predict(
                    x_train,
                    y_train,
                    x_test,
                    float(cost),
                    sigma_value,
                    epsilon,
                    solver_tolerance,
                    int(max_iterations),
                )
                scale = max(float(np.max(np.abs(y_test))), float(np.max(np.abs(fit.prediction))))
                scale = scale if scale > 0.0 else 1.0
                residual = y_test / scale - fit.prediction / scale
                scores[rep, fold, j] = scale * float(np.sqrt(np.mean(residual * residual)))
                kkt[rep, fold, j] = fit.kkt_error
                gaps[rep, fold, j] = fit.duality_gap
                standardized[rep, fold] = fit.standardized
    if not np.isfinite(scores).all():
        raise ArithmeticError("epsilon-SVR cross-validation produced non-finite scores")
    flat = scores.reshape((-1, costs.size))
    score_scale = float(np.max(np.abs(flat)))
    if score_scale == 0.0:
        means = np.zeros(costs.size)
        sds = np.zeros(costs.size)
    else:
        normalized = flat / score_scale
        means = score_scale * normalized.mean(axis=0)
        sds = (
            score_scale * normalized.std(axis=0, ddof=1)
            if flat.shape[0] > 1
            else np.zeros(costs.size)
        )
    best = int(np.argmin(means))
    final = _fit_predict(
        x,
        y,
        x,
        float(costs[best]),
        sigma_value,
        epsilon,
        solver_tolerance,
        int(max_iterations),
    )
    censored = imputation.status == 0
    below = censored & (final.prediction < imputation.observed_time)
    above = censored & (final.prediction > imputation.horizon)
    refined = np.where(censored, final.prediction, imputation.observed_time)
    if enforce_censoring:
        refined[censored] = np.maximum(refined[censored], imputation.observed_time[censored])
    return CondiSSVMRefinement(
        _owned(final.prediction),
        _owned(refined),
        _owned(below),
        _owned(above),
        sigma_value,
        _readonly_int(pairs),
        _owned(costs),
        _owned(scores),
        _owned(means),
        _owned(sds),
        _owned(kkt),
        _owned(gaps),
        _owned(standardized),
        best,
        float(costs[best]),
        final.kkt_error,
        final.duality_gap,
        final.primal_objective,
        final.dual_objective,
        final.iterations,
        _readonly_int(final.support_indices),
        _owned(final.dual_coefficients),
        final.intercept,
        final.response_center,
        final.response_scale,
        _owned(final.predictor_magnitude),
        _owned(final.predictor_center),
        _owned(final.predictor_scale),
        final.standardized,
        _readonly_int(assignments),
        None if random_state is None else int(random_state),
        enforce_censoring,
    )
