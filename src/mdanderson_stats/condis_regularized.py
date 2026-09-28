"""Caret-style ridge, lasso, and kNN refinement for CondiS-X imputations.

The learners predict the imputed time from status and the supplied numeric
covariates.  The status column is intentionally a predictor, as in the
original CondiS-X formula ``pred_time ~ .``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite
from .boin import _owned
from .condis import CondiSImputation

CondiSRegularizedMethod = Literal["ridge", "lasso", "knn"]


@dataclass(frozen=True)
class CondiSRegularizedRefinement:
    """Selected learner, cross-validation scores, and in-sample refinement."""

    method: CondiSRegularizedMethod
    fitted_time: FloatArray
    refined_time: FloatArray
    below_censoring: NDArray[np.bool_]
    above_horizon: NDArray[np.bool_]
    tuning_values: FloatArray
    mean_rmse: FloatArray
    sd_rmse: FloatArray
    fold_rmse: FloatArray
    best_index: int
    best_value: float
    fold_ids: NDArray[np.int64]
    random_state: int | None
    enforce_censoring: bool


def _readonly_int(values: NDArray[np.int64]) -> NDArray[np.int64]:
    result = np.array(values, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _design(
    imputation: CondiSImputation,
    covariates: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    if not isinstance(imputation, CondiSImputation):
        raise TypeError("imputation must be CondiSImputation")
    x = finite(covariates, "covariates")
    n = imputation.imputed_time.size
    if x.ndim != 2 or x.shape[0] != n or not 1 <= x.shape[1] <= 500 or x.size > 2_000_000:
        raise ValueError("covariates must have n rows, 1..500 columns and <=2 million cells")
    # Formula pred_time ~ . includes the event indicator as an ordinary input.
    design = np.column_stack((imputation.status.astype(np.float64), x))
    if not np.isfinite(design).all():
        raise ValueError("predictor matrix must contain only finite real values")
    return design, imputation.imputed_time


def _make_folds(
    n: int,
    target: FloatArray,
    folds: int,
    repeats: int,
    random_state: int | None,
    fold_ids: ArrayLike | None,
) -> NDArray[np.int64]:
    if isinstance(folds, (bool, np.bool_)) or not isinstance(folds, (int, np.integer)):
        raise ValueError("folds must be an integer")
    if isinstance(repeats, (bool, np.bool_)) or not isinstance(repeats, (int, np.integer)):
        raise ValueError("repeats must be an integer")
    if not 2 <= folds <= 100 or not 1 <= repeats <= 10:
        raise ValueError("folds must be 2..100 and repeats must be 1..10")
    folds = min(folds, n)
    if random_state is not None and (
        isinstance(random_state, (bool, np.bool_))
        or not isinstance(random_state, (int, np.integer))
        or not 0 <= random_state <= np.iinfo(np.uint32).max
    ):
        raise ValueError("random_state must be None or a nonnegative 32-bit integer")
    if fold_ids is None:
        rng = np.random.default_rng(random_state)
        result = np.empty((repeats, n), dtype=np.int64)
        cuts = min(5, max(2, n // folds))
        breaks = np.unique(np.quantile(target, np.linspace(0.0, 1.0, cuts)))
        strata = np.searchsorted(breaks[1:-1], target, side="left")
        for rep in range(repeats):
            assignment = np.empty(n, dtype=np.int64)
            for group in np.unique(strata):
                rows = np.flatnonzero(strata == group)
                labels = np.tile(np.arange(folds), rows.size // folds)
                if rows.size % folds:
                    labels = np.r_[labels, rng.permutation(folds)[: rows.size % folds]]
                assignment[rng.permutation(rows)] = labels
            result[rep] = assignment
        return result
    raw = np.asarray(fold_ids)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError("fold_ids must contain integer fold labels")
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.shape != (repeats, n):
        raise ValueError("fold_ids must have shape (repeats, n), or shape (n,) for one repeat")
    result = raw.astype(np.int64, copy=True)
    if np.any(result < 0) or np.any(result >= folds):
        raise ValueError("fold_ids must use labels from 0 through folds-1")
    for row in result:
        if not np.array_equal(np.unique(row), np.arange(folds)):
            raise ValueError("each fold assignment must use every fold and have nonempty folds")
    return result


def _glmnet_path_predict(
    x: FloatArray,
    y: FloatArray,
    x_test: FloatArray,
    lambdas: FloatArray,
    alpha: float,
    max_sweeps: int = 100_000,
) -> FloatArray:
    """Fit the Gaussian glmnet default path and interpolate at requested lambdas.

    glmnet uses L2-normalized centered columns/response internally.  Its
    coordinate objective is ``RSS/2 + lambda * penalty`` in those coordinates;
    the transforms below account for that scale and restore raw predictions.
    """
    n, p = x.shape
    xscale = np.max(np.abs(x), axis=0)
    xscale = np.where(xscale > 0.0, xscale, 1.0)
    xscaled = x / xscale
    xtest_scaled = x_test / xscale
    xmean = xscaled.mean(axis=0)
    yscale = float(np.max(np.abs(y)))
    yscale = yscale if yscale > 0.0 else 1.0
    yscaled = y / yscale
    ymean_scaled = float(yscaled.mean())
    xc = xscaled - xmean
    yc = yscaled - ymean_scaled
    xnorm = np.sqrt(np.einsum("ij,ij->j", xc, xc))
    ynorm = float(np.linalg.norm(yc))
    active = xnorm > 0.0
    if ynorm == 0.0 or not np.any(active):
        return np.full((x_test.shape[0], lambdas.size), yscale * ymean_scaled, dtype=np.float64)
    z = np.zeros_like(xc)
    z[:, active] = xc[:, active] / xnorm[active]
    z_test = np.zeros_like(xtest_scaled)
    z_test[:, active] = (xtest_scaled[:, active] - xmean[active]) / xnorm[active]
    target = yc / ynorm
    correlations = z.T @ target
    lambda_max = float(np.max(np.abs(correlations[active]))) / max(alpha, 1e-3)
    if lambda_max == 0.0:
        return np.full((x_test.shape[0], lambdas.size), yscale * ymean_scaled, dtype=np.float64)
    # glmnet's default nlambda=100 and lambda.min.ratio=(n<p ? 1e-2 : 1e-4).
    # The first internal point is the large-number null fit; fix.lam replaces
    # only its reported lambda by geometric extrapolation from the next points.
    ratio = 1e-2 if n < p else 1e-4
    count = 100
    step = ratio ** (1.0 / (count - 1))
    response_sd = yscale * (ynorm / np.sqrt(n))
    first_finite = lambda_max * response_sd * step
    path_lambdas = np.empty(count, dtype=np.float64)
    path_lambdas[0] = first_finite / step
    path_lambdas[1:] = first_finite * step ** np.arange(count - 1)

    path = np.zeros((p, count), dtype=np.float64)
    coef = np.zeros(p, dtype=np.float64)
    residual = target.copy()
    norms = np.einsum("ij,ij->j", z, z)
    previous_deviance = 0.0
    path_size = count
    for index in range(1, count):
        lam = float(path_lambdas[index] / response_sd)
        for _ in range(max_sweeps):
            max_change = 0.0
            for j in range(p):
                if not active[j]:
                    continue
                old = coef[j]
                partial = residual + z[:, j] * old
                rho = float(z[:, j] @ partial)
                if alpha == 1.0:
                    new = np.sign(rho) * max(abs(rho) - lam, 0.0) / norms[j]
                else:
                    new = rho / (norms[j] + lam)
                if new != old:
                    residual = partial - z[:, j] * new
                    coef[j] = new
                    max_change = max(max_change, abs(new - old))
            if max_change <= 1e-7:
                break
        else:
            raise ArithmeticError("glmnet-style coordinate descent did not converge")
        path[:, index] = coef
        deviance = 1.0 - float(residual @ residual)
        deviance_change = np.inf if deviance == 0.0 else (deviance - previous_deviance) / deviance
        previous_deviance = deviance
        if index + 1 >= 5 and (deviance_change < 1e-5 or deviance > 0.999):
            path_size = index + 1
            break
    path_lambdas = path_lambdas[:path_size]
    path = path[:, :path_size]
    # Prediction stays centered to avoid cancellation between a large intercept
    # and large offset slopes. Interpolation follows glmnet's exact=FALSE rule.
    # glmnet fixes the infinite/null first lambda by geometric extrapolation.
    if path_lambdas.size > 2:
        path_lambdas[0] = path_lambdas[1] * (path_lambdas[1] / path_lambdas[2])
    requested = np.clip(lambdas, path_lambdas[-1], path_lambdas[0])
    predictions = np.empty((x_test.shape[0], lambdas.size), dtype=np.float64)
    for j, requested_lambda in enumerate(requested):
        hi = int(np.searchsorted(-path_lambdas, -requested_lambda, side="right"))
        hi = min(max(hi, 1), path_size - 1)
        lo = hi - 1
        span = path_lambdas[lo] - path_lambdas[hi]
        fraction = 1.0 if span == 0 else (requested_lambda - path_lambdas[hi]) / span
        beta = fraction * path[:, lo] + (1.0 - fraction) * path[:, hi]
        predictions[:, j] = yscale * (ymean_scaled + ynorm * (z_test @ beta))
    return predictions


def _knn_predict(
    x_train: FloatArray, y_train: FloatArray, x_test: FloatArray, k: int
) -> FloatArray:
    actual_k = min(k, x_train.shape[0])
    if actual_k >= 1000:
        raise ValueError("kNN k must be below the native 1000-neighbor storage limit")
    if x_train.shape[0] * x_test.shape[0] > 2_000_000:
        raise ValueError("kNN distance work exceeds 2 million row pairs")
    out = np.empty(x_test.shape[0], dtype=np.float64)
    distance_scale = max(float(np.max(np.abs(x_train))), float(np.max(np.abs(x_test))))
    distance_scale = distance_scale if distance_scale > 0.0 else 1.0
    sentinel = 0.99 * np.finfo(np.float64).max
    for i, query in enumerate(x_test):
        delta = x_train / distance_scale - query / distance_scale
        distance = np.zeros(x_train.shape[0], dtype=np.float64)
        for column in range(x_train.shape[1]):
            difference = delta[:, column]
            distance += difference * difference
        distances: list[float] = [float(sentinel)] * 1000
        rows = [-1] * 1000
        retained = actual_k
        for candidate, candidate_distance in enumerate(distance):
            if candidate_distance > distances[actual_k - 1] * (1.0 + 1e-4):
                continue
            insertion = next(
                (j for j in range(retained + 1) if candidate_distance < distances[j]),
                None,
            )
            if insertion is None:
                continue
            for slot in range(retained, insertion, -1):
                distances[slot] = distances[slot - 1]
                rows[slot] = rows[slot - 1]
            distances[insertion] = float(candidate_distance)
            rows[insertion] = candidate
            if distances[retained] <= distances[actual_k - 1]:
                if retained >= 999:
                    raise ValueError("kNN tie set exceeds the native 1000-neighbor limit")
                retained += 1
            distances[retained] = sentinel
            rows[retained] = -1
        included = actual_k
        while included < retained and distances[included] <= distances[actual_k - 1] * (1 + 1e-4):
            included += 1
        values = y_train[np.asarray(rows[:included], dtype=np.int64)]
        response_scale = float(np.max(np.abs(values)))
        out[i] = (
            0.0
            if response_scale == 0.0
            else response_scale * float(np.mean(values / response_scale))
        )
    return out


def condis_regularized_refine(
    imputation: CondiSImputation,
    covariates: ArrayLike,
    *,
    method: CondiSRegularizedMethod,
    folds: int = 10,
    repeats: int = 1,
    random_state: int | None = 0,
    fold_ids: ArrayLike | None = None,
    lambda_grid: ArrayLike | None = None,
    neighbor_grid: ArrayLike | None = None,
    enforce_censoring: bool = False,
) -> CondiSRegularizedRefinement:
    """Tune ridge, lasso, or raw-distance kNN by repeated fold RMSE.

    Ridge/lasso use the CondiS caret grids ``seq(0.01, 10, length=10)`` by
    default. kNN uses the default caret grid ``5, 7, 9``. Default folds use
    caret's numeric-target quantile stratification, with a deterministic NumPy
    random stream; fold assignments do not claim R RNG parity. Explicit
    ``fold_ids`` permits exact resampling comparisons.
    """
    if method not in ("ridge", "lasso", "knn"):
        raise ValueError("method must be 'ridge', 'lasso', or 'knn'")
    if not isinstance(enforce_censoring, bool):
        raise ValueError("enforce_censoring must be boolean")
    x, y = _design(imputation, covariates)
    n = y.size
    if isinstance(folds, (bool, np.bool_)) or not isinstance(folds, (int, np.integer)):
        raise ValueError("folds must be an integer")
    if not 2 <= folds <= 100:
        raise ValueError("folds must be 2..100")
    fold_count = min(int(folds), n)
    assignments = _make_folds(n, y, fold_count, repeats, random_state, fold_ids)
    if method == "knn":
        if lambda_grid is not None:
            raise ValueError("lambda_grid applies only to ridge and lasso")
        raw_grid = np.asarray([5, 7, 9] if neighbor_grid is None else neighbor_grid)
        if (
            np.iscomplexobj(raw_grid)
            or raw_grid.ndim != 1
            or raw_grid.size == 0
            or raw_grid.dtype.kind not in "iu"
        ):
            raise ValueError("neighbor_grid must be a nonempty vector of positive integers")
        if np.any(raw_grid < 1) or np.any(raw_grid >= 1000) or raw_grid.size > 100:
            raise ValueError("neighbor_grid values must be positive and contain at most 100 values")
        tuning = np.unique(raw_grid.astype(np.int64))[::-1].astype(np.float64)
        smallest_training_fold = min(
            int(np.count_nonzero(assignments[rep] != fold))
            for rep in range(repeats)
            for fold in range(fold_count)
        )
        clipped = tuning[tuning > smallest_training_fold]
        if clipped.size:
            warnings.warn(
                f"k={int(clipped[0])} exceeds a CV training-fold size; "
                "k is clipped to the available rows",
                UserWarning,
                stacklevel=2,
            )
    else:
        if neighbor_grid is not None:
            raise ValueError("neighbor_grid applies only to kNN")
        raw_grid = np.asarray(np.linspace(0.01, 10.0, 10) if lambda_grid is None else lambda_grid)
        if (
            np.iscomplexobj(raw_grid)
            or raw_grid.ndim != 1
            or raw_grid.size == 0
            or raw_grid.dtype.kind not in "fiu"
        ):
            raise ValueError("lambda_grid must be a nonempty vector of finite positive values")
        tuning = np.unique(raw_grid.astype(np.float64))[::-1]
        if tuning.size > 100 or not np.isfinite(tuning).all() or np.any(tuning <= 0):
            raise ValueError("lambda_grid must contain 1..100 finite positive values")
    if n * tuning.size * repeats > 2_000_000:
        raise ValueError("cross-validation work exceeds the 2 million row-tuning-repeat budget")
    if method == "knn":
        work = n * n * repeats * tuning.size * x.shape[1]
        if work > 100_000_000:
            raise ValueError("kNN cross-validation exceeds the bounded distance-work budget")
    else:
        work = n * x.shape[1] * fold_count * repeats * 100
        if work > 250_000_000:
            raise ValueError("regularized regression exceeds the bounded path-work budget")
    max_sweeps = 100_000
    if method != "knn":
        max_sweeps = max(
            1,
            min(
                1_000,
                250_000_000 // (n * x.shape[1] * 100 * (fold_count * repeats + 1)),
            ),
        )

    scores = np.empty((repeats, fold_count, tuning.size), dtype=np.float64)
    for rep in range(repeats):
        for fold in range(fold_count):
            test_mask = assignments[rep] == fold
            train_mask = ~test_mask
            x_train, y_train = x[train_mask], y[train_mask]
            x_test, y_test = x[test_mask], y[test_mask]
            if method == "knn":
                for j, value in enumerate(tuning):
                    prediction = _knn_predict(x_train, y_train, x_test, int(value))
                    if not np.isfinite(prediction).all():
                        raise ArithmeticError(
                            "kNN produced non-finite cross-validation predictions"
                        )
                    scale = max(float(np.max(np.abs(y_test))), float(np.max(np.abs(prediction))))
                    scale = scale if scale > 0.0 else 1.0
                    residual = y_test / scale - prediction / scale
                    scores[rep, fold, j] = scale * float(np.sqrt(np.mean(residual * residual)))
            else:
                alpha = 0.0 if method == "ridge" else 1.0
                predicted = _glmnet_path_predict(
                    x_train, y_train, x_test, tuning, alpha, max_sweeps
                )
                if not np.isfinite(predicted).all():
                    raise ArithmeticError(
                        "regularized regression produced non-finite cross-validation predictions"
                    )
                scale = max(float(np.max(np.abs(y_test))), float(np.max(np.abs(predicted))))
                scale = scale if scale > 0.0 else 1.0
                residual = y_test[:, None] / scale - predicted / scale
                scores[rep, fold] = scale * np.sqrt(np.mean(residual * residual, axis=0))
    if not np.isfinite(scores).all():
        raise ArithmeticError("cross-validation produced non-finite RMSE scores")
    # Caret's MeanSD gives every resample equal weight, rather than pooling
    # held-out rows before computing RMSE.
    flat_scores = scores.reshape((-1, tuning.size))
    score_scale = float(np.max(np.abs(flat_scores)))
    if score_scale == 0.0:
        mean_rmse = np.zeros(tuning.size)
        sd_rmse = np.zeros(tuning.size)
    else:
        normalized_scores = flat_scores / score_scale
        mean_rmse = score_scale * normalized_scores.mean(axis=0)
        sd_rmse = (
            score_scale * normalized_scores.std(axis=0, ddof=1)
            if flat_scores.shape[0] > 1
            else np.zeros(tuning.size)
        )
    best = int(np.argmin(mean_rmse))

    if method == "knn":
        fitted = _knn_predict(x, y, x, int(tuning[best]))
    else:
        alpha = 0.0 if method == "ridge" else 1.0
        fitted = _glmnet_path_predict(x, y, x, tuning[best : best + 1], alpha, max_sweeps)[:, 0]
    if not np.isfinite(fitted).all():
        raise ArithmeticError("selected learner produced non-finite predictions")
    censored = imputation.status == 0
    below = censored & (fitted < imputation.observed_time)
    above = censored & (fitted > imputation.horizon)
    refined = np.where(censored, fitted, imputation.observed_time)
    if enforce_censoring:
        refined[censored] = np.maximum(refined[censored], imputation.observed_time[censored])
    return CondiSRegularizedRefinement(
        method,
        _owned(fitted),
        _owned(refined),
        _owned(below),
        _owned(above),
        _owned(tuning),
        _owned(mean_rmse),
        _owned(sd_rmse),
        _owned(scores),
        best,
        float(tuning[best]),
        _readonly_int(assignments),
        None if random_state is None else int(random_state),
        enforce_censoring,
    )
