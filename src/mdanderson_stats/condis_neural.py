"""Single-hidden-layer neural refinement for CondiS-X imputations.

The network and optimizer follow the bundled ``nnet``/R ``vmmin`` numerical
contract: one logistic hidden layer, linear output, summed squared error and
decay on every weight (biases included). Python random starts and fold
assignments are reproducible but do not claim R RNG-stream parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite
from .boin import _owned
from .condis import CondiSImputation
from .condis_regularized import _design, _make_folds

_GRID_SIZE = np.asarray([1, 3, 5], dtype=np.int64)
_GRID_DECAY = np.asarray([0.1, 0.0001, 0.0], dtype=np.float64)
_MAX_WEIGHTS = 1000
_MAX_FIT_ROWS = 2000
_MAX_CV_FITS = 5000


@dataclass(frozen=True)
class CondiSNeuralFit:
    """One fitted network; convergence code 1 means the iteration cap was hit."""

    hidden_size: int
    decay: float
    weights: FloatArray
    fitted_time: FloatArray
    objective: float
    convergence_code: int
    iterations: int
    function_evaluations: int
    gradient_evaluations: int

    @property
    def status(self) -> str:
        """Readable native status; code zero is a stop signal, not a globality proof."""
        if self.iterations == 0:
            return "not_run"
        return "iteration_limit" if self.convergence_code == 1 else "stopped"


@dataclass(frozen=True)
class CondiSNeuralRefinement:
    """Nine-model CV results and the selected full-data refit."""

    tuning_grid: NDArray[np.float64]
    fold_ids: NDArray[np.int64]
    fold_rmse: FloatArray
    mean_rmse: FloatArray
    sd_rmse: FloatArray
    best_index: int
    selected_hidden_size: int
    selected_decay: float
    selected_fit: CondiSNeuralFit
    fitted_time: FloatArray
    refined_time: FloatArray
    below_censoring: NDArray[np.bool_]
    above_horizon: NDArray[np.bool_]
    fold_convergence: NDArray[np.int64]
    fold_iterations: NDArray[np.int64]
    fold_objective: FloatArray
    random_state: int | None
    enforce_censoring: bool


def _sigmoid(values: FloatArray) -> FloatArray:
    out = np.empty_like(values)
    low, high = values < -15.0, values > 15.0
    mid = ~(low | high)
    out[low] = 0.0
    out[high] = 1.0
    out[mid] = 1.0 / (1.0 + np.exp(-values[mid]))
    return out


def _predict_gradient(
    x: FloatArray, y: FloatArray, weights: FloatArray, hidden_size: int, decay: float
) -> tuple[float, FloatArray, FloatArray]:
    n, p = x.shape
    block = (p + 1) * hidden_size
    hidden_w = weights[:block].reshape(hidden_size, p + 1)
    out_w = weights[block:]
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        activation = _sigmoid(hidden_w[:, 0][None, :] + x @ hidden_w[:, 1:].T)
        prediction = out_w[0] + activation @ out_w[1:]
        residual = prediction - y
        objective = float(residual @ residual)
        if decay:
            objective += float(decay * (weights @ weights))
        delta = 2.0 * residual
        grad_out = np.r_[delta.sum(), activation.T @ delta]
        hidden_delta = (delta[:, None] * out_w[1:][None, :]) * activation * (1.0 - activation)
        grad_hidden = np.column_stack((hidden_delta.sum(axis=0), hidden_delta.T @ x))
        gradient = np.r_[grad_hidden.ravel(), grad_out] + 2.0 * decay * weights
    if (
        not np.isfinite(objective)
        or not np.isfinite(gradient).all()
        or not np.isfinite(prediction).all()
    ):
        raise ArithmeticError("neural network objective or gradient is not representable")
    return objective, gradient, prediction


def _objective(
    x: FloatArray, y: FloatArray, weights: FloatArray, hidden_size: int, decay: float
) -> float:
    p = x.shape[1]
    block = (p + 1) * hidden_size
    hidden_w = weights[:block].reshape(hidden_size, p + 1)
    out_w = weights[block:]
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        activation = _sigmoid(hidden_w[:, 0][None, :] + x @ hidden_w[:, 1:].T)
        residual = out_w[0] + activation @ out_w[1:] - y
        value = float(residual @ residual)
        if decay:
            value += float(decay * (weights @ weights))
    return value


def _vmmin(
    x: FloatArray,
    y: FloatArray,
    start: FloatArray,
    hidden_size: int,
    decay: float,
    max_iterations: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[FloatArray, float, int, int, int, int]:
    """Bounded translation of R's inverse-Hessian BFGS ``vmmin`` loop."""
    w = start.copy()
    if max_iterations <= 0:
        return w, _objective(x, y, w, hidden_size, decay), 0, 0, 0, 0
    f, grad, _ = _predict_gradient(x, y, w, hidden_size, decay)
    fcount, gcount, iteration = 1, 1, 1
    n = w.size
    last_reset = gcount
    count = 0
    while True:
        if last_reset == gcount:
            inverse = np.eye(n)
        direction = -(inverse @ grad)
        slope = float(direction @ grad)
        if not np.isfinite(direction).all() or not np.isfinite(slope):
            raise ArithmeticError("BFGS search direction is not representable")
        if slope < 0.0:
            step = 1.0
            accepted = False
            trial_f = f
            candidate = w.copy()
            for _ in range(100):
                candidate = w + step * direction
                unchanged = (10.0 + w) == (10.0 + candidate)
                if np.all(unchanged):
                    break
                candidate_f = _objective(x, y, candidate, hidden_size, decay)
                fcount += 1
                trial_f = candidate_f
                if np.isfinite(candidate_f) and candidate_f <= f + slope * step * 1e-4:
                    accepted = True
                    break
                step *= 0.2
            enough = (trial_f > absolute_tolerance) and (
                abs(trial_f - f) > relative_tolerance * (abs(f) + relative_tolerance)
            )
            if not enough:
                if accepted:
                    w = candidate
                f = trial_f
                count = n
            else:
                count = 0 if accepted else n
            if enough and accepted:
                new_f, new_grad, _ = _predict_gradient(x, y, candidate, hidden_size, decay)
                gcount += 1
                displacement = step * direction
                change = new_grad - grad
                curvature = float(displacement @ change)
                w, f, grad = candidate, new_f, new_grad
                iteration += 1
                if curvature > 0.0:
                    by = inverse @ change
                    d2 = 1.0 + float(by @ change) / curvature
                    inverse += (
                        d2 * np.outer(displacement, displacement)
                        - np.outer(by, displacement)
                        - np.outer(displacement, by)
                    ) / curvature
                else:
                    last_reset = gcount
            if count == n:
                if last_reset < gcount:
                    count = 0
                    last_reset = gcount
        else:
            count = 0
            if last_reset == gcount:
                count = n
            else:
                last_reset = gcount
        if iteration >= max_iterations:
            break
        if gcount - last_reset > 2 * n:
            last_reset = gcount
        if count == n and last_reset == gcount:
            break
    status = 0 if iteration < max_iterations else 1
    return w, f, status, iteration, fcount, gcount


def fit_condis_neural(
    predictors: ArrayLike,
    target: ArrayLike,
    *,
    hidden_size: int,
    decay: float,
    initial_weights: ArrayLike | None = None,
    random_state: int | None = 0,
    max_iterations: int = 100,
    absolute_tolerance: float = 1e-4,
    relative_tolerance: float = 1e-8,
) -> CondiSNeuralFit:
    """Fit one native-contract network; explicit starts enable R comparisons."""
    raw_x = np.asarray(predictors)
    raw_y = np.asarray(target)
    if raw_x.ndim != 2 or raw_y.ndim != 1 or raw_x.size > 1_000_000 or raw_y.size > _MAX_FIT_ROWS:
        raise ValueError("predictors/target exceed the bounded fit dimensions")
    x, y = finite(predictors, "predictors"), finite(target, "target")
    if (
        x.ndim != 2
        or not 1 <= x.shape[1] <= 500
        or y.ndim != 1
        or x.shape[0] != y.size
        or not 1 <= y.size <= _MAX_FIT_ROWS
    ):
        raise ValueError("predictors and target must have matching rows (1..2000)")
    if (
        isinstance(hidden_size, (bool, np.bool_))
        or not isinstance(hidden_size, (int, np.integer))
        or hidden_size not in (1, 3, 5)
    ):
        raise ValueError("hidden_size must be one of 1, 3, or 5")
    decay = _nonnegative_real(decay, "decay")
    count = (x.shape[1] + 1) * int(hidden_size) + int(hidden_size) + 1
    if count > _MAX_WEIGHTS:
        raise ValueError("network exceeds the native 1000-weight limit")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or not 0 <= max_iterations <= 10000
    ):
        raise ValueError("max_iterations must be an integer from 0 through 10000")
    if int(max_iterations) * count * count > 100_000_000:
        raise ValueError("neural fit exceeds the bounded BFGS matrix-work budget")
    absolute_tolerance = _nonnegative_real(absolute_tolerance, "absolute_tolerance")
    relative_tolerance = _nonnegative_real(relative_tolerance, "relative_tolerance")
    if initial_weights is None:
        if random_state is not None and (
            isinstance(random_state, (bool, np.bool_))
            or not isinstance(random_state, (int, np.integer))
            or random_state < 0
        ):
            raise ValueError("random_state must be None or a nonnegative integer")
        start = np.random.default_rng(random_state).uniform(-0.7, 0.7, count)
    else:
        start = finite(initial_weights, "initial_weights")
        if start.shape != (count,):
            raise ValueError(f"initial_weights must have length {count}")
        if random_state not in (None, 0):
            raise ValueError("random_state is ignored for explicit starts; pass None or 0")
    weights, objective, code, iterations, fc, gc = _vmmin(
        x,
        y,
        start,
        int(hidden_size),
        float(decay),
        int(max_iterations),
        float(absolute_tolerance),
        float(relative_tolerance),
    )
    final_objective, _, fitted = _predict_gradient(x, y, weights, int(hidden_size), float(decay))
    return CondiSNeuralFit(
        int(hidden_size),
        float(decay),
        _owned(weights),
        _owned(fitted),
        final_objective,
        code,
        iterations,
        fc,
        gc,
    )


def condis_neural_refine(
    imputation: CondiSImputation,
    covariates: ArrayLike,
    *,
    folds: int = 10,
    random_state: int | None = 0,
    fold_ids: ArrayLike | None = None,
    initial_weights: ArrayLike | None = None,
    fold_initial_weights: ArrayLike | None = None,
    max_iterations: int = 100,
    enforce_censoring: bool = False,
) -> CondiSNeuralRefinement:
    """Tune all nine native candidates by fold RMSE, then refit the winner.

    Explicit full starts are shaped ``(9, n_weights_for_candidate)`` only
    when all network sizes imply the same number of weights; otherwise supply
    a length-nine sequence of vectors. Fold starts use ``(fold, candidate)``
    nested sequences of vectors and are intended for exact native validation.
    """
    if not isinstance(imputation, CondiSImputation):
        raise TypeError("imputation must be CondiSImputation")
    if imputation.imputed_time.size > _MAX_FIT_ROWS:
        raise ValueError("neural refinement is limited to 2000 rows")
    if not isinstance(enforce_censoring, bool):
        raise ValueError("enforce_censoring must be boolean")
    x, y = _design(imputation, covariates)
    y = finite(y, "imputed_time")
    largest_model_weights = (x.shape[1] + 1) * 5 + 5 + 1
    if largest_model_weights > _MAX_WEIGHTS:
        raise ValueError("size-5 candidate exceeds the native 1000-weight limit")
    if y.size > _MAX_FIT_ROWS:
        raise ValueError("neural refinement is limited to 2000 rows")
    if (
        isinstance(folds, (bool, np.bool_))
        or not isinstance(folds, (int, np.integer))
        or not 2 <= folds <= 100
    ):
        raise ValueError("folds must be an integer from 2 through 100")
    if y.size < 2:
        raise ValueError("neural cross-validation requires at least two rows")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or not 0 <= max_iterations <= 10000
    ):
        raise ValueError("max_iterations must be an integer from 0 through 10000")
    fold_count = min(int(folds), y.size)
    assignments = _make_folds(y.size, y, fold_count, 1, random_state, fold_ids)[0]
    candidates = [(int(s), float(d)) for s in _GRID_SIZE for d in _GRID_DECAY]
    fit_count = fold_count * len(candidates) + len(candidates)
    max_train = y.size - int(np.min(np.bincount(assignments)))
    max_weights = max((x.shape[1] + 1) * s + s + 1 for s, _ in candidates)
    if (
        fit_count > _MAX_CV_FITS
        or fit_count * max_train * max(1, x.shape[1]) * max_iterations > 250_000_000
        or fit_count * max_weights * max_weights * max_iterations > 250_000_000
    ):
        raise ValueError("neural CV exceeds the bounded fit-work budget")
    score = np.empty((fold_count, len(candidates)), dtype=np.float64)
    convergence = np.empty_like(score, dtype=np.int64)
    iterations = np.empty_like(score, dtype=np.int64)
    objectives = np.empty_like(score, dtype=np.float64)
    full_starts = cast(
        list[FloatArray] | None,
        _read_start_set(initial_weights, "initial_weights", x.shape[1]),
    )
    supplied_folds = cast(
        list[list[FloatArray]] | None,
        _read_start_set(fold_initial_weights, "fold_initial_weights", x.shape[1], nested=True),
    )
    if supplied_folds is not None and len(supplied_folds) != fold_count:
        raise ValueError("fold_initial_weights must provide one candidate set per fold")
    if supplied_folds is not None and any(len(row) != len(candidates) for row in supplied_folds):
        raise ValueError("each fold must provide nine initial-weight vectors")
    rng = np.random.default_rng(random_state)
    expected = [(x.shape[1] + 1) * s + s + 1 for s, _ in candidates]
    if full_starts is not None:
        for j, start in enumerate(full_starts):
            if start.shape != (expected[j],):
                raise ValueError(f"initial_weights[{j}] must have length {expected[j]}")
    if supplied_folds is not None:
        for f, row in enumerate(supplied_folds):
            for j, start in enumerate(row):
                if start.shape != (expected[j],):
                    raise ValueError(
                        f"fold_initial_weights[{f}][{j}] must have length {expected[j]}"
                    )
    for f in range(fold_count):
        test = assignments == f
        train = ~test
        fold_starts = None if supplied_folds is None else supplied_folds[f]
        for j, (size, decay) in enumerate(candidates):
            fold_start = fold_starts[j] if fold_starts is not None else None
            if fold_start is None:
                fold_start = rng.uniform(-0.7, 0.7, expected[j])
            fit_result = fit_condis_neural(
                x[train],
                y[train],
                hidden_size=size,
                decay=decay,
                initial_weights=fold_start,
                random_state=0,
                max_iterations=max_iterations,
            )
            pred = _predict_with_weights(x[test], fit_result.weights, size)
            scale = max(float(np.max(np.abs(y[test]))), float(np.max(np.abs(pred))), 1.0)
            score[f, j] = scale * float(np.sqrt(np.mean((y[test] / scale - pred / scale) ** 2)))
            convergence[f, j] = fit_result.convergence_code
            iterations[f, j] = fit_result.iterations
            objectives[f, j] = fit_result.objective
    if not np.isfinite(score).all():
        raise ArithmeticError("neural cross-validation produced non-finite RMSE")
    mean = score.mean(axis=0)
    sd = score.std(axis=0, ddof=1) if fold_count > 1 else np.zeros(len(candidates))
    best = int(np.argmin(mean))
    size, decay = candidates[best]
    full_start = (
        full_starts[best] if full_starts is not None else rng.uniform(-0.7, 0.7, expected[best])
    )
    selected = fit_condis_neural(
        x,
        y,
        hidden_size=size,
        decay=decay,
        initial_weights=full_start,
        random_state=0,
        max_iterations=max_iterations,
    )
    fitted_values = selected.fitted_time
    censored = imputation.status == 0
    below = censored & (fitted_values < imputation.observed_time)
    above = censored & (fitted_values > imputation.horizon)
    refined = np.where(censored, fitted_values, imputation.observed_time)
    if enforce_censoring:
        refined[censored] = np.maximum(refined[censored], imputation.observed_time[censored])
    grid = np.asarray(candidates, dtype=np.float64)
    return CondiSNeuralRefinement(
        _owned(grid),
        _readonly_int(assignments),
        _owned(score),
        _owned(mean),
        _owned(sd),
        best,
        size,
        decay,
        selected,
        _owned(fitted_values),
        _owned(refined),
        _readonly_bool(below),
        _readonly_bool(above),
        _readonly_int(convergence),
        _readonly_int(iterations),
        _owned(objectives),
        None if random_state is None else int(random_state),
        enforce_censoring,
    )


def _predict_with_weights(x: FloatArray, weights: FloatArray, hidden_size: int) -> FloatArray:
    p = x.shape[1]
    block = (p + 1) * hidden_size
    hidden_w = weights[:block].reshape(hidden_size, p + 1)
    out_w = weights[block:]
    return out_w[0] + _sigmoid(hidden_w[:, 0][None, :] + x @ hidden_w[:, 1:].T) @ out_w[1:]


def _read_start_set(
    values: ArrayLike | None, name: str, predictors: int, *, nested: bool = False
) -> list[FloatArray] | list[list[FloatArray]] | None:
    if values is None:
        return None
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be a bounded list, tuple, or array")
    if isinstance(values, np.ndarray) and values.size > 100_000:
        raise ValueError(f"{name} exceeds the bounded start storage")
    outer = list(values)
    if len(outer) > 100:
        raise ValueError(f"{name} has too many start sets")
    result: list[FloatArray] | list[list[FloatArray]] = []
    for i, entry in enumerate(outer):
        if nested:
            if not isinstance(entry, (list, tuple, np.ndarray)) or len(entry) > 9:
                raise ValueError(f"{name}[{i}] must contain nine starts")
            row = list(entry)
            if len(row) != 9:
                raise ValueError(f"{name}[{i}] must contain nine starts")
            converted = [finite(start, f"{name}[{i}][{j}]") for j, start in enumerate(row)]
            result.append(converted)  # type: ignore[arg-type]
        else:
            result.append(finite(entry, f"{name}[{i}]"))  # type: ignore[arg-type]
    if not nested and len(result) != 9:
        raise ValueError(f"{name} must contain nine starts")
    return result


def _readonly_int(values: NDArray[np.int64]) -> NDArray[np.int64]:
    out = np.array(values, dtype=np.int64, copy=True)
    out.setflags(write=False)
    return out


def _readonly_bool(values: NDArray[np.bool_]) -> NDArray[np.bool_]:
    out = np.array(values, dtype=np.bool_, copy=True)
    out.setflags(write=False)
    return out


def _nonnegative_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite nonnegative real scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative real scalar")
    return result
