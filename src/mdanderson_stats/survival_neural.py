"""Small bounded NumPy neural survival models used by SurvivalContour.

This module implements the five model families called by the author wrapper.
Training is a transparent Python full-batch Adam implementation, not a
reproduction of the app's torch/torchtuples optimizer stream or architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .survival_neural_discrete import _loss_gradient as _discrete_loss_gradient

NeuralFamily = Literal["coxtime", "deepsurv", "deephit", "loghaz", "pchazard"]
_FAMILIES = {"coxtime", "deepsurv", "deephit", "loghaz", "pchazard"}
_MAX_ROWS = 10_000
_MAX_FEATURES = 200
_MAX_WORK = 100_000_000
_MAX_TRAIN_BYTES = 128 * 1024 * 1024
_MAX_SURFACE = 2_000_000
_MAX_DEEPHIT_PAIR_TIME = 20_000_000


@dataclass(frozen=True)
class SurvivalNeuralFit:
    """Immutable fitted network and all transforms needed for prediction."""

    family: str
    weights: tuple[FloatArray, ...]
    feature_unit: FloatArray
    feature_center: FloatArray
    feature_scale: FloatArray
    time_center: float
    time_scale: float
    cuts: FloatArray
    baseline_times: FloatArray
    baseline_log_hazard: FloatArray
    loss_history: FloatArray
    best_epoch: int
    epochs_run: int
    stopped_early: bool
    random_state: int
    training_rows: int


@dataclass(frozen=True)
class SurvivalNeuralContour:
    """Survival probabilities over one covariate grid, without intervals."""

    fit: SurvivalNeuralFit
    continuous_column: int
    profile: FloatArray
    grid: FloatArray
    times: FloatArray
    survival: FloatArray


def _integer(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return result


def _inputs(
    time: ArrayLike, event: ArrayLike, x: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    raw_t, raw_e, raw_x = np.asarray(time), np.asarray(event), np.asarray(x)
    if (
        raw_t.ndim != 1
        or raw_e.shape != raw_t.shape
        or raw_t.size > _MAX_ROWS
        or raw_x.size > 1_000_000
    ):
        raise ValueError("input shapes exceed the bounded survival-training dimensions")
    if any(np.iscomplexobj(v) for v in (time, event, x)):
        raise ValueError("time, event and x must be real")
    t, e, design = finite(time, "time"), finite(event, "event"), finite(x, "x")
    if design.ndim == 1:
        design = design[:, None]
    if (
        t.ndim != 1
        or e.shape != t.shape
        or design.ndim != 2
        or design.shape[0] != t.size
        or not 3 <= t.size <= _MAX_ROWS
        or not 1 <= design.shape[1] <= _MAX_FEATURES
        or design.size > 1_000_000
        or np.any(t < 0)
        or np.any((e != 0) & (e != 1))
        or not np.any(e == 1)
    ):
        raise ValueError(
            "expected 3..10,000 aligned rows, finite features, nonnegative times and binary events"
        )
    return t, e, design


def _km_cuts(t: FloatArray, e: FloatArray, n_cuts: int) -> FloatArray:
    """Training-only KM-survival quantile cuts, matching the pycox label transform."""
    unique = np.unique(t)
    at_risk = np.asarray([(t >= value).sum() for value in unique], dtype=np.float64)
    deaths = np.asarray([((t == value) & (e == 1)).sum() for value in unique], dtype=np.float64)
    survival = np.cumprod(1.0 - deaths / at_risk)
    survival_cuts = np.linspace(float(np.min(survival)), float(np.max(survival)), n_cuts)
    reverse_survival = survival[::-1]
    reverse_time = unique[::-1]
    positions = np.searchsorted(reverse_survival, survival_cuts, side="left")
    positions = positions[::-1].clip(0, unique.size - 1)
    cuts = np.unique(reverse_time[positions])
    cuts[0] = 0.0
    if cuts[-1] != float(np.max(t)):
        raise ValueError("KM quantile transform did not include the maximum training time")
    if cuts.size < 2:
        raise ValueError("training outcomes do not identify at least two discrete time cuts")
    return cuts


def _validate_cuts(cuts: ArrayLike | None, t: FloatArray, e: FloatArray, n_cuts: int) -> FloatArray:
    if cuts is None:
        return _km_cuts(t, e, n_cuts)
    if np.iscomplexobj(cuts):
        raise ValueError("cuts must be real")
    raw = np.asarray(cuts)
    if raw.ndim != 1 or raw.size > 201:
        raise ValueError("cuts must be a vector with at most 201 entries")
    value = finite(cuts, "cuts")
    if value.ndim != 1 or value.size < 2 or np.any(np.diff(value) <= 0):
        raise ValueError("cuts must be a strictly increasing vector of at least two values")
    return value


def _network_init(widths: tuple[int, ...], rng: np.random.Generator) -> tuple[FloatArray, ...]:
    arrays: list[FloatArray] = []
    for fan_in, fan_out in zip(widths[:-1], widths[1:]):
        limit = np.sqrt(2.0 / fan_in)
        arrays.append(rng.normal(0.0, limit, (fan_in + 1, fan_out)))
    return tuple(arrays)


def _forward(
    x: FloatArray, weights: tuple[FloatArray, ...]
) -> tuple[list[FloatArray], list[FloatArray]]:
    activations = [x]
    preactivations: list[FloatArray] = []
    current = x
    for index, weight in enumerate(weights):
        augmented = np.column_stack((current, np.ones(current.shape[0])))
        with np.errstate(over="ignore", invalid="ignore"):
            z = augmented @ weight
        if not np.isfinite(z).all():
            raise ArithmeticError("neural network activation is not representable")
        preactivations.append(z)
        current = np.maximum(z, 0.0) if index < len(weights) - 1 else z
        activations.append(current)
    return activations, preactivations


def _backprop(
    activations: list[FloatArray],
    preactivations: list[FloatArray],
    weights: tuple[FloatArray, ...],
    grad: FloatArray,
) -> tuple[FloatArray, ...]:
    gradients: list[FloatArray] = [np.empty_like(w) for w in weights]
    delta = grad
    for i in range(len(weights) - 1, -1, -1):
        augmented = np.column_stack((activations[i], np.ones(activations[i].shape[0])))
        gradients[i] = augmented.T @ delta
        if i:
            delta = (delta @ weights[i][:-1].T) * (preactivations[i - 1] > 0.0)
    return tuple(gradients)


def _survival_from_log_cumulative(log_cumulative: FloatArray) -> FloatArray:
    """Evaluate exp(-H) from log(H) without overflowing H."""
    if np.isnan(log_cumulative).any():
        raise ArithmeticError("cumulative hazard is not representable")
    result = np.ones_like(log_cumulative)
    representable = log_cumulative <= np.log(np.finfo(np.float64).max)
    with np.errstate(under="ignore", over="ignore", invalid="ignore"):
        result[representable] = np.exp(-np.exp(log_cumulative[representable]))
    result[~representable] = 0.0
    return result


def _deepsurv_gradient(
    t: FloatArray, e: FloatArray, z: FloatArray
) -> tuple[float, FloatArray, FloatArray, FloatArray]:
    """Exact full-risk-set Breslow objective and output gradient."""
    n = t.size
    eta = z[:, 0]
    event_times = np.unique(t[e == 1])
    gradient = np.zeros_like(z)
    loss = 0.0
    log_increments = np.empty(event_times.size, dtype=np.float64)
    for k, event_time in enumerate(event_times):
        deaths = np.flatnonzero((t == event_time) & (e == 1))
        risk = np.flatnonzero(t >= event_time)
        log_denominator = float(logsumexp(eta[risk]))
        multiplicity = deaths.size
        loss += (multiplicity * log_denominator - float(eta[deaths].sum())) / n
        gradient[risk, 0] += multiplicity * np.exp(eta[risk] - log_denominator) / n
        gradient[deaths, 0] -= 1.0 / n
        log_increments[k] = np.log(multiplicity) - log_denominator
    return loss, gradient, event_times, log_increments


def _cox_time_gradient(
    x: FloatArray,
    t: FloatArray,
    e: FloatArray,
    weights: tuple[FloatArray, ...],
    time_center: float,
    time_scale: float,
) -> tuple[float, tuple[FloatArray, ...], FloatArray, FloatArray]:
    """Exact full-risk-set Cox-Time loss and gradients at each event time."""
    n = t.size
    times = np.unique(t[e == 1])
    total = [np.zeros_like(w) for w in weights]
    baseline = np.empty(times.size, dtype=np.float64)
    objective = 0.0
    for k, event_time in enumerate(times):
        event_rows = np.flatnonzero((t == event_time) & (e == 1))
        risk_rows = np.flatnonzero(t >= event_time)
        scaled_time = (np.log1p(event_time) - time_center) / time_scale
        inputs = np.column_stack((x, np.full(n, scaled_time)))
        activations, preactivations = _forward(inputs, weights)
        eta = activations[-1][:, 0]
        log_denominator = float(logsumexp(eta[risk_rows]))
        multiplicity = event_rows.size
        objective += (multiplicity * log_denominator - float(eta[event_rows].sum())) / n
        probability = np.exp(eta[risk_rows] - log_denominator)
        output_gradient = np.zeros_like(activations[-1])
        output_gradient[risk_rows, 0] = multiplicity * probability / n
        output_gradient[event_rows, 0] -= 1.0 / n
        gradients = _backprop(activations, preactivations, weights, output_gradient)
        for layer, gradient in enumerate(gradients):
            total[layer] += gradient
        baseline[k] = np.log(multiplicity) - log_denominator
    return float(objective), tuple(total), times, baseline


def fit_survival_neural(
    time: ArrayLike,
    event: ArrayLike,
    x: ArrayLike,
    *,
    family: NeuralFamily,
    cuts: ArrayLike | None = None,
    n_cuts: int = 20,
    hidden_layers: tuple[int, ...] = (32, 32),
    max_epochs: int = 300,
    learning_rate: float = 0.003,
    patience: int = 30,
    alpha: float = 0.2,
    rank_sigma: float = 0.1,
    random_state: int = 0,
) -> SurvivalNeuralFit:
    """Fit one bounded neural survival model; all choices are Python defaults.

    Cut points for discrete families are estimated from training outcomes only.
    Cox objectives use exact full risk sets and Breslow ties. Floats are retained
    in contrast to the survivalmodels wrapper's integer cast.
    """
    if family not in _FAMILIES:
        raise ValueError("family must be one of coxtime, deepsurv, deephit, loghaz or pchazard")
    t, e, design = _inputs(time, event, x)
    epochs = _integer(max_epochs, "max_epochs", 1, 20_000)
    patience_n = _integer(patience, "patience", 1, epochs)
    nc = _integer(n_cuts, "n_cuts", 2, 200)
    if len(hidden_layers) > 4 or any(
        isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 256 for v in hidden_layers
    ):
        raise ValueError("hidden_layers must contain at most four widths in 1..256")
    if not hidden_layers:
        raise ValueError("hidden_layers must not be empty")
    if not np.isfinite(learning_rate) or learning_rate <= 0 or learning_rate > 0.1:
        raise ValueError("learning_rate must be finite and in (0, 0.1]")
    if family == "deephit" and (not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    if not np.isfinite(rank_sigma) or rank_sigma <= 0:
        raise ValueError("rank_sigma must be positive")
    if family in ("loghaz", "deephit", "pchazard"):
        requested_cuts = nc + 1 if family == "pchazard" and cuts is None else nc
        time_cuts = _validate_cuts(cuts, t, e, requested_cuts)
        if family == "pchazard" and np.any((t <= time_cuts[0]) & (e == 1)):
            raise ValueError("PCHazard cannot represent an event at or before its first cut")
    else:
        if cuts is not None:
            raise ValueError("cuts apply only to discrete-time families")
        time_cuts = np.empty(0, dtype=np.float64)
    unit = np.max(np.abs(design), axis=0)
    unit[unit == 0.0] = 1.0
    unit_design = design / unit
    center = np.mean(unit_design, axis=0)
    scale = np.std(unit_design, axis=0)
    scale[scale == 0] = 1.0
    normalized = (unit_design - center) / scale
    log_time = np.log1p(t)
    time_center = float(np.mean(log_time))
    time_scale = float(np.std(log_time))
    if time_scale == 0.0:
        time_scale = 1.0
    network_x = normalized
    n, p = network_x.shape
    if family == "coxtime":
        p += 1
    if family in ("deepsurv", "coxtime"):
        output_n = 1
    elif family == "pchazard":
        output_n = time_cuts.size - 1
    else:
        output_n = time_cuts.size
    widths = (p, *hidden_layers, output_n)
    n_events = int(np.count_nonzero(e))
    repeats = n_events if family == "coxtime" else 1
    parameter_count = sum((widths[i] + 1) * widths[i + 1] for i in range(len(widths) - 1))
    risk_scan_work = n * n_events if family in ("coxtime", "deepsurv") else 0
    pair_time_work = n * n * output_n if family == "deephit" and alpha < 1.0 else 0
    work = epochs * (n * parameter_count * repeats + risk_scan_work + pair_time_work + n * output_n)
    transient_bytes = 8 * (
        n * (3 * p + 4 * output_n + 3 * sum(hidden_layers))
        + parameter_count * 6
        + (n * output_n * 4 if family in ("deephit", "loghaz", "pchazard") else 0)
    )
    if (
        work > _MAX_WORK
        or transient_bytes > _MAX_TRAIN_BYTES
        or (family == "deephit" and alpha < 1.0 and n * n * output_n > _MAX_DEEPHIT_PAIR_TIME)
    ):
        raise ValueError("network training exceeds the bounded work or memory budget")
    seed = _integer(random_state, "random_state", 0, 2**32 - 1)
    rng = np.random.default_rng(seed)
    weights = _network_init(widths, rng)
    first = tuple(np.zeros_like(w) for w in weights)
    second = tuple(np.zeros_like(w) for w in weights)
    best = tuple(w.copy() for w in weights)
    best_loss = np.inf
    history = np.empty(epochs, dtype=np.float64)
    stale = 0
    baseline_times = np.empty(0, dtype=np.float64)
    baseline_log_hazard = np.empty(0, dtype=np.float64)
    for epoch in range(1, epochs + 1):
        if family == "coxtime":
            loss, gradients, baseline_times, baseline_log_hazard = _cox_time_gradient(
                network_x, t, e, weights, time_center, time_scale
            )
        else:
            activations, preactivations = _forward(network_x, weights)
            if family in ("loghaz", "deephit", "pchazard"):
                loss, grad_out, baseline_times, baseline_log_hazard = _discrete_loss_gradient(
                    family,
                    t,
                    e,
                    activations[-1],
                    time_cuts,
                    alpha=alpha,
                    rank_sigma=rank_sigma,
                )
            else:
                loss, grad_out, baseline_times, baseline_log_hazard = _deepsurv_gradient(
                    t, e, activations[-1]
                )
            gradients = _backprop(activations, preactivations, weights, grad_out)
        history[epoch - 1] = loss
        if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in gradients):
            raise ArithmeticError("neural survival objective is not finite")
        if loss < best_loss:
            best_loss = loss
            best = tuple(w.copy() for w in weights)
            stale = 0
        else:
            stale += 1
        if stale >= patience_n:
            break
        next_weights: list[FloatArray] = []
        next_first: list[FloatArray] = []
        next_second: list[FloatArray] = []
        for w, g, m, v in zip(weights, gradients, first, second):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                m_new = 0.9 * m + 0.1 * g
                v_new = 0.999 * v + 0.001 * g * g
                mhat = m_new / (1.0 - 0.9**epoch)
                vhat = v_new / (1.0 - 0.999**epoch)
                candidate = w - learning_rate * mhat / (np.sqrt(vhat) + 1e-8)
            if not all(np.isfinite(value).all() for value in (m_new, v_new, candidate)):
                raise ArithmeticError("Adam moment or weight update exceeds floating-point range")
            next_weights.append(candidate)
            next_first.append(m_new)
            next_second.append(v_new)
        weights, first, second = tuple(next_weights), tuple(next_first), tuple(next_second)
    epochs_run = epoch
    if family == "coxtime":
        _, _, baseline_times, baseline_log_hazard = _cox_time_gradient(
            network_x, t, e, best, time_center, time_scale
        )
    else:
        activations, _ = _forward(network_x, best)
        if family == "deepsurv":
            _, _, baseline_times, baseline_log_hazard = _deepsurv_gradient(t, e, activations[-1])
    return SurvivalNeuralFit(
        family,
        tuple(_freeze(w) for w in best),
        _freeze(unit),
        _freeze(center),
        _freeze(scale),
        time_center,
        time_scale,
        _freeze(time_cuts),
        _freeze(baseline_times),
        _freeze(baseline_log_hazard),
        _freeze(history[:epochs_run]),
        int(np.argmin(history[:epochs_run]) + 1),
        epochs_run,
        epochs_run < epochs,
        seed,
        n,
    )


def predict_survival_neural(fit: SurvivalNeuralFit, x: ArrayLike, times: ArrayLike) -> FloatArray:
    """Predict right-continuous survival at caller-supplied nonnegative times."""
    if not isinstance(fit, SurvivalNeuralFit):
        raise ValueError("fit must be a SurvivalNeuralFit")
    if np.iscomplexobj(x) or np.iscomplexobj(times):
        raise ValueError("x and times must be real")
    raw_design, raw_grid = np.asarray(x), np.asarray(times)
    if raw_design.size > 1_000_000 or raw_grid.size > _MAX_SURFACE:
        raise ValueError("prediction inputs exceed the bounded dimensions")
    design = finite(x, "x")
    grid = finite(times, "times")
    if design.ndim == 1:
        design = design[None, :]
    if (
        design.ndim != 2
        or design.shape[0] == 0
        or design.shape[1] != fit.feature_center.size
        or grid.ndim != 1
        or grid.size == 0
        or np.any(grid < 0)
    ):
        raise ValueError("x and times have incompatible shapes or invalid values")
    if design.shape[0] * grid.size > _MAX_SURFACE:
        raise ValueError("prediction surface exceeds the bounded cell budget")
    hidden_work = sum(weight.shape[0] * weight.shape[1] for weight in fit.weights)
    evaluations = design.shape[0] * max(
        1, fit.baseline_times.size if fit.family == "coxtime" else 1
    )
    if evaluations * hidden_work > _MAX_WORK:
        raise ValueError("prediction network work exceeds the bounded budget")
    output_units = fit.weights[-1].shape[1]
    hidden_units = sum(weight.shape[1] for weight in fit.weights[:-1])
    scratch_cells = (
        3 * design.shape[0] * design.shape[1]
        + 2 * design.shape[0] * grid.size
        + design.shape[0] * (4 * hidden_units + 8 * output_units + 1)
    )
    if scratch_cells * 8 > _MAX_TRAIN_BYTES:
        raise ValueError("prediction network scratch exceeds the bounded memory budget")
    extra_prediction_work = 0
    if fit.family == "coxtime":
        extra_prediction_work = design.shape[0] * fit.baseline_times.size * grid.size
    elif fit.family == "deephit":
        extra_prediction_work = design.shape[0] * grid.size * output_units
    if evaluations * hidden_work + extra_prediction_work > _MAX_WORK:
        raise ValueError("prediction network and time-grid work exceeds the bounded budget")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = (design / fit.feature_unit - fit.feature_center) / fit.feature_scale
    if not np.isfinite(normalized).all():
        raise ArithmeticError("prediction covariates are not representable on the fitted scale")
    if fit.family == "coxtime":
        out = np.ones((design.shape[0], grid.size), dtype=np.float64)
        log_cumulative = np.full(design.shape[0], -np.inf, dtype=np.float64)
        positions = np.searchsorted(fit.baseline_times, grid, side="right") - 1
        for k, (event_time, baseline_log_increment) in enumerate(
            zip(fit.baseline_times, fit.baseline_log_hazard)
        ):
            augmented = np.column_stack(
                (
                    normalized,
                    np.full(
                        design.shape[0],
                        (np.log1p(event_time) - fit.time_center) / fit.time_scale,
                    ),
                )
            )
            z = _forward(augmented, fit.weights)[0][-1][:, 0]
            log_cumulative = np.logaddexp(log_cumulative, baseline_log_increment + z)
            columns = np.flatnonzero(positions == k)
            if columns.size:
                out[:, columns] = _survival_from_log_cumulative(log_cumulative)[:, None]
        return _freeze(out)
    if fit.family == "deepsurv":
        z = _forward(normalized, fit.weights)[0][-1][:, 0]
        log_baseline_cumulative = np.logaddexp.accumulate(fit.baseline_log_hazard)
        positions = np.searchsorted(fit.baseline_times, grid, side="right") - 1
        out = np.ones((design.shape[0], grid.size), dtype=np.float64)
        for j, position in enumerate(positions):
            if position >= 0:
                out[:, j] = _survival_from_log_cumulative(log_baseline_cumulative[position] + z)
        return _freeze(out)
    z = _forward(normalized, fit.weights)[0][-1]
    if fit.family == "loghaz":
        log_survival = np.cumsum(-np.logaddexp(0.0, z), axis=1)
        idx = np.searchsorted(fit.cuts, grid, side="right") - 1
        result = np.ones((design.shape[0], grid.size))
        for j, k in enumerate(idx):
            if k >= 0:
                result[:, j] = np.exp(log_survival[:, min(k, z.shape[1] - 1)])
        return _freeze(result)
    if fit.family == "deephit":
        logits = np.column_stack((z, np.zeros(z.shape[0])))
        logp = logits - logsumexp(logits, axis=1)[:, None]
        log_tail = np.empty((design.shape[0], z.shape[1]), dtype=np.float64)
        tail = logp[:, -1].copy()
        for k in range(z.shape[1] - 1, -1, -1):
            log_tail[:, k] = tail
            tail = np.logaddexp(tail, logp[:, k])
        idx = np.searchsorted(fit.cuts, grid, side="right") - 1
        result = np.ones((design.shape[0], grid.size))
        for j, k in enumerate(idx):
            if k >= 0:
                result[:, j] = np.exp(log_tail[:, min(k, z.shape[1] - 1)])
        return _freeze(result)
    if fit.family == "pchazard":
        increments = np.logaddexp(0.0, z)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            cum = np.column_stack((np.zeros(design.shape[0]), np.cumsum(increments, axis=1)))
        idx = (np.searchsorted(fit.cuts, grid, side="right") - 1).clip(0, fit.cuts.size - 2)
        width = fit.cuts[idx + 1] - fit.cuts[idx]
        fraction = np.clip((grid - fit.cuts[idx]) / width, 0.0, 1.0)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result = np.exp(-cum[:, idx] - increments[:, idx] * fraction[None, :])
        return _freeze(result)
    raise ValueError("fit contains an unsupported family")


def survival_neural_contour(
    time: ArrayLike,
    event: ArrayLike,
    x: ArrayLike,
    continuous_column: int,
    *,
    family: NeuralFamily,
    grid: ArrayLike | None = None,
    profile: ArrayLike | None = None,
    times: ArrayLike | None = None,
    n_grid: int = 30,
    **fit_options: object,
) -> SurvivalNeuralContour:
    """Fit a neural family and form a bounded continuous-covariate contour."""
    if family not in _FAMILIES:
        raise ValueError("unsupported neural survival family")
    t, e, design = _inputs(time, event, x)
    col = _integer(continuous_column, "continuous_column", 0, design.shape[1] - 1)
    ng = _integer(n_grid, "n_grid", 2, 2000)
    if grid is None:
        unit = max(float(np.max(np.abs(design[:, col]))), 1.0)
        q025, q975 = np.quantile(design[:, col] / unit, [0.025, 0.975])
        values = np.linspace(q025, q975, ng) * unit
    else:
        if np.iscomplexobj(grid):
            raise ValueError("grid must be real")
        if np.asarray(grid).size > 2000:
            raise ValueError("grid may contain at most 2,000 values")
        values = finite(grid, "grid")
        if values.ndim != 1 or values.size < 2 or np.any(np.diff(values) <= 0):
            raise ValueError("grid must be an increasing vector with at least two values")
    if profile is None:
        column_unit = np.max(np.abs(design), axis=0)
        column_unit[column_unit == 0.0] = 1.0
        base = np.mean(design / column_unit, axis=0) * column_unit
    else:
        if np.iscomplexobj(profile):
            raise ValueError("profile must be real")
        base = finite(profile, "profile")
    if base.shape != (design.shape[1],):
        raise ValueError("profile must contain one value per covariate")
    if times is None:
        time_grid = np.unique(t)
    else:
        if np.iscomplexobj(times):
            raise ValueError("times must be real")
        if np.asarray(times).size > _MAX_SURFACE:
            raise ValueError("times exceed the bounded output dimensions")
        time_grid = finite(times, "times")
    if (
        time_grid.ndim != 1
        or time_grid.size == 0
        or np.any(time_grid < 0)
        or np.any(np.diff(time_grid) <= 0)
    ):
        raise ValueError("times must be an increasing vector")
    if values.size * time_grid.size > _MAX_SURFACE:
        raise ValueError("contour surface exceeds the bounded cell budget")
    hidden_layers = fit_options.get("hidden_layers", (32, 32))
    if (
        not isinstance(hidden_layers, tuple)
        or not hidden_layers
        or len(hidden_layers) > 4
        or any(
            isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= 256
            for width in hidden_layers
        )
    ):
        raise ValueError("hidden_layers must be a tuple of valid hidden widths")
    units = design.shape[1] + (1 if family == "coxtime" else 0)
    cuts_option = fit_options.get("cuts")
    n_cuts_option = fit_options.get("n_cuts", 20)
    requested_cuts = _integer(cast(int, n_cuts_option), "n_cuts", 2, 200)
    if cuts_option is not None:
        cuts_array = cast(ArrayLike, cuts_option)
        if np.iscomplexobj(cuts_array) or np.asarray(cuts_array).size > 201:
            raise ValueError("cuts must be real and contain at most 201 values")
        if np.asarray(cuts_array).ndim != 1:
            raise ValueError("cuts must be a vector")
    output_units = (
        1
        if family in ("coxtime", "deepsurv")
        else (
            np.asarray(cast(ArrayLike, cuts_option)).size - (1 if family == "pchazard" else 0)
            if cuts_option is not None
            else requested_cuts - (1 if family == "pchazard" else 0)
        )
    )
    parameter_cells = (units + 1) * hidden_layers[0]
    parameter_cells += sum(
        (hidden_layers[i] + 1) * hidden_layers[i + 1] for i in range(len(hidden_layers) - 1)
    )
    parameter_cells += (hidden_layers[-1] + 1) * output_units
    train_rows = design.shape[0]
    train_hidden = sum(hidden_layers)
    train_events = int(np.count_nonzero(e))
    output_cells = train_rows * output_units
    epoch_cap = _integer(cast(int, fit_options.get("max_epochs", 300)), "max_epochs", 1, 20_000)
    alpha_value = float(cast(float, fit_options.get("alpha", 0.2)))
    work = epoch_cap * (
        train_rows * parameter_cells * (train_events if family == "coxtime" else 1)
        + (train_rows * train_events if family in ("coxtime", "deepsurv") else 0)
        + (
            train_rows * train_rows * output_units
            if family == "deephit" and alpha_value < 1.0
            else 0
        )
        + output_cells
    )
    fit_peak_bytes = 8 * (
        train_rows * (3 * units + 4 * output_units + 3 * train_hidden) + 6 * parameter_cells
    )
    prediction_peak_bytes = 8 * (
        3 * values.size * design.shape[1]
        + 2 * values.size * time_grid.size
        + values.size * (4 * train_hidden + 8 * output_units + 1)
        + parameter_cells
        + 3 * train_rows * design.shape[1]
    )
    if (
        work > _MAX_WORK
        or max(fit_peak_bytes, prediction_peak_bytes) > _MAX_TRAIN_BYTES
        or (
            family == "deephit"
            and alpha_value < 1.0
            and train_rows * train_rows * output_units > _MAX_DEEPHIT_PAIR_TIME
        )
    ):
        raise ValueError("combined contour fit and prediction memory exceeds the bounded budget")
    prediction_x = np.tile(base, (values.size, 1))
    prediction_x[:, col] = values
    fit = fit_survival_neural(t, e, design, family=family, **fit_options)  # type: ignore[arg-type]
    surface = predict_survival_neural(fit, prediction_x, time_grid)
    return SurvivalNeuralContour(
        fit, col, _freeze(base), _freeze(values), _freeze(time_grid), surface
    )
