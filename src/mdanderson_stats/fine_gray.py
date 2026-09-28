# SPDX-License-Identifier: GPL-2.0-or-later
"""Fine–Gray proportional subdistribution hazards regression.

The estimating equations and censoring-adjusted sandwich follow ``cmprsk``
2.2-12 (Robert Gray, Copyright (C) 2000); Python adaptation Copyright 2026.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack
from scipy.special import logsumexp

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite, scalar

_MAX_ROWS = 100_000
_MAX_COLUMNS = 100
_MAX_DESIGN = 2_000_000
_MAX_RISK_WORK = 5_000_000


@dataclass(frozen=True)
class FineGrayFit:
    coefficients: FloatArray
    covariance: FloatArray
    observed_information: FloatArray
    sandwich_meat: FloatArray
    score: FloatArray
    log_likelihood: float
    null_log_likelihood: float
    event_times: FloatArray
    time_function_values: FloatArray
    baseline_increments: FloatArray
    cumulative_baseline: FloatArray
    score_residuals: FloatArray
    censoring_survival_left: FloatArray
    log_baseline_increments: FloatArray
    censoring_group_labels: tuple[str | int, ...]
    converged: bool
    iterations: int
    column_scale: FloatArray
    column_center: FloatArray
    scaled_coefficients: FloatArray
    _fixed_columns: int
    _fixed_origin: FloatArray
    _time_origin: FloatArray
    _time_scale: FloatArray
    _event_centers: FloatArray
    _centered_log_baseline: FloatArray


@dataclass(frozen=True)
class FineGrayPrediction:
    times: FloatArray
    cumulative_incidence: FloatArray
    cumulative_subdistribution_hazard: FloatArray
    subdistribution_survival: FloatArray


def fine_gray(
    time: ArrayLike,
    status: ArrayLike,
    x: ArrayLike | None = None,
    *,
    time_covariates: ArrayLike | None = None,
    time_functions: Callable[[FloatArray], ArrayLike] | None = None,
    censoring_groups: ArrayLike | None = None,
    failcode: int = 1,
    cencode: int = 0,
    initial: ArrayLike | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> FineGrayFit:
    """Fit a Fine–Gray model with Breslow target-event ties.

    ``x`` columns precede ``time_covariates * time_functions(t)`` columns.
    The callback is called once with sorted unique target-event times. Censoring
    groups have separate censoring Kaplan–Meier curves but shared target effects.
    """
    if np.iscomplexobj(time) or np.iscomplexobj(status):
        raise ValueError("time and status must be real")
    t, s = finite(time, "time"), count(status, "status")
    if t.ndim != 1 or s.shape != t.shape or not 2 <= t.size <= _MAX_ROWS or np.any(t < 0):
        raise ValueError(
            "time and status must be matching vectors with 2..100000 nonnegative times"
        )
    if (
        not isinstance(failcode, (int, np.integer))
        or isinstance(failcode, (bool, np.bool_))
        or not isinstance(cencode, (int, np.integer))
        or isinstance(cencode, (bool, np.bool_))
        or failcode < 0
        or cencode < 0
        or failcode >= 2**53
        or cencode >= 2**53
        or failcode == cencode
    ):
        raise ValueError("failcode and cencode must be distinct nonnegative integers")
    if isinstance(tolerance, (bool, np.bool_)) or np.iscomplexobj(tolerance):
        raise ValueError("tolerance must be a positive real scalar")
    tolerance = scalar(tolerance, "tolerance")
    if (
        tolerance <= 0
        or isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or not 1 <= max_iterations <= 1000
    ):
        raise ValueError("tolerance must be positive and max_iterations must be in 1..1000")
    target, censor = s == failcode, s == cencode
    competing = ~target & ~censor
    if not target.any():
        raise ValueError("Fine–Gray regression requires target events")
    if x is None:
        fixed = np.empty((t.size, 0))
    else:
        if np.iscomplexobj(x):
            raise ValueError("x must be real")
        fixed = finite(x, "x")
        if fixed.ndim == 1:
            fixed = fixed[:, None]
        if fixed.ndim != 2 or fixed.shape[0] != t.size:
            raise ValueError("x must have one row per observation")
    if time_covariates is None:
        if time_functions is not None:
            raise ValueError("time_functions requires time_covariates")
        time_x = np.empty((t.size, 0))
    else:
        if np.iscomplexobj(time_covariates) or not callable(time_functions):
            raise ValueError(
                "time_covariates must be real and supplied with a callable time_functions"
            )
        time_x = finite(time_covariates, "time_covariates")
        if time_x.ndim == 1:
            time_x = time_x[:, None]
        if time_x.ndim != 2 or time_x.shape[0] != t.size:
            raise ValueError("time_covariates must have one row per observation")
    events = np.unique(t[target])
    n, p_fixed, p_time = t.size, fixed.shape[1], time_x.shape[1]
    p = p_fixed + p_time
    if not 1 <= p <= _MAX_COLUMNS or fixed.size + time_x.size > _MAX_DESIGN:
        raise ValueError("require 1..100 covariates and at most 2,000,000 input design entries")
    if n * events.size > _MAX_RISK_WORK or n * events.size * p > _MAX_DESIGN:
        raise ValueError("Fine–Gray fit exceeds bounded row-event design limits")
    if time_functions is not None:
        raw_tf = np.asarray(time_functions(_freeze(events)))
        if np.iscomplexobj(raw_tf):
            raise ValueError("time_functions output must be real")
        tf = finite(raw_tf, "time_functions output")
        if tf.ndim == 1:
            tf = tf[:, None]
        if tf.shape != (events.size, p_time):
            raise ValueError("time_functions must return (#target event times, #time covariates)")
    else:
        tf = np.empty((events.size, 0))
    groups, labels = _encode_groups(censoring_groups, n)
    km_curves = [_km_curve(t, censor, groups, g) for g in range(len(labels))]
    gleft = np.array([_km_query(km_curves[groups[i]], t[i]) for i in range(n)])
    target_gleft = np.column_stack([_km_query(km_curves[g], events) for g in range(len(labels))])
    fixed_origin = fixed.mean(axis=0) if p_fixed else np.empty(0)
    time_origin = time_x.mean(axis=0) if p_time else np.empty(0)
    scale = np.ones(p)
    if p_fixed:
        scale[:p_fixed] = np.max(np.abs(fixed - fixed_origin), axis=0)
    if p_time:
        scale[p_fixed:] = np.max(np.abs(time_x - time_origin), axis=0)
    if np.any(scale == 0):
        raise ValueError("Fine–Gray design is rank deficient")
    fixed_norm = (fixed - fixed_origin) / scale[:p_fixed] if p_fixed else fixed
    time_norm = (time_x - time_origin) / scale[p_fixed:] if p_time else time_x
    event_centers = np.empty((events.size, p))
    for j in range(events.size):
        z = _event_design(fixed_norm, time_norm, tf, j, p_fixed)
        event_centers[j] = z.mean(axis=0)
    if initial is not None and np.iscomplexobj(initial):
        raise ValueError("initial must be real")
    initial_beta = np.zeros(p) if initial is None else finite(initial, "initial")
    if initial_beta.shape != (p,):
        raise ValueError("initial must contain one value per coefficient")
    beta = initial_beta * scale

    def evaluate(b: FloatArray, keep_risk: bool = False):
        nll = 0.0
        grad, info = np.zeros(p), np.zeros((p, p))
        logbase = np.empty(events.size)
        residuals = np.empty((events.size, p))
        risks = np.empty((events.size, n)) if keep_risk else None
        for j, event_time in enumerate(events):
            dmask = target & (t == event_time)
            d = int(dmask.sum())
            w = np.zeros(n)
            w[t >= event_time] = 1.0
            prior = competing & (t < event_time)
            if np.any(prior):
                numerator = target_gleft[j, groups[prior]]
                denominator = gleft[prior]
                w[prior] = np.divide(
                    numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
                )
            z = _event_design(fixed_norm, time_norm, tf, j, p_fixed) - event_centers[j]
            eta = z @ b
            logw = np.full(n, -np.inf)
            active = w > 0
            logw[active] = np.log(w[active]) + eta[active]
            lden = float(logsumexp(logw))
            prob = np.exp(logw - lden)
            mean = prob @ z
            centered = z - mean
            nll += d * lden - float(np.sum(eta[dmask]))
            grad += d * mean - np.sum(z[dmask], axis=0)
            info += d * (centered.T @ (prob[:, None] * centered))
            logbase[j] = np.log(d) - lden
            residuals[j] = np.sum(z[dmask], axis=0) - d * mean
            if keep_risk:
                assert risks is not None
                risks[j] = prob
        return nll, grad, (info + info.T) / 2, logbase, residuals, risks

    null = evaluate(np.zeros(p))
    nll, grad, info, logbase, residuals, _ = evaluate(beta)
    _check_separation(
        t,
        target,
        competing,
        groups,
        target_gleft,
        events,
        fixed_norm,
        time_norm,
        tf,
        p_fixed,
        event_centers,
        null[1],
    )
    converged = False
    for iteration in range(max_iterations + 1):
        eigen = np.linalg.eigvalsh(info)
        if eigen[-1] <= 0 or eigen[0] <= 64 * np.finfo(float).eps * p * eigen[-1]:
            raise ArithmeticError("Fine–Gray information is numerically singular")
        step = np.linalg.solve(info, grad)
        decrement = float(grad @ step)
        if decrement <= tolerance * tolerance:
            converged = True
            break
        if iteration == max_iterations:
            break
        fraction = 1.0
        for _ in range(50):
            candidate = beta - fraction * step
            trial = evaluate(candidate)
            slack = 8 * np.finfo(float).eps * max(1.0, abs(nll))
            if trial[0] <= nll - 1e-4 * fraction * decrement + slack:
                beta = candidate
                nll, grad, info, logbase, residuals = trial[:5]
                break
            fraction *= 0.5
        else:
            if np.max(np.abs(grad)) <= 2e-9:
                converged = True
                break
            raise ArithmeticError("Fine–Gray likelihood line search failed")
    if not converged:
        raise ArithmeticError(
            f"Fine–Gray optimization did not converge in {max_iterations} iterations"
        )
    nll, grad, info, logbase, residuals, risks = evaluate(beta, keep_risk=True)
    meat = _sandwich_meat(
        t,
        target,
        censor,
        competing,
        groups,
        events,
        fixed_norm,
        time_norm,
        tf,
        p_fixed,
        event_centers,
        risks,
        labels,
    )
    scaled_cov = np.linalg.solve(info, meat)
    scaled_cov = np.linalg.solve(info, scaled_cov.T).T
    with np.errstate(over="ignore", invalid="ignore"):
        coefficients = beta / scale
        covariance = scaled_cov / scale[:, None] / scale[None, :]
    if not np.isfinite(coefficients).all() or not np.isfinite(covariance).all():
        raise ArithmeticError("Fine–Gray coefficient units exceed numerical range")
    baseline_log = logbase - event_centers @ beta
    if p_fixed:
        baseline_log -= float((fixed_origin / scale[:p_fixed]) @ beta[:p_fixed])
    if p_time:
        time_origin_scaled = time_origin / scale[p_fixed:]
        baseline_log -= np.sum(tf * time_origin_scaled[None, :] * beta[None, p_fixed:], axis=1)
    with np.errstate(over="ignore", under="ignore"):
        increments = np.exp(baseline_log)
    score_resid = residuals * scale
    return FineGrayFit(
        _freeze(coefficients),
        _freeze((covariance + covariance.T) / 2),
        _freeze(info * scale[:, None] * scale[None, :]),
        _freeze(meat * scale[:, None] * scale[None, :]),
        _freeze(-grad * scale),
        -float(nll),
        -float(null[0]),
        _freeze(events),
        _freeze(tf),
        _freeze(increments),
        _freeze(np.cumsum(increments)),
        _freeze(score_resid),
        _freeze(gleft),
        _freeze(baseline_log),
        labels,
        True,
        iteration,
        _freeze(scale),
        _freeze(event_centers[0]),
        _freeze(beta),
        p_fixed,
        _freeze(fixed_origin),
        _freeze(time_origin),
        _freeze(scale[p_fixed:]),
        _freeze(event_centers),
        _freeze(logbase),
    )


def fine_gray_predict(
    fit: FineGrayFit,
    x: ArrayLike | None = None,
    *,
    times: ArrayLike | None = None,
    time_covariates: ArrayLike | None = None,
) -> FineGrayPrediction:
    """Predict cumulative incidence at requested times using fitted event steps."""
    if any(a is not None and np.iscomplexobj(a) for a in (x, time_covariates, times)):
        raise ValueError("prediction covariates and times must be real")
    p_fixed = fit._fixed_columns
    p_time = fit.time_function_values.shape[1]
    if p_fixed:
        if x is None:
            raise ValueError("x is required for this fitted model")
        xx = finite(x, "x")
        if xx.ndim == 1:
            xx = xx[None, :]
        if xx.ndim != 2 or xx.shape[1] != p_fixed:
            raise ValueError("x must have the fitted fixed-covariate count")
    else:
        if x is not None and np.asarray(x).size:
            raise ValueError("this model has no fixed covariates")
        xx = np.empty(
            (
                1
                if time_covariates is None
                else np.asarray(time_covariates).reshape(-1, p_time).shape[0],
                0,
            )
        )
    if p_time:
        if time_covariates is None:
            raise ValueError("time_covariates is required for this fitted model")
        if np.iscomplexobj(time_covariates):
            raise ValueError("time_covariates must be real")
        zraw = finite(time_covariates, "time_covariates")
        if zraw.ndim == 1:
            zraw = zraw[None, :]
        if zraw.ndim != 2 or zraw.shape != (xx.shape[0], p_time):
            raise ValueError("time_covariates must match profile rows and fitted columns")
    else:
        if time_covariates is not None:
            raise ValueError("this model has no time-varying covariates")
        zraw = np.empty((xx.shape[0], 0))
    if not 1 <= xx.shape[0] <= _MAX_ROWS or xx.size + zraw.size > _MAX_DESIGN:
        raise ValueError(
            "prediction requires 1..100000 profiles and at most 2,000,000 design entries"
        )
    requested = fit.event_times if times is None else finite(times, "times")
    if requested.ndim != 1 or np.any(requested < 0) or requested.size > _MAX_ROWS:
        raise ValueError("times must be a nonnegative vector of at most 100,000 values")
    if xx.shape[0] * max(1, requested.size + fit.event_times.size) > _MAX_DESIGN:
        raise ValueError("prediction exceeds the bounded profile-time/event limit")
    loghazards_at_events = np.full((xx.shape[0], fit.event_times.size), -np.inf)
    beta = fit.scaled_coefficients
    for j, tf in enumerate(fit.time_function_values):
        design = np.empty((xx.shape[0], beta.size))
        with np.errstate(over="ignore", invalid="ignore"):
            design[:, :p_fixed] = (
                ((xx - fit._fixed_origin) / fit.column_scale[:p_fixed]) if p_fixed else xx
            )
            if p_time:
                design[:, p_fixed:] = ((zraw - fit._time_origin) / fit._time_scale) * tf
            eta = (design - fit._event_centers[j]) @ beta
        if not np.isfinite(design).all() or not np.isfinite(eta).all():
            raise ArithmeticError("Fine–Gray prediction linear predictor exceeds numerical range")
        with np.errstate(over="ignore", under="ignore"):
            loghazards_at_events[:, j] = fit._centered_log_baseline[j] + eta
    log_cumulative_at_events = np.logaddexp.accumulate(loghazards_at_events, axis=1)
    idx = np.searchsorted(fit.event_times, requested, side="right") - 1
    hazards = np.zeros((xx.shape[0], requested.size))
    valid = idx >= 0
    if np.any(valid):
        with np.errstate(over="ignore", under="ignore"):
            hazards[:, valid] = np.exp(log_cumulative_at_events[:, idx[valid]])
    incidence = -np.expm1(-hazards)
    with np.errstate(under="ignore"):
        survival = np.exp(-hazards)
    return FineGrayPrediction(
        _freeze(requested), _freeze(incidence), _freeze(hazards), _freeze(survival)
    )


def _event_design(fixed, time_x, tf, j, p_fixed):
    n = fixed.shape[0]
    p_time = time_x.shape[1]
    out = np.empty((n, p_fixed + p_time))
    if p_fixed:
        out[:, :p_fixed] = fixed
    if p_time:
        out[:, p_fixed:] = time_x * tf[j]
    return out


def _encode_groups(values: ArrayLike | None, n: int) -> tuple[np.ndarray, tuple[str | int, ...]]:
    if values is None:
        return np.zeros(n, dtype=int), ("all",)
    if isinstance(values, (str, bytes)):
        raise ValueError("censoring_groups must be a label sequence")
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or raw.size != n:
        raise ValueError("censoring_groups must contain one label per row")
    labels: list[str | int] = []
    codes = np.empty(n, dtype=int)
    for i, item in enumerate(raw):
        if isinstance(item, (bool, np.bool_)):
            raise ValueError("censoring_groups labels must be strings or integers, not bool")
        if isinstance(item, (str, np.str_)):
            label: str | int = str(item)
        elif isinstance(item, (int, np.integer)) and abs(int(item)) < 2**53:
            label = int(item)
        else:
            raise ValueError("censoring_groups labels must be strings or exact integers")
        if label not in labels:
            labels.append(label)
        codes[i] = labels.index(label)
    if len(labels) > 100:
        raise ValueError("at most 100 censoring groups are supported")
    return codes, tuple(labels)


def _km_curve(t, censor, groups, group):
    rows = np.flatnonzero(groups == group)
    order = rows[np.argsort(t[rows], kind="stable")]
    risk, surv, j, table_times, after = order.size, 1.0, 0, [], []
    while j < order.size:
        k = j + 1
        while k < order.size and t[order[k]] == t[order[j]]:
            k += 1
        table_times.append(float(t[order[j]]))
        d = int(np.count_nonzero(censor[order[j:k]]))
        if d:
            surv *= (risk - d) / risk
        after.append(surv)
        risk -= k - j
        j = k
    return np.asarray(table_times), np.asarray(after)


def _km_query(curve, query):
    times, after = curve
    positions = np.searchsorted(times, query, side="left") - 1
    if not times.size:
        return np.ones(np.shape(query), dtype=float)
    return np.where(positions >= 0, after[np.maximum(positions, 0)], 1.0)


def _check_separation(
    t,
    target,
    competing,
    groups,
    target_gleft,
    events,
    fixed,
    time_x,
    tf,
    p_fixed,
    centers,
    null_gradient,
):
    """Reject a monotone likelihood using a bounded sparse feasibility LP."""
    blocks = []
    for j, event_time in enumerate(events):
        z = _event_design(fixed, time_x, tf, j, p_fixed) - centers[j]
        deaths = target & (t == event_time)
        mean_event = np.mean(z[deaths], axis=0)
        rows = z - mean_event
        # Risk rows with positive Fine–Gray weight: followed subjects and prior
        # competing events. Target-event ties force equal fitted directions.
        risk = (t >= event_time) | (competing & (t < event_time) & (target_gleft[j, groups] > 0))
        if np.any(risk):
            blocks.append(csr_matrix(rows[risk]))
    constraints = vstack(blocks, format="csr")
    objective = null_gradient / max(1.0, float(np.max(np.abs(null_gradient))))
    result = linprog(
        objective,
        A_ub=constraints,
        b_ub=np.zeros(constraints.shape[0]),
        bounds=[(-1, 1)] * objective.size,
        method="highs",
    )
    if not result.success:
        raise ArithmeticError(f"Fine–Gray separation check failed: {result.message}")
    if result.fun < -1e-7:
        raise ValueError("Fine–Gray monotone likelihood: no finite coefficient maximum")


def _sandwich_meat(
    t, target, censor, competing, groups, events, fixed, time_x, tf, p_fixed, centers, probs, labels
):
    n, p = t.size, fixed.shape[1] + time_x.shape[1]
    influence = np.zeros((n, p))
    q = np.zeros((n, p))
    for j, event_time in enumerate(events):
        z = _event_design(fixed, time_x, tf, j, p_fixed) - centers[j]
        d = int(np.count_nonzero(target & (t == event_time)))
        influence -= d * probs[j, :, None] * (z - probs[j] @ z)
        deaths = target & (t == event_time)
        influence[deaths] += z[deaths] - probs[j] @ z
    # Censoring-distribution correction, grouped as in native crrvv. For each
    # event time, strict-left prefix sums enforce competing time < censor time.
    for group in range(len(labels)):
        cmask = competing & (groups == group)
        for j, event_time in enumerate(events):
            dg = int(np.count_nonzero(target & (t == event_time) & (groups == group)))
            if not dg:
                continue
            z = _event_design(fixed, time_x, tf, j, p_fixed) - centers[j]
            rows = np.flatnonzero(cmask & (t < event_time))
            if not rows.size:
                continue
            rows = rows[np.argsort(t[rows], kind="stable")]
            vals = probs[j, rows, None] * (z[rows] - probs[j] @ z)
            prefix = np.cumsum(vals, axis=0)
            censor_rows = np.flatnonzero(censor & (groups == group) & (t <= event_time))
            positions = np.searchsorted(t[rows], t[censor_rows], side="left") - 1
            valid = positions >= 0
            q[censor_rows[valid]] += dg * prefix[positions[valid]]
    for group in range(len(labels)):
        risk = int(np.count_nonzero(groups == group))
        ss2 = np.zeros(p)
        for u in np.unique(t):
            rows = (t == u) & (groups == group)
            cens = rows & censor
            nc = int(cens.sum())
            if nc and risk:
                qr = q[np.flatnonzero(cens)[0]]
                ss2 -= nc * qr / (risk * risk)
            influence[rows] += ss2
            if nc and risk:
                influence[cens] += q[cens] / risk
            risk -= int(rows.sum())
    return influence.T @ influence
