"""Bounded Cox survival surfaces and covariate-contour summaries.

This Python workflow fits a numeric Cox design, predicts at caller-encoded
covariate profiles, and returns pointwise log-scale confidence bands. It does
not encode categorical variables or estimate a nonparametric substitute
baseline. Breslow fitting remains available for compatibility; Efron is the
default here to match ``survival::coxph``. The five covariate summaries use
explicit probabilities (.10, .25, .50, .75, .90), a Python convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp, ndtri

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite
from .survan_cox import SurvanCox, survan_cox
from .survan_cox_likelihood import _combine_moments, _CoxLikelihood, _weighted_moments

_MAX_SURFACE_CELLS = 2_000_000
_DEFAULT_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


@dataclass(frozen=True)
class SurvivalCoxContour:
    """Cox fit and its primary/quantile survival prediction surfaces."""

    fit: SurvanCox
    continuous_column: int
    confidence: float
    profile: FloatArray
    grid: FloatArray
    times: FloatArray
    survival: FloatArray
    lower: FloatArray
    upper: FloatArray
    cumulative_hazard: FloatArray
    se_log_survival: FloatArray
    quantile_probabilities: FloatArray
    covariate_quantiles: FloatArray
    quantile_profiles: FloatArray
    quantile_survival: FloatArray
    quantile_lower: FloatArray
    quantile_upper: FloatArray
    quantile_cumulative_hazard: FloatArray
    quantile_se_log_survival: FloatArray


def survival_cox_contour(
    time: ArrayLike,
    event: ArrayLike,
    x: ArrayLike,
    continuous_column: int,
    *,
    grid: ArrayLike | None = None,
    profile: ArrayLike | None = None,
    times: ArrayLike | None = None,
    n_grid: int = 30,
    confidence: float = 0.95,
    quantile_probabilities: ArrayLike = _DEFAULT_QUANTILES,
    ties: str = "efron",
) -> SurvivalCoxContour:
    """Fit Cox regression and predict a continuous-covariate survival contour.

    The default grid spans type-7 empirical quantiles .025 through .975.
    Other covariates default to training means; ``profile`` replaces that
    complete mean profile before the continuous coordinate is varied. The
    numeric design matrix must already encode categorical variables. Native
    default prediction times are all distinct observed times; supplied times
    must be increasing and use a right-continuous step curve, flat before the
    first observed time and after the last.

    Main surfaces have shape ``(n_grid, n_times)``. Quantile curves use the
    explicitly selected covariate quantiles and have shape
    ``(n_quantiles, n_times)``. A combined cap bounds returned surface cells.
    """
    if isinstance(continuous_column, (bool, np.bool_)) or not isinstance(
        continuous_column, (int, np.integer)
    ):
        raise ValueError("continuous_column must be an integer column index")
    if isinstance(n_grid, (bool, np.bool_)) or not isinstance(n_grid, (int, np.integer)):
        raise ValueError("n_grid must be an integer")
    if not 2 <= n_grid <= 2000:
        raise ValueError("n_grid must be in 2..2000")
    conf = _scalar_probability(confidence, "confidence", open_interval=True)
    if ties not in ("breslow", "efron"):
        raise ValueError("ties must be 'breslow' or 'efron'")

    if np.iscomplexobj(x):
        raise ValueError("x must be real")
    design = finite(x, "x")
    if design.ndim == 1:
        design = design[:, None]
    if design.ndim != 2 or not 0 <= continuous_column < design.shape[1]:
        raise ValueError("x must be a matrix and continuous_column must index a column")
    if design.shape[0] > 100_000 or design.shape[1] > 100:
        raise ValueError("x is limited to 100,000 rows and 100 numeric covariates")
    if any(np.iscomplexobj(a) for a in (time, event)):
        raise ValueError("time and event must be real")
    t = finite(time, "time")
    e = count(event, "event")
    if t.ndim != 1 or e.shape != t.shape or t.size != design.shape[0]:
        raise ValueError("time, event and x must have matching row counts")
    if np.any(t < 0) or np.any(e > 1):
        raise ValueError("time must be nonnegative and event binary")
    scale = np.max(np.abs(design), axis=0)
    scale[scale == 0] = 1.0
    normalized = design / scale
    center = np.mean(normalized, axis=0)
    mean_profile = scale * center
    if not np.isfinite(mean_profile).all():
        raise ArithmeticError("mean covariate profile is not representable")
    if profile is None:
        base_profile = mean_profile
    else:
        if np.iscomplexobj(profile):
            raise ValueError("profile must be real")
        raw = np.asarray(profile)
        if raw.ndim != 1 or raw.size != design.shape[1]:
            raise ValueError("profile must contain one value per covariate")
        base_profile = finite(raw, "profile")
    if grid is None:
        column = normalized[:, continuous_column]
        quantile_grid = np.quantile(column, [0.025, 0.975], method="linear")
        grid_values = (
            np.linspace(quantile_grid[0], quantile_grid[1], int(n_grid)) * scale[continuous_column]
        )
    else:
        if np.iscomplexobj(grid):
            raise ValueError("grid must be real")
        grid_values = finite(grid, "grid")
        if grid_values.ndim != 1 or not 2 <= grid_values.size <= 2000:
            raise ValueError("grid must be a one-dimensional array of 2..2000 values")
        if np.any(np.diff(grid_values) <= 0):
            raise ValueError("grid values must be strictly increasing")
    if not np.isfinite(grid_values).all():
        raise ArithmeticError("continuous-covariate grid is not representable")
    if np.iscomplexobj(quantile_probabilities):
        raise ValueError("quantile_probabilities must be real")
    probabilities = finite(quantile_probabilities, "quantile_probabilities")
    if (
        probabilities.ndim != 1
        or not 1 <= probabilities.size <= 20
        or np.any((probabilities <= 0) | (probabilities >= 1))
    ):
        raise ValueError("quantile_probabilities must contain 1..20 values in (0,1)")
    unique_probabilities = np.unique(probabilities)
    if unique_probabilities.size != probabilities.size:
        raise ValueError("quantile_probabilities must be distinct")

    if times is None:
        prediction_times = np.unique(t)
    else:
        if np.iscomplexobj(times):
            raise ValueError("times must be real")
        prediction_times = finite(times, "times")
        if (
            prediction_times.ndim != 1
            or prediction_times.size == 0
            or prediction_times.size > 100_000
            or np.any(prediction_times < 0)
            or np.any(np.diff(prediction_times) <= 0)
        ):
            raise ValueError("times must be increasing nonnegative values (maximum 100,000)")
    cells = (
        5 * (grid_values.size + probabilities.size) * prediction_times.size
        + (grid_values.size + probabilities.size) * design.shape[1]
        + grid_values.size
        + probabilities.size
        + prediction_times.size
    )
    if cells > _MAX_SURFACE_CELLS:
        raise ValueError("combined Cox contour output exceeds the 2,000,000-cell limit")

    fit = survan_cox(t, e, design, ties=ties)

    event_times, baseline = _cox_baseline(t, e, normalized - center, fit.scaled_coefficients, ties)
    zcrit = float(-ndtri((1 - conf) / 2))
    scaled_covariance = fit.covariance * scale[:, None] * scale[None, :]

    main_profiles = np.repeat(base_profile[None, :], grid_values.size, axis=0)
    main_profiles[:, int(continuous_column)] = grid_values
    survival, lower, upper, hazard, standard_error = _predict_profiles(
        main_profiles,
        normalized,
        center,
        scale,
        fit.scaled_coefficients,
        scaled_covariance,
        event_times,
        baseline,
        prediction_times,
        zcrit,
    )

    column_norm = normalized[:, int(continuous_column)]
    q_norm = np.quantile(column_norm, probabilities, method="linear")
    q_values = q_norm * scale[int(continuous_column)]
    if not np.isfinite(q_values).all():
        raise ArithmeticError("covariate quantiles are not representable")
    quantile_profiles = np.repeat(base_profile[None, :], probabilities.size, axis=0)
    quantile_profiles[:, int(continuous_column)] = q_values
    q_survival, q_lower, q_upper, q_hazard, q_se = _predict_profiles(
        quantile_profiles,
        normalized,
        center,
        scale,
        fit.scaled_coefficients,
        scaled_covariance,
        event_times,
        baseline,
        prediction_times,
        zcrit,
    )
    return SurvivalCoxContour(
        fit,
        int(continuous_column),
        conf,
        _freeze(base_profile),
        _freeze(grid_values),
        _freeze(prediction_times),
        _freeze(survival),
        _freeze(lower),
        _freeze(upper),
        _freeze(hazard),
        _freeze(standard_error),
        _freeze(probabilities),
        _freeze(q_values),
        _freeze(quantile_profiles),
        _freeze(q_survival),
        _freeze(q_lower),
        _freeze(q_upper),
        _freeze(q_hazard),
        _freeze(q_se),
    )


def plot_survival_contour_2d(
    result: SurvivalCoxContour, *, ax: Any | None = None, levels: int = 12
) -> Any:
    """Lazily draw survival contours; requires the optional Matplotlib extra."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    contours = ax.contourf(result.grid, result.times, result.survival.T, levels=levels)  # type: ignore[attr-defined]
    ax.figure.colorbar(contours, ax=ax, label="Survival probability")  # type: ignore[attr-defined]
    ax.set_xlabel(f"Covariate column {result.continuous_column}")  # type: ignore[attr-defined]
    ax.set_ylabel("Time")  # type: ignore[attr-defined]
    return ax


def plot_survival_contour_3d(
    result: SurvivalCoxContour, *, ax: Any | None = None, surface: str = "survival"
) -> Any:
    """Lazily draw the returned survival or confidence-limit surface."""
    import matplotlib.pyplot as plt

    if surface not in ("survival", "lower", "upper"):
        raise ValueError("surface must be survival, lower or upper")
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")
    values = getattr(result, surface)
    xx, tt = np.meshgrid(result.grid, result.times)
    ax.plot_surface(xx, tt, values.T)  # type: ignore[attr-defined]
    ax.set_xlabel(f"Covariate column {result.continuous_column}")  # type: ignore[attr-defined]
    ax.set_ylabel("Time")  # type: ignore[attr-defined]
    ax.set_zlabel(surface)  # type: ignore[attr-defined]
    return ax


def _cox_baseline(
    time: FloatArray,
    event: FloatArray,
    design: FloatArray,
    beta: FloatArray,
    ties: str,
) -> tuple[FloatArray, tuple[FloatArray, FloatArray, FloatArray]]:
    model = _CoxLikelihood(time, event, design, ties=ties)
    eta = model.x @ beta
    if not np.isfinite(eta).all():
        raise ArithmeticError("Cox baseline linear predictors are not finite")
    p = design.shape[1]
    zero_mean, zero_cov = np.zeros(p), np.zeros((p, p))
    risk_log = -np.inf
    risk_mean, risk_cov = zero_mean.copy(), zero_cov.copy()
    hazard_log_increments = np.full(model.starts.size, -np.inf)
    variance_log_increments = np.full(model.starts.size, -np.inf)
    gradient_mean_increments = np.zeros((model.starts.size, p))
    for group, (start, end) in enumerate(zip(model.starts, model.ends, strict=True)):
        block_x, block_eta = model.x[start : end + 1], eta[start : end + 1]
        block_event = model.event[start : end + 1].astype(bool)
        block_shift = float(np.max(block_eta))
        block_log, block_mean, block_cov = _weighted_moments(
            block_x, np.exp(block_eta - block_shift), block_shift
        )
        deaths = int(model.deaths[group])
        if deaths:
            log_denominators: list[float] = []
            means: list[FloatArray] = []
            if ties == "breslow":
                logden, mean, _ = _combine_moments(
                    [(risk_log, risk_mean, risk_cov), (block_log, block_mean, block_cov)]
                )
                log_denominators = [logden] * deaths
                means = [mean] * deaths
            else:
                dx, de = block_x[block_event], block_eta[block_event]
                dshift = float(np.max(de))
                dlog, dmean, dcov = _weighted_moments(dx, np.exp(de - dshift), dshift)
                other = ~block_event
                if np.any(other):
                    ox, oe = block_x[other], block_eta[other]
                    oshift = float(np.max(oe))
                    olog, omean, ocov = _weighted_moments(ox, np.exp(oe - oshift), oshift)
                else:
                    olog, omean, ocov = -np.inf, zero_mean, zero_cov
                for tied_index in range(deaths):
                    q = tied_index / deaths
                    components = [(risk_log, risk_mean, risk_cov), (olog, omean, ocov)]
                    if q < 1:
                        components.append((dlog + float(np.log1p(-q)), dmean, dcov))
                    logden, mean, _ = _combine_moments(components)
                    log_denominators.append(logden)
                    means.append(mean)
            log_h_increment = float(logsumexp(-np.asarray(log_denominators)))
            log_v_increment = float(logsumexp(-2 * np.asarray(log_denominators)))
            inverse_weights = np.exp(-np.asarray(log_denominators) - log_h_increment)
            mean_increment = sum(
                (weight * value for weight, value in zip(inverse_weights, means, strict=True)),
                np.zeros(p),
            )
            hazard_log_increments[group] = log_h_increment
            variance_log_increments[group] = log_v_increment
            gradient_mean_increments[group] = mean_increment
        risk_log, risk_mean, risk_cov = _combine_moments(
            [(risk_log, risk_mean, risk_cov), (block_log, block_mean, block_cov)]
        )

    # The descending scan accumulates risk sets; prediction uses ascending time.
    base_log_h = np.empty(model.starts.size)
    base_log_v = np.empty(model.starts.size)
    base_a_over_h = np.zeros((model.starts.size, p))
    cumulative_h = -np.inf
    cumulative_v = -np.inf
    a_over_h = np.zeros(p)
    for index, group in enumerate(range(model.starts.size - 1, -1, -1)):
        increment_h = hazard_log_increments[group]
        increment_v = variance_log_increments[group]
        if np.isfinite(increment_h):
            new_h = float(np.logaddexp(cumulative_h, increment_h))
            previous_fraction = (
                0.0 if not np.isfinite(cumulative_h) else float(np.exp(cumulative_h - new_h))
            )
            increment_fraction = float(np.exp(increment_h - new_h))
            a_over_h = (
                previous_fraction * a_over_h + increment_fraction * gradient_mean_increments[group]
            )
            cumulative_h = new_h
        if np.isfinite(increment_v):
            cumulative_v = float(np.logaddexp(cumulative_v, increment_v))
        base_log_h[index] = cumulative_h
        base_log_v[index] = cumulative_v
        base_a_over_h[index] = a_over_h
    event_times = model.time[model.ends[::-1]]
    return event_times, (_freeze(base_log_h), _freeze(base_log_v), _freeze(base_a_over_h))


def _predict_profiles(
    profiles: FloatArray,
    normalized_design: FloatArray,
    center: FloatArray,
    scale: FloatArray,
    beta: FloatArray,
    covariance: FloatArray,
    event_times: FloatArray,
    baseline: tuple[FloatArray, FloatArray, FloatArray],
    times: FloatArray,
    zcrit: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    log_h0, log_v0, a_over_h0 = baseline
    indices = np.searchsorted(event_times, times, side="right") - 1
    active = indices >= 0
    selected = np.maximum(indices, 0)
    log_hazards = np.where(active, log_h0[selected], -np.inf)
    log_variances = np.where(active, log_v0[selected], -np.inf)
    gradients = np.where(active[:, None], a_over_h0[selected], 0.0)
    nprofile, ntime = profiles.shape[0], times.size
    out = [np.empty((nprofile, ntime)) for _ in range(5)]
    scaled_profile = profiles / scale - center
    eta_profiles = scaled_profile @ beta
    for i, (z, eta) in enumerate(zip(scaled_profile, eta_profiles, strict=True)):
        log_h = log_hazards + eta
        log_base_se = 0.5 * log_variances + eta
        h0_z_minus_a = z[None, :] - gradients
        coordinate_variance = np.einsum("ti,ij,tj->t", h0_z_minus_a, covariance, h0_z_minus_a)
        coordinate_variance = np.maximum(0.0, coordinate_variance)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            h = np.exp(log_h)
            se = np.hypot(np.exp(log_base_se), h * np.sqrt(coordinate_variance))
            surv = np.exp(-h)
            lower = np.exp(-h - zcrit * se)
            upper = np.exp(np.minimum(0.0, -h + zcrit * se))
        if any(np.isnan(value).any() for value in (h, se, surv, lower, upper)):
            raise ArithmeticError("Cox prediction produced NaN values")
        for target, value in zip(out, (surv, lower, upper, h, se), strict=True):
            target[i] = value
    if not all(np.isfinite(value).all() for value in out):
        raise ArithmeticError("Cox prediction surface exceeds numerical range")
    return tuple(out)  # type: ignore[return-value]


def _scalar_probability(value: float, name: str, *, open_interval: bool) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind == "b":
        raise ValueError(f"{name} must be a scalar probability")
    result = float(raw)
    if not isfinite(result) or (not 0 < result < 1 if open_interval else not 0 <= result <= 1):
        raise ValueError(f"{name} must lie in the probability range")
    return result
