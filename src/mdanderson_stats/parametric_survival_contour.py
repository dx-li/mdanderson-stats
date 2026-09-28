"""Continuous-covariate contours for exact parametric survival AFT fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar
from .parametric_survival import (
    ParametricSurvivalFit,
    ParametricSurvivalPrediction,
    fit_parametric_survival,
    predict_parametric_survival,
)

if TYPE_CHECKING:
    from .generalized_gamma import GeneralizedGammaFit, GeneralizedGammaPrediction

_DEFAULT_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
_MAX_CELLS = 2_000_000


@dataclass(frozen=True)
class ParametricSurvivalContour:
    """Parametric fit with primary and continuous-covariate quantile surfaces."""

    fit: ParametricSurvivalFit | GeneralizedGammaFit
    continuous_column: int
    confidence: float
    profile: FloatArray
    grid: FloatArray
    times: FloatArray
    survival: FloatArray
    lower: FloatArray
    upper: FloatArray
    cumulative_hazard: FloatArray
    se_log_cumulative_hazard: FloatArray
    quantile_probabilities: FloatArray
    covariate_quantiles: FloatArray
    quantile_profiles: FloatArray
    quantile_survival: FloatArray
    quantile_lower: FloatArray
    quantile_upper: FloatArray
    quantile_cumulative_hazard: FloatArray
    quantile_se_log_cumulative_hazard: FloatArray


def parametric_survival_contour(
    time: ArrayLike,
    event: ArrayLike,
    x: ArrayLike,
    continuous_column: int,
    *,
    distribution: str = "weibull",
    grid: ArrayLike | None = None,
    profile: ArrayLike | None = None,
    times: ArrayLike | None = None,
    n_grid: int = 30,
    confidence: float = 0.95,
    quantile_probabilities: ArrayLike = _DEFAULT_QUANTILES,
) -> ParametricSurvivalContour:
    """Fit a parametric AFT model and predict its continuous-covariate contour.

    Other covariates default to their training means, or to ``profile`` when
    supplied. Default prediction times are zero followed by distinct failures.
    Main surfaces have shape ``(n_grid, n_times)``; quantile summaries use
    the requested probabilities of the continuous covariate. Confidence
    limits use a deterministic delta method on log cumulative hazard.
    ``distribution="gengamma"`` selects Prentice generalized gamma;
    ``"gengamma.orig"`` selects the original Stacy parameterization.
    """
    if isinstance(continuous_column, (bool, np.bool_)) or not isinstance(
        continuous_column, (int, np.integer)
    ):
        raise ValueError("continuous_column must be an integer column index")
    if isinstance(n_grid, (bool, np.bool_)) or not isinstance(n_grid, (int, np.integer)):
        raise ValueError("n_grid must be an integer")
    if not 2 <= n_grid <= 2000:
        raise ValueError("n_grid must be in 2..2000")
    if np.iscomplexobj(confidence):
        raise ValueError("confidence must be real")
    conf = scalar(confidence, "confidence")
    if not 0 < conf < 1:
        raise ValueError("confidence must be in (0, 1)")

    if any(np.iscomplexobj(value) for value in (time, event, x)):
        raise ValueError("time, event and x must be real")
    t, e, design = finite(time, "time"), finite(event, "event"), finite(x, "x")
    if design.ndim == 1:
        design = design[:, None]
    if (
        t.ndim != 1
        or e.shape != t.shape
        or design.ndim != 2
        or design.shape[0] != t.size
        or not 1 <= t.size <= 20_000
        or not 1 <= design.shape[1] <= 16
        or design.size > 2_000_000
        or np.any(t < 0)
        or np.any((e != 0) & (e != 1))
    ):
        raise ValueError(
            "require aligned nonnegative times, binary events and at most 16 covariates"
        )
    if not 0 <= continuous_column < design.shape[1]:
        raise ValueError("continuous_column must index a covariate")

    if profile is None:
        mean_scales = np.max(np.abs(design), axis=0)
        mean_scales[mean_scales == 0] = 1.0
        base_profile = np.mean(design / mean_scales, axis=0) * mean_scales
    else:
        if np.iscomplexobj(profile):
            raise ValueError("profile must be real")
        base_profile = finite(profile, "profile")
        if base_profile.ndim != 1 or base_profile.size != design.shape[1]:
            raise ValueError("profile must contain one value per covariate")
    if not np.isfinite(base_profile).all():
        raise ArithmeticError("mean covariate profile is not representable")

    if grid is None:
        grid_scale = float(np.max(np.abs(design[:, continuous_column])))
        if grid_scale == 0:
            grid_scale = 1.0
        column = design[:, continuous_column] / grid_scale
        endpoints = np.quantile(column, [0.025, 0.975], method="linear")
        if endpoints[1] <= endpoints[0]:
            raise ValueError("continuous covariate must vary to form a contour grid")
        grid_values = np.linspace(endpoints[0], endpoints[1], int(n_grid)) * grid_scale
    else:
        if np.iscomplexobj(grid):
            raise ValueError("grid must be real")
        grid_values = finite(grid, "grid")
        if (
            grid_values.ndim != 1
            or not 2 <= grid_values.size <= 2000
            or np.any(np.diff(grid_values) <= 0)
        ):
            raise ValueError("grid must be strictly increasing with 2..2000 values")
    if not np.isfinite(grid_values).all():
        raise ArithmeticError("continuous-covariate grid is not representable")

    if times is None:
        prediction_times = np.r_[0.0, np.unique(t[e == 1])]
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
            raise ValueError("times must be strictly increasing nonnegative values")

    if np.iscomplexobj(quantile_probabilities):
        raise ValueError("quantile_probabilities must be real")
    probabilities = finite(quantile_probabilities, "quantile_probabilities")
    if (
        probabilities.ndim != 1
        or not 1 <= probabilities.size <= 20
        or np.any((probabilities <= 0) | (probabilities >= 1))
        or np.unique(probabilities).size != probabilities.size
    ):
        raise ValueError("quantile_probabilities must contain 1..20 distinct values in (0,1)")
    cells = (
        8 * (grid_values.size + probabilities.size) * prediction_times.size
        + (grid_values.size + probabilities.size) * design.shape[1]
        + grid_values.size
        + probabilities.size
        + prediction_times.size
    )
    if cells > _MAX_CELLS:
        raise ValueError("combined parametric contour output exceeds the 2,000,000-cell limit")

    main_profiles = np.repeat(base_profile[None, :], grid_values.size, axis=0)
    main_profiles[:, continuous_column] = grid_values

    quantile_scale = float(np.max(np.abs(design[:, continuous_column])))
    if quantile_scale == 0:
        quantile_scale = 1.0
    column = design[:, continuous_column] / quantile_scale
    covariate_quantiles = np.quantile(column, probabilities, method="linear") * quantile_scale
    if not np.isfinite(covariate_quantiles).all():
        raise ArithmeticError("continuous-covariate quantiles are not representable")
    quantile_profiles = np.repeat(base_profile[None, :], probabilities.size, axis=0)
    quantile_profiles[:, continuous_column] = covariate_quantiles
    # Fit only after both profile grids and their aggregate allocation are valid.
    fit: ParametricSurvivalFit | GeneralizedGammaFit
    main: ParametricSurvivalPrediction | GeneralizedGammaPrediction
    quantile: ParametricSurvivalPrediction | GeneralizedGammaPrediction
    if distribution in ("gengamma", "gengamma.orig"):
        from .generalized_gamma import fit_generalized_gamma, predict_generalized_gamma

        generalized_fit = fit_generalized_gamma(
            t,
            e,
            design,
            parameterization="prentice" if distribution == "gengamma" else "stacy",
        )
        main = predict_generalized_gamma(
            generalized_fit, prediction_times, main_profiles, confidence=conf
        )
        quantile = predict_generalized_gamma(
            generalized_fit, prediction_times, quantile_profiles, confidence=conf
        )
        fit = generalized_fit
    else:
        ordinary_fit = fit_parametric_survival(t, e, design, distribution=distribution)
        main = predict_parametric_survival(
            ordinary_fit, prediction_times, main_profiles, confidence=conf
        )
        quantile = predict_parametric_survival(
            ordinary_fit, prediction_times, quantile_profiles, confidence=conf
        )
        fit = ordinary_fit
    return ParametricSurvivalContour(
        fit,
        int(continuous_column),
        conf,
        _freeze(base_profile),
        _freeze(grid_values),
        _freeze(prediction_times),
        main.survival,
        main.lower,
        main.upper,
        main.cumulative_hazard,
        main.se_log_cumulative_hazard,
        _freeze(probabilities),
        _freeze(covariate_quantiles),
        _freeze(quantile_profiles),
        quantile.survival,
        quantile.lower,
        quantile.upper,
        quantile.cumulative_hazard,
        quantile.se_log_cumulative_hazard,
    )
