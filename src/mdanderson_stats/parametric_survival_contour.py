"""Continuous-covariate contours for parametric and spline survival fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

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
from .survival_uncertainty import (
    _MAX_PARAMETER_DRAWS,
    ParametricSurvivalMCPrediction,
    predict_parametric_survival_mc,
)
from .survival_uncertainty import (
    _MAX_SURFACE_CELLS as _MC_MAX_CELLS,
)
from .survival_uncertainty import (
    _MAX_WORK_CELLS as _MC_MAX_WORK,
)

if TYPE_CHECKING:
    from .generalized_gamma import GeneralizedGammaFit, GeneralizedGammaPrediction
    from .survival_spline import SurvivalSplineFit, SurvivalSplinePrediction

_DEFAULT_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
_MAX_CELLS = 2_000_000


@dataclass(frozen=True)
class ParametricSurvivalContour:
    """Parametric fit with primary and continuous-covariate quantile surfaces.

    ``lower`` and ``upper`` follow ``interval_method``. Both standard-error
    surfaces remain delta-method quantities; ``monte_carlo`` retains the
    joint draws and pointwise Monte Carlo diagnostics when requested.
    """

    fit: ParametricSurvivalFit | GeneralizedGammaFit | SurvivalSplineFit
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
    interval_method: Literal["delta", "monte_carlo"] = "delta"
    monte_carlo: ParametricSurvivalMCPrediction | None = None


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
    spline_k: int | None = None,
    spline_internal_knots: ArrayLike | None = None,
    interval_method: Literal["delta", "monte_carlo"] = "delta",
    draws: int = 1000,
    rng: np.random.Generator | int | None = None,
    parameter_draws: ArrayLike | None = None,
) -> ParametricSurvivalContour:
    """Fit a survival model and predict its continuous-covariate contour.

    Other covariates default to their training means, or to ``profile`` when
    supplied. Default prediction times are zero followed by distinct failures.
    Main surfaces have shape ``(n_grid, n_times)``; quantile summaries use
    the requested probabilities of the continuous covariate. By default,
    confidence limits use a deterministic delta method on log
    cumulative hazard. Set ``interval_method="monte_carlo"`` for joint-normal
    parameter draws and pointwise type-7 limits. In that mode ``draws`` and
    ``rng`` control generated draws; they are unused in delta mode. Optional
    ``parameter_draws`` reuses draws in the fitted model's scaled parameter
    coordinates and is accepted only for Monte Carlo intervals. The attached
    ``monte_carlo`` result exposes the draws and valid-count diagnostics.
    ``se_log_cumulative_hazard`` remains delta-derived in either mode.
    ``distribution="gengamma"`` selects Prentice generalized gamma;
    ``"gengamma.orig"`` selects the original Stacy parameterization.
    ``"spline_hazard"``, ``"spline_odds"`` and ``"spline_normal"`` select
    Royston–Parmar models. They default to four internal log-time knots;
    ``spline_k`` or ``spline_internal_knots`` can specify the knot choice.
    """
    spline_scales: dict[str, Literal["hazard", "odds", "normal"]] = {
        "spline_hazard": "hazard",
        "spline_odds": "odds",
        "spline_normal": "normal",
    }
    is_spline = distribution in tuple(spline_scales)
    if not is_spline and (spline_k is not None or spline_internal_knots is not None):
        raise ValueError("spline knot options require a spline distribution")
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
    if interval_method not in ("delta", "monte_carlo"):
        raise ValueError("interval_method must be 'delta' or 'monte_carlo'")
    if parameter_draws is not None and interval_method != "monte_carlo":
        raise ValueError("parameter_draws require interval_method='monte_carlo'")

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

    mc_draw_count = 0
    parameter_count = 0
    spline_basis_count = 0
    combined_profile_count = int(grid_values.size + probabilities.size)
    if interval_method == "monte_carlo":
        if distribution in ("weibull", "lognormal", "loglogistic"):
            parameter_count = int(design.shape[1] + 2)
        elif distribution in ("gengamma", "gengamma.orig"):
            parameter_count = int(design.shape[1] + 3)
        elif is_spline:
            from .survival_spline import _MAX_INTERNAL_KNOTS

            if spline_internal_knots is None:
                knot_count = 4 if spline_k is None else spline_k
                if (
                    isinstance(knot_count, (bool, np.bool_))
                    or not isinstance(knot_count, (int, np.integer))
                    or not 0 <= knot_count <= _MAX_INTERNAL_KNOTS
                ):
                    raise ValueError(f"spline_k must be an integer in [0, {_MAX_INTERNAL_KNOTS}]")
            else:
                if np.iscomplexobj(spline_internal_knots):
                    raise ValueError("spline_internal_knots must be real")
                inner = finite(spline_internal_knots, "spline_internal_knots")
                if inner.ndim != 1 or inner.size > _MAX_INTERNAL_KNOTS:
                    raise ValueError(
                        f"spline_internal_knots must contain at most {_MAX_INTERNAL_KNOTS} values"
                    )
                if spline_k is not None:
                    if (
                        isinstance(spline_k, (bool, np.bool_))
                        or not isinstance(spline_k, (int, np.integer))
                        or not 0 <= spline_k <= _MAX_INTERNAL_KNOTS
                    ):
                        raise ValueError(
                            f"spline_k must be an integer in [0, {_MAX_INTERNAL_KNOTS}]"
                        )
                    if spline_k != inner.size:
                        raise ValueError(
                            "spline_k must equal the number of explicit spline_internal_knots"
                        )
                if inner.size > 1 and np.any(np.diff(inner) <= 0):
                    raise ValueError("spline_internal_knots must be strictly increasing")
                knot_count = int(inner.size)
            spline_basis_count = int(knot_count + 2)
            parameter_count = spline_basis_count + int(design.shape[1])
        else:
            raise ValueError(f"unsupported survival distribution: {distribution}")

        if parameter_draws is None:
            if isinstance(draws, (bool, np.bool_)) or not isinstance(draws, (int, np.integer)):
                raise ValueError("draws must be an integer")
            if not 2 <= draws <= _MAX_PARAMETER_DRAWS:
                raise ValueError(f"draws must be in [2, {_MAX_PARAMETER_DRAWS}]")
            mc_draw_count = int(draws)
            if rng is not None and not isinstance(rng, (int, np.integer, np.random.Generator)):
                raise ValueError("rng must be an integer, Generator or None")
            if isinstance(rng, (int, np.integer)) and rng < 0:
                raise ValueError("rng integer seed must be nonnegative")
        else:
            if np.iscomplexobj(parameter_draws):
                raise ValueError("parameter_draws must be real")
            raw_parameter_draws = np.asarray(parameter_draws)
            if (
                raw_parameter_draws.ndim != 2
                or raw_parameter_draws.shape[1] != parameter_count
                or not 2 <= raw_parameter_draws.shape[0] <= _MAX_PARAMETER_DRAWS
                or raw_parameter_draws.dtype.kind not in "iuf"
                or raw_parameter_draws.size > _MC_MAX_CELLS
            ):
                raise ValueError("parameter_draws must have shape (B, parameter_count)")
            finite(raw_parameter_draws, "parameter_draws")
            mc_draw_count = int(raw_parameter_draws.shape[0])

        combined_surface = combined_profile_count * int(prediction_times.size)
        parameter_cells = mc_draw_count * parameter_count
        predictor_cells = (
            8 * combined_surface
            + combined_profile_count * parameter_count
            + parameter_cells
            + (mc_draw_count if is_spline else 0)
        )
        mc_extra_cells = (
            5 * combined_surface
            + parameter_cells
            + (mc_draw_count if is_spline else 0)
            + combined_profile_count * int(design.shape[1])
            + int(prediction_times.size)
            + int(design.shape[1])
        )
        spline_work = (
            mc_draw_count * spline_basis_count * spline_basis_count
            + mc_draw_count * combined_surface * spline_basis_count
        )
        if (
            predictor_cells > _MC_MAX_CELLS
            or cells + mc_extra_cells > _MC_MAX_CELLS
            or (mc_draw_count * combined_surface + spline_work > _MC_MAX_WORK)
        ):
            raise ValueError("combined Monte Carlo contour exceeds the bounded work or cell limit")

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
    fit: ParametricSurvivalFit | GeneralizedGammaFit | SurvivalSplineFit
    main: ParametricSurvivalPrediction | GeneralizedGammaPrediction | SurvivalSplinePrediction
    quantile: ParametricSurvivalPrediction | GeneralizedGammaPrediction | SurvivalSplinePrediction
    if is_spline:
        from .survival_spline import fit_survival_spline, predict_survival_spline

        spline_fit = fit_survival_spline(
            t,
            e,
            design,
            scale=spline_scales[distribution],
            k=spline_k,
            internal_knots=spline_internal_knots,
        )
        main = predict_survival_spline(spline_fit, prediction_times, main_profiles, confidence=conf)
        quantile = predict_survival_spline(
            spline_fit, prediction_times, quantile_profiles, confidence=conf
        )
        fit = spline_fit
    elif distribution in ("gengamma", "gengamma.orig"):
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
    monte_carlo: ParametricSurvivalMCPrediction | None = None
    if interval_method == "monte_carlo":
        main_cumulative_hazard = main.cumulative_hazard
        main_delta_se = main.se_log_cumulative_hazard
        quantile_cumulative_hazard = quantile.cumulative_hazard
        quantile_delta_se = quantile.se_log_cumulative_hazard
        del main, quantile
        combined_profiles = np.concatenate((main_profiles, quantile_profiles), axis=0)
        monte_carlo = predict_parametric_survival_mc(
            fit,
            prediction_times,
            combined_profiles,
            confidence=conf,
            draws=int(draws) if parameter_draws is None else draws,
            rng=rng,
            parameter_draws=parameter_draws,
        )
        del combined_profiles
        main_slice = slice(0, int(grid_values.size))
        quantile_slice = slice(int(grid_values.size), combined_profile_count)
        main_survival = monte_carlo.survival[main_slice]
        main_lower = monte_carlo.lower[main_slice]
        main_upper = monte_carlo.upper[main_slice]
        quantile_survival = monte_carlo.survival[quantile_slice]
        quantile_lower = monte_carlo.lower[quantile_slice]
        quantile_upper = monte_carlo.upper[quantile_slice]
    else:
        main_survival = main.survival
        main_lower = main.lower
        main_upper = main.upper
        main_cumulative_hazard = main.cumulative_hazard
        main_delta_se = main.se_log_cumulative_hazard
        quantile_survival = quantile.survival
        quantile_lower = quantile.lower
        quantile_upper = quantile.upper
        quantile_cumulative_hazard = quantile.cumulative_hazard
        quantile_delta_se = quantile.se_log_cumulative_hazard

    return ParametricSurvivalContour(
        fit,
        int(continuous_column),
        conf,
        _freeze(base_profile),
        _freeze(grid_values),
        _freeze(prediction_times),
        main_survival,
        main_lower,
        main_upper,
        main_cumulative_hazard,
        main_delta_se,
        _freeze(probabilities),
        _freeze(covariate_quantiles),
        _freeze(quantile_profiles),
        quantile_survival,
        quantile_lower,
        quantile_upper,
        quantile_cumulative_hazard,
        quantile_delta_se,
        interval_method,
        monte_carlo,
    )
