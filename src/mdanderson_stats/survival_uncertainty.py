"""Joint-normal Monte Carlo uncertainty for parametric survival curves."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import log_ndtr

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar
from .generalized_gamma import (
    GeneralizedGammaFit,
    _near_zero_mask,
    _prentice_logtails,
)
from .parametric_survival import (
    ParametricSurvivalFit,
    _terms,
)
from .survival_spline import (
    SurvivalSplineFit,
    _basis,
    _minimum_slope,
)

_MAX_SURFACE_CELLS = 2_000_000
_MAX_WORK_CELLS = 20_000_000
_MAX_PARAMETER_DRAWS = 100_000
_MAX_DRAW_CELL_BUFFER = 200_000


@dataclass(frozen=True)
class ParametricSurvivalMCPrediction:
    """Pointwise simulation limits from joint asymptotic-normal parameter draws.

    ``parameter_draws`` are in the fit's retained scaled coordinates. The
    per-cell ``valid_draws`` count excludes only NaN survival values, matching
    flexsurv's pointwise quantile ``na.rm=TRUE`` behavior. For spline fits,
    evaluable nonmonotone coefficient draws are retained and their minimum
    slopes are reported separately. ``simulated_sd`` follows the source's
    ordinary sample-SD convention and becomes NaN if any draw for that cell
    is NaN.
    """

    profile: FloatArray
    times: FloatArray
    log_survival: FloatArray
    survival: FloatArray
    lower: FloatArray
    upper: FloatArray
    simulated_sd: FloatArray
    valid_draws: NDArray[np.int64]
    parameter_draws: FloatArray
    confidence: float
    spline_minimum_slope: FloatArray | None = None

    @property
    def draws(self) -> int:
        return int(self.parameter_draws.shape[0])


def _integer(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return result


def _fit_parameters(
    fit: ParametricSurvivalFit | GeneralizedGammaFit | SurvivalSplineFit,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, str | None]:
    mean = np.asarray(fit.scaled_parameters, dtype=np.float64)
    covariance = np.asarray(fit.scaled_covariance, dtype=np.float64)
    if mean.ndim != 1 or mean.size < 2 or covariance.shape != (mean.size, mean.size):
        raise ValueError("fit parameter and covariance dimensions are inconsistent")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("fit parameters and covariance must be finite")
    cov_mean = np.asarray(fit.covariate_mean, dtype=np.float64)
    cov_scale = np.asarray(fit.covariate_scale, dtype=np.float64)
    if (
        cov_mean.ndim != 1
        or cov_scale.shape != cov_mean.shape
        or not np.isfinite(cov_mean).all()
        or not np.isfinite(cov_scale).all()
        or np.any(cov_scale <= 0)
    ):
        raise ValueError("fit covariate normalization is inconsistent")
    parameterization: str | None
    if isinstance(fit, ParametricSurvivalFit):
        if fit.distribution not in ("weibull", "lognormal", "loglogistic"):
            raise ValueError("fit has an unsupported parametric distribution")
        expected = cov_mean.size + 2
        parameterization = None
    elif isinstance(fit, GeneralizedGammaFit):
        if fit.parameterization not in ("prentice", "stacy"):
            raise ValueError("fit has an unsupported generalized-gamma parameterization")
        expected = cov_mean.size + 3
        parameterization = fit.parameterization
    elif isinstance(fit, SurvivalSplineFit):
        if fit.scale not in ("hazard", "odds", "normal"):
            raise ValueError("fit has an unsupported spline scale")
        knots = np.asarray(fit.scaled_knots, dtype=np.float64)
        if (
            knots.ndim != 1
            or knots.size != fit.k + 2
            or knots.size < 2
            or not np.isfinite(knots).all()
            or np.any(np.diff(knots) <= 0)
        ):
            raise ValueError("fit spline knots are inconsistent")
        expected = knots.size + cov_mean.size
        parameterization = fit.scale
    else:
        raise ValueError("fit must be a parametric, generalized-gamma or spline survival fit")
    if mean.size != expected:
        raise ValueError("fit parameters do not match the covariate count")
    if (
        not np.isfinite(fit.log_time_center)
        or not np.isfinite(fit.log_time_scale)
        or fit.log_time_scale <= 0
    ):
        raise ValueError("fit time normalization is inconsistent")
    return mean, covariance, cov_mean, cov_scale, parameterization


def _draw_is_representable(
    parameters: FloatArray,
    fit: ParametricSurvivalFit | GeneralizedGammaFit | SurvivalSplineFit,
    parameterization: str | None,
) -> bool:
    """Whether the coefficient draw can be evaluated in the fitted model."""
    if not np.isfinite(parameters).all():
        return False
    if isinstance(fit, SurvivalSplineFit):
        # flexsurv samples these coordinates without a monotonicity restriction.
        return True
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if isinstance(fit, ParametricSurvivalFit):
            sigma = np.exp(parameters[-1])
            return bool(np.isfinite(sigma) and sigma > 0)
        if parameterization == "prentice":
            sigma = np.exp(parameters[-2])
            q = parameters[-1]
            return bool(np.isfinite(sigma) and sigma > 0 and abs(q) <= 1e150)
        log_shape, log_k = parameters[-2:]
        shape = np.exp(log_shape)
        sqrt_k = np.exp(0.5 * log_k)
        q = np.exp(-0.5 * log_k)
        return bool(
            np.isfinite(shape)
            and shape > 0
            and np.isfinite(sqrt_k)
            and sqrt_k > 0
            and np.isfinite(q)
            and 0 < q <= 1e150
        )


def _draw_parameters(
    mean: FloatArray,
    covariance: FloatArray,
    draws: int,
    rng: np.random.Generator | int | None,
) -> FloatArray:
    symmetric = (covariance + covariance.T) / 2
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14):
        raise ValueError("fit covariance is not symmetric")
    try:
        factor = np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as error:
        raise ValueError("fit covariance must be positive definite") from error
    generator = np.random.default_rng(rng)
    with np.errstate(over="ignore", invalid="ignore"):
        standard = generator.standard_normal((draws, mean.size))
        result = mean[None, :] + standard @ factor.T
    if not np.isfinite(result).all():
        raise ArithmeticError("joint normal parameter draws exceed numerical range")
    return result


def _freeze_int(value: ArrayLike) -> NDArray[np.int64]:
    array = np.ascontiguousarray(value, dtype=np.int64)
    return np.frombuffer(array.tobytes(), dtype=np.int64).reshape(array.shape)


def _survival_draw(
    fit: ParametricSurvivalFit | GeneralizedGammaFit | SurvivalSplineFit,
    parameters: FloatArray,
    design: FloatArray,
    log_time: FloatArray,
    parameterization: str | None,
    spline_basis: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Evaluate one parameter draw at one profile and a time block."""
    if not np.isfinite(parameters).all():
        nan = np.full(log_time.shape, np.nan)
        return nan, nan.copy()
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        if isinstance(fit, SurvivalSplineFit):
            basis = _basis(fit.scaled_knots, log_time) if spline_basis is None else spline_basis
            covariates = np.broadcast_to(design[1:], (log_time.size, design.size - 1))
            spline_design = np.column_stack((basis, covariates))
            eta = spline_design @ parameters
            if fit.scale == "hazard":
                log_survival = -np.exp(eta)
            elif fit.scale == "odds":
                log_survival = -np.logaddexp(0.0, eta)
            else:
                log_survival = log_ndtr(-eta)
            survival = np.exp(log_survival)
            return log_survival, survival

        residual = log_time - float(design @ parameters[: design.size])
        if isinstance(fit, ParametricSurvivalFit):
            sigma = float(np.exp(parameters[-1]))
            if not np.isfinite(sigma) or sigma <= 0:
                nan = np.full(log_time.shape, np.nan)
                return nan, nan.copy()
            z = residual / sigma
            log_survival = np.full(log_time.shape, np.nan)
            negative_infinite = np.isneginf(z)
            positive_infinite = np.isposinf(z)
            log_survival[negative_infinite] = 0.0
            log_survival[positive_infinite] = -np.inf
            finite_z = np.isfinite(z)
            if np.any(finite_z):
                log_survival[finite_z] = _terms(z[finite_z], fit.distribution)[3]
            survival = np.exp(log_survival)
            return log_survival, survival

        if parameterization == "prentice":
            sigma = float(np.exp(parameters[design.size]))
            q = float(parameters[design.size + 1])
            if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(q) or abs(q) > 1e150:
                nan = np.full(log_time.shape, np.nan)
                return nan, nan.copy()
            standardized = residual / sigma
        else:
            log_shape = float(parameters[design.size])
            log_k = float(parameters[design.size + 1])
            shape = float(np.exp(log_shape))
            sqrt_k = float(np.exp(0.5 * log_k))
            q = float(np.exp(-0.5 * log_k))
            if (
                not np.isfinite(shape)
                or shape <= 0
                or not np.isfinite(sqrt_k)
                or sqrt_k <= 0
                or not np.isfinite(q)
                or not np.isfinite(log_k)
            ):
                nan = np.full(log_time.shape, np.nan)
                return nan, nan.copy()
            standardized = sqrt_k * (shape * residual - log_k)
        log_sf = np.full(log_time.shape, np.nan)
        negative_infinite = np.isneginf(standardized)
        positive_infinite = np.isposinf(standardized)
        log_sf[negative_infinite] = 0.0
        log_sf[positive_infinite] = -np.inf
        finite_z = np.isfinite(standardized)
        if np.any(finite_z):
            finite_values = standardized[finite_z]
            local_near = _near_zero_mask(q, finite_values)
            supported = ~((0 < abs(q) < 1e-154) & ~local_near)
            finite_sf = np.full(finite_values.shape, np.nan)
            if np.any(supported):
                _, finite_sf[supported] = _prentice_logtails(finite_values[supported], q)
            log_sf[finite_z] = finite_sf
        # The Prentice kernel returns log survival directly. Keeping that
        # quantity avoids a lossy log-S -> cumulative-hazard -> log-S roundtrip.
        log_survival = log_sf.copy()
        survival = np.exp(log_survival)
    invalid = np.isnan(log_survival) | np.isnan(survival)
    log_survival[invalid] = np.nan
    survival[invalid] = np.nan
    return log_survival, survival


def _type7_quantiles(samples: FloatArray, probabilities: tuple[float, float]) -> FloatArray:
    result = np.full((2, samples.shape[1]), np.nan)
    for column in range(samples.shape[1]):
        values = samples[:, column]
        values = np.sort(values[~np.isnan(values)])
        if values.size == 0:
            continue
        for row, probability in enumerate(probabilities):
            position = (values.size - 1) * probability
            lower_index = int(np.floor(position))
            fraction = position - lower_index
            if fraction == 0 or lower_index == values.size - 1:
                result[row, column] = values[lower_index]
            else:
                lower, upper = values[lower_index], values[lower_index + 1]
                result[row, column] = (1 - fraction) * lower + fraction * upper
    return result


def predict_parametric_survival_mc(
    fit: ParametricSurvivalFit | GeneralizedGammaFit | SurvivalSplineFit,
    times: ArrayLike,
    profiles: ArrayLike | None = None,
    *,
    draws: int = 1000,
    confidence: float = 0.95,
    rng: np.random.Generator | int | None = None,
    parameter_draws: ArrayLike | None = None,
) -> ParametricSurvivalMCPrediction:
    """Return pointwise survival limits from full-covariance normal draws.

    Generated draws are joint multivariate-normal samples in the fit's
    ``scaled_parameters`` coordinate order, using ``scaled_covariance``. Supply
    ``parameter_draws`` to reuse a `(B, parameter_count)` array in that same
    coordinate system; when supplied, its row count determines B and ``draws``
    and ``rng`` are unused. Pointwise limits use type-7 quantiles and omit only
    NaN predictions per cell. No draws are truncated or redrawn.

    Positive-infinite times are supported and have survival zero; zero times
    have survival one for valid draws. For spline fits, the native unrestricted
    normal draws are not filtered for monotonicity. ``spline_minimum_slope``
    reports each draw's exact minimum baseline derivative over the full spline
    support, making any rising sampled curves visible. Other time/profile
    conventions match the deterministic prediction APIs.
    """
    if np.iscomplexobj(confidence):
        raise ValueError("confidence must be real")
    conf = scalar(confidence, "confidence")
    if not 0 < conf < 1:
        raise ValueError("confidence must be in (0, 1)")
    mean, covariance, covariate_mean, covariate_scale, parameterization = _fit_parameters(fit)
    n_generated = (
        _integer(draws, "draws", 2, _MAX_PARAMETER_DRAWS) if parameter_draws is None else 0
    )

    if np.iscomplexobj(times):
        raise ValueError("times must be real")
    raw_times = np.asarray(times)
    if (
        raw_times.ndim != 1
        or raw_times.size < 1
        or raw_times.size > 100_000
        or raw_times.dtype.kind not in "iuf"
        or np.any(np.isnan(raw_times))
        or np.any(raw_times < 0)
        or np.any(np.isneginf(raw_times))
    ):
        raise ValueError("times must be a nonempty nonnegative vector")
    times_copy = np.asarray(raw_times, dtype=np.float64)
    if np.any(np.isnan(times_copy)):
        raise ValueError("times could not be represented")

    if profiles is None:
        profile_values = np.zeros((1, covariate_mean.size))
    else:
        if np.iscomplexobj(profiles):
            raise ValueError("profiles must be real")
        raw_profiles = np.asarray(profiles)
        if raw_profiles.dtype.kind not in "iuf":
            raise ValueError("profiles must be real numeric values")
        if raw_profiles.ndim == 1:
            raw_profiles = raw_profiles[None, :]
        if raw_profiles.ndim != 2 or raw_profiles.shape[1] != covariate_mean.size:
            raise ValueError("profiles must have one column per fitted covariate")
        if not 1 <= raw_profiles.shape[0] <= 100_000 or raw_profiles.size > _MAX_SURFACE_CELLS:
            raise ValueError("profiles exceed the bounded profile limit")
        profile_values = finite(raw_profiles, "profiles")

    n_profiles, n_times = profile_values.shape[0], times_copy.size
    surface_cells = prod((int(n_profiles), int(n_times)))
    if not 1 <= n_profiles <= 100_000:
        raise ValueError("profiles must contain between 1 and 100000 rows")
    if parameter_draws is None:
        n_draws = n_generated
        if rng is not None and not isinstance(rng, (int, np.integer, np.random.Generator)):
            raise ValueError("rng must be an integer, Generator or None")
    else:
        if np.iscomplexobj(parameter_draws):
            raise ValueError("parameter_draws must be real")
        raw_draws = np.asarray(parameter_draws)
        if (
            raw_draws.ndim != 2
            or raw_draws.shape[1] != mean.size
            or not 2 <= raw_draws.shape[0] <= _MAX_PARAMETER_DRAWS
            or raw_draws.size > _MAX_SURFACE_CELLS
            or raw_draws.dtype.kind not in "iuf"
        ):
            raise ValueError("parameter_draws must have shape (B, parameter_count)")
        n_draws = int(raw_draws.shape[0])
        parameter_sample = raw_draws
    parameter_cells = prod((int(n_draws), int(mean.size)))
    basis_count = fit.scaled_knots.size if isinstance(fit, SurvivalSplineFit) else 0
    spline_work = n_draws * basis_count * basis_count + n_draws * surface_cells * basis_count
    if (
        8 * surface_cells
        + n_profiles * mean.size
        + parameter_cells
        + (n_draws if isinstance(fit, SurvivalSplineFit) else 0)
        > _MAX_SURFACE_CELLS
        or surface_cells * n_draws + spline_work > _MAX_WORK_CELLS
        or parameter_cells > _MAX_SURFACE_CELLS
    ):
        raise ValueError("Monte Carlo prediction exceeds the bounded work or cell limit")
    per_time_cells = n_draws + 4 * basis_count + mean.size
    draw_chunk = min(n_times, max(1, _MAX_DRAW_CELL_BUFFER // per_time_cells))
    if draw_chunk * per_time_cells > _MAX_DRAW_CELL_BUFFER:
        raise ValueError("Monte Carlo draw workspace exceeds the bounded cell limit")

    if parameter_draws is None:
        parameter_sample = _draw_parameters(mean, covariance, n_draws, rng)
    else:
        parameter_sample = finite(parameter_sample, "parameter_draws")
    normalized_profiles = (
        (profile_values - covariate_mean) / covariate_scale
        if covariate_mean.size
        else np.empty((n_profiles, 0))
    )
    if not np.isfinite(normalized_profiles).all():
        raise ArithmeticError("normalized prediction profiles exceed numerical range")
    design = np.column_stack((np.ones(n_profiles), normalized_profiles))
    spline_minimum_slope = None
    if isinstance(fit, SurvivalSplineFit):
        spline_minimum_slope = np.empty(n_draws)
        for draw_index, parameters in enumerate(parameter_sample):
            with np.errstate(over="ignore", invalid="ignore"):
                spline_minimum_slope[draw_index] = _minimum_slope(
                    parameters[:basis_count], fit.scaled_knots
                )
    log_time = np.full(times_copy.shape, np.inf)
    finite_positive = np.isfinite(times_copy) & (times_copy > 0)
    log_time[finite_positive] = (
        np.log(times_copy[finite_positive]) - fit.log_time_center
    ) / fit.log_time_scale
    log_time[times_copy == 0] = -np.inf
    if not np.isfinite(log_time[finite_positive]).all():
        raise ArithmeticError("normalized prediction times exceed numerical range")

    point_log_survival = np.empty((n_profiles, n_times))
    point_survival = np.empty((n_profiles, n_times))
    lower = np.empty((n_profiles, n_times))
    upper = np.empty((n_profiles, n_times))
    simulated_sd = np.empty((n_profiles, n_times))
    valid_draws = np.empty((n_profiles, n_times), dtype=np.int64)
    probabilities = ((1 - conf) / 2, 1 - (1 - conf) / 2)

    for start in range(0, n_times, draw_chunk):
        stop = min(n_times, start + draw_chunk)
        block_times = times_copy[start:stop]
        block_log_time = log_time[start:stop]
        zero = block_times == 0
        infinite = np.isposinf(block_times)
        evaluable = np.isfinite(block_times) & (block_times > 0)
        evaluable_log_time = block_log_time[evaluable]
        spline_basis = (
            _basis(fit.scaled_knots, evaluable_log_time)
            if isinstance(fit, SurvivalSplineFit)
            else None
        )
        for profile_index in range(n_profiles):
            row_design = design[profile_index]
            point_log = np.full(block_times.shape, np.nan)
            point_s = np.full(block_times.shape, np.nan)
            point_valid = _draw_is_representable(mean, fit, parameterization)
            if point_valid:
                point_log[zero] = 0.0
                point_s[zero] = 1.0
                point_log[infinite] = -np.inf
                point_s[infinite] = 0.0
            if np.any(evaluable):
                point_values_log, point_values = _survival_draw(
                    fit,
                    mean,
                    row_design,
                    evaluable_log_time,
                    parameterization,
                    spline_basis,
                )
                point_log[evaluable] = point_values_log
                point_s[evaluable] = point_values
            point_log_survival[profile_index, start:stop] = point_log
            point_survival[profile_index, start:stop] = point_s
            draws_surface = np.full((n_draws, stop - start), np.nan)
            for draw_index, parameters in enumerate(parameter_sample):
                if not _draw_is_representable(parameters, fit, parameterization):
                    continue
                draws_surface[draw_index, zero] = 1.0
                draws_surface[draw_index, infinite] = 0.0
                if np.any(evaluable):
                    _, values = _survival_draw(
                        fit,
                        parameters,
                        row_design,
                        evaluable_log_time,
                        parameterization,
                        spline_basis,
                    )
                    draws_surface[draw_index, evaluable] = values
            quantiles = _type7_quantiles(draws_surface, probabilities)
            lower[profile_index, start:stop] = quantiles[0]
            upper[profile_index, start:stop] = quantiles[1]
            valid_draws[profile_index, start:stop] = np.sum(~np.isnan(draws_surface), axis=0)
            with np.errstate(invalid="ignore", over="ignore"):
                simulated_sd[profile_index, start:stop] = np.std(draws_surface, axis=0, ddof=1)

    for name, values in (
        ("point survival", point_survival),
        ("survival limits", lower),
        ("survival limits", upper),
    ):
        valid = ~np.isnan(values)
        if np.any((values[valid] < 0) | (values[valid] > 1)):
            raise ArithmeticError(f"{name} fell outside [0, 1]")
    return ParametricSurvivalMCPrediction(
        _freeze(profile_values),
        _freeze(times_copy),
        _freeze(point_log_survival),
        _freeze(point_survival),
        _freeze(lower),
        _freeze(upper),
        _freeze(simulated_sd),
        _freeze_int(valid_draws),
        _freeze(parameter_sample),
        conf,
        None if spline_minimum_slope is None else _freeze(spline_minimum_slope),
    )
