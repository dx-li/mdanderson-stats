"""RMC-COMPASS point predictions from the rounded public model panel.

These are approximations to version V1.0.2.0, not the undisclosed full fit.
Parameter-rounding envelopes are not statistical confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

from ._cdflib import _freeze
from ._validation import FloatArray

_BETA = np.array([-0.3932, -0.1104, -0.0261, -0.2636])
_INTERCEPT = 5.7273
_SHRINKAGE = 0.895
_SIGMA = 0.8334
_MAX_TIMES = 100_000


@dataclass(frozen=True)
class RMCCompassReportedPrediction:
    """Immutable predictions using reported, rounded parameters only.

    The two-column rounding envelopes assume nearest rounding at the displayed
    parameter precision, conditional on the effective covariates. They exclude
    estimation uncertainty and uncertainty about native clamping precision.
    """

    ecog_performance_status: int
    disease_sites: int
    neutrophil_lymphocyte_ratio: float
    corrected_calcium_mg_dl: float
    effective_neutrophil_lymphocyte_ratio: float
    effective_corrected_calcium_mg_dl: float
    times_months: FloatArray
    linear_predictor: float
    median_survival_months: float
    survival_probability: FloatArray
    death_probability: FloatArray
    survival_parameter_rounding_envelope: FloatArray
    median_parameter_rounding_envelope: tuple[float, float]
    risk_group: Literal["High", "Intermediate", "Low"]
    risk_group_stable_under_parameter_rounding: bool
    extrapolated_predictors: tuple[str, ...]
    clamped_predictors: tuple[str, ...]
    model_version: str = "V1.0.2.0 (rounded public parameters)"


def _integer(value: int, name: str, maximum: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not 0 <= value <= maximum
    ):
        raise ValueError(f"{name} must be an integer from 0 to {maximum}")
    return int(value)


def _nonnegative(value: float, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite nonnegative real scalar")
    result = float(raw)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative real scalar")
    return result


def _times(values: ArrayLike) -> FloatArray:
    if (
        isinstance(values, np.ndarray)
        and values.size > _MAX_TIMES
        or isinstance(values, (list, tuple))
        and len(values) > _MAX_TIMES
    ):
        raise ValueError(f"times_months exceeds {_MAX_TIMES} values")
    raw = np.asarray(values)
    if raw.ndim > 1 or not 0 < raw.size <= _MAX_TIMES or raw.dtype.kind not in "iuf":
        raise ValueError("times_months must be a nonempty real scalar or vector")
    result = np.asarray(raw, dtype=float).reshape(-1)
    if np.any(~np.isfinite(result)) or np.any(result < 0) or np.any(result > 36):
        raise ValueError("times_months must be finite and between 0 and 36")
    return result


def _risk_group(probability: float) -> Literal["High", "Intermediate", "Low"]:
    if probability < 0.25:
        return "High"
    if probability > 0.5:
        return "Low"
    return "Intermediate"


def _survival(log_time: FloatArray, predictor: float, sigma: float) -> FloatArray:
    return np.asarray(ndtr((predictor - log_time) / sigma), dtype=float)


def rmc_compass_reported_prediction(
    ecog_performance_status: int,
    disease_sites: int,
    *,
    neutrophil_lymphocyte_ratio: float,
    corrected_calcium_mg_dl: float,
    times_months: ArrayLike = (6, 12, 18, 24, 36),
) -> RMCCompassReportedPrediction:
    """Approximate the app's log-normal AFT point predictions in months.

    Inputs are the already computed ANC/ALC ratio and corrected calcium in
    mg/dL. The public development limits clamp them to [0.84, 22] and
    [8.3, 11.2]. ECOG 4 and site counts 8--11 are accepted and flagged, without
    clamping. Times beyond the app's 36-month horizon are rejected.

    Confidence intervals cannot be recovered from the public coefficients and
    are deliberately absent. See docs/rmc-compass.md for precision limitations.
    """
    ecog = _integer(ecog_performance_status, "ecog_performance_status", 4)
    sites = _integer(disease_sites, "disease_sites", 11)
    nlr = _nonnegative(neutrophil_lymphocyte_ratio, "neutrophil_lymphocyte_ratio")
    calcium = _nonnegative(corrected_calcium_mg_dl, "corrected_calcium_mg_dl")
    times = _times(times_months)
    effective_nlr = float(np.clip(nlr, 0.84, 22.0))
    effective_calcium = float(np.clip(calcium, 8.3, 11.2))
    covariates = np.array([ecog, sites, effective_nlr, effective_calcium])
    linear_predictor = float(_INTERCEPT + _SHRINKAGE * (_BETA @ covariates))

    # All covariates are nonnegative and all coefficient intervals negative.
    # The extrema therefore use opposite shrinkage/negative-slope endpoints.
    lower_lp = float(5.72725 + 0.8955 * ((_BETA - 0.00005) @ covariates))
    upper_lp = float(5.72735 + 0.8945 * ((_BETA + 0.00005) @ covariates))
    log_time = np.full(times.shape, -np.inf)
    np.log(times, out=log_time, where=times > 0)
    survival = _survival(log_time, linear_predictor, _SIGMA)
    death = np.asarray(ndtr((log_time - linear_predictor) / _SIGMA), dtype=float)
    lower_survival = np.minimum(
        _survival(log_time, lower_lp, 0.83335),
        _survival(log_time, lower_lp, 0.83345),
    )
    upper_survival = np.maximum(
        _survival(log_time, upper_lp, 0.83335),
        _survival(log_time, upper_lp, 0.83345),
    )
    log24 = np.array([np.log(24.0)])
    survival24 = float(_survival(log24, linear_predictor, _SIGMA)[0])
    lower24 = min(float(_survival(log24, lower_lp, s)[0]) for s in (0.83335, 0.83345))
    upper24 = max(float(_survival(log24, upper_lp, s)[0]) for s in (0.83335, 0.83345))
    group = _risk_group(survival24)
    extrapolated = tuple(
        name
        for name, is_outside in (
            ("ecog_performance_status", ecog > 3),
            ("disease_sites", sites > 7),
        )
        if is_outside
    )
    clamped = tuple(
        name
        for name, is_clamped in (
            ("neutrophil_lymphocyte_ratio", nlr != effective_nlr),
            ("corrected_calcium_mg_dl", calcium != effective_calcium),
        )
        if is_clamped
    )
    return RMCCompassReportedPrediction(
        ecog,
        sites,
        nlr,
        calcium,
        effective_nlr,
        effective_calcium,
        _freeze(times),
        linear_predictor,
        float(np.exp(linear_predictor)),
        _freeze(survival),
        _freeze(death),
        _freeze(np.column_stack((lower_survival, upper_survival))),
        (float(np.exp(lower_lp)), float(np.exp(upper_lp))),
        group,
        _risk_group(lower24) == group == _risk_group(upper24),
        extrapolated,
        clamped,
    )
