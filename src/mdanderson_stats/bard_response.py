"""Source-grounded categorical logistic response model used by BARD.

The response margins and joint factor-profile distribution are explicit inputs.
This module does not infer independence or generate factor profiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import log_expit, logsumexp

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_MAX_DOSES = 100
_MAX_FACTORS = 5
_MAX_LEVELS = 20
_MAX_PROFILES = 100_000
_MAX_PROFILE_CELLS = 1_000_000
_MAX_EVALUATION_WORK = 100_000_000
_BISECTION_STEPS = 160
_BISECTION_TOLERANCE = 2e-13


def _readonly(value: ArrayLike, *, dtype: type = np.float64) -> NDArray:
    result: NDArray = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _bounded_vector(value: ArrayLike, name: str, maximum: int) -> FloatArray:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.size < 1 or value.size > maximum:
            raise ValueError(f"{name} must be a one-dimensional array of length 1..{maximum}")
        if value.dtype.kind not in "iuf":
            raise ValueError(f"{name} must contain real numeric values")
    elif isinstance(value, (list, tuple)):
        if not 1 <= len(value) <= maximum:
            raise ValueError(f"{name} must have length 1..{maximum}")
        if any(
            isinstance(item, (list, tuple, np.ndarray, complex, np.bool_)) or isinstance(item, bool)
            for item in value
        ):
            raise ValueError(f"{name} must contain only real scalar values")
    else:
        raise ValueError(f"{name} must be a one-dimensional numeric sequence")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite real values")
    return result


def _bounded_matrix(value: ArrayLike, name: str, *, max_rows: int, max_cols: int) -> NDArray:
    raw: ArrayLike
    if isinstance(value, np.ndarray):
        if value.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional")
        rows, cols = value.shape
        if not 1 <= rows <= max_rows or not 1 <= cols <= max_cols:
            raise ValueError(f"{name} dimensions exceed supported bounds")
        if value.dtype.kind not in "iuf":
            raise ValueError(f"{name} must contain real numeric values")
        raw = value
    elif isinstance(value, (list, tuple)):
        rows = len(value)
        if not 1 <= rows <= max_rows:
            raise ValueError(f"{name} row count exceeds supported bounds")
        if not isinstance(value[0], (list, tuple, np.ndarray)):
            raise ValueError(f"{name} must be a rectangular two-dimensional sequence")
        cols = len(value[0])
        if not 1 <= cols <= max_cols or rows * cols > _MAX_PROFILE_CELLS:
            raise ValueError(f"{name} dimensions exceed supported bounds")
        for row in value:
            if not isinstance(row, (list, tuple, np.ndarray)) or getattr(row, "ndim", 1) != 1:
                raise ValueError(f"{name} must be rectangular with one-dimensional rows")
            if len(row) != cols:
                raise ValueError(f"{name} must be rectangular")
            if any(
                isinstance(item, (list, tuple, np.ndarray, complex, np.bool_))
                or isinstance(item, bool)
                for item in row
            ):
                raise ValueError(f"{name} must contain only real scalar values")
        raw = value
    else:
        raise ValueError(f"{name} must be a two-dimensional numeric sequence")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(raw, dtype=np.float64)
    if result.shape != (rows, cols) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite real values")
    return result


def _profiles(value: ArrayLike, *, expected_factors: int | None = None) -> IntArray:
    raw = _bounded_matrix(value, "factor_profiles", max_rows=_MAX_PROFILES, max_cols=_MAX_FACTORS)
    if expected_factors is not None and raw.shape[1] != expected_factors:
        raise ValueError("factor_profiles must match the number of odds-ratio rows")
    if np.any(raw != np.floor(raw)) or np.any(raw < 1) or np.any(raw > _MAX_LEVELS):
        raise ValueError("factor_profiles must contain integer category labels in 1..20")
    return raw.astype(np.int64)


def _odds_ratios(value: ArrayLike, *, factors: int | None = None) -> FloatArray:
    result = _bounded_matrix(
        value, "response_odds_ratios", max_rows=_MAX_FACTORS, max_cols=_MAX_LEVELS
    )
    if factors is not None and result.shape[0] != factors:
        raise ValueError("response_odds_ratios must have one row per factor")
    if np.any(result <= 0) or np.any(result[:, 0] != 1):
        raise ValueError("odds ratios must be positive and the level-1 column must equal 1")
    return result


def _conditional(intercepts: FloatArray, profiles: IntArray, odds: FloatArray) -> FloatArray:
    if profiles.shape[1] != odds.shape[0] or np.max(profiles) > odds.shape[1]:
        raise ValueError("factor profile categories must fit the odds-ratio matrix")
    log_or = np.log(odds)
    offsets = np.zeros(profiles.shape[0], dtype=np.float64)
    for factor in range(profiles.shape[1]):
        offsets += log_or[factor, profiles[:, factor] - 1]
    # exp(log_expit(x)) preserves representable subnormal lower-tail values;
    # a direct expit implementation may underflow while forming exp(-x).
    return np.asarray(np.exp(log_expit(intercepts[:, None] + offsets[None, :])), dtype=np.float64)


def _preflight_output(doses: int, profiles: int, *, iterations: int = 1) -> None:
    cells = doses * profiles
    if cells > _MAX_PROFILE_CELLS:
        raise ValueError("dose-by-profile output exceeds the 1,000,000-cell limit")
    if cells * iterations > _MAX_EVALUATION_WORK:
        raise ValueError("response-model work exceeds the 100,000,000 profile-iteration limit")


def bard_response_probabilities(
    intercepts: ArrayLike,
    factor_profiles: ArrayLike,
    response_odds_ratios: ArrayLike,
) -> FloatArray:
    """Evaluate dose-by-profile response probabilities from logistic effects.

    ``factor_profiles`` uses one-based category labels.  The odds-ratio matrix
    has one row per factor and one column per category, with reference-level
    odds ratios of exactly one in its first column.
    """
    if isinstance(intercepts, np.ndarray):
        if intercepts.ndim != 1 or not 1 <= intercepts.size <= _MAX_DOSES:
            raise ValueError(f"intercepts must be one-dimensional with length 1..{_MAX_DOSES}")
        if intercepts.dtype.kind not in "iuf":
            raise ValueError("intercepts must contain real numeric values")
    elif isinstance(intercepts, (list, tuple)):
        if not 1 <= len(intercepts) <= _MAX_DOSES or any(
            isinstance(item, (list, tuple, np.ndarray, complex, bool, np.bool_))
            for item in intercepts
        ):
            raise ValueError("intercepts must be a bounded one-dimensional real sequence")
    else:
        raise ValueError("intercepts must be a bounded one-dimensional real sequence")
    with np.errstate(over="ignore", invalid="ignore"):
        alpha = np.asarray(intercepts, dtype=np.float64)
    if np.any(np.isnan(alpha)):
        raise ValueError("intercepts may be finite or infinite, but not NaN")
    profiles = _profiles(factor_profiles)
    odds = _odds_ratios(response_odds_ratios, factors=profiles.shape[1])
    if np.max(profiles) > odds.shape[1]:
        raise ValueError("factor profile categories must fit the odds-ratio matrix")
    _preflight_output(alpha.size, profiles.shape[0])
    return _readonly(_conditional(alpha, profiles, odds))


@dataclass(frozen=True)
class BARDResponseModel:
    """Calibrated dose-specific intercepts and their profile probabilities."""

    population_response: FloatArray
    factor_profiles: IntArray
    profile_probabilities: FloatArray
    response_odds_ratios: FloatArray
    intercepts: FloatArray
    conditional_probabilities: FloatArray
    marginal_residuals: FloatArray

    def probability(self, dose: int, factor_profile: ArrayLike) -> float:
        """Evaluate one profile at a one-based dose index."""
        if isinstance(dose, (bool, np.bool_)) or not isinstance(dose, (int, np.integer)):
            raise ValueError("dose must be an integer one-based dose index")
        if not 1 <= int(dose) <= self.intercepts.size:
            raise ValueError("dose is outside the calibrated range")
        profile = _bounded_vector(factor_profile, "factor_profile", _MAX_FACTORS)
        if profile.size != self.response_odds_ratios.shape[0]:
            raise ValueError("factor_profile must contain one category per factor")
        if (
            np.any(profile != np.floor(profile))
            or np.any(profile < 1)
            or np.any(profile > self.response_odds_ratios.shape[1])
        ):
            raise ValueError("factor_profile categories are outside the odds-ratio matrix")
        one = profile.astype(np.int64)[None, :]
        return float(
            _conditional(
                self.intercepts[int(dose) - 1 : int(dose)], one, self.response_odds_ratios
            )[0, 0]
        )


def bard_response_model(
    population_response: ArrayLike,
    factor_profiles: ArrayLike,
    profile_probabilities: ArrayLike,
    response_odds_ratios: ArrayLike,
) -> BARDResponseModel:
    """Calibrate dose intercepts to explicit population response margins.

    The weighted factor-profile probabilities define the population mixture.
    Each dose has its own intercept; the categorical odds ratios are shared
    across doses. Margins exactly zero or one produce deterministic fitted
    probabilities and negative or positive infinite intercepts respectively.
    """
    target = _bounded_vector(population_response, "population_response", _MAX_DOSES)
    if np.any((target < 0) | (target > 1)):
        raise ValueError("population_response values must lie in [0,1]")
    profiles = _profiles(factor_profiles)
    _preflight_output(target.size, profiles.shape[0], iterations=_BISECTION_STEPS)
    weights = _bounded_vector(profile_probabilities, "profile_probabilities", _MAX_PROFILES)
    odds = _odds_ratios(response_odds_ratios, factors=profiles.shape[1])
    if profiles.shape[0] != weights.size:
        raise ValueError("profile_probabilities must have one entry per factor profile")
    if np.max(profiles) > odds.shape[1]:
        raise ValueError("factor profile categories must fit the odds-ratio matrix")
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("profile_probabilities must be nonnegative with positive total mass")
    total = float(np.sum(weights, dtype=np.float64))
    if not np.isfinite(total) or abs(total - 1.0) > 1e-12:
        raise ValueError("profile_probabilities must sum to one within 1e-12")
    weights = weights / total

    offsets = np.zeros(profiles.shape[0], dtype=np.float64)
    log_or = np.log(odds)
    for factor in range(profiles.shape[1]):
        offsets += log_or[factor, profiles[:, factor] - 1]
    intercepts = np.empty(target.size, dtype=np.float64)
    positive_weights = weights > 0
    for dose, margin in enumerate(target):
        if margin == 0:
            intercepts[dose] = -np.inf
        elif margin == 1:
            intercepts[dose] = np.inf
        else:
            logit_margin = log(margin) - log(1.0 - margin)
            active_offsets = offsets[positive_weights]
            low = logit_margin - float(np.max(active_offsets))
            high = logit_margin - float(np.min(active_offsets))
            log_weights = np.full(weights.shape, -np.inf, dtype=np.float64)
            positive_weights = weights > 0
            log_weights[positive_weights] = np.log(weights[positive_weights])
            lower_log_target = log(margin)
            upper_log_target = float(np.log1p(-margin))
            target_log_tail = lower_log_target if margin <= 0.5 else upper_log_target

            def objective(intercept: float) -> float:
                linear = intercept + offsets
                if margin <= 0.5:
                    return float(logsumexp(log_weights + log_expit(linear)))
                return float(logsumexp(log_weights + log_expit(-linear)))

            low_value = objective(low)
            high_value = objective(high)
            if margin <= 0.5:
                bracketed = (
                    low_value <= target_log_tail + 1e-12 and high_value >= target_log_tail - 1e-12
                )
            else:
                bracketed = (
                    low_value >= target_log_tail - 1e-12 and high_value <= target_log_tail + 1e-12
                )
            if not bracketed:
                raise ArithmeticError(
                    "could not bracket the response intercept at the requested margin"
                )
            for _ in range(_BISECTION_STEPS):
                middle = low + (high - low) * 0.5
                log_fitted_tail = objective(middle)
                if margin <= 0.5:
                    if log_fitted_tail < lower_log_target:
                        low = middle
                    else:
                        high = middle
                else:
                    if log_fitted_tail > upper_log_target:
                        low = middle
                    else:
                        high = middle
                if high - low <= _BISECTION_TOLERANCE * max(1.0, abs(middle)):
                    break
            intercept = low + (high - low) * 0.5
            final_log_tail = objective(intercept)
            if not np.isfinite(intercept) or abs(final_log_tail - target_log_tail) > 5e-11:
                raise ArithmeticError(
                    "response-intercept inversion did not meet its log-tail tolerance"
                )
            active_linear = intercept + offsets[positive_weights]
            log_slope = float(
                logsumexp(
                    log_weights[positive_weights]
                    + log_expit(active_linear)
                    + log_expit(-active_linear)
                )
            )
            if log_slope - target_log_tail < log(1e-8):
                raise ArithmeticError(
                    "response-intercept inversion is numerically unresolved because the "
                    "weighted response curve is too flat"
                )
            intercepts[dose] = intercept
    conditional = _conditional(intercepts, profiles, odds)
    residuals = conditional @ weights - target
    return BARDResponseModel(
        _readonly(target),
        _readonly(profiles, dtype=np.int64),
        _readonly(weights),
        _readonly(odds),
        _readonly(intercepts),
        _readonly(conditional),
        _readonly(residuals),
    )
