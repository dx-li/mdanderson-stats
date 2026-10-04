"""Source-specific global-logistic coefficient fit and dose decision.

The coefficient routine follows the recovered `arm::bayesglm.fit` IRLS
working-prior update. It is not a Cauchy-prior MAP optimizer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, log_expit

from ._cdflib import _freeze
from .mtadf import _count_vector
from .mtadf_author import _author_safety_state

FloatArray = NDArray[np.float64]
_MAX_DOSES = 20
_MAX_SUBJECTS = 10_000
_MAX_IRLS_ITERATIONS = 100
_IRLS_EPSILON = 1e-8
_BASE_SCALE = 2.5
_PRIOR_DF = 1.0


def _dose_count(value: int, name: str, size: int) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iu" or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer dose index")
    result = int(raw)
    if not 0 <= result < size:
        raise ValueError(f"{name} must be in [0, {size - 1}]")
    return result


def _count_data(subjects: ArrayLike, responses: ArrayLike) -> tuple[FloatArray, FloatArray]:
    n_size = _preflight_count_vector(subjects, "subjects")
    y_size = _preflight_count_vector(responses, "responses")
    if n_size != y_size:
        raise ValueError("subjects and responses must be matching one-dimensional vectors")
    n = _count_vector(subjects, "subjects")
    y = _count_vector(responses, "responses")
    if n.shape != y.shape or np.any(y > n) or int(n.sum()) > _MAX_SUBJECTS:
        raise ValueError(
            "responses cannot exceed subjects and total subjects are limited to 10,000"
        )
    if not np.any(n):
        raise ValueError("global logistic fit requires at least one observed subject")
    return n.astype(float), y.astype(float)


def _preflight_count_vector(value: ArrayLike, name: str) -> int:
    """Check bounded one-dimensional shape before NumPy materializes sequences."""
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional count vector")
        length = value.size
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        length = len(value)
        if not 2 <= length <= _MAX_DOSES:
            raise ValueError(f"{name} must have 2..{_MAX_DOSES} dose entries")
        for item in value:
            if not np.isscalar(item):
                raise ValueError(f"{name} must be a one-dimensional count vector")
            scalar = np.asarray(item)
            if scalar.ndim != 0 or scalar.dtype.kind not in "iuf":
                raise ValueError(f"{name} must contain real numeric counts")
    else:
        raise ValueError(f"{name} must be a bounded one-dimensional array or sequence")
    if not 2 <= length <= _MAX_DOSES:
        raise ValueError(f"{name} must have 2..{_MAX_DOSES} dose entries")
    return int(length)


def _bounded_real_vector(value: object, name: str, length: int) -> FloatArray:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.size != length:
            raise ValueError(f"{name} must be a matching one-dimensional vector")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != length or any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must be a matching one-dimensional vector")
    else:
        raise ValueError(f"{name} must be a matching one-dimensional vector")
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    return np.asarray(raw, dtype=float)


def _standardized_grid(dose_count: int) -> FloatArray:
    grid = np.arange(1, dose_count + 1, dtype=float)
    return (grid - float(np.mean(grid))) / (2.0 * float(np.std(grid, ddof=1)))


def _source_prior_scales(design: FloatArray, subjects: FloatArray) -> FloatArray:
    """Apply arm's scaled prior rule to patient-expanded design columns."""
    observed = subjects > 0
    n = subjects[observed]
    x = design[observed]
    total = int(n.sum())
    scales = np.full(design.shape[1], _BASE_SCALE, dtype=float)
    for column in range(design.shape[1]):
        values = x[:, column]
        unique = np.unique(values)
        if unique.size == 2:
            x_scale = float(unique[-1] - unique[0])
        elif unique.size > 2:
            mean = float(np.dot(n, values) / total)
            sum_squares = float(np.dot(n, (values - mean) ** 2))
            sd = np.sqrt(sum_squares / (total - 1)) if total > 1 else 0.0
            x_scale = 2.0 * float(sd)
        else:
            x_scale = 1.0
        scales[column] = max(_BASE_SCALE / x_scale, 1e-12)
    return scales


def _weighted_qr(
    design: FloatArray, weights: FloatArray, working_response: FloatArray
) -> tuple[FloatArray, FloatArray]:
    if (
        design.ndim != 2
        or weights.shape != (design.shape[0],)
        or working_response.shape != weights.shape
    ):
        raise ArithmeticError("internal weighted least-squares dimensions do not match")
    if np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise ArithmeticError("IRLS generated nonpositive or nonfinite working weights")
    root_weight = np.sqrt(weights)
    weighted_design = design * root_weight[:, None]
    weighted_response = working_response * root_weight
    q, r = np.linalg.qr(weighted_design, mode="reduced")
    diagonal = np.abs(np.diag(r))
    if diagonal.size != design.shape[1] or np.any(diagonal <= np.finfo(float).tiny):
        raise ArithmeticError("prior-augmented IRLS design is numerically rank deficient")
    coefficient = np.linalg.solve(r, q.T @ weighted_response)
    r_inverse = np.linalg.solve(r, np.eye(r.shape[0]))
    covariance = r_inverse @ r_inverse.T
    if not np.all(np.isfinite(coefficient)) or not np.all(np.isfinite(covariance)):
        raise ArithmeticError("weighted QR produced nonfinite coefficients or covariance")
    return coefficient, covariance


def _initial_working_rows(
    design: FloatArray, subjects: FloatArray, responses: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    rows: list[FloatArray] = []
    weights: list[float] = []
    working: list[float] = []
    start_eta = np.log(3.0) + 4.0 / 3.0
    for index, (n, y) in enumerate(zip(subjects, responses, strict=True)):
        for count, sign in ((y, 1.0), (n - y, -1.0)):
            if count > 0:
                rows.append(design[index])
                weights.append(float(count) * 0.1875)
                working.append(sign * start_eta)
    return np.asarray(rows), np.asarray(weights), np.asarray(working)


def _prior_augmented_fit(
    design_rows: FloatArray,
    weight_rows: FloatArray,
    working_rows: FloatArray,
    prior_scales: FloatArray,
    observed_design_mean: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    dimension = prior_scales.size
    prior_rows = np.eye(dimension)
    prior_rows[0] = observed_design_mean
    augmented_design = np.vstack((design_rows, prior_rows))
    augmented_weights = np.r_[weight_rows, 1.0 / prior_scales**2]
    augmented_working = np.r_[working_rows, np.zeros(dimension)]
    return _weighted_qr(augmented_design, augmented_weights, augmented_working)


def _deviance(subjects: FloatArray, responses: FloatArray, eta: FloatArray) -> float:
    return float(-2 * np.sum(responses * log_expit(eta) + (subjects - responses) * log_expit(-eta)))


@dataclass(frozen=True)
class MTADFAuthorGlobalFit:
    """Fitted global author quadratic model using the recovered IRLS update."""

    subjects: FloatArray
    responses: FloatArray
    standardized_doses: FloatArray
    coefficients: FloatArray
    fitted_efficacy: FloatArray
    prior_scales: FloatArray
    final_prior_sd: FloatArray
    deviance: float
    iterations: int
    converged: bool


def mtadf_author_global_fit(subjects: ArrayLike, responses: ArrayLike) -> MTADFAuthorGlobalFit:
    """Fit the author global quadratic logistic model to binomial counts.

    The implementation groups the Bernoulli-expanded source data by dose,
    including its outcome-dependent first family initialization. It applies
    `arm::bayesglm.fit`'s patient-expanded predictor scales and adaptive
    prior-SD update, then solves each weighted problem with QR. The fit is a
    converged working-prior IRLS estimate, not an ordinary Cauchy MAP.
    """
    n, y = _count_data(subjects, responses)
    xs = _standardized_grid(n.size)
    design = np.column_stack((np.ones(n.size), xs, xs * xs))
    prior_scales = _source_prior_scales(design, n)
    observed = n > 0
    observed_design = design[observed]
    observed_n = n[observed]
    observed_y = y[observed]
    total = float(n.sum())
    design_mean = np.average(observed_design, axis=0, weights=observed_n)

    initial_dev = -2.0 * total * np.log(0.75)
    previous_dev = initial_dev
    prior_sd = prior_scales.copy()
    coefficients = np.zeros(3, dtype=float)
    converged = False
    final_deviance = initial_dev
    iteration_count = 0
    for iteration in range(1, _MAX_IRLS_ITERATIONS + 1):
        if iteration == 1:
            work_design, work_weights, work_response = _initial_working_rows(
                observed_design, observed_n, observed_y
            )
        else:
            eta = observed_design @ coefficients
            mu = expit(eta)
            variance = mu * (1.0 - mu)
            informative = variance > 0
            if not np.any(informative):
                raise ArithmeticError("IRLS has no informative observations")
            work_design = observed_design[informative]
            work_weights = observed_n[informative] * variance[informative]
            fraction = observed_y[informative] / observed_n[informative]
            work_response = eta[informative] + (fraction - mu[informative]) / variance[informative]
        coefficients, covariance = _prior_augmented_fit(
            work_design, work_weights, work_response, prior_sd, design_mean
        )

        centered_coefficients = coefficients.copy()
        centered_coefficients[0] = float(np.dot(coefficients, design_mean))
        sampling_variance = np.diag(covariance).copy()
        sampling_variance[0] = float(design_mean @ covariance @ design_mean)
        updated_prior_sd = np.sqrt(
            (centered_coefficients**2 + sampling_variance + _PRIOR_DF * prior_scales**2)
            / (1.0 + _PRIOR_DF)
        )
        if not np.all(np.isfinite(updated_prior_sd)) or np.any(updated_prior_sd <= 0):
            raise ArithmeticError("adaptive prior scales are not representable")

        observed_eta = observed_design @ coefficients
        final_deviance = _deviance(observed_n, observed_y, observed_eta)
        iteration_count = iteration
        if (
            iteration > 1
            and abs(final_deviance - previous_dev) / (0.1 + abs(final_deviance)) < _IRLS_EPSILON
        ):
            converged = True
            prior_sd = updated_prior_sd
            break
        previous_dev = final_deviance
        prior_sd = updated_prior_sd

    # With one observed dose, the prior-augmented model has an exact flat
    # solution; remove QR roundoff in the two otherwise unidentified slopes.
    if int(np.count_nonzero(observed)) == 1:
        coefficients[1:] = 0.0
        final_deviance = _deviance(observed_n, observed_y, observed_design @ coefficients)
    fitted = expit(design @ coefficients)
    if not np.all(np.isfinite(fitted)):
        raise ArithmeticError("global efficacy predictions are not finite")
    return MTADFAuthorGlobalFit(
        _freeze(n),
        _freeze(y),
        _freeze(xs),
        _freeze(coefficients),
        _freeze(fitted),
        _freeze(prior_scales),
        _freeze(prior_sd),
        float(final_deviance),
        iteration_count,
        converged,
    )


def _author_admissible_count(
    subjects: FloatArray,
    toxicities: FloatArray,
    toxicity_limit: float,
    safety_cutoff: float,
) -> tuple[int, FloatArray, FloatArray]:
    """Adapt the shared author-source safety implementation for global decisions."""
    _, _, raw, adjusted, count = _author_safety_state(
        subjects,
        toxicities,
        toxicity_limit=toxicity_limit,
        safety_cutoff=safety_cutoff,
    )
    return count, raw, adjusted


def _author_global_next_dose(current: int, target: int, admissible_count: int) -> int:
    """One-step movement toward peak, capped by the supplied (possibly lagged) cap."""
    proposed = current + (1 if target > current else -1 if target < current else 0)
    return min(max(proposed, 0), admissible_count - 1)


@dataclass(frozen=True)
class MTADFAuthorGlobalDecision:
    """Author-global interim movement or final dose recommendation."""

    action: str
    dose: int
    reason: str
    current_dose: int | None
    target_dose: int
    admissible_count: int
    raw_overdose_probability: FloatArray
    adjusted_overdose_probability: FloatArray
    fitted_efficacy: FloatArray
    fit: MTADFAuthorGlobalFit


def mtadf_author_global_decision(
    subjects: ArrayLike,
    toxicities: ArrayLike,
    responses: ArrayLike,
    *,
    current_dose: int | None = None,
    final: bool = False,
    toxicity_limit: float = 0.3,
    safety_cutoff: float = 0.8,
    fit: MTADFAuthorGlobalFit | None = None,
) -> MTADFAuthorGlobalDecision:
    """Apply the author's rightmost efficacy peak and fresh isotonic safety cap.

    Dose indexes are zero-based. Interim conduct moves one dose toward the
    rightmost global fitted maximum, then caps by the current admissible count.
    ``final=True`` selects the fitted peak directly before applying that cap.
    """
    n, y = _count_data(subjects, responses)
    _preflight_count_vector(toxicities, "toxicities")
    tox = _count_vector(toxicities, "toxicities").astype(float)
    if tox.shape != n.shape or np.any(tox > n):
        raise ValueError("toxicities must match subjects and cannot exceed them")
    if not isinstance(final, (bool, np.bool_)):
        raise ValueError("final must be boolean")
    if current_dose is None:
        if not final:
            raise ValueError("current_dose is required for an interim decision")
        current = None
    else:
        current = _dose_count(current_dose, "current_dose", n.size)
        if not final and n[current] == 0:
            raise ValueError("current_dose must have observed subjects for interim conduct")
    phi_raw = np.asarray(toxicity_limit)
    cutoff_raw = np.asarray(safety_cutoff)
    if (
        phi_raw.ndim != 0
        or phi_raw.dtype.kind not in "iuf"
        or cutoff_raw.ndim != 0
        or cutoff_raw.dtype.kind not in "iuf"
    ):
        raise ValueError("toxicity_limit and safety_cutoff must be real scalars")
    phi = float(phi_raw)
    cutoff = float(cutoff_raw)
    if not isfinite(phi) or not 0 < phi < 1 or not isfinite(cutoff) or not 0 < cutoff < 1:
        raise ValueError("toxicity_limit and safety_cutoff must lie in (0,1)")
    actual_fit = mtadf_author_global_fit(n, y) if fit is None else fit
    if not isinstance(actual_fit, MTADFAuthorGlobalFit):
        raise ValueError("fit must be an MTADFAuthorGlobalFit")
    fit_subjects = _bounded_real_vector(actual_fit.subjects, "fit subjects", n.size)
    fit_responses = _bounded_real_vector(actual_fit.responses, "fit responses", n.size)
    fit_grid = _bounded_real_vector(actual_fit.standardized_doses, "fit dose grid", n.size)
    coefficients = _bounded_real_vector(actual_fit.coefficients, "fit coefficients", 3)
    fitted = _bounded_real_vector(actual_fit.fitted_efficacy, "fit efficacy", n.size)
    prior_scales = _bounded_real_vector(actual_fit.prior_scales, "fit prior scales", 3)
    final_prior_sd = _bounded_real_vector(actual_fit.final_prior_sd, "fit final prior scales", 3)
    if not np.array_equal(fit_subjects, n) or not np.array_equal(fit_responses, y):
        raise ValueError("fit does not match the supplied subject and response counts")
    expected_grid = _standardized_grid(n.size)
    expected_fitted = expit(
        np.column_stack((np.ones(n.size), expected_grid, expected_grid**2)) @ coefficients
    )
    if (
        not np.array_equal(fit_grid, expected_grid)
        or not np.all(np.isfinite(coefficients))
        or not np.all(np.isfinite(fitted))
        or np.any((fitted < 0) | (fitted > 1))
        or not np.allclose(fitted, expected_fitted, rtol=0, atol=2e-14)
        or not np.all(np.isfinite(prior_scales))
        or np.any(prior_scales <= 0)
        or not np.all(np.isfinite(final_prior_sd))
        or np.any(final_prior_sd <= 0)
        or not isfinite(float(actual_fit.deviance))
        or not 1 <= actual_fit.iterations <= _MAX_IRLS_ITERATIONS
        or not isinstance(actual_fit.converged, (bool, np.bool_))
    ):
        raise ValueError("fit fields are inconsistent with the author global model")
    if not actual_fit.converged:
        raise ArithmeticError("author global fit did not converge; dose decision is withheld")

    admissible_count, raw_overdose, adjusted = _author_admissible_count(n, tox, phi, cutoff)
    efficacy = actual_fit.fitted_efficacy
    maximum = float(np.max(efficacy))
    target = int(np.flatnonzero(efficacy == maximum)[-1])
    if final:
        dose = min(target, admissible_count - 1)
        action = "recommend"
        reason = "rightmost_peak_capped_by_fresh_safety"
    else:
        assert current is not None
        dose = _author_global_next_dose(current, target, admissible_count)
        action = "escalate" if dose > current else "deescalate" if dose < current else "stay"
        proposed = current + (1 if target > current else -1 if target < current else 0)
        if dose != proposed:
            reason = "one_step_peak_movement_capped_by_fresh_safety"
        else:
            reason = "one_step_toward_rightmost_fitted_peak"
    return MTADFAuthorGlobalDecision(
        action,
        int(dose),
        reason,
        current,
        target,
        admissible_count,
        _freeze(raw_overdose),
        _freeze(adjusted),
        efficacy,
        actual_fit,
    )
