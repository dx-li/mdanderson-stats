"""Empirical-Bayes spike-and-slab calibration for WFMM coefficients.

This calibrates coefficient-specific inclusion probabilities and slab
variances conditional on supplied random and residual variance components. It
does not estimate those variance components or the inverse-gamma priors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve

from ._validation import FloatArray
from .wfmm_model import (
    _MAX_COEFFICIENTS,
    _MAX_DESIGN_CELLS,
    _MAX_FIXED_EFFECTS,
    _MAX_RANDOM_EFFECTS,
    _MAX_ROWS,
    WFMMPrior,
    _component_array,
    _finite_real,
    _marginal_covariance,
    _strata,
)

_MAX_WORK = 250_000_000
_MAX_INFORMATION_CONDITION = 1.0e12


@dataclass(frozen=True)
class WFMMShrinkage:
    """GLS estimates and calibrated spike-and-slab prior parameters.

    ``group_*`` arrays are indexed by fixed-effect row and the sorted distinct
    values of ``partition_labels``. If the EM estimate reaches ``U=0``, the
    canonical equivalent prior is all-spike (``pi=0, tau=0``), since pi is
    unidentified at that boundary. The reported group objective is the log
    marginal likelihood ratio relative to a standard-normal sampling density.
    """

    prior: WFMMPrior
    fixed_estimates: FloatArray
    conditional_variances: FloatArray
    standardized_estimates: FloatArray
    inclusion_probability: FloatArray
    slab_to_sampling_variance: FloatArray
    group_inclusion_probability: FloatArray
    group_slab_to_sampling_variance: FloatArray
    group_iterations: NDArray[np.int64]
    group_converged: NDArray[np.bool_]
    group_log_marginal_likelihood: FloatArray
    partition_labels: NDArray[np.int64]


def _coefficient_labels(value: ArrayLike | None, count: int) -> NDArray[np.int64]:
    if value is None:
        return np.zeros(count, dtype=np.int64)
    raw = np.asarray(value)
    if raw.shape != (count,) or raw.dtype.kind not in "iuf":
        raise ValueError("coefficient_partition must be a real integer vector of length K")
    labels = _finite_real(raw, "coefficient_partition")
    if np.any(labels < 0) or np.any(labels != np.floor(labels)):
        raise ValueError("coefficient_partition must contain nonnegative integers")
    if np.any(labels > np.iinfo(np.int64).max):
        raise ValueError("coefficient_partition labels exceed int64")
    return labels.astype(np.int64)


def _em_group(
    z: FloatArray,
    *,
    initial_pi: float,
    initial_u: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[float, float, int, bool, float]:
    pi, u = initial_pi, initial_u
    with np.errstate(over="ignore", invalid="ignore"):
        z_squared = z * z
    if not np.all(np.isfinite(z_squared)):
        raise ArithmeticError("standardized WFMM estimates exceed the EM numeric range")
    converged = False
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        if u == 0.0:
            posterior_inclusion = np.full(z.shape, pi, dtype=np.float64)
        elif pi <= 0.0:
            posterior_inclusion = np.zeros(z.shape, dtype=np.float64)
        elif pi >= 1.0:
            posterior_inclusion = np.ones(z.shape, dtype=np.float64)
        else:
            logit_pi = np.log(pi) - np.log1p(-pi)
            log_odds = logit_pi - 0.5 * np.log1p(u) + 0.5 * z_squared * (u / (1.0 + u))
            posterior_inclusion = 1.0 / (1.0 + np.exp(-np.clip(log_odds, -745.0, 709.0)))
        inclusion_sum = float(np.sum(posterior_inclusion))
        if inclusion_sum == 0.0:
            # Numerically equivalent to the all-spike boundary.
            pi, u = 0.0, 0.0
            converged = True
            iterations = iteration
            break
        new_u = max(0.0, float(np.dot(posterior_inclusion, z_squared) / inclusion_sum - 1.0))
        new_pi = float(np.mean(posterior_inclusion))
        if not np.isfinite(new_u) or not np.isfinite(new_pi):
            raise ArithmeticError("WFMM shrinkage EM update is nonfinite")
        iterations = iteration
        if new_u == 0.0:
            pi, u = 0.0, 0.0
            converged = True
            break
        pi_change = abs(new_pi - pi)
        u_change = abs(new_u - u) / max(1.0, abs(u))
        pi, u = new_pi, new_u
        if max(pi_change, u_change) <= tolerance:
            converged = True
            break

    if u == 0.0:
        log_likelihood = 0.0
    else:
        log_pi = -np.inf if pi == 0.0 else np.log(pi)
        log_one_minus_pi = -np.inf if pi == 1.0 else np.log1p(-pi)
        log_slab_ratio = -0.5 * np.log1p(u) + 0.5 * z_squared * (u / (1.0 + u))
        log_likelihood = float(np.sum(np.logaddexp(log_one_minus_pi, log_pi + log_slab_ratio)))
    if not np.isfinite(log_likelihood):
        raise ArithmeticError("WFMM shrinkage marginal likelihood is nonfinite")
    return pi, u, iterations, converged, log_likelihood


def calibrate_wfmm_shrinkage(
    coefficients: ArrayLike,
    fixed_design: ArrayLike,
    random_design: ArrayLike | None = None,
    *,
    random_variance: ArrayLike | None,
    residual_variance: ArrayLike,
    random_strata: ArrayLike | None = None,
    residual_strata: ArrayLike | None = None,
    coefficient_partition: ArrayLike | None = None,
    initial_inclusion_probability: float = 0.5,
    initial_slab_to_sampling_variance: float = 1.0,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
) -> WFMMShrinkage:
    """Estimate full-GLS fixed effects and groupwise spike-and-slab priors.

    The conditional sampling variance for coefficient ``i,k`` is
    ``1 / (X_i' Sigma_k^-1 X_i)``; it is not the corresponding diagonal entry
    of the inverse joint information matrix. EM groups are fixed-effect rows
    crossed with coefficient partitions. ``random_variance`` and
    ``residual_variance`` are explicit starting/fixed covariance components.
    To avoid unstable full GLS estimates, the diagonal-normalized information
    matrix must have condition number at most ``1e12``.
    """
    raw_d, raw_x = np.asarray(coefficients), np.asarray(fixed_design)
    if (
        raw_d.ndim != 2
        or not 1 <= raw_d.shape[0] <= _MAX_ROWS
        or not 1 <= raw_d.shape[1] <= _MAX_COEFFICIENTS
    ):
        raise ValueError("coefficients must be an N-by-K matrix within supported limits")
    rows, coefficient_count = raw_d.shape
    if raw_x.ndim != 2 or raw_x.shape[0] != rows or not 1 <= raw_x.shape[1] <= _MAX_FIXED_EFFECTS:
        raise ValueError("fixed_design must be an N-by-P matrix with 1..100 columns")
    fixed_count = raw_x.shape[1]
    if random_design is None:
        z_design = np.empty((rows, 0), dtype=np.float64)
    else:
        raw_z = np.asarray(random_design)
        if raw_z.ndim != 2 or raw_z.shape[0] != rows or raw_z.shape[1] > _MAX_RANDOM_EFFECTS:
            raise ValueError("random_design must be N-by-M with at most 500 columns")
        z_design = _finite_real(raw_z, "random_design")
    if rows * (fixed_count + z_design.shape[1] + coefficient_count) > _MAX_DESIGN_CELLS:
        raise ValueError("WFMM calibration design exceeds the 2000000-cell limit")
    d = _finite_real(raw_d, "coefficients")
    x = _finite_real(raw_x, "fixed_design")
    column_scale = np.max(np.abs(x), axis=0)
    if np.any(column_scale == 0.0):
        raise ArithmeticError("fixed-effect GLS design is not full rank")
    scaled_x = x / column_scale
    scaled_x /= np.sqrt(np.sum(scaled_x * scaled_x, axis=0))
    if np.linalg.matrix_rank(scaled_x) != fixed_count:
        raise ArithmeticError("fixed-effect GLS design is not full rank")

    random_groups = (
        _strata(random_strata, z_design.shape[1], "random_strata")
        if z_design.shape[1]
        else np.empty(0, dtype=np.int64)
    )
    if z_design.shape[1] == 0 and random_strata is not None:
        raise ValueError("random_strata requires a nonempty random_design")
    residual_groups = _strata(residual_strata, rows, "residual_strata")
    random_group_count = int(random_groups.max()) + 1 if random_groups.size else 0
    residual_group_count = int(residual_groups.max()) + 1
    q = _component_array(
        random_variance,
        (random_group_count, coefficient_count),
        "random_variance",
        positive=True,
        default_empty=random_group_count == 0,
    )
    s = _component_array(
        residual_variance,
        (residual_group_count, coefficient_count),
        "residual_variance",
        positive=True,
    )
    labels = _coefficient_labels(coefficient_partition, coefficient_count)
    partitions = np.unique(labels)
    raw_initial_pi = np.asarray(initial_inclusion_probability)
    raw_initial_u = np.asarray(initial_slab_to_sampling_variance)
    if raw_initial_pi.ndim != 0 or raw_initial_pi.dtype.kind not in "iuf":
        raise ValueError("initial_inclusion_probability must be a real scalar")
    if raw_initial_u.ndim != 0 or raw_initial_u.dtype.kind not in "iuf":
        raise ValueError("initial_slab_to_sampling_variance must be a real scalar")
    initial_pi = float(initial_inclusion_probability)
    initial_u = float(initial_slab_to_sampling_variance)
    if not np.isfinite(initial_pi) or not 0.0 < initial_pi < 1.0:
        raise ValueError("initial_inclusion_probability must lie in (0,1)")
    if not np.isfinite(initial_u) or initial_u < 0.0:
        raise ValueError("initial_slab_to_sampling_variance must be finite and nonnegative")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(
        max_iterations, (int, np.integer)
    ):
        raise ValueError("max_iterations must be an integer in [1,10000]")
    max_iter = int(max_iterations)
    if not 1 <= max_iter <= 10_000:
        raise ValueError("max_iterations must be an integer in [1,10000]")
    raw_tolerance = np.asarray(tolerance)
    if raw_tolerance.ndim != 0 or raw_tolerance.dtype.kind not in "iuf":
        raise ValueError("tolerance must be a real scalar")
    tolerance_value = float(tolerance)
    if not np.isfinite(tolerance_value) or not 0.0 < tolerance_value < 1.0:
        raise ValueError("tolerance must be finite and lie in (0,1)")

    work = coefficient_count * (
        rows**3 + rows**2 * (fixed_count + z_design.shape[1]) + fixed_count**3
    )
    em_work = fixed_count * coefficient_count * max_iter * 8
    if work + em_work > _MAX_WORK:
        raise ValueError("WFMM shrinkage calibration exceeds the bounded work limit")

    estimates = np.empty((fixed_count, coefficient_count), dtype=np.float64)
    conditional_variances = np.empty_like(estimates)
    for coefficient in range(coefficient_count):
        covariance = _marginal_covariance(
            z_design,
            random_groups,
            q,
            residual_groups,
            s,
            coefficient,
        )
        try:
            factor = cho_factor(covariance, lower=True, check_finite=False)
            inverse_d = cho_solve(factor, d[:, coefficient], check_finite=False)
            inverse_x = cho_solve(factor, x, check_finite=False)
        except np.linalg.LinAlgError as exc:
            raise ArithmeticError("WFMM covariance or GLS information is singular") from exc
        information = x.T @ inverse_x
        if not np.all(np.isfinite(information)):
            raise ArithmeticError("WFMM GLS information is nonfinite")
        diagonal_information = np.diag(information)
        if np.any(diagonal_information <= 0.0):
            raise ArithmeticError("fixed-effect conditional information is not positive")
        diagonal_root = np.sqrt(diagonal_information)
        normalized_information = information / diagonal_root[:, None] / diagonal_root[None, :]
        eigenvalues = np.linalg.eigvalsh(normalized_information)
        if (
            not np.all(np.isfinite(eigenvalues))
            or eigenvalues[0] <= 0.0
            or eigenvalues[-1] / eigenvalues[0] > _MAX_INFORMATION_CONDITION
        ):
            raise ArithmeticError("fixed-effect GLS information is numerically ill-conditioned")
        try:
            info_factor = cho_factor(information, lower=True, check_finite=False)
            estimates[:, coefficient] = cho_solve(info_factor, x.T @ inverse_d, check_finite=False)
        except np.linalg.LinAlgError as exc:
            raise ArithmeticError("fixed-effect GLS design is not full rank") from exc
        conditional_variances[:, coefficient] = 1.0 / diagonal_information
    if not np.all(np.isfinite(estimates)) or not np.all(np.isfinite(conditional_variances)):
        raise ArithmeticError("WFMM GLS estimates or conditional variances are not representable")
    standardized = estimates / np.sqrt(conditional_variances)
    if not np.all(np.isfinite(standardized)):
        raise ArithmeticError("standardized WFMM GLS estimates are not representable")

    inclusion = np.empty_like(estimates)
    ratio = np.empty_like(estimates)
    group_pi = np.empty((fixed_count, partitions.size), dtype=np.float64)
    group_u = np.empty_like(group_pi)
    iterations = np.empty(group_pi.shape, dtype=np.int64)
    converged = np.empty(group_pi.shape, dtype=np.bool_)
    marginal = np.empty(group_pi.shape, dtype=np.float64)
    for effect in range(fixed_count):
        for group_index, partition in enumerate(partitions):
            indices = np.flatnonzero(labels == partition)
            pi, u, iteration_count, success, log_likelihood = _em_group(
                standardized[effect, indices],
                initial_pi=initial_pi,
                initial_u=initial_u,
                max_iterations=max_iter,
                tolerance=tolerance_value,
            )
            group_pi[effect, group_index] = pi
            group_u[effect, group_index] = u
            iterations[effect, group_index] = iteration_count
            converged[effect, group_index] = success
            marginal[effect, group_index] = log_likelihood
            inclusion[effect, indices] = pi
            ratio[effect, indices] = u
    with np.errstate(over="ignore", invalid="ignore"):
        slab = conditional_variances * ratio
    if not np.all(np.isfinite(slab)):
        raise ArithmeticError("calibrated WFMM slab variances are not representable")
    calibrated_prior = WFMMPrior(inclusion, slab)
    partitions.flags.writeable = False
    return WFMMShrinkage(
        calibrated_prior,
        _freeze(estimates),
        _freeze(conditional_variances),
        _freeze(standardized),
        _freeze(inclusion),
        _freeze(ratio),
        _freeze(group_pi),
        _freeze(group_u),
        _freeze_int(iterations),
        _freeze_bool(converged),
        _freeze(marginal),
        partitions,
    )


def _freeze(value: ArrayLike) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    array.flags.writeable = False
    return array


def _freeze_int(value: ArrayLike) -> NDArray[np.int64]:
    array = np.array(value, dtype=np.int64, copy=True)
    array.flags.writeable = False
    return array


def _freeze_bool(value: ArrayLike) -> NDArray[np.bool_]:
    array = np.array(value, dtype=np.bool_, copy=True)
    array.flags.writeable = False
    return array
