"""Case-weighted coefficient bootstrap for interval-censored PH regression.

This follows the ordinary ``icenReg::ic_sp`` coefficient-bootstrap contract.
It does not bootstrap or attach confidence limits to the nonparametric
baseline: locations of mass inside Turnbull support intervals remain an
identification issue, not a sampling interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, prod

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, scalar
from .interval_survival import (
    _MAX_COVARIATES,
    _MAX_ITERATIONS,
    _MAX_OUTPUT_CELLS,
    _MAX_ROWS,
    _MAX_SUPPORT,
    _MAX_WORK,
    IntervalSurvivalFit,
    _interval_data,
    _maximal_intersections,
    fit_interval_survival,
)

_MAX_BOOTSTRAP_REPLICATES = 5_000
_MAX_RESAMPLE_SIZE = 100_000
_MAX_RESAMPLE_WORK = 50_000_000
_MAX_BOOTSTRAP_FIT_WORK = 2_000_000_000


@dataclass(frozen=True)
class IntervalSurvivalBootstrap:
    """Original fit and source-style coefficient resamples.

    ``coefficient_samples`` has one row per requested replicate; failed rows
    are NaN and remain aligned with ``replicate_status``/``replicate_errors``.
    Covariance and standard errors use successful rows only, matching the
    native omission rule, and are explicitly survivor-conditional. They are
    undefined (NaN) when fewer than two replicates succeed.
    """

    fit: IntervalSurvivalFit
    coefficient_samples: FloatArray
    replicate_status: tuple[str, ...]
    replicate_errors: tuple[str | None, ...]
    successful_replicates: int
    failed_replicates: int
    covariance: FloatArray
    standard_error: FloatArray
    covariance_conditional_on_success: bool
    resample_size: int
    estimated_resample_work: int
    estimated_fit_work: int
    # Successful fits contribute measured work; a failed fit is charged its
    # conservative full-iteration allowance because the fitter exposes no
    # iteration count on exceptions.
    accounted_fit_work: int


def _positive_int(value: int, name: str, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not 1 <= result <= upper:
        raise ValueError(f"{name} must be in [1, {upper}]")
    return result


def _index_tape(
    value: ArrayLike, *, replicates: int, resample_size: int, n_rows: int
) -> np.ndarray:
    """Validate a bounded integer tape before copying it."""
    expected = (replicates, resample_size)
    if isinstance(value, np.ndarray):
        if value.shape != expected:
            raise ValueError(f"resample_indices must have shape {expected}")
        if value.dtype.kind not in "iu":
            raise ValueError("resample_indices must contain integers")
        if value.size and (np.any(value < 0) or np.any(value >= n_rows)):
            raise ValueError("resample_indices contains an out-of-range row index")
        return np.array(value, dtype=np.int64, order="C", copy=True)
    if not isinstance(value, (list, tuple)):
        raise ValueError("resample_indices must be a two-dimensional integer array or nested list")
    rows = value
    if len(rows) != replicates:
        raise ValueError(f"resample_indices must have shape {expected}")
    for row in rows:
        if isinstance(row, np.ndarray):
            if row.ndim != 1 or row.dtype.kind not in "iu":
                raise ValueError("resample_indices must contain one-dimensional integer rows")
        elif not isinstance(row, (list, tuple)):
            raise ValueError("resample_indices must contain integer row indices")
        if len(row) != resample_size:
            raise ValueError(f"resample_indices must have shape {expected}")
        if isinstance(row, (list, tuple)):
            for item in row:
                if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
                    raise ValueError(
                        "resample_indices must contain integers, not booleans or floats"
                    )
                if item < 0 or item >= n_rows:
                    raise ValueError("resample_indices contains an out-of-range row index")
        elif row.size and (np.any(row < 0) or np.any(row >= n_rows)):
            raise ValueError("resample_indices contains an out-of-range row index")
    try:
        return np.array(rows, dtype=np.int64, order="C", copy=True)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("resample_indices must contain representable integer row indices") from exc


def _sample_covariance(samples: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Compute ddof=1 covariance after column scaling to avoid square overflow."""
    n, p = samples.shape
    if n < 2:
        missing = np.full((p, p), np.nan)
        return missing, np.full(p, np.nan)
    scale = np.max(np.abs(samples), axis=0)
    scale[scale == 0] = 1.0
    normalized = samples / scale
    centered = normalized - np.mean(normalized, axis=0)
    cov_normalized = (centered.T @ centered) / (n - 1)
    cov_normalized = 0.5 * (cov_normalized + cov_normalized.T)
    # Combine mantissas and exponents before rescaling. Multiplying by one
    # column scale first could underflow even when the full covariance entry
    # becomes representable after multiplication by the other scale.
    cov_mantissa, cov_exponent = np.frexp(cov_normalized)
    scale_mantissa, scale_exponent = np.frexp(scale)
    covariance_mantissa = cov_mantissa * scale_mantissa[:, None] * scale_mantissa[None, :]
    covariance_exponent = cov_exponent + scale_exponent[:, None] + scale_exponent[None, :]
    with np.errstate(over="ignore", invalid="ignore"):
        covariance = np.ldexp(covariance_mantissa, covariance_exponent)
    if not np.isfinite(covariance).all():
        raise ArithmeticError("bootstrap coefficient covariance exceeds float64 range")
    if np.any((covariance_mantissa != 0.0) & (covariance == 0.0)):
        raise ArithmeticError(
            "bootstrap coefficient covariance is below float64 range after rescaling"
        )
    diagonal = np.diag(covariance)
    if np.any(diagonal < -1e-12 * max(float(np.max(np.abs(diagonal), initial=0.0)), 1.0)):
        raise ArithmeticError("bootstrap coefficient covariance has a negative variance")
    standard_error = np.sqrt(np.maximum(diagonal, 0.0))
    return covariance, standard_error


def bootstrap_interval_survival_coefficients(
    lower: ArrayLike,
    upper: ArrayLike,
    covariates: ArrayLike | None = None,
    *,
    weights: ArrayLike | None = None,
    replicates: int = 100,
    resample_indices: ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 500,
    max_support: int = _MAX_SUPPORT,
    max_work: int = _MAX_WORK,
) -> IntervalSurvivalBootstrap:
    """Fit interval PH and estimate coefficient covariance by case bootstrap.

    Each replicate draws ``ceil(sum(weights))`` original row indices with
    replacement and probabilities proportional to the original case weights.
    Duplicate sampled rows are collapsed to integer frequency weights before
    fitting. An explicit ``resample_indices`` tape uses zero-based row indices
    and has shape ``(replicates, ceil(sum(weights)))``; it is mutually
    exclusive with ``rng``. A seed, ``Generator`` or ``None`` follows the
    package's NumPy random-state convention and does not match R's RNG stream.

    Failed fits remain in the status ledger and as NaN sample rows; no
    replacement draws are generated. The returned covariance and standard
    errors use successful draws only, as in ``ic_sp``, and are conditional on
    that success set. Baseline survival confidence bands are not produced.
    Stratified and cluster bootstrap policies are not inferred from this
    ordinary-fit source contract.
    """
    b = _positive_int(replicates, "replicates", _MAX_BOOTSTRAP_REPLICATES)
    if resample_indices is not None and rng is not None:
        raise ValueError("rng cannot be supplied with resample_indices")
    if rng is not None and not isinstance(rng, np.random.Generator):
        if isinstance(rng, (bool, np.bool_)) or not isinstance(rng, (int, np.integer)) or rng < 0:
            raise ValueError("rng must be a nonnegative integer, Generator or None")
    tol = scalar(tolerance, "tolerance")
    if tol <= 0:
        raise ValueError("tolerance must be positive")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or not 1 <= int(max_iterations) <= _MAX_ITERATIONS
    ):
        raise ValueError(f"max_iterations must be in [1, {_MAX_ITERATIONS}]")
    if (
        isinstance(max_support, (bool, np.bool_))
        or not isinstance(max_support, (int, np.integer))
        or not 2 <= int(max_support) <= _MAX_SUPPORT
    ):
        raise ValueError(f"max_support must be in [2, {_MAX_SUPPORT}]")
    if (
        isinstance(max_work, (bool, np.bool_))
        or not isinstance(max_work, (int, np.integer))
        or not 1 <= int(max_work) <= _MAX_WORK
    ):
        raise ValueError(f"max_work must be in [1, {_MAX_WORK}]")

    lo, hi, x, w = _interval_data(lower, upper, covariates, weights)
    n_rows, n_covariates = x.shape
    if n_covariates == 0:
        raise ValueError("coefficient bootstrap requires at least one covariate")
    if n_rows > _MAX_ROWS or n_covariates > _MAX_COVARIATES:
        raise ValueError("bootstrap input exceeds interval PH limits")
    weight_total = float(np.sum(w))
    if not np.isfinite(weight_total) or weight_total <= 0:
        raise ValueError("weights must have a finite positive total")
    resample_size = ceil(weight_total)
    if resample_size > _MAX_RESAMPLE_SIZE:
        raise ValueError(f"weighted bootstrap sample size exceeds {_MAX_RESAMPLE_SIZE}")
    estimated_resample_work = prod((b, resample_size))
    if estimated_resample_work > _MAX_RESAMPLE_WORK:
        raise ValueError("bootstrap sampling and row-collapse work exceeds the 50000000-unit limit")
    tape_cells = prod((b, resample_size)) if resample_indices is not None else 0
    result_cells = (
        5 * prod((b, n_covariates)) + n_covariates * n_covariates + 2 * n_covariates + 2 * b
    )
    scratch_cells = resample_size + 4 * n_rows + 4 * n_covariates
    if 2 * tape_cells + result_cells + scratch_cells > _MAX_OUTPUT_CELLS:
        raise ValueError("bootstrap tape, results and scratch exceed the 2000000-cell limit")

    support_lower, support_upper, _, _ = _maximal_intersections(lo, hi)
    original_support = support_lower.size
    if original_support > int(max_support):
        raise ValueError("original Turnbull support exceeds max_support")
    if n_rows * original_support > int(max_work):
        raise ValueError("original interval likelihood work estimate exceeds max_work")
    # A resampled subset can have up to two distinct endpoints per source row.
    replicate_support_bound = min(int(max_support), 2 * n_rows)
    per_fit_bound = n_rows * replicate_support_bound
    if per_fit_bound > int(max_work):
        raise ValueError("bootstrap replicate interval likelihood work exceeds max_work")
    estimated_fit_work = int(max_iterations) * (n_rows * original_support + b * per_fit_bound)
    if estimated_fit_work > _MAX_BOOTSTRAP_FIT_WORK:
        raise ValueError("worst-case bootstrap fitting work exceeds the 2000000000-unit hard limit")
    tape = (
        None
        if resample_indices is None
        else _index_tape(resample_indices, replicates=b, resample_size=resample_size, n_rows=n_rows)
    )

    # The original fit is deterministic. Complete it before creating/advancing
    # a random generator so an invalid fit never consumes caller randomness.
    fit = fit_interval_survival(
        lo,
        hi,
        x,
        weights=w,
        tolerance=tol,
        max_iterations=int(max_iterations),
        max_support=int(max_support),
        max_work=int(max_work),
    )
    generator = np.random.default_rng(rng) if tape is None else None
    probabilities = w / weight_total
    samples = np.full((b, n_covariates), np.nan)
    statuses: list[str] = []
    errors: list[str | None] = []
    accounted_fit_work = fit.iterations * n_rows * original_support
    for replicate in range(b):
        if tape is not None:
            sampled = tape[replicate]
        else:
            assert generator is not None
            sampled = generator.choice(n_rows, size=resample_size, replace=True, p=probabilities)
        multiplicity = np.bincount(sampled, minlength=n_rows)
        retained = np.flatnonzero(multiplicity)
        counts = multiplicity[retained].astype(np.float64)
        try:
            boot_fit = fit_interval_survival(
                lo[retained],
                hi[retained],
                x[retained],
                weights=counts,
                tolerance=tol,
                max_iterations=int(max_iterations),
                max_support=int(max_support),
                max_work=int(max_work),
            )
        except (ValueError, ArithmeticError) as exc:
            statuses.append("fit_failed")
            errors.append(str(exc))
            accounted_fit_work += int(max_iterations) * retained.size * replicate_support_bound
            continue
        samples[replicate] = boot_fit.coefficients
        statuses.append("ok")
        errors.append(None)
        accounted_fit_work += boot_fit.iterations * retained.size * boot_fit.support_mass.size

    successful = samples[np.isfinite(samples).all(axis=1)]
    covariance, standard_error = _sample_covariance(successful)
    return IntervalSurvivalBootstrap(
        fit,
        _freeze(samples),
        tuple(statuses),
        tuple(errors),
        int(successful.shape[0]),
        b - int(successful.shape[0]),
        _freeze(covariance),
        _freeze(standard_error),
        True,
        resample_size,
        estimated_resample_work,
        estimated_fit_work,
        int(accounted_fit_work),
    )
