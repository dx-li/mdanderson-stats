"""Row-bootstrap regression uncertainty for interval competing risks.

This follows ``intccr::bssmle_se``'s ordinary row-resampling design and
returns uncertainty for cause-specific regression slopes only. Every
replicate is refit from its full sampled row sequence so empirical event-time
knots are recomputed; duplicate rows are not collapsed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, prod

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, scalar
from .interval_competing_risk import (
    _ALPHA_MAX,
    _MAX_COVARIATES,
    _MAX_ITERATIONS,
    _MAX_OUTPUT_CELLS,
    _MAX_ROWS,
    _MAX_WORK,
    IntervalCompetingRiskFit,
    _choose_knots,
    _fit_interval_competing_risk,
    _input_data,
)
from .interval_survival_bootstrap import _index_tape, _sample_covariance

_MAX_BOOTSTRAP_REPLICATES = 5_000
_MAX_RESAMPLE_WORK = 50_000_000
_MAX_TOTAL_FIT_WORK = 2_000_000_000


@dataclass(frozen=True)
class IntervalCompetingRiskBootstrap:
    """Original fit and aligned cause-specific coefficient resamples.

    Failed optimizer rows remain present as NaN and retain their status and
    error. Covariance and standard errors use successful rows only and are
    explicitly conditional on that subset. They are undefined when fewer
    than two refits succeed. The original fit's separate score-based
    ``fit.covariance`` is unavailable (NaN) because native ``bssmle_se`` does
    not calculate it; use this result's bootstrap covariance instead.
    """

    fit: IntervalCompetingRiskFit
    coefficient_samples: FloatArray
    replicate_status: tuple[str, ...]
    replicate_errors: tuple[str | None, ...]
    successful_replicates: int
    failed_replicates: int
    covariance: FloatArray
    standard_error: FloatArray
    covariance_conditional_on_success: bool
    alpha: FloatArray
    k: float
    on_invalid_resample: str
    boundary_cif_tolerance: float
    tolerance: float
    max_iterations: int
    resample_size: int
    estimated_resample_work: int
    estimated_fit_work: int
    accounted_fit_work: int


def _positive_int(value: int, name: str, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not 1 <= result <= upper:
        raise ValueError(f"{name} must be in [1, {upper}]")
    return result


def _fit_work(n: int, p: int, n_basis: int) -> int:
    # Mirrors the bounded design/corner work preflight in the fitted model.
    corners = 1 if p == 0 else 2**p
    dimension = 2 * n_basis + 2 * p
    return (corners + 2 * n) * dimension


def bootstrap_interval_competing_risk_coefficients(
    lower: ArrayLike,
    upper: ArrayLike,
    event: ArrayLike,
    covariates: ArrayLike | None = None,
    *,
    alpha: ArrayLike = (0.0, 0.0),
    k: float = 1.0,
    replicates: int = 100,
    resample_indices: ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
    on_invalid_resample: str = "raise",
    boundary_cif_tolerance: float = 1e-7,
    tolerance: float = 1e-7,
    max_iterations: int = 1_000,
) -> IntervalCompetingRiskBootstrap:
    """Bootstrap cause-specific slopes by ordinary row resampling.

    The explicit zero-based tape has shape ``(replicates, n_observations)``.
    Its row order and duplicates are passed unchanged to each refit, which
    recalculates empirical spline knots. Missing either event cause is a
    source-level input error and raises by default; ``on_invalid_resample=
    'record'`` is a Python extension that records missing-cause and resampled
    input-validation failures instead. Other unexpected errors are not caught.
    Optimizer nonconvergence is always recorded and is not redrawn.

    A random seed or ``Generator`` follows NumPy's stream and does not match R
    random-stream parity. Only coefficient covariance and standard errors are
    returned; no baseline or cumulative-incidence confidence band is implied.
    """
    b = _positive_int(replicates, "replicates", _MAX_BOOTSTRAP_REPLICATES)
    if resample_indices is not None and rng is not None:
        raise ValueError("rng cannot be supplied with resample_indices")
    if rng is not None and not isinstance(rng, np.random.Generator):
        if isinstance(rng, (bool, np.bool_)) or not isinstance(rng, (int, np.integer)) or rng < 0:
            raise ValueError("rng must be a nonnegative integer, Generator or None")
    if on_invalid_resample not in ("raise", "record"):
        raise ValueError("on_invalid_resample must be 'raise' or 'record'")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or not 20 <= int(max_iterations) <= _MAX_ITERATIONS
    ):
        raise ValueError(f"max_iterations must be in [20, {_MAX_ITERATIONS}]")
    tol = scalar(tolerance, "tolerance")
    if not 1e-9 <= tol <= 1e-3:
        raise ValueError("tolerance must be in [1e-9, 1e-3]")

    lo, hi, code, x = _input_data(lower, upper, event, covariates)
    n, p = x.shape
    if p == 0:
        raise ValueError("coefficient bootstrap requires at least one covariate")
    if n > _MAX_ROWS or p > _MAX_COVARIATES:
        raise ValueError("bootstrap input exceeds interval competing-risk limits")
    if isinstance(alpha, np.ndarray):
        if alpha.shape != (2,):
            raise ValueError(f"alpha must contain two values in [0, {_ALPHA_MAX:g}]")
    elif isinstance(alpha, (list, tuple)):
        if len(alpha) != 2 or any(isinstance(item, (list, tuple, np.ndarray)) for item in alpha):
            raise ValueError(f"alpha must contain two values in [0, {_ALPHA_MAX:g}]")
    else:
        alpha_shape = getattr(alpha, "shape", None)
        if alpha_shape is not None and tuple(alpha_shape) != (2,):
            raise ValueError(f"alpha must contain two values in [0, {_ALPHA_MAX:g}]")
    if np.iscomplexobj(alpha):
        raise ValueError("alpha must be real")
    alpha_values = np.asarray(alpha, dtype=np.float64)
    if (
        alpha_values.shape != (2,)
        or not np.isfinite(alpha_values).all()
        or np.any(alpha_values < 0.0)
        or np.any(alpha_values > _ALPHA_MAX)
    ):
        raise ValueError(f"alpha must contain two values in [0, {_ALPHA_MAX:g}]")
    knot_rate = scalar(k, "k")
    if not 0.5 <= knot_rate <= 1.0:
        raise ValueError("k must be between 0.5 and 1")
    boundary_tol = scalar(boundary_cif_tolerance, "boundary_cif_tolerance")
    if not 1e-12 <= boundary_tol <= 1e-3:
        raise ValueError("boundary_cif_tolerance must be in [1e-12, 1e-3]")

    estimated_resample_work = prod((b, n))
    if estimated_resample_work > _MAX_RESAMPLE_WORK:
        raise ValueError("bootstrap sampling and row-copy work exceeds the 50000000-unit limit")
    p_out = 2 * p
    tape_cells = estimated_resample_work if resample_indices is not None else 0
    result_cells = 5 * b * p_out + p_out * p_out + 2 * p_out + 3 * b

    endpoint = np.r_[lo, hi[code > 0]]
    original_knots, _ = _choose_knots(endpoint, knot_rate)
    original_basis = original_knots.size + 4
    # A replicate has at most n lower endpoints and n event upper endpoints.
    replicate_basis = floor(knot_rate * (2 * n) ** (1.0 / 3.0)) + 4
    corners = 2**p
    fit_scratch_cells = (
        n * (2 * replicate_basis + 20 + 4 * p)
        + 2 * replicate_basis
        + 4 * p
        + corners * (4 * replicate_basis + 2 * p + 4)
    )
    if 2 * tape_cells + result_cells + fit_scratch_cells > _MAX_OUTPUT_CELLS:
        raise ValueError("bootstrap tape, results and fit scratch exceed the 2000000-cell limit")
    original_work = _fit_work(n, p, original_basis)
    replicate_work = _fit_work(n, p, replicate_basis)
    if max(original_work, replicate_work) > _MAX_WORK:
        raise ValueError("interval competing-risk bootstrap fit exceeds the per-fit work limit")
    estimated_fit_work = int(max_iterations) * (original_work + b * replicate_work)
    if estimated_fit_work > _MAX_TOTAL_FIT_WORK:
        raise ValueError("worst-case bootstrap fitting work exceeds the 2000000000-unit hard limit")

    tape = (
        None
        if resample_indices is None
        else _index_tape(resample_indices, replicates=b, resample_size=n, n_rows=n)
    )

    # Fit deterministic original data before creating or advancing a generator.
    fit = _fit_interval_competing_risk(
        lo,
        hi,
        code,
        x,
        alpha=alpha_values,
        k=knot_rate,
        boundary_cif_tolerance=boundary_tol,
        tolerance=tol,
        max_iterations=int(max_iterations),
        compute_covariance=False,
    )
    generator = np.random.default_rng(rng) if tape is None else None
    samples = np.full((b, p_out), np.nan, dtype=np.float64)
    statuses: list[str] = []
    errors: list[str | None] = []
    accounted_fit_work = fit.iterations * original_work
    for replicate in range(b):
        if tape is None:
            assert generator is not None
            indices = generator.integers(0, n, size=n, endpoint=False)
        else:
            indices = tape[replicate]
        sampled_events = code[indices]
        present = (np.any(sampled_events == 1), np.any(sampled_events == 2))
        if not all(present):
            message = f"resample {replicate} omits cause {1 if not present[0] else 2}"
            if on_invalid_resample == "raise":
                raise ValueError(message)
            statuses.append("missing_cause")
            errors.append(message)
            continue
        try:
            boot_fit = _fit_interval_competing_risk(
                lo[indices],
                hi[indices],
                sampled_events,
                x[indices],
                alpha=alpha_values,
                k=knot_rate,
                boundary_cif_tolerance=boundary_tol,
                tolerance=tol,
                max_iterations=int(max_iterations),
                compute_covariance=False,
            )
        except ArithmeticError as exc:
            statuses.append("fit_failed")
            errors.append(str(exc))
            accounted_fit_work += int(max_iterations) * replicate_work
            continue
        except ValueError as exc:
            if on_invalid_resample == "raise":
                raise ValueError(f"resample {replicate} could not be fit: {exc}") from exc
            statuses.append("invalid_data")
            errors.append(str(exc))
            accounted_fit_work += int(max_iterations) * replicate_work
            continue
        if not boot_fit.converged:
            statuses.append("not_converged")
            errors.append("optimizer did not satisfy the fitted convergence check")
            accounted_fit_work += int(max_iterations) * replicate_work
            continue
        if not np.isfinite(boot_fit.coefficients).all():
            raise ArithmeticError(f"resample {replicate} produced non-finite slope coefficients")
        samples[replicate] = boot_fit.coefficients
        statuses.append("ok")
        errors.append(None)
        actual_basis = boot_fit.knots.size + 4
        accounted_fit_work += boot_fit.iterations * _fit_work(n, p, actual_basis)

    successful = samples[np.isfinite(samples).all(axis=1)]
    covariance, standard_error = _sample_covariance(successful)
    alpha_copy = np.array(alpha_values, dtype=np.float64, copy=True)
    return IntervalCompetingRiskBootstrap(
        fit=fit,
        coefficient_samples=_freeze(samples),
        replicate_status=tuple(statuses),
        replicate_errors=tuple(errors),
        successful_replicates=int(successful.shape[0]),
        failed_replicates=b - int(successful.shape[0]),
        covariance=_freeze(covariance),
        standard_error=_freeze(standard_error),
        covariance_conditional_on_success=True,
        alpha=_freeze(alpha_copy),
        k=knot_rate,
        on_invalid_resample=on_invalid_resample,
        boundary_cif_tolerance=boundary_tol,
        tolerance=tol,
        max_iterations=int(max_iterations),
        resample_size=n,
        estimated_resample_work=estimated_resample_work,
        estimated_fit_work=estimated_fit_work,
        accounted_fit_work=int(accounted_fit_work),
    )
