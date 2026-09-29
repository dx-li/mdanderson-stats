"""Coefficient bootstrap for shared-coefficient stratified interval PH fits.

The stratified fit is a Python interval-likelihood extension. The available
native sources do not define a stratified bootstrap policy, so callers choose
between fixed-size within-stratum draws and the ordinary pooled weighted-row
draw explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, fsum, prod
from typing import Literal

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
    _interval_data,
    _maximal_intersections,
)
from .interval_survival_bootstrap import (
    _MAX_BOOTSTRAP_FIT_WORK,
    _MAX_BOOTSTRAP_REPLICATES,
    _MAX_RESAMPLE_SIZE,
    _MAX_RESAMPLE_WORK,
    _index_tape,
    _positive_int,
    _sample_covariance,
)
from .interval_survival_stratified import (
    StratifiedIntervalSurvivalFit,
    fit_stratified_interval_survival,
)
from .survan_cox import _encode_strata


@dataclass(frozen=True)
class StratifiedIntervalSurvivalBootstrap:
    """Original fit and coefficient draws from a stratified row bootstrap.

    ``resample_indices`` is retained as an immutable zero-based global-row
    tape, including when generated from ``rng``, so every refit is replayable.
    Failure rows remain aligned as NaNs. Covariance and standard errors use
    successful refits only and are undefined with fewer than two successes.
    """

    fit: StratifiedIntervalSurvivalFit
    stratum_labels: tuple[str | int, ...]
    resampling: Literal["within_stratum", "pooled"]
    within_stratum_sample_sizes: tuple[int, ...]
    resample_size: int
    resample_indices: np.ndarray
    resampled_stratum_counts: np.ndarray
    coefficient_samples: FloatArray
    replicate_status: tuple[str, ...]
    replicate_errors: tuple[str | None, ...]
    successful_replicates: int
    failed_replicates: int
    covariance: FloatArray
    standard_error: FloatArray
    covariance_conditional_on_success: bool
    estimated_resample_work: int
    estimated_fit_work: int
    accounted_fit_work: int


def _finite_positive_weight_sum(weights: FloatArray, name: str) -> float:
    """Return an accurately summed positive weight total for exact ceil sizes."""
    try:
        total = fsum(float(weight) for weight in weights)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} weights must have a finite positive total") from exc
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"{name} weights must have a finite positive total")
    return total


def _probabilities(weights: FloatArray, name: str) -> FloatArray:
    """Normalize weights without overflowing their sum."""
    scale = float(np.max(weights))
    normalized = weights / scale
    if np.any(normalized == 0.0):
        raise ValueError(f"{name} weight range is too wide for representable resampling")
    total_scaled = float(np.sum(normalized))
    probabilities = normalized / total_scaled
    probabilities /= float(np.sum(probabilities))
    if not np.isfinite(probabilities).all() or np.any(probabilities <= 0):
        raise ValueError(f"{name} resampling probabilities are not representable")
    return probabilities


def _validate_tape_membership(
    tape: np.ndarray,
    *,
    policy: Literal["within_stratum", "pooled"],
    codes: np.ndarray,
    sample_sizes: tuple[int, ...],
) -> None:
    if policy != "within_stratum":
        return
    start = 0
    for group, size in enumerate(sample_sizes):
        block = tape[:, start : start + size]
        if block.size and np.any(codes[block] != group):
            raise ValueError(
                "within_stratum resample_indices blocks must contain rows from "
                "their matching stratum"
            )
        start += size


def _readonly_array(value: np.ndarray) -> np.ndarray:
    """Return an immutable copy without changing integer tape/count dtypes."""
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


def bootstrap_stratified_interval_survival_coefficients(
    lower: ArrayLike,
    upper: ArrayLike,
    covariates: ArrayLike | None = None,
    *,
    strata: ArrayLike,
    resampling: Literal["within_stratum", "pooled"],
    weights: ArrayLike | None = None,
    replicates: int = 100,
    resample_indices: ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 500,
    max_support: int = _MAX_SUPPORT,
    max_work: int = _MAX_WORK,
) -> StratifiedIntervalSurvivalBootstrap:
    """Bootstrap the shared coefficient vector of a stratified interval-PH fit.

    ``resampling`` is required. ``within_stratum`` draws
    ``ceil(sum(weights_g))`` rows independently in each first-seen stratum,
    with probabilities proportional to that stratum's positive case weights.
    ``pooled`` follows the ordinary ``icenReg::ic_sp`` row-bootstrap design:
    draw ``ceil(sum(weights))`` rows globally with probabilities proportional
    to case weights. A pooled draw omitting an original stratum is recorded as
    failed so it cannot silently refit a different set of baselines.

    A replay tape has shape ``(replicates, sum_g ceil(sum(weights_g)))`` for
    ``within_stratum`` or ``(replicates, ceil(sum(weights)))`` for ``pooled``.
    Values are zero-based indices into the original rows. Within-stratum tape
    columns are concatenated blocks in first-seen stratum order and must index
    rows from the corresponding stratum. Repeated indices become sampled
    frequency weights; original case weights are not multiplied into them.
    A seed or generator reproduces a chosen NumPy stream but does not match R.

    Refit failures are retained as NaN rows and are not redrawn. Covariance
    and standard errors are sample summaries conditional on successful fits;
    no baseline confidence band is inferred from them.
    """
    b = _positive_int(replicates, "replicates", _MAX_BOOTSTRAP_REPLICATES)
    if resampling not in ("within_stratum", "pooled"):
        raise ValueError("resampling must be 'within_stratum' or 'pooled'")
    if resample_indices is not None and rng is not None:
        raise ValueError("rng cannot be supplied with resample_indices")
    if rng is not None and not isinstance(rng, np.random.Generator):
        if isinstance(rng, (bool, np.bool_)) or not isinstance(rng, (int, np.integer)) or rng < 0:
            raise ValueError("rng must be a nonnegative integer, Generator or None")
    tol = scalar(tolerance, "tolerance")
    if tol <= 0:
        raise ValueError("tolerance must be positive")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(
        max_iterations, (int, np.integer)
    ):
        raise ValueError("max_iterations must be an integer")
    if not 1 <= int(max_iterations) <= _MAX_ITERATIONS:
        raise ValueError(f"max_iterations must be in [1, {_MAX_ITERATIONS}]")
    if isinstance(max_support, (bool, np.bool_)) or not isinstance(max_support, (int, np.integer)):
        raise ValueError("max_support must be an integer")
    if not 2 <= int(max_support) <= _MAX_SUPPORT:
        raise ValueError(f"max_support must be in [2, {_MAX_SUPPORT}]")
    if isinstance(max_work, (bool, np.bool_)) or not isinstance(max_work, (int, np.integer)):
        raise ValueError("max_work must be an integer")
    if not 1 <= int(max_work) <= _MAX_WORK:
        raise ValueError(f"max_work must be in [1, {_MAX_WORK}]")

    if any(
        np.iscomplexobj(value)
        for value in (lower, upper, covariates, strata, weights)
        if value is not None
    ):
        raise ValueError("interval, covariate, weight, and stratum data must be real")
    lo, hi, x, w = _interval_data(lower, upper, covariates, weights)
    n_rows, n_covariates = x.shape
    if not 1 <= n_rows <= _MAX_ROWS or n_covariates > _MAX_COVARIATES:
        raise ValueError("stratified bootstrap input exceeds interval PH limits")
    if n_covariates == 0:
        raise ValueError("coefficient bootstrap requires at least one covariate")
    if x.size > _MAX_OUTPUT_CELLS:
        raise ValueError("covariate design exceeds the two-million-cell limit")
    codes, labels = _encode_strata(strata)
    if codes.shape != lo.shape:
        raise ValueError("strata must contain one label per interval")

    row_groups = tuple(np.flatnonzero(codes == group) for group in range(len(labels)))
    probability_groups: list[FloatArray] = []
    within_sizes: list[int] = []
    for group, rows in enumerate(row_groups):
        group_weights = w[rows]
        total = _finite_positive_weight_sum(group_weights, f"stratum {labels[group]!r}")
        size = ceil(total)
        if size > _MAX_RESAMPLE_SIZE:
            raise ValueError(
                f"stratum {labels[group]!r} resample size exceeds {_MAX_RESAMPLE_SIZE}"
            )
        probability_groups.append(_probabilities(group_weights, f"stratum {labels[group]!r}"))
        within_sizes.append(size)
    within_sample_sizes = tuple(within_sizes)
    within_total_size = sum(within_sample_sizes)
    total_weight = _finite_positive_weight_sum(w, "pooled")
    pooled_size = ceil(total_weight)
    if pooled_size > _MAX_RESAMPLE_SIZE:
        raise ValueError(f"pooled resample size exceeds {_MAX_RESAMPLE_SIZE}")
    resample_size = within_total_size if resampling == "within_stratum" else pooled_size
    if resample_size > _MAX_RESAMPLE_SIZE:
        raise ValueError(f"total resample size exceeds {_MAX_RESAMPLE_SIZE}")
    pooled_probabilities = _probabilities(w, "pooled") if resampling == "pooled" else None
    estimated_resample_work = prod((b, resample_size + n_rows))
    if estimated_resample_work > _MAX_RESAMPLE_WORK:
        raise ValueError("stratified bootstrap sampling work exceeds the 50000000-unit limit")

    support_by_group: list[int] = []
    original_fit_work = 0
    for group, rows in enumerate(row_groups):
        support = int(_maximal_intersections(lo[rows], hi[rows])[0].size)
        support_by_group.append(support)
        original_fit_work += int(rows.size) * support
    original_support = sum(support_by_group)
    if original_support > int(max_support):
        raise ValueError("original combined Turnbull support exceeds max_support")
    if original_fit_work > int(max_work):
        raise ValueError("original stratified likelihood work estimate exceeds max_work")

    if resampling == "within_stratum":
        maximum_unique_rows = sum(
            min(int(rows.size), size)
            for rows, size in zip(row_groups, within_sample_sizes, strict=True)
        )
    else:
        maximum_unique_rows = sum(min(int(rows.size), pooled_size) for rows in row_groups)
    replicate_support_bound = min(int(max_support), 2 * maximum_unique_rows)
    replicate_fit_work = maximum_unique_rows * replicate_support_bound
    if replicate_fit_work > int(max_work):
        raise ValueError("stratified bootstrap replicate likelihood work exceeds max_work")
    estimated_fit_work = int(max_iterations) * (original_fit_work + b * replicate_fit_work)
    if estimated_fit_work > _MAX_BOOTSTRAP_FIT_WORK:
        raise ValueError(
            "worst-case stratified bootstrap fitting work exceeds the 2000000000-unit limit"
        )

    tape_cells = prod((b, resample_size))
    coefficient_cells = prod((b, n_covariates))
    n_groups = len(labels)
    result_cells = (
        2 * tape_cells
        + 5 * coefficient_cells
        + prod((b, n_groups))
        + n_covariates * n_covariates
        + 2 * n_covariates
        + 2 * b
        + n_groups
    )
    scratch_cells = (
        4 * n_rows * n_covariates
        + n_rows * 8
        + 8 * max(original_support, replicate_support_bound)
        + 8 * n_covariates
        + 8 * n_groups
        + resample_size
    )
    if result_cells + scratch_cells > _MAX_OUTPUT_CELLS:
        raise ValueError(
            "stratified bootstrap tape, results and fit scratch exceed the 2000000-cell limit"
        )

    if resample_indices is None:
        tape = None
    else:
        tape = _index_tape(
            resample_indices,
            replicates=b,
            resample_size=resample_size,
            n_rows=n_rows,
        )
        _validate_tape_membership(
            tape,
            policy=resampling,
            codes=codes,
            sample_sizes=within_sample_sizes,
        )

    # Fit original data first so invalid designs fail before RNG consumption.
    fit = fit_stratified_interval_survival(
        lo,
        hi,
        x,
        strata=strata,
        weights=w,
        tolerance=tol,
        max_iterations=int(max_iterations),
        max_support=int(max_support),
        max_work=int(max_work),
    )
    generator = np.random.default_rng(rng) if tape is None else None
    if tape is None:
        assert generator is not None
        tape = np.empty((b, resample_size), dtype=np.int64)
        if resampling == "pooled":
            assert pooled_probabilities is not None
            tape[:] = generator.choice(
                n_rows,
                size=(b, resample_size),
                replace=True,
                p=pooled_probabilities,
            )
    if resample_indices is None and resampling == "within_stratum":
        assert generator is not None
        # Draw each replicate separately, preserving a straightforward stream
        # order and independent within-group sampling calls.
        for replicate in range(b):
            start = 0
            for rows, size, probability in zip(
                row_groups, within_sample_sizes, probability_groups, strict=True
            ):
                stop = start + size
                local = generator.choice(rows.size, size=size, replace=True, p=probability)
                tape[replicate, start:stop] = rows[local]
                start = stop

    resampled_counts = np.empty((b, n_groups), dtype=np.int64)
    coefficient_samples = np.full((b, n_covariates), np.nan, dtype=np.float64)
    statuses: list[str] = []
    errors: list[str | None] = []
    accounted_fit_work = fit.iterations * original_fit_work
    for replicate in range(b):
        sampled = tape[replicate]
        multiplicity = np.bincount(sampled, minlength=n_rows)
        retained = np.flatnonzero(multiplicity)
        counts = multiplicity[retained].astype(np.float64)
        replicate_codes = codes[retained]
        resampled_counts[replicate] = np.bincount(codes[sampled], minlength=n_groups)
        if resampling == "pooled" and np.any(resampled_counts[replicate] == 0):
            missing = tuple(
                labels[group] for group in np.flatnonzero(resampled_counts[replicate] == 0)
            )
            statuses.append("fit_failed")
            errors.append(f"pooled resample omitted strata {missing!r}")
            accounted_fit_work += int(max_iterations) * replicate_fit_work
            continue
        replicate_strata = np.asarray([labels[int(code)] for code in replicate_codes], dtype=object)
        try:
            bootstrap_fit = fit_stratified_interval_survival(
                lo[retained],
                hi[retained],
                x[retained],
                strata=replicate_strata,
                weights=counts,
                tolerance=tol,
                max_iterations=int(max_iterations),
                max_support=int(max_support),
                max_work=int(max_work),
            )
        except (ArithmeticError, ValueError) as exc:
            statuses.append("fit_failed")
            errors.append(str(exc))
            accounted_fit_work += int(max_iterations) * replicate_fit_work
            continue
        coefficient_samples[replicate] = bootstrap_fit.coefficients
        statuses.append("ok")
        errors.append(None)
        support_used = sum(item.support_lower.size for item in bootstrap_fit.strata)
        accounted_fit_work += bootstrap_fit.iterations * retained.size * support_used
        del bootstrap_fit

    successful = coefficient_samples[np.isfinite(coefficient_samples).all(axis=1)]
    covariance, standard_error = _sample_covariance(successful)
    assert tape is not None
    return StratifiedIntervalSurvivalBootstrap(
        fit=fit,
        stratum_labels=labels,
        resampling=resampling,
        within_stratum_sample_sizes=(within_sample_sizes if resampling == "within_stratum" else ()),
        resample_size=resample_size,
        resample_indices=_readonly_array(tape),
        resampled_stratum_counts=_readonly_array(resampled_counts),
        coefficient_samples=_freeze(coefficient_samples),
        replicate_status=tuple(statuses),
        replicate_errors=tuple(errors),
        successful_replicates=int(successful.shape[0]),
        failed_replicates=b - int(successful.shape[0]),
        covariance=_freeze(covariance),
        standard_error=_freeze(standard_error),
        covariance_conditional_on_success=True,
        estimated_resample_work=estimated_resample_work,
        estimated_fit_work=estimated_fit_work,
        accounted_fit_work=int(accounted_fit_work),
    )
