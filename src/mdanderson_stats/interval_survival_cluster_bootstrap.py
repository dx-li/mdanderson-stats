"""Cluster-resampled coefficient uncertainty for interval-censored PH fits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

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
from .interval_survival_bootstrap import _index_tape, _positive_int, _sample_covariance

_MAX_CLUSTER_BOOTSTRAP_REPLICATES = 5_000
_MAX_CLUSTER_RESAMPLE_WORK = 50_000_000
_MAX_CLUSTER_BOOTSTRAP_FIT_WORK = 2_000_000_000


@dataclass(frozen=True)
class IntervalSurvivalClusterBootstrap:
    """Cluster bootstrap fits and survivor-conditional coefficient covariance.

    ``cluster_draw_indices`` indexes ``cluster_labels`` in its sorted order.
    Each row contains exactly one draw per original cluster; its
    ``resampled_row_counts`` entry is the expanded observation count before
    equivalent repeated rows are collapsed to integer frequency weights.
    Failed fits remain aligned as NaN coefficient rows. Covariance is based
    only on successful refits and is undefined with fewer than two successes.
    """

    fit: IntervalSurvivalFit
    cluster_labels: tuple[str | int, ...]
    cluster_draw_indices: np.ndarray
    resampled_row_counts: np.ndarray
    coefficient_samples: FloatArray
    replicate_status: tuple[str, ...]
    replicate_errors: tuple[str | None, ...]
    successful_replicates: int
    failed_replicates: int
    covariance: FloatArray
    standard_error: FloatArray
    covariance_conditional_on_success: bool
    on_fit_failure: str
    maximum_possible_resampled_rows: int
    tolerance: float
    max_iterations: int
    max_support: int
    max_work: int
    estimated_resample_work: int
    estimated_fit_work: int
    accounted_fit_work: int


def _cluster_map(values: ArrayLike, n_rows: int) -> tuple[np.ndarray, tuple[str | int, ...]]:
    """Encode homogeneous integer or string cluster IDs in sorted source order."""
    if isinstance(values, (str, bytes)):
        raise ValueError("cluster_ids must be a one-dimensional label sequence")
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or values.size != n_rows:
            raise ValueError("cluster_ids must contain one label per observation")
        raw = values.astype(object, copy=False)
    else:
        if not isinstance(values, (Sequence,)):
            raise ValueError("cluster_ids must be a bounded one-dimensional label sequence")
        if len(values) != n_rows:
            raise ValueError("cluster_ids must contain one label per observation")
        raw = np.fromiter(values, dtype=object, count=n_rows)
    if raw.ndim != 1 or raw.size != n_rows:
        raise ValueError("cluster_ids must contain one label per observation")

    labels: list[str | int] = []
    integer_labels: bool | None = None
    for value in raw.tolist():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("cluster IDs cannot be booleans")
        if isinstance(value, (int, np.integer)):
            label: str | int = int(value)
            is_integer = True
        elif isinstance(value, (str, np.str_)) and str(value):
            label = str(value)
            is_integer = False
        else:
            raise ValueError("cluster IDs must be nonempty strings or integers")
        if integer_labels is None:
            integer_labels = is_integer
        elif integer_labels != is_integer:
            raise ValueError("cluster IDs must use one homogeneous label type")
        labels.append(label)

    ordered = tuple(sorted(set(labels)))
    if len(ordered) < 2:
        raise ValueError("cluster bootstrap requires at least two clusters")
    lookup = {label: index for index, label in enumerate(ordered)}
    codes = np.fromiter((lookup[label] for label in labels), dtype=np.int64, count=n_rows)
    return codes, ordered


def bootstrap_interval_survival_cluster_coefficients(
    lower: ArrayLike,
    upper: ArrayLike,
    cluster_ids: ArrayLike,
    covariates: ArrayLike | None = None,
    *,
    replicates: int = 100,
    cluster_draw_indices: ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
    on_fit_failure: str = "raise",
    tolerance: float = 1e-8,
    max_iterations: int = 500,
    max_support: int = _MAX_SUPPORT,
    max_work: int = _MAX_WORK,
) -> IntervalSurvivalClusterBootstrap:
    """Refit interval PH after sampling the original clusters with replacement.

    Every replicate draws ``G`` clusters from the ``G`` observed clusters,
    where labels are sorted as in the source's factor-based split. The
    zero-based ``cluster_draw_indices`` tape has shape ``(replicates, G)`` and
    indexes that sorted label tuple. Repeated cluster draws multiply each
    member observation's integer case weight; this is likelihood-equivalent
    to concatenating the repeated rows, while preserving their contribution
    to the newly computed support. This explicit-weight implementation does
    not claim random-stream parity with R.

    Source behavior aborts if any refit fails. ``on_fit_failure='record'`` is
    a Python extension that retains the failed replicate without redrawing.
    The original fit's likelihood remains the independent-observation fit;
    the bootstrap covariance is the cluster-robust uncertainty summary.
    """
    b = _positive_int(replicates, "replicates", _MAX_CLUSTER_BOOTSTRAP_REPLICATES)
    if cluster_draw_indices is not None and rng is not None:
        raise ValueError("rng cannot be supplied with cluster_draw_indices")
    if rng is not None and not isinstance(rng, np.random.Generator):
        if isinstance(rng, (bool, np.bool_)) or not isinstance(rng, (int, np.integer)) or rng < 0:
            raise ValueError("rng must be a nonnegative integer, Generator or None")
    if on_fit_failure not in ("raise", "record"):
        raise ValueError("on_fit_failure must be 'raise' or 'record'")
    tol = scalar(tolerance, "tolerance")
    if tol <= 0:
        raise ValueError("tolerance must be positive")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(
        max_iterations, (int, np.integer)
    ):
        raise ValueError("max_iterations must be an integer")
    if not 1 <= int(max_iterations) <= _MAX_ITERATIONS:
        raise ValueError(f"max_iterations must be in [1, {_MAX_ITERATIONS}]")
    if isinstance(max_support, (bool, np.bool_)) or not isinstance(
        max_support, (int, np.integer)
    ):
        raise ValueError("max_support must be an integer")
    if not 2 <= int(max_support) <= _MAX_SUPPORT:
        raise ValueError(f"max_support must be in [2, {_MAX_SUPPORT}]")
    if isinstance(max_work, (bool, np.bool_)) or not isinstance(max_work, (int, np.integer)):
        raise ValueError("max_work must be an integer")
    if not 1 <= int(max_work) <= _MAX_WORK:
        raise ValueError(f"max_work must be in [1, {_MAX_WORK}]")

    lo, hi, x, _ = _interval_data(lower, upper, covariates, None)
    n_rows, n_covariates = x.shape
    if n_rows > _MAX_ROWS or n_covariates > _MAX_COVARIATES:
        raise ValueError("cluster bootstrap input exceeds interval PH limits")
    if n_covariates == 0:
        raise ValueError("cluster coefficient bootstrap requires at least one covariate")
    codes, labels = _cluster_map(cluster_ids, n_rows)
    n_clusters = len(labels)
    sizes = np.bincount(codes, minlength=n_clusters)

    # A replicate can select the largest group G times. Frequency weights
    # avoid materializing that expanded row matrix, but record its true size.
    max_expanded_rows = prod((n_clusters, int(np.max(sizes))))
    sample_work = prod((b, n_clusters + n_rows))
    if sample_work > _MAX_CLUSTER_RESAMPLE_WORK:
        raise ValueError(
            "cluster bootstrap sampling and frequency work exceeds the 50000000-unit limit"
        )

    support_lower, support_upper, _, _ = _maximal_intersections(lo, hi)
    original_support = int(support_lower.size)
    if original_support > int(max_support):
        raise ValueError("original Turnbull support exceeds max_support")
    replicate_support_bound = min(int(max_support), 2 * n_rows)
    replicate_fit_work = n_rows * replicate_support_bound
    original_fit_work = n_rows * original_support
    if max(replicate_fit_work, original_fit_work) > int(max_work):
        raise ValueError("cluster bootstrap interval likelihood work exceeds max_work")
    estimated_fit_work = int(max_iterations) * (
        original_fit_work + b * replicate_fit_work
    )
    if estimated_fit_work > _MAX_CLUSTER_BOOTSTRAP_FIT_WORK:
        raise ValueError("worst-case cluster bootstrap fit work exceeds the 2000000000-unit limit")

    tape_cells = prod((b, n_clusters))
    coefficient_cells = prod((b, n_covariates))
    result_cells = (
        5 * coefficient_cells
        + n_covariates * n_covariates
        + 2 * n_covariates
        + 3 * b
        + tape_cells
        + n_clusters
    )
    # Include a routed-row copy, the fitter's covariate scaling/design arrays,
    # and compact row/cluster counters while the original input remains live.
    scratch_cells = (
        4 * n_rows * n_covariates
        + n_covariates * n_covariates
        + 16 * n_rows
        + 8 * max(original_support, replicate_support_bound)
        + 6 * n_covariates
    )
    if result_cells + scratch_cells > _MAX_OUTPUT_CELLS:
        raise ValueError("cluster tape, results and fit scratch exceed the 2000000-cell limit")

    tape = (
        None
        if cluster_draw_indices is None
        else _index_tape(
            cluster_draw_indices,
            replicates=b,
            resample_size=n_clusters,
            n_rows=n_clusters,
        )
    )
    original_fit = fit_interval_survival(
        lo,
        hi,
        x,
        tolerance=tol,
        max_iterations=int(max_iterations),
        max_support=int(max_support),
        max_work=int(max_work),
    )
    generator = np.random.default_rng(rng) if tape is None else None
    if tape is None:
        assert generator is not None
        tape = generator.integers(0, n_clusters, size=(b, n_clusters), endpoint=False)

    row_counts = np.empty(b, dtype=np.int64)
    coefficient_samples = np.full((b, n_covariates), np.nan, dtype=np.float64)
    statuses: list[str] = []
    errors: list[str | None] = []
    accounted_fit_work = original_fit.iterations * original_fit_work
    for replicate in range(b):
        draw_counts = np.bincount(tape[replicate], minlength=n_clusters)
        row_weights = draw_counts[codes]
        selected = row_weights > 0
        row_counts[replicate] = int(np.sum(row_weights, dtype=np.int64))
        try:
            fit = fit_interval_survival(
                lo[selected],
                hi[selected],
                x[selected],
                weights=row_weights[selected].astype(np.float64),
                tolerance=tol,
                max_iterations=int(max_iterations),
                max_support=int(max_support),
                max_work=int(max_work),
            )
        except (ArithmeticError, ValueError) as exc:
            if on_fit_failure == "raise":
                raise type(exc)(f"cluster bootstrap replicate {replicate} failed: {exc}") from exc
            statuses.append("fit_failed" if isinstance(exc, ArithmeticError) else "invalid_data")
            errors.append(str(exc))
            accounted_fit_work += int(max_iterations) * replicate_fit_work
            continue
        coefficient_samples[replicate] = fit.coefficients
        statuses.append("ok")
        errors.append(None)
        accounted_fit_work += (
            fit.iterations * int(np.count_nonzero(selected)) * int(fit.support_lower.size)
        )

    success_rows = coefficient_samples[np.isfinite(coefficient_samples).all(axis=1)]
    covariance, standard_error = _sample_covariance(success_rows)
    tape.setflags(write=False)
    row_counts.setflags(write=False)
    return IntervalSurvivalClusterBootstrap(
        fit=original_fit,
        cluster_labels=labels,
        cluster_draw_indices=tape,
        resampled_row_counts=row_counts,
        coefficient_samples=_freeze(coefficient_samples),
        replicate_status=tuple(statuses),
        replicate_errors=tuple(errors),
        successful_replicates=int(success_rows.shape[0]),
        failed_replicates=b - int(success_rows.shape[0]),
        covariance=_freeze(covariance),
        standard_error=_freeze(standard_error),
        covariance_conditional_on_success=True,
        on_fit_failure=on_fit_failure,
        maximum_possible_resampled_rows=max_expanded_rows,
        tolerance=tol,
        max_iterations=int(max_iterations),
        max_support=int(max_support),
        max_work=int(max_work),
        estimated_resample_work=sample_work,
        estimated_fit_work=estimated_fit_work,
        accounted_fit_work=int(accounted_fit_work),
    )
