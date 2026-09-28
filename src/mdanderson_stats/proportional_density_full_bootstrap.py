"""Full-data goodness-of-fit bootstrap for proportional-density curves."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, scalar
from .proportional_density import proportional_density

_MAX_REPLICATES = 10_000
_MAX_RESAMPLED_ROWS = 50_000_000
_MAX_TAPE_CELLS = 2_000_000


@dataclass(frozen=True)
class ProportionalDensityFullBootstrapTape:
    """Fixed-status resampling indices for replayable full-data replicates.

    Event indices address the original fitted pooled failure-time support;
    censor indices address raw censored times within each original arm. Each
    matrix has one row per replicate. Replacement sampling permits repeated
    indices. The four arrays are validated by the bootstrap function.
    """

    control_event: ArrayLike
    treatment_event: ArrayLike
    control_censor: ArrayLike
    treatment_censor: ArrayLike


@dataclass(frozen=True)
class ProportionalDensityFullBootstrap:
    """Full-data Delta_n bootstrap results; failed replicates remain unresolved."""

    statistic: float
    tau: float
    bootstrap_statistics: FloatArray
    pvalue: float | None
    pvalue_lower: float
    pvalue_upper: float
    monte_carlo_standard_error: float | None
    failed_replicates: int
    failure_reasons: tuple[tuple[str, int], ...]


def _full_curve_area(
    time: FloatArray, fitted_survival: FloatArray, empirical_survival: FloatArray, tau: float
) -> float:
    """Integrate squared right-continuous survival-curve differences exactly."""
    keep = time < tau
    points = time[keep]
    difference = fitted_survival[keep] - empirical_survival[keep]
    if not points.size:
        return 0.0
    # Each post-jump value applies from its event time through the next event.
    widths = np.diff(np.r_[points, tau])
    return float(np.dot(widths, difference * difference))


def _matrix_shape(value: ArrayLike, name: str) -> tuple[int, int]:
    if isinstance(value, np.ndarray):
        if value.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional integer matrix")
        return value.shape
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an integer matrix")
    rows = len(value)
    if rows == 0:
        return (0, 0)
    first = value[0]
    if isinstance(first, np.ndarray):
        if first.ndim != 1:
            raise ValueError(f"{name} must be a two-dimensional integer matrix")
        columns = first.size
    elif isinstance(first, (list, tuple)):
        columns = len(first)
    else:
        raise ValueError(f"{name} must be a two-dimensional integer matrix")
    if rows * columns > _MAX_TAPE_CELLS:
        raise ValueError(f"{name} exceeds the {_MAX_TAPE_CELLS:,}-cell tape limit")
    for row in value[1:]:
        if isinstance(row, np.ndarray):
            if row.ndim != 1 or row.size != columns:
                raise ValueError(f"{name} has inconsistent row lengths")
        elif not isinstance(row, (list, tuple)) or len(row) != columns:
            raise ValueError(f"{name} has inconsistent row lengths")
    return (rows, columns)


def _index_matrix(value: ArrayLike, name: str, shape: tuple[int, int], upper: int) -> np.ndarray:
    if _matrix_shape(value, name) != shape:
        raise ValueError(f"{name} must be an integer matrix with shape {shape}")
    if shape[1] == 0:
        return np.empty(shape, dtype=np.int64)
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer matrix with shape {shape}")
    if raw.dtype.kind == "u" and raw.size and int(raw.max()) > np.iinfo(np.int64).max:
        raise ValueError(f"{name} contains an out-of-range index")
    indices = raw.astype(np.int64, copy=False)
    if np.any((indices < 0) | (indices >= upper)):
        raise ValueError(f"{name} contains an out-of-range index")
    return indices


def _validated_tape(
    tape: ProportionalDensityFullBootstrapTape,
    replicates: int,
    event_counts: np.ndarray,
    censor_counts: np.ndarray,
    support_size: int,
    censor_sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        _index_matrix(
            tape.control_event, "control_event", (replicates, int(event_counts[0])), support_size
        ),
        _index_matrix(
            tape.treatment_event,
            "treatment_event",
            (replicates, int(event_counts[1])),
            support_size,
        ),
        _index_matrix(
            tape.control_censor,
            "control_censor",
            (replicates, int(censor_counts[0])),
            int(censor_sizes[0]),
        ),
        _index_matrix(
            tape.treatment_censor,
            "treatment_censor",
            (replicates, int(censor_counts[1])),
            int(censor_sizes[1]),
        ),
    )


def proportional_density_full_bootstrap(
    time: ArrayLike,
    event: ArrayLike,
    treatment: ArrayLike,
    *,
    replicates: int = 999,
    seed: int | None = None,
    tau: float | None = None,
    equal_censoring: bool = False,
    resample_tape: ProportionalDensityFullBootstrapTape | None = None,
) -> ProportionalDensityFullBootstrap:
    """Calibrate the disease-curve Delta_n statistic by full-data resampling.

    Per arm, the original number of observed failures is drawn with replacement
    from the fitted observed-failure mass, and the original number of censoring
    records is drawn with replacement from that arm's raw censor times. Status
    counts remain fixed; these separately sampled, status-labeled records need
    no latent failure/censor pairing. Every replicate refits both curves.

    Integration uses unit weight and the exact right-continuous step integral
    through the original-data endpoint ``tau`` (default: the smaller arm maximum
    follow-up). This differs from the archive's right-endpoint rectangle
    implementation of the same curve-area statistic. It is not a treatment-
    effect null calibration or a parameter confidence interval. NumPy's random
    stream is reproducible for a seed but is not claimed to match R.

    A supplied ``resample_tape`` has four integer matrices with shapes
    ``(replicates, event_count_arm)`` and ``(replicates, censor_count_arm)``.
    It replaces RNG sampling, so a seed cannot be supplied with a tape.
    """
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or not 1 <= replicates <= _MAX_REPLICATES
    ):
        raise ValueError(f"replicates must be an integer in [1,{_MAX_REPLICATES}]")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    if not isinstance(equal_censoring, bool):
        raise ValueError("equal_censoring must be boolean")
    if resample_tape is not None and not isinstance(
        resample_tape, ProportionalDensityFullBootstrapTape
    ):
        raise ValueError("resample_tape must be a ProportionalDensityFullBootstrapTape or None")
    if resample_tape is not None and seed is not None:
        raise ValueError("seed cannot be supplied with an explicit resample_tape")

    shapes = tuple(np.shape(value) for value in (time, event, treatment))
    if (
        len(shapes[0]) != 1
        or not 4 <= shapes[0][0] <= 1_000_000
        or any(shape != shapes[0] for shape in shapes[1:])
    ):
        raise ValueError("require equal-length vectors with 4 to 1,000,000 observations")
    if shapes[0][0] * replicates > _MAX_RESAMPLED_ROWS:
        raise ValueError(f"bootstrap requires at most {_MAX_RESAMPLED_ROWS:,} resampled records")
    if resample_tape is not None and shapes[0][0] * replicates > _MAX_TAPE_CELLS:
        raise ValueError(f"resample_tape requires at most {_MAX_TAPE_CELLS:,} index cells")

    original = proportional_density(time, event, treatment, equal_censoring=equal_censoring)
    t = np.asarray(time, dtype=float)
    d = np.asarray(event, dtype=float)
    z = np.asarray(treatment, dtype=float)
    arm_size = np.array([np.count_nonzero(z == arm) for arm in (0, 1)], dtype=np.int64)
    event_counts = np.array(
        [np.count_nonzero((z == arm) & (d == 1)) for arm in (0, 1)], dtype=np.int64
    )
    censor_counts = arm_size - event_counts
    common_end = float(min(np.max(t[z == 0]), np.max(t[z == 1])))
    endpoint = common_end if tau is None else scalar(tau, "tau")
    if not 0 < endpoint <= common_end:
        raise ValueError("tau must be positive and no greater than common follow-up")
    statistic = _full_curve_area(
        original.time,
        original.disease_survival[:, 0],
        original.nonparametric_survival[:, 0],
        endpoint,
    )

    censor_times = (t[(z == 0) & (d == 0)], t[(z == 1) & (d == 0)])
    probabilities = original.observed_mass / original.observed_mass.sum(axis=0)
    tape_arrays = None
    if resample_tape is not None:
        tape_arrays = _validated_tape(
            resample_tape,
            replicates,
            event_counts,
            censor_counts,
            original.time.size,
            np.array([censor_times[0].size, censor_times[1].size], dtype=np.int64),
        )
        if any(np.any(probabilities[tape_arrays[arm], arm] == 0) for arm in (0, 1)):
            raise ValueError("resample_tape selects an event time with zero fitted probability")

    rng = None if tape_arrays is not None else np.random.default_rng(seed)
    statistics = np.full(replicates, np.nan)
    reasons: dict[str, int] = {}
    for index in range(replicates):
        if tape_arrays is None:
            assert rng is not None
            event_indices = (
                rng.choice(original.time.size, size=int(event_counts[0]), p=probabilities[:, 0]),
                rng.choice(original.time.size, size=int(event_counts[1]), p=probabilities[:, 1]),
            )
            censor_indices = (
                rng.integers(censor_times[0].size, size=int(censor_counts[0]))
                if censor_counts[0]
                else np.empty(0, dtype=np.int64),
                rng.integers(censor_times[1].size, size=int(censor_counts[1]))
                if censor_counts[1]
                else np.empty(0, dtype=np.int64),
            )
        else:
            event_indices = (tape_arrays[0][index], tape_arrays[1][index])
            censor_indices = (tape_arrays[2][index], tape_arrays[3][index])

        event_times = (original.time[event_indices[0]], original.time[event_indices[1]])
        sampled_censors = (
            censor_times[0][censor_indices[0]],
            censor_times[1][censor_indices[1]],
        )
        sample_time = np.concatenate(
            (event_times[0], sampled_censors[0], event_times[1], sampled_censors[1])
        )
        sample_event = np.concatenate(
            (
                np.ones(event_times[0].size),
                np.zeros(sampled_censors[0].size),
                np.ones(event_times[1].size),
                np.zeros(sampled_censors[1].size),
            )
        )
        sample_arm = np.concatenate(
            (
                np.zeros(event_times[0].size + sampled_censors[0].size),
                np.ones(event_times[1].size + sampled_censors[1].size),
            )
        )
        try:
            bootstrap_fit = proportional_density(
                sample_time,
                sample_event,
                sample_arm,
                equal_censoring=equal_censoring,
            )
        except (ValueError, ArithmeticError) as exc:
            reason = str(exc)
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        statistics[index] = _full_curve_area(
            bootstrap_fit.time,
            bootstrap_fit.disease_survival[:, 0],
            bootstrap_fit.nonparametric_survival[:, 0],
            endpoint,
        )

    failed = int(np.isnan(statistics).sum())
    exceedances = int(np.sum(statistics >= statistic))
    lower = (1 + exceedances) / (replicates + 1)
    upper = (1 + exceedances + failed) / (replicates + 1)
    return ProportionalDensityFullBootstrap(
        statistic,
        endpoint,
        _freeze(statistics),
        lower if failed == 0 else None,
        lower,
        upper,
        float(np.sqrt(lower * (1 - lower) / replicates)) if failed == 0 else None,
        failed,
        tuple(sorted(reasons.items())),
    )
