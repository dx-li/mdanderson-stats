"""Convert longitudinal visit records to interval competing-risk observations.

The converter follows the intended ``intccr::dataprep`` first-event rule while
using deterministic ID/time ordering (the pinned R helper accidentally sorts by
logical ``ID & time``). Covariates are baseline values taken from each subject's
earliest retained visit; this is not a time-varying-covariate model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray

_MAX_ROWS = 20_000
_MAX_COVARIATES = 10
_MAX_CELLS = 2_000_000

type SubjectID = str | int | float
type IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class IntervalCompetingRiskVisitData:
    """Immutable subject-level observations produced from visit histories.

    Row indices are zero-based indices into the supplied long-format input.
    ``upper_source_row == -1`` denotes a right-censored record;
    ``lower_source_row == -1`` denotes the zero-time boundary. Rows after the
    first observed event are listed in ``ignored_post_event_rows`` and do not
    affect the interval or baseline covariates.
    """

    subject_id: tuple[SubjectID, ...]
    lower: FloatArray
    upper: FloatArray
    event: IntArray
    covariates: FloatArray
    dropped_missing_time_rows: IntArray
    ignored_post_event_rows: IntArray
    lower_source_row: IntArray
    upper_source_row: IntArray
    covariate_source_row: IntArray


def _shape_before_array(value: object, name: str, *, max_rows: int) -> tuple[int, ...]:
    """Inspect common container shapes and row bounds before NumPy conversion."""
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        n = len(value)
        if n > max_rows:
            raise ValueError(f"{name} exceeds the {max_rows}-row input limit")
        if n == 0 or not any(isinstance(item, (list, tuple, np.ndarray)) for item in value):
            shape = (n,)
        else:
            if any(not isinstance(item, (list, tuple, np.ndarray)) for item in value):
                raise ValueError(f"{name} must be a rectangular array")
            if any(isinstance(item, np.ndarray) and item.ndim != 1 for item in value):
                raise ValueError(f"{name} must be at most two-dimensional")
            lengths = [len(item) for item in value]
            if len(set(lengths)) > 1:
                raise ValueError(f"{name} must be a rectangular array")
            width = lengths[0] if lengths else 0
            if n * width > _MAX_CELLS:
                raise ValueError(f"{name} exceeds the {_MAX_CELLS}-cell input limit")
            for row in value:
                cells = row.tolist() if isinstance(row, np.ndarray) else row
                if any(isinstance(cell, (list, tuple, np.ndarray)) for cell in cells):
                    raise ValueError(f"{name} must be at most two-dimensional")
            shape = (n, width)
    else:
        reported_shape = getattr(value, "shape", None)
        if reported_shape is None:
            raise ValueError(f"{name} must be an array-like value")
        shape = tuple(int(size) for size in reported_shape)
    if not shape or shape[0] > max_rows:
        raise ValueError(f"{name} must have at most {max_rows} rows")
    return tuple(int(size) for size in shape)


def _owned_float(value: np.ndarray) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _owned_int(value: np.ndarray) -> IntArray:
    array = np.asarray(value, dtype=np.int64)
    return np.frombuffer(array.tobytes(), dtype=np.int64).reshape(array.shape)


def _identifier_values(value: object, n: int, retained: np.ndarray) -> tuple[SubjectID | None, ...]:
    shape = _shape_before_array(value, "subject_id", max_rows=_MAX_ROWS)
    if shape != (n,):
        raise ValueError("subject_id must be a one-dimensional vector aligned with visit_time")
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            raise ValueError("subject_id must contain real numeric or string identifiers")
        values = value.tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError("subject_id must contain real numeric or string identifiers")
        values = raw.tolist()
    kinds: set[str] = set()
    normalized: list[SubjectID | None] = []
    for index, item in enumerate(values):
        if not retained[index]:
            normalized.append(None)
            continue
        if isinstance(item, (bool, np.bool_)):
            raise ValueError("boolean subject identifiers are not supported")
        if isinstance(item, (str, np.str_)):
            kinds.add("string")
            normalized.append(str(item))
        elif isinstance(item, (int, np.integer)):
            kinds.add("numeric")
            normalized.append(int(item))
        elif isinstance(item, (float, np.floating)) and np.isfinite(item):
            kinds.add("numeric")
            normalized.append(float(item))
        else:
            raise ValueError("subject_id values must be nonmissing finite numbers or strings")
    if len(kinds) > 1:
        raise ValueError("subject_id values must use one sortable type: numeric or string")
    return tuple(normalized)


def _real_vector(
    value: object, name: str, n: int, *, allow_nan: bool, allow_infinite: bool = False
) -> FloatArray:
    shape = _shape_before_array(value, name, max_rows=_MAX_ROWS)
    if shape != (n,):
        raise ValueError(f"{name} must be a one-dimensional vector aligned with visit_time")
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "iuf" and raw.dtype.kind != "O":
        raise ValueError(f"{name} must contain real numeric values")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if (not allow_infinite and np.any(np.isinf(result))) or (
        not allow_nan and np.any(np.isnan(result))
    ):
        qualifier = "finite values" if not allow_nan else "finite values or missing values"
        raise ValueError(f"{name} must contain {qualifier}")
    return result


def _real_covariates(value: object | None, n: int) -> FloatArray:
    if value is None:
        return np.empty((n, 0), dtype=np.float64)
    shape = _shape_before_array(value, "covariates", max_rows=_MAX_ROWS)
    if len(shape) == 1:
        if shape[0] != n:
            raise ValueError("covariates must have one row per visit")
    elif len(shape) == 2:
        if shape[0] != n or shape[1] > _MAX_COVARIATES:
            raise ValueError(
                f"covariates must have one row per visit and at most {_MAX_COVARIATES} columns"
            )
    else:
        raise ValueError("covariates must be a vector or a two-dimensional matrix")
    if int(np.prod(shape, dtype=np.int64)) > _MAX_CELLS:
        raise ValueError(f"covariates exceed the {_MAX_CELLS}-cell input limit")
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "iuf" and raw.dtype.kind != "O":
        raise ValueError("covariates must contain real numeric values")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("covariates must contain real numeric values") from exc
    if result.ndim == 1:
        result = result[:, None]
    return result


def prepare_interval_competing_risk_visits(
    subject_id: ArrayLike,
    visit_time: ArrayLike,
    status: ArrayLike,
    covariates: ArrayLike | None = None,
) -> IntervalCompetingRiskVisitData:
    """Convert visit-level status histories to ``(lower, upper, cause)`` rows.

    ``status=0`` denotes an event-free visit, while the first status 1 or 2
    marks the first observed cause. The preceding event-free visit is the lower
    endpoint and the event visit is the upper endpoint; if the event is at the
    first visit, the lower endpoint is zero. A subject with no event is
    right-censored at its last visit. Rows with missing visit times are dropped
    and reported. Status arrays must be real numeric; status values after the
    first event are not interpreted. Tied visit times
    within a subject are rejected because their event ordering is unspecified.

    Baseline covariates are taken from the earliest retained visit and must be
    finite there. Later covariate values are not used or checked for finiteness;
    the supplied covariate array must still have a real numeric dtype.
    IDs must be finite numeric values or strings of one homogeneous type. The
    returned arrays are read-only; the existing fitter can consume ``lower``,
    ``upper``, ``event`` and ``covariates`` directly.
    """
    nshape = _shape_before_array(visit_time, "visit_time", max_rows=_MAX_ROWS)
    if len(nshape) != 1 or not 1 <= nshape[0] <= _MAX_ROWS:
        raise ValueError(f"visit_time must be a vector with 1..{_MAX_ROWS} rows")
    n = nshape[0]
    times = _real_vector(visit_time, "visit_time", n, allow_nan=True)
    retained = ~np.isnan(times)
    ids = _identifier_values(subject_id, n, retained)
    statuses = _real_vector(status, "status", n, allow_nan=True, allow_infinite=True)
    x = _real_covariates(covariates, n)

    if np.any(times[~np.isnan(times)] < 0):
        raise ValueError("observed visit times must be nonnegative")
    if np.any(np.isinf(times)):
        raise ValueError("observed visit times must be finite or missing")
    dropped = np.flatnonzero(np.isnan(times)).astype(np.int64)
    retained_rows = np.flatnonzero(retained)
    if retained_rows.size == 0:
        raise ValueError("no visits remain after dropping missing visit times")

    ids_retained: list[SubjectID] = []
    for index in retained_rows:
        identifier = ids[int(index)]
        if identifier is None:
            raise ValueError("subject_id must be present for every retained visit")
        ids_retained.append(identifier)
    ordered_ids = sorted(set(ids_retained))
    group_rows: dict[SubjectID, list[int]] = {identifier: [] for identifier in ordered_ids}
    for index in retained_rows:
        identifier = ids[int(index)]
        assert identifier is not None
        group_rows[identifier].append(int(index))

    output_ids: list[SubjectID] = []
    lower: list[float] = []
    upper: list[float] = []
    event: list[int] = []
    baseline: list[np.ndarray] = []
    lower_rows: list[int] = []
    upper_rows: list[int] = []
    baseline_rows: list[int] = []
    ignored: list[int] = []

    for identifier in ordered_ids:
        rows = sorted(group_rows[identifier], key=lambda row: float(times[row]))
        ordered_times = times[rows]
        if ordered_times.size > 1 and np.any(np.diff(ordered_times) == 0):
            raise ValueError(f"subject {identifier!r} has tied visit times")
        baseline_row = rows[0]
        baseline_values = x[baseline_row]
        if not np.all(np.isfinite(baseline_values)):
            raise ValueError(
                f"baseline covariates must be finite at subject {identifier!r}'s first visit"
            )

        previous_censor: int | None = None
        event_row: int | None = None
        for position, row in enumerate(rows):
            code = statuses[row]
            if not np.isfinite(code) or code not in (0.0, 1.0, 2.0):
                raise ValueError(f"status must be 0, 1, or 2 through the first event (row {row})")
            if code == 0.0:
                previous_censor = row
                continue
            event_row = row
            if position + 1 < len(rows):
                ignored.extend(rows[position + 1 :])
            break

        output_ids.append(identifier)
        baseline.append(np.array(baseline_values, copy=True))
        baseline_rows.append(baseline_row)
        if event_row is None:
            assert previous_censor is not None
            lower.append(float(times[previous_censor]))
            upper.append(np.inf)
            event.append(0)
            lower_rows.append(previous_censor)
            upper_rows.append(-1)
        else:
            event_time = float(times[event_row])
            lower_time = 0.0 if previous_censor is None else float(times[previous_censor])
            if event_time <= lower_time:
                raise ValueError(
                    f"event at row {event_row} does not produce a positive-length interval"
                )
            lower.append(lower_time)
            upper.append(event_time)
            event.append(int(statuses[event_row]))
            lower_rows.append(-1 if previous_censor is None else previous_censor)
            upper_rows.append(event_row)

    output_covariates = (
        np.vstack(baseline).reshape(len(output_ids), x.shape[1])
        if x.shape[1]
        else np.empty((len(output_ids), 0), dtype=np.float64)
    )
    return IntervalCompetingRiskVisitData(
        subject_id=tuple(output_ids),
        lower=_owned_float(np.asarray(lower, dtype=np.float64)),
        upper=_owned_float(np.asarray(upper, dtype=np.float64)),
        event=_owned_int(np.asarray(event, dtype=np.int64)),
        covariates=_owned_float(output_covariates),
        dropped_missing_time_rows=_owned_int(dropped),
        ignored_post_event_rows=_owned_int(np.asarray(sorted(ignored), dtype=np.int64)),
        lower_source_row=_owned_int(np.asarray(lower_rows, dtype=np.int64)),
        upper_source_row=_owned_int(np.asarray(upper_rows, dtype=np.int64)),
        covariate_source_row=_owned_int(np.asarray(baseline_rows, dtype=np.int64)),
    )
