"""MUHAZ Nelson and product-limit estimates over failure-time intervals."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite
from .gray_test import _labels


@dataclass(frozen=True)
class KPHazard:
    time: FloatArray
    hazard: FloatArray
    variance: FloatArray
    stratum_index: NDArray[np.int64]
    strata: tuple[str | float, ...]
    q: int
    method: str
    legacy: bool


def _windows(values: FloatArray, q: int) -> FloatArray:
    """Sum consecutive increments without subtracting infinite cumulative totals."""
    prefix = np.r_[
        np.longdouble(0), np.cumsum(np.where(np.isfinite(values), values, 0), dtype=np.longdouble)
    ]
    result = np.asarray(prefix[q:] - prefix[:-q], dtype=float)
    for mask, value in [(np.isinf(values), np.inf), (np.isnan(values), np.nan)]:
        counts = np.r_[0, np.cumsum(mask)]
        result[counts[q:] > counts[:-q]] = value
    return result


def kphaz(
    time: ArrayLike,
    status: ArrayLike,
    *,
    strata: ArrayLike | None = None,
    q: int = 1,
    method: Literal["nelson", "product-limit"] = "nelson",
    legacy: bool = False,
) -> KPHazard:
    """Difference cumulative hazards across q consecutive distinct failure gaps.

    Estimates are located at interval midpoints, excluding the initial failure's
    increment. Default tied-event increments use the common risk set, including
    tied censoring. legacy=True reproduces sequential source weights and its
    terminal-censor 0/0 variance behavior. Strata with <=q failure times contribute
    no output rows. Missing values are rejected; status must be binary.
    """
    t, d = finite(time, "time"), count(status, "status")
    if t.ndim != 1 or t.size == 0 or d.shape != t.shape or np.any(t < 0) or np.any(d > 1):
        raise ValueError(
            "Require nonempty matching time/status vectors, nonnegative times and binary status"
        )
    if not np.any(d):
        raise ValueError("No events occur in this dataset")
    if isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)) or q < 1:
        raise ValueError("q must be a positive integer")
    if method not in ("nelson", "product-limit"):
        raise ValueError("method must be nelson or product-limit")
    if not isinstance(legacy, (bool, np.bool_)):
        raise ValueError("legacy must be boolean")
    labels, groups = _labels(np.ones(t.size) if strata is None else strata, t.shape, "strata")
    outputs = []
    for group in range(len(labels)):
        selected = groups == group
        times, events = t[selected], d[selected]
        order = np.argsort(times, kind="stable")
        times, events = times[order], events[order]
        unique, starts = np.unique(times, return_index=True)
        deaths = np.add.reduceat(events, starts)
        risk = times.size - starts
        if legacy:
            r = np.arange(times.size, 0, -1, dtype=float)
            if method == "nelson":
                increments, variances = events / r, events / r**2
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    increments = -np.log1p(-events / r)
                    variances = events / (r * (r - 1))
            increments = np.add.reduceat(increments, starts)
            variances = np.add.reduceat(variances, starts)
        elif method == "nelson":
            increments, variances = deaths / risk, deaths / risk**2
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                increments = -np.log1p(-deaths / risk)
                variances = deaths / (risk * (risk - deaths))
        failures = deaths > 0
        failure_times = unique[failures]
        if failure_times.size <= q:
            continue
        # Include all increments after the left failure through the right failure.
        # Only failure-time increments matter except source 0/0 censoring terms.
        increment = increments[failures]
        variance = variances[failures]
        duration = failure_times[q:] - failure_times[:-q]
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            hz = _windows(increment[1:], q) / duration
            vr = _windows(variance[1:], q) / duration / duration
        midpoint = failure_times[:-q] + duration / 2
        outputs.append((midpoint, hz, vr, np.full(midpoint.size, group, dtype=np.int64)))
    if outputs:
        times_out, hazard, var, index = (np.concatenate([o[i] for o in outputs]) for i in range(4))
    else:
        times_out, hazard, var, index = (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([], dtype=np.int64),
        )
    for a in (times_out, hazard, var, index):
        a.flags.writeable = False
    return KPHazard(times_out, hazard, var, index, labels, int(q), method, bool(legacy))
