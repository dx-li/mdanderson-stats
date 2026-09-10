"""Patient-level reconstruction from cleaned Kaplan–Meier coordinates."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, count, finite
from .bayesian_monitoring import _integer
from .boin import _owned


@dataclass(frozen=True)
class ReconstructedIPD:
    time: FloatArray
    event: FloatArray
    curve_time: FloatArray
    survival: FloatArray
    risk: FloatArray
    events: FloatArray
    censored: FloatArray
    risk_time: FloatArray
    reported_risk: FloatArray
    reconstructed_risk: FloatArray
    remaining: int
    requested_total_events: int | None
    rmse: float
    mean_absolute_error: float
    max_absolute_error: float


def _coordinates(time: ArrayLike, survival: ArrayLike) -> tuple[FloatArray, FloatArray]:
    t, s = finite(time, "time"), finite(survival, "survival")
    if t.ndim != 1 or s.shape != t.shape or not 2 <= t.size <= 100_000:
        raise ValueError("time and survival must be equal-length vectors of 2..100000 points")
    if np.any(t < 0) or np.any(np.diff(t) < 0) or t[-1] == 0:
        raise ValueError("time must be nonnegative, nondecreasing, and end after zero")
    if np.any((s < 0) | (s > 1)) or np.any(np.diff(s) > 0):
        raise ValueError("survival must be nonincreasing in [0, 1]")
    if t[0] != 0 or s[0] != 1:
        raise ValueError("the first coordinate must be (0, 1)")
    return t, s


def _censor_counts(t: FloatArray, low: int, high: int, number: float) -> FloatArray:
    """Bin uniformly spaced latent censors without an O(points * patients) loop."""
    if number <= 0:
        return np.zeros(high - low)
    fraction = np.arange(1, int(number) + 1, dtype=float) / (number + 1)
    times = t[low] + fraction * (t[high - 1] - t[low])
    bins = np.searchsorted(t, times, side="right") - 1
    return np.bincount(bins - low, minlength=high - low).astype(float)


def _interval(
    s: FloatArray,
    low: int,
    high: int,
    initial_risk: int,
    initial_survival: float,
    censor: FloatArray,
    *,
    trim_censor: bool,
) -> tuple[FloatArray, FloatArray, FloatArray, int, float]:
    risks, events, fitted = (np.empty(high - low) for _ in range(3))
    remaining, probability = initial_risk, initial_survival
    for offset, index in enumerate(range(low, high)):
        risks[offset] = remaining
        deaths = (
            0
            if index == 0 or probability == 0
            else int(np.rint(remaining * (1 - s[index] / probability)))
        )
        if not 0 <= deaths <= remaining:
            raise ValueError("curve and risk counts imply a negative or excessive event count")
        if remaining:
            probability *= 1 - deaths / remaining
        if censor[offset] > remaining - deaths:
            if not trim_censor:
                raise ValueError("interval censoring exceeds available patients")
            censor[offset] = remaining - deaths
        events[offset], fitted[offset] = deaths, probability
        remaining -= deaths + int(censor[offset])
    return risks, events, fitted, remaining, probability


def reconstruct_ipd(
    time: ArrayLike,
    survival: ArrayLike,
    *,
    patients: int | None = None,
    risk_time: ArrayLike | None = None,
    at_risk: ArrayLike | None = None,
    total_events: int | None = None,
) -> ReconstructedIPD:
    """Modified iterative KM reconstruction from already-cleaned coordinates.

    Supply either patients or paired risk_time/at_risk beginning at time zero.
    Risk counts refer to the start of each interval, before events at that time.
    total_events informs the final censor estimate; it is not an enforced total.
    Reconstructed records are approximate synthetic records, not recovered data.
    See docs/ipdfromkm.md for censor assumptions and source differences.
    """
    t, s = _coordinates(time, survival)
    if (risk_time is None) != (at_risk is None):
        raise ValueError("risk_time and at_risk must be supplied together")
    if risk_time is None:
        if patients is None:
            raise ValueError("patients is required when no risk table is supplied")
        rt, nr = np.array([0.0]), np.array([_integer(patients, "patients")], dtype=float)
    else:
        assert at_risk is not None
        rt, nr = finite(risk_time, "risk_time"), count(at_risk, "at_risk")
        if patients is not None:
            raise ValueError("supply patients or a risk table, not both")
    if rt.ndim != 1 or nr.shape != rt.shape or not 1 <= rt.size <= t.size:
        raise ValueError("risk times and counts must be equal-length nonempty vectors")
    if rt[0] != 0 or np.any(np.diff(rt) <= 0) or np.any(np.diff(nr) > 0):
        raise ValueError("risk times must increase from zero and counts must not increase")
    if not 1 <= nr[0] <= 1_000_000:
        raise ValueError("initial patient count must be in 1..1000000")
    requested = None if total_events is None else _integer(total_events, "total_events")
    if requested is not None and not 0 <= requested <= nr[0]:
        raise ValueError("total_events must be between zero and the initial count")
    # A reported count beyond the last curve point constrains terminal censoring.
    after = rt > t[-1]
    end_count = int(nr[-1]) if np.any(after) else 0
    rt, nr = rt[~after], nr[~after]
    starts = np.searchsorted(t, rt, side="left")
    ends = np.append(starts[1:], t.size)
    if np.any(ends <= starts):
        raise ValueError("each risk interval must contain at least one curve coordinate")
    risks, events, fitted, censored = (np.zeros(t.size) for _ in range(4))
    interval_risk = np.empty(rt.size)
    censor_estimates: list[float] = []
    remaining, probability = int(nr[0]), 1.0
    for interval, (low, high) in enumerate(zip(starts, ends, strict=True)):
        interval_risk[interval] = remaining
        final = interval == rt.size - 1
        guess: float
        if not final:
            if s[low] == 0:
                guess = remaining - int(nr[interval + 1])
            else:
                guess = int(np.rint(remaining * s[high] / s[low] - nr[interval + 1]))
        elif interval == 0:
            guess = 0 if requested is None else remaining - requested
        else:
            left_events = 0 if requested is None else max(0, requested - int(events[:low].sum()))
            mean_censor = float(np.mean(censor_estimates))
            with np.errstate(over="ignore"):
                duration_ratio = (t[-1] - rt[interval]) / (rt[interval] - rt[interval - 1])
                estimate = 0.0 if mean_censor == 0 else min(remaining, mean_censor * duration_ratio)
            # R truncates a fractional final estimate. Snap roundoff near an
            # integer so that changing time units does not lose one censor.
            nearest = float(np.rint(estimate))
            if abs(estimate - nearest) <= 16 * np.finfo(float).eps * max(1, abs(estimate)):
                estimate = nearest
            guess = max(0, min(estimate, remaining - end_count - left_events))
        guess = max(0, guess)
        visited: set[float] = set()
        while True:
            if guess in visited or len(visited) >= 10_000:
                raise ArithmeticError(
                    "censor reconstruction did not converge; review curve and risk table"
                )
            visited.add(guess)
            censor = _censor_counts(t, int(low), int(high), guess)
            r, d, km, next_risk, next_probability = _interval(
                s,
                int(low),
                int(high),
                remaining,
                probability,
                censor,
                trim_censor=final,
            )
            if final:
                break
            updated = int(censor.sum()) + next_risk - int(nr[interval + 1])
            adjust = (
                next_risk > nr[interval + 1] and updated < remaining - nr[interval + 1] + 1
            ) or (next_risk < nr[interval + 1] and updated > 0)
            if not adjust:
                guess = updated
                break
            guess = max(0, updated)
        censor_estimates.append(guess)
        risks[low:high], events[low:high], fitted[low:high], censored[low:high] = r, d, km, censor
        remaining, probability = next_risk, next_probability
    # Midpoint censors reproduce the source's records; the latent uniform times
    # above determine counts, rather than being emitted as patient times.
    midpoint = t.copy()
    midpoint[:-1] = t[:-1] + (t[1:] - t[:-1]) / 2
    if np.any((censored[:-1] > 0) & (midpoint[:-1] >= t[1:])):
        raise ArithmeticError("curve times are too close to represent midpoint censoring")
    record_time = np.concatenate(
        (
            np.repeat(t, events.astype(int)),
            np.repeat(midpoint, censored.astype(int)),
            np.repeat(t[-1], remaining),
        )
    )
    record_event = np.concatenate(
        (np.ones(int(events.sum())), np.zeros(int(censored.sum()) + remaining))
    )
    order = np.argsort(record_time, kind="stable")
    # At repeated times the right-continuous KM value is the last value there.
    fitted = fitted[np.searchsorted(t, t, side="right") - 1]
    error = fitted - s
    return ReconstructedIPD(
        _owned(record_time[order]),
        _owned(record_event[order]),
        _owned(t),
        _owned(fitted),
        _owned(risks),
        _owned(events),
        _owned(censored),
        _owned(rt),
        _owned(nr),
        _owned(interval_risk),
        remaining,
        requested,
        float(np.sqrt(np.mean(error * error))),
        float(np.mean(np.abs(error))),
        float(np.max(np.abs(error))),
    )
