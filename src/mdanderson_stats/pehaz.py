"""Piecewise-exponential hazard estimation from right-censored observations."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite, scalar


@dataclass(frozen=True)
class PiecewiseHazard:
    cuts: FloatArray
    hazard: FloatArray
    events: NDArray[np.int64]
    at_risk: NDArray[np.int64]
    follow_up: FloatArray
    width: float
    bounds: tuple[float, float]
    legacy: bool

    def report(self, *, digits: int = 6) -> str:
        """Tab-separated bin estimates and sufficient statistics; NA=no exposure."""
        if (
            isinstance(digits, (bool, np.bool_))
            or not isinstance(digits, (int, np.integer))
            or not 1 <= digits <= 17
        ):
            raise ValueError("digits must be an integer from 1 to 17")
        lines = [
            f"Bin width: {self.width:.{digits}g}",
            "left\tright\thazard\tevents\tat_risk\tfollow_up",
        ]
        for i, h in enumerate(self.hazard):
            value = "NA" if np.isnan(h) else format(h, f".{digits}g")
            lines.append(
                "\t".join(
                    [
                        format(self.cuts[i], f".{digits}g"),
                        format(self.cuts[i + 1], f".{digits}g"),
                        value,
                        str(self.events[i]),
                        str(self.at_risk[i]),
                        format(self.follow_up[i], f".{digits}g"),
                    ]
                )
            )
        return "\n".join(lines) + "\n"

    def write_report(self, path: str | Path, *, digits: int = 6) -> None:
        Path(path).write_text(self.report(digits=digits), encoding="utf-8")


def pehaz(
    times: ArrayLike,
    delta: ArrayLike | None = None,
    *,
    width: float | None = None,
    bounds: tuple[float, float] | None = None,
    legacy: bool = False,
) -> PiecewiseHazard:
    """Estimate each bin's constant hazard as events divided by person-time.

    Bins are left-closed/right-open, except the final bin includes its right
    endpoint. The final cut is truncated to bounds[1]. legacy=True instead uses
    equal-width bins that can overshoot bounds[1], with every right endpoint
    excluded, as in the archived S function. At-risk counts include observations
    ending exactly at the left endpoint. No exposure and no events gives NaN;
    positive events with zero exposure gives +inf (unbounded likelihood).
    """
    t = finite(times, "times")
    if t.ndim != 1 or t.size == 0 or np.any(t < 0):
        raise ValueError("times must be a nonempty nonnegative vector")
    status = np.ones(t.size) if delta is None else count(delta, "delta")
    if status.shape != t.shape or np.any(status > 1):
        raise ValueError("delta must match times and contain only 0 or 1")
    if not isinstance(legacy, (bool, np.bool_)):
        raise ValueError("legacy must be boolean")
    interval = finite((0, float(t.max())) if bounds is None else bounds, "bounds")
    if interval.shape != (2,) or not 0 <= interval[0] < interval[1]:
        raise ValueError("bounds must be two increasing nonnegative values")
    left, right = map(float, interval)
    if width is None:
        events = int(status.sum())
        if events == 0:
            raise ValueError("width is required when there are no events")
        w = (right - left) / (8 * events**0.2)
    else:
        w = scalar(width, "width")
    if w <= 0 or not np.isfinite(w):
        raise ValueError("width must be positive and finite")
    ratio = (right - left) / w
    if not np.isfinite(ratio) or ratio > np.iinfo(np.intp).max - 1:
        raise ValueError("width produces an unrepresentable number of bins")
    n = max(1, int(np.ceil(ratio)))
    cuts = left + np.arange(n + 1) * w
    if not legacy:
        cuts[-1] = right
    if not np.all(np.isfinite(cuts)) or np.any(np.diff(cuts) <= 0):
        raise ValueError("width cannot be represented at these time bounds")
    order = np.argsort(t, kind="stable")
    t, status = t[order], status[order].astype(np.int64)
    starts = np.searchsorted(t, cuts[:-1], side="left")
    ends = np.searchsorted(t, cuts[1:], side="left")
    if not legacy:
        ends[-1] = np.searchsorted(t, cuts[-1], side="right")
    cumulative = np.r_[0, np.cumsum(status)]
    events_per_bin = cumulative[ends] - cumulative[starts]
    risk = t.size - starts
    # Ending observations belong to disjoint bins: the total sliced work is O(N).
    # Sum local time differences, avoiding subtraction of large cumulative sums.
    exposure = np.array(
        [
            np.sum(t[a:b] - lo) + (t.size - b) * (hi - lo)
            for lo, hi, a, b in zip(cuts[:-1], cuts[1:], starts, ends, strict=True)
        ]
    )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            hazard = np.divide(events_per_bin, exposure, out=np.full(n, np.nan), where=exposure > 0)
    except FloatingPointError as exc:
        raise RuntimeError("Hazard estimate exceeds numerical range") from exc
    hazard[(exposure == 0) & (events_per_bin > 0)] = np.inf
    if not np.all(np.isfinite(exposure)):
        raise RuntimeError("Total person-time exceeds numerical range")
    for array in [cuts, hazard, events_per_bin, risk, exposure]:
        array.flags.writeable = False
    return PiecewiseHazard(
        cuts, hazard, events_per_bin, risk, exposure, w, (left, right), bool(legacy)
    )
