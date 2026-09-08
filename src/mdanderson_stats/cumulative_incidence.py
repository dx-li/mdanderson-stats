"""Competing-risk cumulative incidence curves and Aalen variance estimates."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtri

from ._validation import FloatArray, count, finite, scalar


@dataclass(frozen=True)
class IncidenceSummary:
    """Read-only time, incidence, standard error and pointwise interval columns."""

    rows: FloatArray
    confidence: float

    @property
    def columns(self) -> tuple[str, ...]:
        percent = str(100 * self.confidence).removesuffix(".0")
        return ("time", "incidence", "std.err", f"lower {percent}% CI", f"upper {percent}% CI")

    def report(self, *, digits: int = 6) -> str:
        """Tab-separated summary using the requested significant-digit precision."""
        if (
            isinstance(digits, (bool, np.bool_))
            or not isinstance(digits, (int, np.integer))
            or not 1 <= digits <= 17
        ):
            raise ValueError("digits must be an integer from 1 to 17")
        lines = ["\t".join(self.columns)]
        lines.extend("\t".join(f"{value:.{digits}g}" for value in row) for row in self.rows)
        return "\n".join(lines) + "\n"

    def write_report(self, path: str | Path, *, digits: int = 6) -> None:
        """Write a UTF-8 tab-separated report, replacing an existing destination."""
        Path(path).write_text(self.report(digits=digits), encoding="utf-8")


@dataclass(frozen=True)
class CumulativeIncidence:
    """Step-function corners, including initial zero and maximum follow-up.

    Duplicate event times hold the left and right limits. Observations censored
    at an event time remain in that time's risk set. n_events counts individuals,
    not distinct event times. Arrays are read-only.
    """

    time: FloatArray
    estimate: FloatArray
    variance: FloatArray
    n_observations: int
    n_events: int
    event_of_interest: int
    censor: int

    def at(self, times: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Right-continuous estimates/variances at nonnegative follow-up times."""
        t = finite(times, "times")
        if np.any(t < 0):
            raise ValueError("times must be nonnegative")
        index = np.searchsorted(self.time, t, side="right") - 1
        return self.estimate[index], self.variance[index]

    def summary(
        self, times: ArrayLike | None = None, *, confidence: float = 0.95
    ) -> IncidenceSummary:
        """Pointwise normal intervals clipped to [0, 1], as in archived CUMINC.

        Omitted times preserve every step corner, including left/right limits.
        Explicit scalar or vector times use right-continuous values in the given
        order. Values after maximum follow-up follow the curve's flat extension.
        These are pointwise asymptotic intervals, not simultaneous bands.
        """
        level = scalar(confidence, "confidence")
        if not 0 < level < 1:
            raise ValueError("confidence must be strictly between 0 and 1")
        if times is None:
            t, estimate, variance = self.time, self.estimate, self.variance
        else:
            t = finite(times, "times")
            if t.ndim > 1:
                raise ValueError("summary times must be a scalar or vector")
            t = t.reshape(-1)
            estimate, variance = self.at(t)
        standard_error = np.sqrt(variance)
        # Inverting the small tail avoids rounding the upper probability to 1.
        half_width = -ndtri((1 - level) / 2) * standard_error
        rows = np.column_stack(
            [
                t,
                estimate,
                standard_error,
                np.clip(estimate - half_width, 0, 1),
                np.clip(estimate + half_width, 0, 1),
            ]
        )
        rows.flags.writeable = False
        return IncidenceSummary(rows, level)


def cumulative_incidence(
    time: ArrayLike,
    event: ArrayLike,
    *,
    event_of_interest: int = 1,
    censor: int = 0,
) -> CumulativeIncidence:
    """Estimate one cause's cumulative incidence under right censoring.

    Event codes are nonnegative integers; every code except censor denotes an
    event. Other event codes are competing causes, not censored observations.
    Missing/nonfinite values are rejected. The variance uses CUMINC's Aalen
    finite-risk-set adjustment, including its tied-event convention.
    """
    t, code = finite(time, "time"), count(event, "event")
    if t.ndim != 1 or t.size == 0 or code.shape != t.shape or np.any(t < 0):
        raise ValueError("Require nonempty matching time/event vectors and nonnegative times")
    for label, name in [(event_of_interest, "event_of_interest"), (censor, "censor")]:
        if (
            isinstance(label, (bool, np.bool_))
            or not isinstance(label, (int, np.integer))
            or not 0 <= label < 2**53
        ):
            raise ValueError(f"{name} must be a nonnegative integer smaller than 2**53")
    if event_of_interest == censor:
        raise ValueError("event_of_interest must differ from censor")
    order = np.argsort(t, kind="stable")
    t, code = t[order], code[order]
    times, starts, sizes = np.unique(t, return_index=True, return_counts=True)
    interest = np.add.reduceat((code == event_of_interest).astype(float), starts)
    failures = np.add.reduceat((code != censor).astype(float), starts)
    risk = t.size - np.r_[0, np.cumsum(sizes[:-1])]
    survival_after = np.cumprod(1 - failures / risk)
    survival_before = np.r_[1.0, survival_after[:-1]]
    increments = survival_before * interest / risk
    incidence = np.cumsum(increments)
    variance = np.zeros(times.size)
    # Center the accumulated variance polynomial at the current incidence.
    # This avoids subtracting three large raw-moment terms at every output.
    value, covariance, curvature = 0.0, 0.0, 0.0
    for i in range(times.size):
        r, d1, d2 = risk[i], interest[i], failures[i] - interest[i]
        delta = increments[i]
        base = (survival_before[i] / r) ** 2
        w1 = base * d1 * ((r - d1) / (r - 1) if d1 > 1 else 1.0)
        w2 = base * d2 * ((r - d2) / (r - 1) if d2 > 1 else 1.0)
        value += 2 * delta * covariance + delta**2 * curvature + w1
        covariance += delta * curvature
        if survival_after[i] > 0:
            covariance -= w1 / survival_after[i]
            curvature += (w1 + w2) / survival_after[i] ** 2
        variance[i] = value
    # Roundoff at a degenerate terminal estimate can be slightly negative.
    tolerance = 128 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(variance))))
    if np.any(variance < -tolerance) or not np.all(np.isfinite(variance)):
        raise RuntimeError("Cumulative-incidence variance calculation is numerically invalid")
    variance = np.maximum(variance, 0)
    selected = interest > 0
    jumps, estimates, variances = times[selected], incidence[selected], variance[selected]
    x = np.r_[0.0, np.repeat(jumps, 2), t[-1]]
    f = (
        np.r_[0.0, np.column_stack([np.r_[0.0, estimates[:-1]], estimates]).ravel(), incidence[-1]]
        if estimates.size
        else np.zeros(2)
    )
    v = (
        np.r_[0.0, np.column_stack([np.r_[0.0, variances[:-1]], variances]).ravel(), variances[-1]]
        if variances.size
        else np.zeros(2)
    )
    for array in (x, f, v):
        array.flags.writeable = False
    return CumulativeIncidence(
        x, f, v, t.size, int(interest.sum()), int(event_of_interest), int(censor)
    )
