"""Survival curves and inverse-survival queries for EXPSURV exploration."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, count, finite


@dataclass(frozen=True)
class ExploratorySurvival:
    time: FloatArray
    survival: FloatArray
    at_risk: FloatArray
    events: FloatArray
    censored: FloatArray
    step_time: FloatArray
    step_survival: FloatArray
    n_observations: int
    legacy: bool

    def at(self, time: ArrayLike) -> FloatArray:
        """Right-continuous survival, including flat extension beyond follow-up."""
        query = finite(time, "time")
        index = np.searchsorted(self.time, query, side="right") - 1
        result = np.where(index < 0, 1.0, self.survival[np.maximum(index, 0)])
        result.flags.writeable = False
        return result

    def survival_quantile(
        self, level: ArrayLike, *, method: Literal["step", "source"] = "step"
    ) -> FloatArray:
        """Time at a survival level q (not cumulative probability 1-q).

        Step mode returns the first time S(t)<=q, with q=1 at time zero.
        Source mode reproduces EXPSURV QUANT's linear interpolation and returns
        the last plateau time at an exact level. Unreached levels return NaN.
        """
        q = finite(level, "level")
        if np.any((q < 0) | (q > 1)):
            raise ValueError("Survival levels must be between zero and one")
        if method not in ("step", "source"):
            raise ValueError("method must be step or source")
        upper = np.searchsorted(-self.survival, -q, side="left")
        reached = upper < self.time.size
        upper = np.minimum(upper, self.time.size - 1)
        if method == "step":
            result = np.where(q == 1, 0, np.where(reached, self.time[upper], np.nan))
        else:
            lower = np.searchsorted(-self.survival, -q, side="right") - 1
            t0 = np.where(lower < 0, 0, self.time[np.maximum(lower, 0)])
            s0 = np.where(lower < 0, 1, self.survival[np.maximum(lower, 0)])
            t1, s1 = self.time[upper], self.survival[upper]
            fraction = np.divide(s0 - q, s0 - s1, out=np.zeros_like(q), where=s0 != s1)
            result = np.where(reached, t0 + (t1 - t0) * fraction, np.nan)
        result.flags.writeable = False
        return result


def exploratory_survival(
    time: ArrayLike, status: ArrayLike | None = None, *, legacy: bool = False
) -> ExploratorySurvival:
    """Fit the EXPSURV Kaplan–Meier curve and exact plotting corners.

    Stable sorting keeps inputs aligned. The default groups tied observations,
    with tied censoring still at risk for tied failures. Legacy mode retains
    KMEST's per-record product and input tie order. It requires sorted input in
    the original program; Python performs that preparation explicitly.
    """
    t = finite(time, "time")
    if t.ndim != 1 or t.size == 0 or np.any(t < 0):
        raise ValueError("time must be a nonempty nonnegative vector")
    d = np.ones(t.size) if status is None else count(status, "status")
    if d.shape != t.shape or np.any(d > 1):
        raise ValueError("status must match time and contain zero or one")
    if not isinstance(legacy, (bool, np.bool_)):
        raise ValueError("legacy must be boolean")
    order = np.argsort(t, kind="stable")
    t, d = t[order], d[order]
    n = t.size
    if legacy:
        risk = np.arange(n, 0, -1, dtype=float)
        deaths, censored = d, 1 - d
    else:
        t, starts, totals = np.unique(t, return_index=True, return_counts=True)
        risk = (n - starts).astype(float)
        deaths = np.add.reduceat(d, starts)
        censored = totals - deaths
    survival = np.cumprod(1 - deaths / risk)
    before = np.r_[1.0, survival[:-1]]
    step_time = np.r_[0.0, np.repeat(t, 2)]
    step_survival = np.r_[1.0, np.column_stack((before, survival)).ravel()]
    for array in (t, risk, deaths, censored, survival, step_time, step_survival):
        array.flags.writeable = False
    return ExploratorySurvival(
        t, survival, risk, deaths, censored, step_time, step_survival, n, bool(legacy)
    )
