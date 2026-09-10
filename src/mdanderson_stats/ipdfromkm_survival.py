"""KM confidence, cumulative hazard and quantile summaries for IPDfromKM."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtri

from ._validation import FloatArray, finite, scalar
from .boin import _owned
from .expsurv import ExploratorySurvival, exploratory_survival


@dataclass(frozen=True)
class IPDSurvivalQuantiles:
    survival_probability: FloatArray
    time: FloatArray
    lower: FloatArray
    upper: FloatArray


@dataclass(frozen=True)
class IPDSurvivalSummary:
    time: FloatArray
    survival: FloatArray
    standard_error: FloatArray
    lower: FloatArray
    upper: FloatArray
    cumulative_hazard: FloatArray
    hazard_standard_error: FloatArray


def _inverse(time: FloatArray, survival: FloatArray, level: FloatArray, tol: float) -> FloatArray:
    """Invert the step curve, taking the midpoint of a matching flat segment."""
    p = 1 - level
    cdf = np.r_[0.0, 1 - survival]
    t = np.r_[0.0, time]
    valid = np.isfinite(cdf)
    y, first = np.unique(cdf[valid], return_index=True)
    x = t[valid][first]
    result = np.full(level.shape, np.nan)
    if len(y) < 2 or not p.size or y[-1] < np.min(p):
        return result
    lo = np.searchsorted(y + tol, p, side="left")
    hi = np.searchsorted(y - tol, p, side="left")
    supported = (p >= y[0] + tol) & (p <= y[-1] - tol)
    left, right = x[np.minimum(lo, len(x) - 1)], x[np.minimum(hi, len(x) - 1)]
    result = np.where(supported, left + (right - left) / 2, np.nan)
    if np.isfinite(survival[-1]):
        terminal = np.abs(p - y[-1]) < tol
        result = np.where(terminal, left + (time[-1] - left) / 2, result)
    return result


@dataclass(frozen=True)
class IPDSurvivalCurve:
    km: ExploratorySurvival
    estimates: IPDSurvivalSummary
    confidence: float

    def at(self, time: ArrayLike) -> IPDSurvivalSummary:
        """Right-continuous summaries at landmarks within observed follow-up."""
        query = finite(time, "time")
        if np.any((query < 0) | (query > self.km.time[-1])):
            raise ValueError("landmark times must be within observed follow-up")
        index = np.searchsorted(self.km.time, query, side="right") - 1
        bounded = np.maximum(index, 0)

        def take(values: FloatArray, initial: float) -> FloatArray:
            return _owned(np.where(index < 0, initial, values[bounded]))

        e = self.estimates
        return IPDSurvivalSummary(
            _owned(query),
            take(e.survival, 1),
            take(e.standard_error, 0),
            take(e.lower, 1),
            take(e.upper, 1),
            take(e.cumulative_hazard, 0),
            take(e.hazard_standard_error, 0),
        )

    def quantiles(
        self,
        survival_probability: ArrayLike = (0.75, 0.5, 0.25),
        *,
        tolerance: float = np.sqrt(np.finfo(float).eps),
    ) -> IPDSurvivalQuantiles:
        """Times and inverted confidence limits at survival probabilities in (0,1).

        Match R's midpoint convention on plateaus; unreachable values are NaN.
        As in native survreport, both limits are NaN when the point is unreached.
        """
        q = finite(survival_probability, "survival_probability")
        tol = scalar(tolerance, "tolerance")
        if np.any((q <= 0) | (q >= 1)) or not 0 < tol < 0.01:
            raise ValueError("survival probabilities must be in (0,1); tolerance in (0,.01)")
        time = _inverse(self.km.time, self.estimates.survival, q, tol)
        lower = _inverse(self.km.time, self.estimates.lower, q, tol)
        upper = _inverse(self.km.time, self.estimates.upper, q, tol)
        return IPDSurvivalQuantiles(
            _owned(q),
            _owned(time),
            _owned(np.where(np.isnan(time), np.nan, lower)),
            _owned(np.where(np.isnan(time), np.nan, upper)),
        )


def ipd_survival_summary(
    time: ArrayLike, event: ArrayLike, *, confidence: float = 0.95
) -> IPDSurvivalCurve:
    """KM with Greenwood log confidence limits and Nelson–Aalen cumulative hazard.

    Uses R survfit's default log confidence transform. Confidence limits and KM
    standard errors at zero survival are NaN, matching the native summary.
    """
    km = exploratory_survival(time, event)
    level = scalar(confidence, "confidence")
    if not 1e-6 <= level <= 1 - 1e-12:
        raise ValueError("confidence must be in [1e-6, 1-1e-12]")
    n, d = km.at_risk, km.events
    greenwood = np.divide(d, n * (n - d), out=np.full_like(n, np.inf), where=n > d)
    log_se = np.sqrt(np.cumsum(greenwood))
    positive = km.survival > 0
    se = np.full_like(n, np.nan)
    se[positive] = km.survival[positive] * log_se[positive]
    z = float(ndtri((1 + level) / 2))
    lower, upper = np.full_like(n, np.nan), np.full_like(n, np.nan)
    log_s = np.log(km.survival[positive])
    lower[positive] = np.exp(log_s - z * log_se[positive])
    upper[positive] = np.exp(np.minimum(0, log_s + z * log_se[positive]))
    hazard, hazard_se = np.cumsum(d / n), np.sqrt(np.cumsum(d / (n * n)))
    estimates = IPDSurvivalSummary(
        km.time,
        km.survival,
        _owned(se),
        _owned(lower),
        _owned(upper),
        _owned(hazard),
        _owned(hazard_se),
    )
    return IPDSurvivalCurve(km, estimates, level)
