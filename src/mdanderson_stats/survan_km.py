"""SURVAN Kaplan–Meier tables and Simon–Lee confidence calculations."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtri

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar
from .expsurv import exploratory_survival


@dataclass(frozen=True)
class SurvanQuantiles:
    survival_probability: FloatArray
    time: FloatArray
    lower: FloatArray
    upper: FloatArray
    flag: NDArray[np.int64]


@dataclass(frozen=True)
class SurvanKM:
    time: FloatArray
    survival: FloatArray
    at_risk: FloatArray
    events: FloatArray
    censored: FloatArray
    standard_error: FloatArray
    lower: FloatArray
    upper: FloatArray
    confidence: float

    def quantiles(self, survival_probability: ArrayLike = (0.75, 0.5, 0.25)) -> SurvanQuantiles:
        """Native interpolated quantiles and Simon–Lee time-interval flags.

        Flags: 0=no interval, 1=lower only, 2=both limits, 3=re-entry makes
        interval unreliable, 4=survival never strictly crosses the probability.
        Missing values are NaN. Flag 3 retains candidate bounds for inspection.
        """
        q = finite(survival_probability, "survival_probability")
        if np.any((q <= 0) | (q >= 1)) or q.size > 1000 or q.size * self.time.size > 20_000_000:
            raise ValueError(
                "require probabilities in (0,1), <=1000 values and <=2e7 table queries"
            )
        times, low, high = (np.full(q.size, np.nan) for _ in range(3))
        flags = np.full(q.size, 4, dtype=np.int64)
        zsq = float(ndtri((1 + self.confidence) / 2)) ** 2
        after = (self.at_risk - self.events - self.censored)[:-1]
        surv = self.survival[:-1]
        for j, probability in enumerate(q.ravel()):
            index = int(np.searchsorted(-self.survival, -probability, side="right"))
            if index == self.time.size:
                continue
            if index == 0:
                # Events at zero can cross the target before the first table row.
                times[j] = self.time[0]
            else:
                left, right = self.survival[index - 1 : index + 1]
                fraction = (left - probability) / (left - right)
                times[j] = self.time[index - 1] + fraction * (
                    self.time[index] - self.time[index - 1]
                )
            flags[j] = 0
            # Algebraically the native statistic. S=0 is outside for q>0;
            # explicit handling avoids the source's 0/0 on exhausted risk sets.
            log_value = np.full(surv.shape, np.inf)
            positive = surv > 0
            with np.errstate(divide="ignore"):
                log_value[positive] = (
                    np.log(after[positive])
                    - np.log(surv[positive])
                    + 2 * np.log(np.abs(surv[positive] - probability))
                    - np.log(probability)
                    - np.log1p(-probability)
                )
            inside = log_value <= np.log(zsq)
            starts = np.flatnonzero(inside)
            if not starts.size:
                continue
            start = int(starts[0])
            low[j], flags[j] = self.time[start], 1
            ends = np.flatnonzero(~inside[start + 1 :])
            if ends.size:
                end = start + 1 + int(ends[0])
                high[j], flags[j] = self.time[end], 2
                if inside[end + 1 :].any():
                    flags[j] = 3
        shape = q.shape
        return SurvanQuantiles(
            _freeze(q),
            _freeze(times.reshape(shape)),
            _freeze(low.reshape(shape)),
            _freeze(high.reshape(shape)),
            np.frombuffer(flags.tobytes(), dtype=np.int64).reshape(shape),
        )


def survan_km(time: ArrayLike, event: ArrayLike, *, confidence: float = 0.95) -> SurvanKM:
    """KM/Greenwood table with SURVAN's risk-based Simon–Lee pointwise limits.

    Includes censor-only times and an initial zero row when needed. Events=1,
    right censors=0; exact ties are pooled with censors at risk for tied deaths.
    """
    if np.iscomplexobj(time) or np.iscomplexobj(event):
        raise ValueError("time and event must be real")
    km = exploratory_survival(time, event)
    level = scalar(confidence, "confidence")
    if not 1e-6 <= level <= 1 - 1e-12:
        raise ValueError("confidence must be in [1e-6,1-1e-12]")
    t, s, n, d, c = km.time, km.survival, km.at_risk, km.events, km.censored
    if t[0] > 0:
        t, s, n, d, c = (
            np.r_[v, a]
            for v, a in zip([0, 1, km.n_observations, 0, 0], [t, s, n, d, c], strict=True)
        )
    increment = np.divide(d, n * (n - d), out=np.zeros_like(n), where=n > d)
    se = s * np.sqrt(np.cumsum(increment))
    z = float(ndtri((1 + level) / 2))
    # Factor the discriminant and obtain the smaller root from the product
    # of roots: no subtraction of two nearly equal quadratic terms.
    denominator = n + z * z / 2 + z * np.sqrt(z * z + 4 * n * (1 - s)) / 2
    lower = n * s / denominator
    upper = s * denominator / (n + z * z * s)
    lower = np.where(s == 1, 1, lower)
    upper = np.where(s == 1, 1, np.minimum(1, upper))
    return SurvanKM(
        _freeze(t),
        _freeze(s),
        _freeze(n),
        _freeze(d),
        _freeze(c),
        _freeze(se),
        _freeze(lower),
        _freeze(upper),
        level,
    )
