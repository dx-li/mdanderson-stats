"""Kalbfleisch–Prentice baseline survivor estimates used by SURVAN."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.special import logsumexp

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite


def _survival(log_hazard: FloatArray) -> FloatArray:
    with np.errstate(over="ignore", under="ignore"):
        return _freeze(np.exp(-np.exp(log_hazard)))


def _log_step(log_weights: FloatArray, log_nonfailure: float) -> float:
    """Log of risk-normalized cumulative-hazard increment."""
    if log_nonfailure == -np.inf:
        return np.inf
    if log_weights.size == 1:
        w = float(log_weights[0])
        if w < -30:
            return 0.0  # -log(1-r)/r = 1 + O(r).
        log_complement = float(np.log1p(-np.exp(w))) if w < -np.log(2) else log_nonfailure
        return float(np.log(-log_complement) - w)

    def residual(log_h: float) -> float:
        z = log_h + log_weights
        terms = np.full(z.shape, -log_h)
        middle = (z >= -30) & (z <= np.log(50))
        terms[middle] = log_weights[middle] - np.log(np.expm1(np.exp(z[middle])))
        large = z > np.log(50)
        with np.errstate(over="ignore"):
            terms[large] = log_weights[large] - np.exp(z[large])
        return float(logsumexp(terms) - log_nonfailure)

    lower = float(np.log(log_weights.size))
    if residual(lower) <= 1e-13:
        return lower
    step = 2.0
    for _ in range(64):
        upper = lower + step
        if residual(upper) < 0:
            return float(brentq(residual, lower, upper, xtol=1e-12, rtol=1e-14))
        step *= 2
    raise ArithmeticError("baseline survival root could not be bracketed")


@dataclass(frozen=True)
class SurvanBaseline:
    time: FloatArray
    reference_log_hazard: float
    log_hazard_increments: FloatArray
    log_cumulative_hazard: FloatArray

    def predict(self, linear_predictor: ArrayLike = 0.0) -> FloatArray:
        """Survival at all table times; output shape is predictor.shape + (times,)."""
        if np.iscomplexobj(linear_predictor):
            raise ValueError("linear_predictor must be real")
        eta = finite(linear_predictor, "linear_predictor")
        if eta.size * self.time.size > 20_000_000:
            raise ValueError("prediction output must not exceed 20000000 values")
        with np.errstate(over="ignore", invalid="ignore"):
            relative = eta - self.reference_log_hazard
            log_hazard = self.log_cumulative_hazard + relative[..., None]
        # Before any failures and after exhaustion have exact survival limits,
        # including when a finite predictor difference overflows.
        log_hazard = np.where(self.log_cumulative_hazard == -np.inf, -np.inf, log_hazard)
        log_hazard = np.where(self.log_cumulative_hazard == np.inf, np.inf, log_hazard)
        return _survival(log_hazard)

    @property
    def survival(self) -> FloatArray:
        return self.predict(0.0)

    @property
    def alpha(self) -> FloatArray:
        return _survival(self.log_hazard_increments - self.reference_log_hazard)

    @property
    def hazard_probability(self) -> FloatArray:
        """Native per-time 1-alpha, evaluated without cancellation near alpha=1."""
        with np.errstate(over="ignore", under="ignore"):
            log_step = self.log_hazard_increments - self.reference_log_hazard
            return _freeze(-np.expm1(-np.exp(log_step)))


def survan_baseline(
    time: ArrayLike, event: ArrayLike, linear_predictor: ArrayLike
) -> SurvanBaseline:
    """SURVAN baseline survival given fixed log relative-risk predictors.

    Linear predictors must use the same reference for estimation and prediction.
    For the native x=0 baseline, supply x @ beta. For a centered fit reference,
    supply fit.log_relative_hazard(x), including for future prediction profiles.
    """
    if any(np.iscomplexobj(a) for a in (time, event, linear_predictor)):
        raise ValueError("time, event and linear_predictor must be real")
    t, e, eta = (
        finite(time, "time"),
        count(event, "event"),
        finite(linear_predictor, "linear_predictor"),
    )
    if (
        t.ndim != 1
        or not 1 <= t.size <= 100_000
        or e.shape != t.shape
        or eta.shape != t.shape
        or np.any(t < 0)
        or np.any(e > 1)
    ):
        raise ValueError(
            "require matching vectors, 1..100000 records, nonnegative times and binary events"
        )
    shift = float(eta.max())
    with np.errstate(over="ignore"):
        normalized = eta - shift
    if not np.isfinite(normalized).all():
        raise ArithmeticError("linear predictor spread exceeds numerical range")
    order = np.argsort(-t, kind="stable")
    t, e, eta = t[order], e[order], normalized[order]
    starts = np.r_[0, np.flatnonzero(np.diff(t)) + 1]
    ends = np.r_[starts[1:], t.size]
    prior = -np.inf
    times, increments = [], []
    for start, end in zip(starts, ends, strict=True):
        values, dead = eta[start:end], e[start:end].astype(bool)
        current = float(logsumexp(values))
        total = float(np.logaddexp(prior, current))
        if dead.any():
            nonfailure = float(np.logaddexp(prior, logsumexp(values[~dead])))
            step = _log_step(values[dead] - total, nonfailure - total)
            times.append(float(t[start]))
            increments.append(step - total)
        prior = total
    tt, inc = np.asarray(times[::-1]), np.asarray(increments[::-1])
    if not tt.size or tt[0] > 0:
        tt, inc = np.r_[0.0, tt], np.r_[-np.inf, inc]
    cumulative = np.logaddexp.accumulate(inc)
    return SurvanBaseline(_freeze(tt), shift, _freeze(inc), _freeze(cumulative))
