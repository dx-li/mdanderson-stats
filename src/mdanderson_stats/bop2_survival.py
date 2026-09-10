"""Single-arm exponential/inverse-gamma BOP2 survival monitoring."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammainc, gammaincinv

from ._validation import FloatArray, count, finite, scalar
from .bayesian_monitoring import _inputs, _owned

_LOG2 = np.log(2.0)


@dataclass(frozen=True)
class BOP2SurvivalState:
    sample_size: FloatArray
    events: FloatArray
    total_observation_time: FloatArray
    posterior_shape: FloatArray
    posterior_scale_ratio: FloatArray
    success_probability: FloatArray
    decision: NDArray[np.str_]


@dataclass(frozen=True)
class BOP2SurvivalDesign:
    max_subjects: int
    null_median: float
    cutoff_scale: float
    gamma: float
    looks: NDArray[np.int64]
    prior_shape: float
    prior_scale_ratio: float
    equality_continues: bool

    def _state_inputs(
        self, events: ArrayLike, sample_size: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        d, n = np.broadcast_arrays(count(events, "events"), count(sample_size, "sample_size"))
        if np.any((d > n) | (n > self.max_subjects)):
            raise ValueError("require 0<=events<=sample_size<=max_subjects")
        return d, n

    def _cutoff(self, n: FloatArray) -> FloatArray:
        cutoff = self.cutoff_scale * (n / self.max_subjects) ** self.gamma
        if np.any((n > 0) & (cutoff == 0)):
            raise ArithmeticError("posterior cutoff underflows; increase cutoff_scale")
        return np.asarray(cutoff)

    def monitor(
        self, events: ArrayLike, total_observation_time: ArrayLike, sample_size: ArrayLike
    ) -> BOP2SurvivalState:
        d, n = self._state_inputs(events, sample_size)
        d, n, total = np.broadcast_arrays(
            d, n, finite(total_observation_time, "total_observation_time")
        )
        if np.any((total < 0) | ((n == 0) & (total != 0))):
            raise ValueError(
                "total observation time must be nonnegative and zero before enrollment"
            )
        # Divide first for scale invariance, using the alternate order only on overflow.
        with np.errstate(over="ignore", under="ignore"):
            ratio = total / self.null_median
            exposure = np.where(
                np.isfinite(ratio), ratio * _LOG2, (total * _LOG2) / self.null_median
            )
            scale = self.prior_scale_ratio + exposure
        shape = self.prior_shape + d
        probability = gammainc(shape, scale)
        if np.any(~np.isfinite(probability)):
            raise ArithmeticError("posterior gamma probability evaluation failed")
        cutoff = self._cutoff(n)
        bad = probability < cutoff if self.equality_continues else probability <= cutoff
        decision = np.full(n.shape, "continue", dtype="U24")
        scheduled = np.isin(n, self.looks)
        decision[scheduled & bad] = "stop_futility"
        decision[(n == self.max_subjects) & bad] = "final_negative"
        decision[(n == self.max_subjects) & ~bad] = "final_positive"
        decision.flags.writeable = False
        return BOP2SurvivalState(
            _owned(n),
            _owned(d),
            _owned(total),
            _owned(shape),
            _owned(scale),
            _owned(probability),
            decision,
        )

    def total_time_boundary(self, events: ArrayLike, sample_size: ArrayLike) -> FloatArray:
        """Analytic total-follow-up threshold; negative means no attainable futility stop.

        This rounded threshold is for reporting. Use monitor for actual decisions.
        """
        d, n = self._state_inputs(events, sample_size)
        if np.any(n == 0):
            raise ValueError("boundary requires positive sample_size")
        q = gammaincinv(self.prior_shape + d, self._cutoff(n))
        with np.errstate(over="ignore", under="ignore"):
            result = ((q - self.prior_scale_ratio) / _LOG2) * self.null_median
        if np.any(~np.isfinite(result)):
            raise ArithmeticError("total-time boundary is not representable in these time units")
        return _owned(result)

    def monitor_records(self, followup: ArrayLike, event_observed: ArrayLike) -> BOP2SurvivalState:
        """Observed follow-up and 0/1 censoring indicators, final axis indexing patients."""
        times, events = np.broadcast_arrays(
            finite(followup, "followup"), count(event_observed, "event_observed")
        )
        if times.ndim == 0 or np.any(times < 0) or np.any(events > 1):
            raise ValueError(
                "require a patient axis, nonnegative follow-up and 0/1 event indicators"
            )
        return self.monitor(events.sum(axis=-1), times.sum(axis=-1), times.shape[-1])


def bop2_survival_design(
    max_subjects: int,
    null_median: float,
    *,
    cutoff_scale: float,
    gamma: float,
    looks: ArrayLike | None = None,
    prior_effective_events: float = 0.05,
    prior: ArrayLike | None = None,
    equality_continues: bool = False,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2SurvivalDesign:
    """Construct a specified-parameter survival design, not an automatically calibrated one.

    Median survival m has IG(a,b) prior. Default a=1+ESS, b=ESS*null_median,
    so E[m]=null_median. Explicit prior is [shape, scale in time units].
    """
    median, scale, power = (
        scalar(null_median, "null_median"),
        scalar(cutoff_scale, "cutoff_scale"),
        scalar(gamma, "gamma"),
    )
    if median <= 0 or not 0 < scale < 1 or not 0 <= power <= 1:
        raise ValueError("require positive null_median, cutoff_scale in (0,1), gamma in [0,1]")
    if not isinstance(equality_continues, (bool, np.bool_)):
        raise ValueError("equality_continues must be boolean")
    n, _, schedule = _inputs(max_subjects, [1, 1], looks, min_subjects, cohort_size)
    if n > 200:
        raise ValueError("survival monitoring supports at most 200 subjects")
    if prior is None:
        ess = scalar(prior_effective_events, "prior_effective_events")
        if ess <= 0:
            raise ValueError("prior_effective_events must be positive")
        a, b = 1 + ess, ess
        if not np.isfinite(a) or a <= 1:
            raise ArithmeticError("effective prior event count cannot be represented accurately")
    else:
        pair = finite(prior, "prior")
        if pair.shape != (2,) or np.any(pair <= 0):
            raise ValueError("prior must contain positive inverse-gamma shape and median scale")
        a, b = float(pair[0]), float(pair[1] / median)
        if not np.isfinite(b) or b <= 0:
            raise ArithmeticError("prior scale ratio is not representable")
    return BOP2SurvivalDesign(n, median, scale, power, schedule, a, b, bool(equality_continues))
