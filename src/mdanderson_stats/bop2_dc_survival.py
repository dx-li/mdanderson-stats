"""Time-to-event BOP2-DC monitoring with an inverse-gamma prior."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammainc

from ._validation import FloatArray, count, finite, scalar
from .boin import _owned

_LOG2 = np.log(2.0)


@dataclass(frozen=True)
class BOP2DCSurvivalState:
    sample_size: NDArray[np.int64]
    events: NDArray[np.int64]
    total_time: FloatArray
    posterior_shape: FloatArray
    posterior_scale: FloatArray
    posterior_median_scale: FloatArray
    posterior_lrv: FloatArray
    posterior_cmv: FloatArray
    decision: NDArray[np.str_]


@dataclass(frozen=True)
class BOP2DCSurvivalDesign:
    max_subjects: int
    lrv: float
    cmv: float
    lambda_lrv: float
    lambda_cmv: float
    gamma_lrv: float
    gamma_cmv: float
    prior_shape: float
    prior_scale: float
    looks: NDArray[np.int64]

    def _posterior(self, events: np.ndarray, total_time: np.ndarray) -> tuple[np.ndarray, ...]:
        shape = self.prior_shape + events
        with np.errstate(over="ignore", invalid="ignore"):
            scale = self.prior_scale + total_time
            anchor = np.maximum(self.prior_scale, total_time)
            normalized_scale = self.prior_scale / anchor + total_time / anchor
            ratio_lrv = (anchor / self.lrv) * (normalized_scale * _LOG2)
            ratio_cmv = (anchor / self.cmv) * (normalized_scale * _LOG2)
            ratio_lrv = np.where(
                np.isfinite(ratio_lrv), ratio_lrv, (anchor * _LOG2) / self.lrv * normalized_scale
            )
            ratio_cmv = np.where(
                np.isfinite(ratio_cmv), ratio_cmv, (anchor * _LOG2) / self.cmv * normalized_scale
            )
            median_scale = (anchor * _LOG2) * normalized_scale
        if (
            np.any(~np.isfinite(shape))
            or np.any((ratio_lrv <= 0) | ~np.isfinite(ratio_lrv))
            or np.any((ratio_cmv <= 0) | ~np.isfinite(ratio_cmv))
        ):
            raise ArithmeticError("posterior inverse-gamma scale is not representable")
        return (
            shape,
            scale,
            median_scale,
            gammainc(shape, ratio_lrv),
            gammainc(shape, ratio_cmv),
        )

    def monitor(
        self, sample_size: ArrayLike, events: ArrayLike, total_time: ArrayLike
    ) -> BOP2DCSurvivalState:
        n, d, exposure = np.broadcast_arrays(
            count(sample_size, "sample_size"),
            count(events, "events"),
            finite(total_time, "total_time"),
        )
        if n.size > 100_000:
            raise ValueError("count batch exceeds 100000 scenarios")
        if np.any((d > n) | (n > self.max_subjects)):
            raise ValueError("require 0<=events<=sample_size<=max_subjects")
        if np.any((exposure < 0) | ((n == 0) & (exposure != 0))):
            raise ValueError("total_time must be nonnegative and zero before enrollment")
        n, d = n.astype(np.int64), d.astype(np.int64)
        shape, scale, median_scale, posterior_lrv, posterior_cmv = self._posterior(d, exposure)
        decision = np.full(n.shape, "continue", dtype="U24")
        for look in self.looks[:-1]:
            at = n == look
            bad = (
                posterior_lrv < self.lambda_lrv * (look / self.max_subjects) ** self.gamma_lrv
            ) & (posterior_cmv < self.lambda_cmv * (look / self.max_subjects) ** self.gamma_cmv)
            decision[at & bad] = "stop_no_go"
        final = n == self.max_subjects
        go = final & (posterior_lrv > self.lambda_lrv) & (posterior_cmv > self.lambda_cmv)
        no_go = final & (posterior_lrv < self.lambda_lrv) & (posterior_cmv < self.lambda_cmv)
        decision[go] = "final_go"
        decision[no_go] = "final_no_go"
        decision[final & ~(go | no_go)] = "final_consider"
        decision.flags.writeable = False
        return BOP2DCSurvivalState(
            _owned(n),
            _owned(d),
            _owned(exposure),
            _owned(shape),
            _owned(scale),
            _owned(median_scale),
            _owned(posterior_lrv),
            _owned(posterior_cmv),
            decision,
        )


def bop2_dc_survival_design(
    max_subjects: int,
    lrv: float,
    cmv: float,
    *,
    lambda_lrv: float = 0.9,
    lambda_cmv: float = 0.5,
    gamma_lrv: float = 0.5,
    gamma_cmv: float = 0.5,
    prior_shape: float = 1e-6,
    prior_scale: float = 1e-6,
    looks: ArrayLike | None = None,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2DCSurvivalDesign:
    n_value = scalar(max_subjects, "max_subjects")
    n = int(n_value)
    if n_value != n or not 1 <= n <= 1000:
        raise ValueError("max_subjects must be an integer in [1,1000]")
    lower, clinical = scalar(lrv, "lrv"), scalar(cmv, "cmv")
    if not 0 < lower < clinical or not np.isfinite(clinical):
        raise ValueError("require 0 < lrv < cmv")
    ll, lc = scalar(lambda_lrv, "lambda_lrv"), scalar(lambda_cmv, "lambda_cmv")
    gl, gc = scalar(gamma_lrv, "gamma_lrv"), scalar(gamma_cmv, "gamma_cmv")
    if not 0 < ll < 1 or not 0 < lc < 1 or not 0 <= gl <= 1 or not 0 <= gc <= 1:
        raise ValueError("cutoffs must be in (0,1) and gamma values in [0,1]")
    a, b = scalar(prior_shape, "prior_shape"), scalar(prior_scale, "prior_scale")
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        raise ValueError("prior_shape and prior_scale must be positive and finite")
    if looks is None:
        first_value, step_value = (
            scalar(min_subjects, "min_subjects"),
            scalar(cohort_size, "cohort_size"),
        )
        first, step = int(first_value), int(step_value)
        if first_value != first or step_value != step or not 1 <= first <= n or step < 1:
            raise ValueError("invalid interim schedule")
        schedule = np.unique(np.r_[np.arange(first, n, step), n]).astype(np.int64)
    else:
        raw = count(looks, "looks")
        if raw.ndim != 1 or not raw.size or np.any(raw < 1) or np.any(raw > n):
            raise ValueError("looks must be nonempty and lie in [1,max_subjects]")
        schedule = raw.astype(np.int64)
        if schedule[-1] != n or np.any(np.diff(schedule) <= 0):
            raise ValueError("looks must increase and end at max_subjects")
    if ll * (schedule[0] / n) ** gl == 0 or lc * (schedule[0] / n) ** gc == 0:
        raise ArithmeticError("interim cutoff underflows; increase cutoff scales")
    return BOP2DCSurvivalDesign(n, lower, clinical, ll, lc, gl, gc, a, b, _owned(schedule))
