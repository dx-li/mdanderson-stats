"""Binary efficacy BOP2-DC monitoring and exact operating characteristics."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betaincc

from ._validation import FloatArray, count, finite, scalar
from .boin import _owned


@dataclass(frozen=True)
class BOP2DCState:
    sample_size: NDArray[np.int64]
    responses: NDArray[np.int64]
    posterior_lrv: FloatArray
    posterior_cmv: FloatArray
    decision: NDArray[np.str_]


@dataclass(frozen=True)
class BOP2DCOperatingCharacteristics:
    probability: FloatArray
    looks: NDArray[np.int64]
    stop_no_go: FloatArray
    final_go: FloatArray
    final_consider: FloatArray
    final_no_go: FloatArray
    sample_size_probability: FloatArray
    expected_sample_size: FloatArray

    @property
    def no_go_probability(self) -> FloatArray:
        """Total no-go probability, including interim futility stops."""
        return self.stop_no_go.sum(axis=-1) + self.final_no_go


@dataclass(frozen=True)
class BOP2DCDesign:
    max_subjects: int
    lrv: float
    cmv: float
    lambda_lrv: float
    lambda_cmv: float
    gamma_lrv: float
    gamma_cmv: float
    prior: tuple[float, float]
    looks: NDArray[np.int64]
    no_go_lrv: NDArray[np.int64]
    no_go_cmv: NDArray[np.int64]

    def _probabilities(
        self, responses: np.ndarray, sample_size: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.prior[0] + responses, self.prior[1] + (sample_size - responses)
        return betaincc(a, b, self.lrv), betaincc(a, b, self.cmv)

    def monitor(self, responses: ArrayLike, sample_size: ArrayLike) -> BOP2DCState:
        y, n = np.broadcast_arrays(count(responses, "responses"), count(sample_size, "sample_size"))
        if np.any((y > n) | (n > self.max_subjects)):
            raise ValueError("require 0 <= responses <= sample_size <= max_subjects")
        y, n = y.astype(np.int64), n.astype(np.int64)
        pl, pc = self._probabilities(y, n)
        decision = np.full(n.shape, "continue", dtype="U16")
        for look in self.looks[:-1]:
            at = n == look
            decision[
                at
                & (pl < self.lambda_lrv * (look / self.max_subjects) ** self.gamma_lrv)
                & (pc < self.lambda_cmv * (look / self.max_subjects) ** self.gamma_cmv)
            ] = "stop_no_go"
        final = n == self.max_subjects
        go = final & (pl > self.lambda_lrv) & (pc > self.lambda_cmv)
        no_go = final & (pl < self.lambda_lrv) & (pc < self.lambda_cmv)
        decision[go] = "final_go"
        decision[no_go] = "final_no_go"
        decision[final & ~(go | no_go)] = "final_consider"
        decision.flags.writeable = False
        return BOP2DCState(
            _owned(n.astype(np.int64)), _owned(y.astype(np.int64)), _owned(pl), _owned(pc), decision
        )

    def boundaries(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        return _owned(self.no_go_lrv.astype(np.int64)), _owned(self.no_go_cmv.astype(np.int64))

    def operating_characteristics(self, probability: ArrayLike) -> BOP2DCOperatingCharacteristics:
        p = finite(probability, "probability")
        if np.any((p < 0) | (p > 1)):
            raise ValueError("probability must lie in [0,1]")
        if p.size * (self.max_subjects + 1) ** 2 > 5_000_000:
            raise ValueError("scenario batch too large; split into smaller batches")
        shape = p.shape
        surviving = np.ones((*shape, 1))
        stop = np.zeros((*shape, self.looks.size))
        for n in range(1, self.max_subjects + 1):
            arriving = np.zeros((*shape, n + 1))
            arriving[..., :-1] += surviving * (1 - p[..., None])
            arriving[..., 1:] += surviving * p[..., None]
            if n in self.looks[:-1]:
                j = int(np.flatnonzero(self.looks == n)[0])
                lr = self.lambda_lrv * (n / self.max_subjects) ** self.gamma_lrv
                lc = self.lambda_cmv * (n / self.max_subjects) ** self.gamma_cmv
                y = np.arange(n + 1)
                pl, pc = self._probabilities(y, np.full(n + 1, n))
                bad = (pl < lr) & (pc < lc)
                stop[..., j] = arriving[..., bad].sum(axis=-1)
                arriving[..., bad] = 0
            surviving = arriving
        y = np.arange(self.max_subjects + 1)
        pl, pc = self._probabilities(y, np.full(y.shape, self.max_subjects))
        go = (pl > self.lambda_lrv) & (pc > self.lambda_cmv)
        no_go = (pl < self.lambda_lrv) & (pc < self.lambda_cmv)
        consider = ~(go | no_go)
        final_go = surviving[..., go].sum(axis=-1)
        final_no_go = surviving[..., no_go].sum(axis=-1)
        final_consider = surviving[..., consider].sum(axis=-1)
        pmf = np.concatenate(
            [stop[..., :-1], (final_go + final_consider + final_no_go)[..., None]], axis=-1
        )
        expected = pmf @ np.r_[self.looks[:-1], self.max_subjects]
        return BOP2DCOperatingCharacteristics(
            _owned(p),
            self.looks,
            _owned(stop),
            _owned(final_go),
            _owned(final_consider),
            _owned(final_no_go),
            _owned(pmf),
            _owned(expected),
        )


def bop2_dc_design(
    max_subjects: int,
    lrv: float,
    cmv: float,
    *,
    lambda_lrv: float = 0.9,
    lambda_cmv: float = 0.5,
    gamma_lrv: float = 0.5,
    gamma_cmv: float = 0.5,
    prior_probability: float = 0.5,
    prior_ess: float = 1.0,
    prior: ArrayLike | None = None,
    looks: ArrayLike | None = None,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2DCDesign:
    n_value = scalar(max_subjects, "max_subjects")
    n = int(n_value)
    if n_value != n or not 1 <= n <= 1000:
        raise ValueError("max_subjects must be an integer in [1,1000]")
    r, c = scalar(lrv, "lrv"), scalar(cmv, "cmv")
    if not 0 < r < c < 1:
        raise ValueError("require 0 < lrv < cmv < 1")
    ll, lc = scalar(lambda_lrv, "lambda_lrv"), scalar(lambda_cmv, "lambda_cmv")
    gl, gc = scalar(gamma_lrv, "gamma_lrv"), scalar(gamma_cmv, "gamma_cmv")
    if not 0 < ll < 1 or not 0 < lc < 1 or not 0 <= gl <= 1 or not 0 <= gc <= 1:
        raise ValueError("cutoffs must be in (0,1) and gamma values in [0,1]")
    if prior is not None:
        shapes = finite(prior, "prior")
        if shapes.shape != (2,) or np.any(shapes <= 0) or not np.isfinite(shapes.sum()):
            raise ValueError("prior must contain two positive finite shapes")
        prior_tuple = (float(shapes[0]), float(shapes[1]))
    else:
        pp, ess = scalar(prior_probability, "prior_probability"), scalar(prior_ess, "prior_ess")
        if not 0 < pp < 1 or not 0 < ess or not np.isfinite(ess):
            raise ValueError("prior_probability must be in (0,1) and prior_ess positive")
        prior_tuple = (pp * ess, (1 - pp) * ess)
    if not all(np.isfinite(x) and x > 0 for x in prior_tuple):
        raise ValueError("prior shapes must remain positive and finite")
    if looks is None:
        first_value, step_value = (
            scalar(min_subjects, "min_subjects"),
            scalar(cohort_size, "cohort_size"),
        )
        first, step = int(first_value), int(step_value)
        if first_value != first or step_value != step:
            raise ValueError("min_subjects and cohort_size must be integers")
        if not 1 <= first <= n or step < 1:
            raise ValueError("invalid interim schedule")
        schedule = np.unique(np.r_[np.arange(first, n, step), n]).astype(np.int64)
    else:
        schedule = count(looks, "looks")
        if np.any(schedule > n):
            raise ValueError("looks cannot exceed max_subjects")
        schedule = schedule.astype(np.int64)
        if (
            schedule.ndim != 1
            or not schedule.size
            or schedule[-1] != n
            or np.any(schedule < 1)
            or np.any(np.diff(schedule) <= 0)
        ):
            raise ValueError("looks must increase and end at max_subjects")
    if ll * (schedule[0] / n) ** gl == 0 or lc * (schedule[0] / n) ** gc == 0:
        raise ArithmeticError("interim cutoff underflows; increase cutoff scales")
    no_l, no_c = [], []
    for look in schedule[:-1]:
        y = np.arange(look + 1)
        pl = betaincc(prior_tuple[0] + y, prior_tuple[1] + (look - y), r)
        pc = betaincc(prior_tuple[0] + y, prior_tuple[1] + (look - y), c)
        lr = ll * (look / n) ** gl
        lc_cut = lc * (look / n) ** gc
        no_l.append(int(np.flatnonzero(pl < lr)[-1]) if np.any(pl < lr) else -1)
        no_c.append(int(np.flatnonzero(pc < lc_cut)[-1]) if np.any(pc < lc_cut) else -1)
    return BOP2DCDesign(
        n,
        r,
        c,
        ll,
        lc,
        gl,
        gc,
        prior_tuple,
        _owned(schedule),
        _owned(np.asarray(no_l, dtype=np.int64)),
        _owned(np.asarray(no_c, dtype=np.int64)),
    )
