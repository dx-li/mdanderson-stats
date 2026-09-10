"""BTOX, BEMPO and BEMPR: Bayesian binary monitoring with exact design probabilities."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betainc, betaincc

from ._validation import FloatArray, count, finite, scalar
from .beta_binomial import BetaBinomialPosterior


def _owned(value: ArrayLike) -> FloatArray:
    x = np.array(value, dtype=float, copy=True)
    x.flags.writeable = False
    return x


def _prob(value: float, name: str) -> float:
    p = scalar(value, name)
    if not 0 <= p <= 1:
        raise ValueError(f"{name} must lie in [0,1]")
    return p


def _integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _inputs(
    max_subjects: int,
    prior: ArrayLike,
    looks: ArrayLike | None,
    min_subjects: int,
    cohort_size: int,
) -> tuple[int, tuple[float, float], NDArray[np.int64]]:
    n = _integer(max_subjects, "max_subjects")
    if not 1 <= n <= 1000:
        raise ValueError("max_subjects must be in [1,1000]")
    p = finite(prior, "prior")
    if p.shape != (2,) or np.any(p <= 0) or not np.isfinite(p.sum()):
        raise ValueError("prior must contain two positive shapes with a finite sum")
    if looks is None:
        first, step = _integer(min_subjects, "min_subjects"), _integer(cohort_size, "cohort_size")
        if not 1 <= first <= n or step < 1:
            raise ValueError("require 1 <= min_subjects <= max_subjects and cohort_size >= 1")
        schedule = np.unique(np.r_[np.arange(first, n, step), n]).astype(np.int64)
    else:
        schedule = count(looks, "looks").astype(np.int64)
        if schedule.ndim != 1 or not schedule.size or schedule[-1] != n:
            raise ValueError("looks must be a nonempty vector ending at max_subjects")
        if np.any(schedule < 1) or np.any(np.diff(schedule) <= 0):
            raise ValueError("looks must increase strictly")
    schedule.flags.writeable = False
    return n, (float(p[0]), float(p[1])), schedule


def _tail_table(n: int, prior: tuple[float, float], rate: float, *, upper: bool) -> FloatArray:
    table = np.full((n + 1, n + 1), np.nan)
    for size in range(n + 1):
        r = np.arange(size + 1)
        a, b = prior[0] + r, prior[1] + size - r
        table[size, : size + 1] = betaincc(a, b, rate) if upper else betainc(a, b, rate)
    if np.any(~np.isfinite(table[np.tril_indices(n + 1)])):
        raise ArithmeticError("posterior beta probability evaluation failed")
    return table


@dataclass(frozen=True)
class MonitoringState:
    sample_size: FloatArray
    events: FloatArray
    posterior: BetaBinomialPosterior
    low_probability: FloatArray
    high_probability: FloatArray
    final_probability: FloatArray
    decision: NDArray[np.str_]


@dataclass(frozen=True)
class MonitoringOperatingCharacteristics:
    probability: FloatArray
    looks: NDArray[np.int64]
    stop_low: FloatArray
    stop_high: FloatArray
    complete_positive: FloatArray
    complete_negative: FloatArray
    sample_size_probability: FloatArray
    expected_sample_size: FloatArray
    sample_size_sd: FloatArray
    expected_events: FloatArray
    expected_observed_rate: FloatArray

    @property
    def positive_conclusion(self) -> FloatArray:
        """Efficacy for BEMPO/BEMPR; excessive toxicity for BTOX."""
        return self.stop_high.sum(axis=-1) + self.complete_positive

    def sample_size_quantile(self, probability: float) -> FloatArray:
        p = _prob(probability, "probability")
        cumulative = np.cumsum(self.sample_size_probability, axis=-1)
        cumulative[..., -1] = 1
        if p == 0:
            index = np.argmax(self.sample_size_probability > 0, axis=-1)
        elif p == 1:
            index = (
                self.looks.size
                - 1
                - np.argmax(self.sample_size_probability[..., ::-1] > 0, axis=-1)
            )
        else:
            index = np.argmax(cumulative >= p, axis=-1)
        return np.asarray(self.looks[index], dtype=float)


@dataclass(frozen=True)
class BayesianMonitoringDesign:
    """Use the three design factories; final analysis supersedes early rules.

    Low denotes futility. High denotes efficacy or excessive toxicity according
    to method. Tables include every attainable (sample size, event count) pair;
    entries above the triangular support are NaN. Boundaries apply at looks only.
    """

    method: str
    max_subjects: int
    prior: tuple[float, float]
    looks: NDArray[np.int64]
    futility_max: NDArray[np.int64]
    positive_min: NDArray[np.int64]
    final_positive_min: int
    low_probability: FloatArray
    high_probability: FloatArray
    final_probability: FloatArray

    def monitor(self, events: ArrayLike, sample_size: ArrayLike) -> MonitoringState:
        r, n = np.broadcast_arrays(count(events, "events"), count(sample_size, "sample_size"))
        if np.any((r > n) | (n > self.max_subjects)):
            raise ValueError("require 0 <= events <= sample_size <= max_subjects")
        rr, nn = r.astype(np.int64), n.astype(np.int64)
        decision = np.full(n.shape, "continue", dtype="U24")
        for index, look in enumerate(self.looks[:-1]):
            decision[(nn == look) & (rr <= self.futility_max[index])] = "stop_futility"
            decision[(nn == look) & (rr >= self.positive_min[index])] = (
                "stop_toxicity" if self.method == "toxicity" else "stop_efficacy"
            )
        decision[(nn == self.max_subjects) & (rr >= self.final_positive_min)] = "final_positive"
        decision[(nn == self.max_subjects) & (rr < self.final_positive_min)] = "final_negative"
        decision.flags.writeable = False
        return MonitoringState(
            _owned(n),
            _owned(r),
            BetaBinomialPosterior(*self.prior).update(r, n - r),
            _owned(self.low_probability[nn, rr]),
            _owned(self.high_probability[nn, rr]),
            _owned(self.final_probability[nn, rr]),
            decision,
        )

    def monitor_outcomes(self, outcomes: ArrayLike) -> MonitoringState:
        """Full observed path, with the first stopping decision remaining in force."""
        y = count(outcomes, "outcomes")
        if y.ndim == 0 or np.any(y > 1) or y.shape[-1] > self.max_subjects:
            raise ValueError("outcomes require a final patient axis of zeros and ones, at most N")
        state = self.monitor(np.cumsum(y, axis=-1), np.arange(1, y.shape[-1] + 1))
        decisions = state.decision.copy()
        for j in range(1, y.shape[-1]):
            stopped = decisions[..., j - 1] != "continue"
            decisions[..., j] = np.where(stopped, decisions[..., j - 1], decisions[..., j])
        decisions.flags.writeable = False
        return MonitoringState(
            state.sample_size,
            state.events,
            state.posterior,
            state.low_probability,
            state.high_probability,
            state.final_probability,
            decisions,
        )

    def operating_characteristics(
        self, probability: ArrayLike
    ) -> MonitoringOperatingCharacteristics:
        p = finite(probability, "probability")
        if np.any((p < 0) | (p > 1)):
            raise ValueError("probability must lie in [0,1]")
        low = np.zeros((*p.shape, self.looks.size))
        high = np.zeros_like(low)
        surviving = np.ones((*p.shape, 1))
        index = 0
        expected_events, observed_rate = np.zeros_like(p), np.zeros_like(p)
        for n in range(1, self.max_subjects + 1):
            arriving = np.zeros((*p.shape, n + 1))
            arriving[..., :-1] = surviving * (1 - p[..., None])
            arriving[..., 1:] += surviving * p[..., None]
            if n == self.looks[index] and n < self.max_subjects:
                boundary_low, boundary_high = self.futility_max[index], self.positive_min[index]
                low[..., index] = arriving[..., : boundary_low + 1].sum(axis=-1)
                high[..., index] = arriving[..., boundary_high:].sum(axis=-1)
                stopped_events = arriving[..., : boundary_low + 1] @ np.arange(
                    boundary_low + 1
                ) + arriving[..., boundary_high:] @ np.arange(boundary_high, n + 1)
                expected_events += stopped_events
                observed_rate += stopped_events / n
                arriving[..., : boundary_low + 1] = 0
                arriving[..., boundary_high:] = 0
                index += 1
            surviving = arriving
        final_events = surviving @ np.arange(self.max_subjects + 1)
        expected_events += final_events
        observed_rate += final_events / self.max_subjects
        positive = surviving[..., self.final_positive_min :].sum(axis=-1)
        negative = surviving[..., : self.final_positive_min].sum(axis=-1)
        sample_pmf = low + high
        sample_pmf[..., -1] += positive + negative
        expected = sample_pmf @ self.looks
        sd = np.sqrt(np.sum(sample_pmf * (self.looks - expected[..., None]) ** 2, axis=-1))
        return MonitoringOperatingCharacteristics(
            _owned(p),
            self.looks,
            _owned(low),
            _owned(high),
            _owned(positive),
            _owned(negative),
            _owned(sample_pmf),
            _owned(expected),
            _owned(sd),
            _owned(expected_events),
            _owned(observed_rate),
        )


def _design(
    method: str,
    n: int,
    prior: tuple[float, float],
    looks: NDArray[np.int64],
    low: FloatArray,
    high: FloatArray,
    final: FloatArray,
    low_stop: NDArray[np.bool_],
    high_stop: NDArray[np.bool_],
    final_cutoff: float,
) -> BayesianMonitoringDesign:
    lo, hi = np.full(looks.size, -1, dtype=np.int64), looks.copy() + 1
    for i, size in enumerate(looks):
        lows, highs = (
            np.flatnonzero(low_stop[size, : size + 1]),
            np.flatnonzero(high_stop[size, : size + 1]),
        )
        if lows.size:
            lo[i] = lows[-1]
        if highs.size:
            hi[i] = highs[0]
        if size < n and lo[i] >= hi[i]:
            raise ValueError(
                "futility and positive early-stopping rules overlap at a scheduled look"
            )
    final_events = np.flatnonzero(final[n, : n + 1] >= final_cutoff)
    final_min = int(final_events[0]) if final_events.size else n + 1
    for value in (lo, hi):
        value.flags.writeable = False
    return BayesianMonitoringDesign(
        method, n, prior, looks, lo, hi, final_min, _owned(low), _owned(high), _owned(final)
    )


def posterior_efficacy_design(
    max_subjects: int,
    *,
    prior: ArrayLike = (0.5, 0.5),
    looks: ArrayLike | None = None,
    min_subjects: int = 5,
    cohort_size: int = 5,
    futility_rate: float = 0.3,
    futility_probability: float = 0.7,
    efficacy_rate: float = 0.3,
    efficacy_probability: float = 0.9,
    final_rate: float = 0.3,
    final_probability: float = 0.8,
    stop_futility: bool = True,
    stop_efficacy: bool = True,
) -> BayesianMonitoringDesign:
    """BEMPO: lower tail > futility cutoff; upper tail >= efficacy cutoffs."""
    if not isinstance(stop_futility, bool) or not isinstance(stop_efficacy, bool):
        raise ValueError("stopping switches must be boolean")
    n, prior_pair, schedule = _inputs(max_subjects, prior, looks, min_subjects, cohort_size)
    f, e, z = (
        _prob(v, name)
        for v, name in (
            (futility_probability, "futility_probability"),
            (efficacy_probability, "efficacy_probability"),
            (final_probability, "final_probability"),
        )
    )
    low = _tail_table(n, prior_pair, _prob(futility_rate, "futility_rate"), upper=False)
    high = _tail_table(n, prior_pair, _prob(efficacy_rate, "efficacy_rate"), upper=True)
    final = _tail_table(n, prior_pair, _prob(final_rate, "final_rate"), upper=True)
    return _design(
        "posterior",
        n,
        prior_pair,
        schedule,
        low,
        high,
        final,
        (low > f) & stop_futility,
        (high >= e) & stop_efficacy,
        z,
    )


def predictive_efficacy_design(
    max_subjects: int,
    *,
    prior: ArrayLike = (0.5, 0.5),
    looks: ArrayLike | None = None,
    min_subjects: int = 5,
    cohort_size: int = 5,
    target_rate: float = 0.3,
    final_probability: float = 0.7,
    predictive_lower: float = 0.3,
    predictive_upper: float = 0.9,
    stop_futility: bool = True,
    stop_efficacy: bool = True,
) -> BayesianMonitoringDesign:
    """BEMPR: predictive probability < lower or >= upper; exact backward recursion."""
    if not isinstance(stop_futility, bool) or not isinstance(stop_efficacy, bool):
        raise ValueError("stopping switches must be boolean")
    n, prior_pair, schedule = _inputs(max_subjects, prior, looks, min_subjects, cohort_size)
    lower, upper = (
        _prob(predictive_lower, "predictive_lower"),
        _prob(predictive_upper, "predictive_upper"),
    )
    if lower > upper:
        raise ValueError("predictive_lower must not exceed predictive_upper")
    cutoff = _prob(final_probability, "final_probability")
    final = _tail_table(n, prior_pair, _prob(target_rate, "target_rate"), upper=True)
    prediction = np.full_like(final, np.nan)
    prediction[n] = (final[n] >= cutoff).astype(float)
    a, b = prior_pair
    for size in range(n - 1, -1, -1):
        r = np.arange(size + 1)
        success = (a + r) / (a + b + size)
        failure = (b + size - r) / (a + b + size)
        prediction[size, : size + 1] = (
            success * prediction[size + 1, 1 : size + 2]
            + failure * prediction[size + 1, : size + 1]
        )
        both = (prediction[size + 1, 1 : size + 2] == 1) & (prediction[size + 1, : size + 1] == 1)
        prediction[size, : size + 1][both] = 1
    return _design(
        "predictive",
        n,
        prior_pair,
        schedule,
        prediction,
        prediction,
        final,
        (prediction < lower) & stop_futility,
        (prediction >= upper) & stop_efficacy,
        cutoff,
    )


def toxicity_monitoring_design(
    max_subjects: int,
    *,
    prior: ArrayLike = (0.5, 0.5),
    looks: ArrayLike | None = None,
    min_subjects: int = 5,
    cohort_size: int = 5,
    toxicity_rate: float = 0.3,
    toxicity_probability: float = 0.7,
) -> BayesianMonitoringDesign:
    """BTOX: stop when the posterior upper toxicity tail is >= the cutoff."""
    n, prior_pair, schedule = _inputs(max_subjects, prior, looks, min_subjects, cohort_size)
    cutoff = _prob(toxicity_probability, "toxicity_probability")
    high = _tail_table(n, prior_pair, _prob(toxicity_rate, "toxicity_rate"), upper=True)
    low = np.where(np.isnan(high), np.nan, 0.0)
    return _design(
        "toxicity",
        n,
        prior_pair,
        schedule,
        low,
        high,
        high,
        np.zeros_like(high, dtype=bool),
        high >= cutoff,
        cutoff,
    )
