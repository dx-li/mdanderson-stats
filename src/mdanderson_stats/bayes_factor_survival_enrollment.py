"""Recovered BayesFactorTTE v1.1 integer-day enrollment workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit, logit

from ._validation import FloatArray, scalar
from .bayes_factor_survival import _log_bf, _parameters, bayes_factor_survival_boundaries
from .beta_binomial import _owned
from .interaction_index_fixed_ray_study import _integer
from .multc_study import _atomic_write

_DAYS_PER_MONTH = 30.4375
_MAX_DAY = 2**31 - 1


def _integer_days(values: ArrayLike, name: str) -> np.ndarray:
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or values.size > 500:
            raise ValueError(f"{name} must be a one-dimensional tape of at most 500 days")
    elif isinstance(values, (list, tuple)):
        if len(values) > 500 or any(not np.isscalar(v) for v in values):
            raise ValueError(f"{name} must be a scalar day tape of at most 500 days")
        if any(isinstance(v, (bool, np.bool_)) for v in values):
            raise ValueError(f"{name} must contain numeric days, not booleans")
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be a finite real one-dimensional day tape")
    if np.any(raw < 0) or np.any(raw > _MAX_DAY) or np.any(raw != np.floor(raw)):
        raise ValueError(f"{name} must contain nonnegative int32 days")
    return np.array(raw, dtype=np.int64, copy=True)


def _monitor(events: int, exposure: int, median_days: float, log_scale: float) -> float:
    # Zero-day events arise from the source's truncation of positive durations.
    # The ordinary continuous-data API correctly rejects that observation pattern.
    return _log_bf(events, exposure / median_days * np.log(2), log_scale)


def _hypothesis(log_bf: float, patients: int, maximum: int, low: float, high: float) -> int:
    if log_bf > logit(high):
        return 2
    if log_bf < logit(low):
        return 0
    return 3 if patients >= maximum else 1


@dataclass(frozen=True)
class BayesFactorEnrollmentLook:
    patients: int
    calendar_day: int
    events: int
    exposure_days: int
    log_bayes_factor: float
    alternative_probability: float
    hypothesis: int


@dataclass(frozen=True)
class BayesFactorEnrollmentTrial:
    """Planned day tapes and actual native-scheduled monitoring history."""

    enrollment_days: np.ndarray
    event_duration_days: np.ndarray
    null_median_months: float
    alternative_median_months: float
    inferiority_cutoff: float
    superiority_cutoff: float
    history: tuple[BayesFactorEnrollmentLook, ...]

    @property
    def patients(self) -> int:
        return self.history[-1].patients

    @property
    def decision(self) -> str:
        return {0: "inferiority", 2: "superiority", 3: "inconclusive"}[self.history[-1].hypothesis]

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    "format_version": 1,
                    "workflow": "BayesFactorTTE-v1.1-integer-day-enrollment",
                    "enrollment_days": self.enrollment_days.tolist(),
                    "event_duration_days": self.event_duration_days.tolist(),
                    "null_median_months": self.null_median_months,
                    "alternative_median_months": self.alternative_median_months,
                    "inferiority_cutoff": self.inferiority_cutoff,
                    "superiority_cutoff": self.superiority_cutoff,
                    "patients": self.patients,
                    "decision": self.decision,
                    "history": [look.__dict__ for look in self.history],
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )

    def write_json(self, path: str | Path) -> Path:
        return _atomic_write(path, self.to_json())


def bayes_factor_survival_enrollment_trial(
    enrollment_days: ArrayLike,
    event_duration_days: ArrayLike,
    *,
    null_median_months: float,
    alternative_median_months: float,
    inferiority_cutoff: float = 0.15,
    superiority_cutoff: float = 0.8,
) -> BayesFactorEnrollmentTrial:
    """Replay recovered v1.1 monitoring, distinct from explicit-calendar replay.

    Arrivals may tie and event durations may be zero. Checks occur just after
    enrollment of patients 2 through N, in input order, with no extra final
    follow-up. Only earlier patients contribute, and an event tied with the
    check remains censored for that look. Months are exactly 30.4375 days.
    Supports 2..500 patients and nonnegative int32 event calendar dates.
    """
    median, scale, low, high = _parameters(
        null_median_months, alternative_median_months, inferiority_cutoff, superiority_cutoff
    )
    arrivals = _integer_days(enrollment_days, "enrollment_days")
    durations = _integer_days(event_duration_days, "event_duration_days")
    if arrivals.size < 2 or arrivals.size > 500 or durations.shape != arrivals.shape:
        raise ValueError("require matching tapes of 2..500 patients")
    if arrivals[0] != 0 or np.any(np.diff(arrivals) < 0):
        raise ValueError("enrollment_days must start at zero and be nondecreasing")
    calendar_events = arrivals + durations
    if np.any(calendar_events > _MAX_DAY):
        raise ValueError("event calendar dates exceed int32 days")
    history = []
    median_days = median * _DAYS_PER_MONTH
    if not np.isfinite(median_days):
        raise ValueError("median day conversion must be finite")
    for i in range(1, arrivals.size):
        now = int(arrivals[i])
        occurred = calendar_events[:i] < now
        events = int(occurred.sum())
        times = np.where(occurred, durations[:i], now - arrivals[:i])
        exposure = int(times.sum())
        if exposure > _MAX_DAY:
            raise ArithmeticError("native total time on test exceeds int32 days")
        log_bf = _monitor(events, exposure, median_days, scale)
        hypothesis = _hypothesis(log_bf, i + 1, arrivals.size, low, high)
        history.append(
            BayesFactorEnrollmentLook(
                i + 1, now, events, exposure, log_bf, float(expit(log_bf)), hypothesis
            )
        )
        if hypothesis != 1:
            break
    return BayesFactorEnrollmentTrial(
        _owned(arrivals),
        _owned(durations),
        median,
        float(alternative_median_months),
        low,
        high,
        tuple(history),
    )


@dataclass(frozen=True)
class BayesFactorEnrollmentStudy:
    seed: int
    accrual_rate_per_month: float
    true_median_months: float
    trial_seeds: tuple[int, ...]
    trials: tuple[BayesFactorEnrollmentTrial, ...]

    @property
    def stopping_probability(self) -> FloatArray:
        """Inferiority, superiority, inconclusive in that order; full precision."""
        return _owned(
            np.array(
                [
                    np.mean([trial.decision == decision for trial in self.trials])
                    for decision in ("inferiority", "superiority", "inconclusive")
                ]
            )
        )

    @property
    def mean_patients(self) -> float:
        return float(np.mean([trial.patients for trial in self.trials]))

    def patient_quantile(self, probability: float) -> int:
        """Original sorted-count index floor(probability*repetitions), no interpolation."""
        p = scalar(probability, "probability")
        if not 0 <= p < 1:
            raise ValueError("probability must lie in [0,1)")
        ordered = sorted(trial.patients for trial in self.trials)
        return ordered[int(p * len(ordered))]

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    "format_version": 1,
                    "workflow": "BayesFactorTTE-v1.1-integer-day-enrollment",
                    "seed": self.seed,
                    "accrual_rate_per_month": self.accrual_rate_per_month,
                    "true_median_months": self.true_median_months,
                    "trial_seeds": self.trial_seeds,
                    "stopping_probability": self.stopping_probability.tolist(),
                    "mean_patients": self.mean_patients,
                    "patient_count_quantiles": {
                        str(p): self.patient_quantile(p) for p in (0.05, 0.1, 0.5, 0.9, 0.95)
                    },
                    "trials": [json.loads(trial.to_json()) for trial in self.trials],
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )

    def write_json(self, path: str | Path) -> Path:
        return _atomic_write(path, self.to_json())


def simulate_bayes_factor_survival_enrollment(
    *,
    null_median_months: float,
    alternative_median_months: float,
    true_median_months: float,
    accrual_rate_per_month: float,
    max_patients: int = 50,
    repetitions: int = 100,
    seed: int = 12345,
    inferiority_cutoff: float = 0.15,
    superiority_cutoff: float = 0.8,
    max_total_work: int = 1_000_000,
    max_total_quadratures: int = 10_000,
    max_storage_bytes: int = 96_000_000,
) -> BayesFactorEnrollmentStudy:
    """Bounded native-scheduled serial study with PCG64 rather than native RNG.

    Positive exponential event durations and interarrival gaps are truncated
    individually to integer days. The first arrival is zero; tied arrivals and
    zero-day events are retained. Each trial records a replayable child seed and
    both full planned tapes. Limits bound worst-case ledgers/evaluations/storage
    before any RNG. Native int32 overflow raises instead of wrapping.
    """
    _parameters(
        null_median_months, alternative_median_months, inferiority_cutoff, superiority_cutoff
    )
    n = _integer(max_patients, "max_patients", 500)
    if n < 2:
        raise ValueError("max_patients must be at least two")
    reps = _integer(repetitions, "repetitions", 10_000)
    raw_seed = np.asarray(seed)
    if raw_seed.ndim or raw_seed.dtype.kind not in "iu" or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0,2**64)")
    work = _integer(max_total_work, "max_total_work", 100_000_000)
    quadratures = _integer(max_total_quadratures, "max_total_quadratures", 100_000)
    storage = _integer(max_storage_bytes, "max_storage_bytes", 128 * 1024 * 1024)
    if reps * n * (n - 1) // 2 > work:
        raise ValueError("study exceeds max_total_work")
    if reps * (n - 1) > quadratures:
        raise ValueError("study exceeds max_total_quadratures")
    if reps * n * 512 + 64 * 1024 * 1024 > storage:
        raise ValueError("study exceeds max_storage_bytes")
    median = scalar(true_median_months, "true_median_months")
    rate = scalar(accrual_rate_per_month, "accrual_rate_per_month")
    if median <= 0 or rate <= 0:
        raise ValueError("true median and accrual rate must be positive")
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        means = np.array([median / np.log(2) * _DAYS_PER_MONTH, _DAYS_PER_MONTH / rate])
    if not np.isfinite(means).all() or np.any(means <= 0):
        raise ValueError("day-scale means must be finite and positive")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    children = tuple(
        int(v) for v in generator.integers(0, 2**64 - 1, size=reps, dtype=np.uint64, endpoint=True)
    )
    trials = []
    for child in children:
        stream = np.random.Generator(np.random.PCG64(child))
        generated = stream.exponential(means, size=(n, 2))
        if not np.isfinite(generated).all() or np.any(generated > _MAX_DAY):
            raise ArithmeticError("generated times exceed finite int32 days")
        days = generated.astype(np.int64)
        arrivals = np.r_[0, np.cumsum(days[:-1, 1])]
        trials.append(
            bayes_factor_survival_enrollment_trial(
                arrivals,
                days[:, 0],
                null_median_months=null_median_months,
                alternative_median_months=alternative_median_months,
                inferiority_cutoff=inferiority_cutoff,
                superiority_cutoff=superiority_cutoff,
            )
        )
    return BayesFactorEnrollmentStudy(int(seed), rate, median, children, tuple(trials))


def bayes_factor_survival_day_boundaries(
    *,
    max_patients: int,
    null_median_months: float,
    alternative_median_months: float,
    inferiority_cutoff: float = 0.15,
    superiority_cutoff: float = 0.8,
) -> tuple[FloatArray, FloatArray]:
    """Uncapped integer-day thresholds for event counts 0..max_patients-1.

    Inferiority holds for integer exposure < the first noninferiority day;
    superiority holds for exposure >= the first superiority day. Disabled sides
    have -inf/+inf respectively. Resolves source discretization without its
    finite search horizon and silently incomplete boundary lists.
    """
    n = _integer(max_patients, "max_patients", 500)
    roots = bayes_factor_survival_boundaries(
        np.arange(n),
        null_median=null_median_months * _DAYS_PER_MONTH,
        alternative_median_mode=alternative_median_months * _DAYS_PER_MONTH,
        inferiority_cutoff=inferiority_cutoff,
        superiority_cutoff=superiority_cutoff,
    )
    if np.any(np.isfinite(roots.inferiority_time) & (roots.inferiority_time >= _MAX_DAY)) or np.any(
        np.isfinite(roots.superiority_time) & (roots.superiority_time >= _MAX_DAY)
    ):
        raise ArithmeticError("integer boundary exceeds the supported int32 day range")
    low = np.where(roots.inferiority_time == -np.inf, 0, np.ceil(roots.inferiority_time))
    if inferiority_cutoff == 0:
        low[:] = -np.inf
    high = np.where(roots.superiority_time == -np.inf, 0, np.floor(roots.superiority_time) + 1)
    return _owned(low), _owned(high)
