"""Bounded complete-outcome replay and serial simulation for general Multc99.

The recovered Multc99 source's noncommercial terms apply to adapted workflows;
see notices/mdanderson-multc99-readme.txt. All enrolled outcomes are retained,
including the cap patient omitted by the legacy simulator.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray
from .multc99 import Multc99Design, Multc99State, _immutable, _integer, _real

_MAX_TRIALS = 10_000
_MAX_WORK = 20_000_000


@dataclass(frozen=True)
class Multc99Trial:
    sample_size: int
    elementary_counts: NDArray[np.int64]
    history: tuple[Multc99State, ...]
    look_sizes: NDArray[np.int64]

    @property
    def final_state(self) -> Multc99State:
        return self.history[-1]


@dataclass(frozen=True)
class Multc99Simulation:
    design: Multc99Design
    elementary_probabilities: FloatArray
    trials: int
    seed: int
    sample_sizes: NDArray[np.int64]
    elementary_counts: NDArray[np.int64]
    lower_hits: NDArray[np.bool_]
    upper_hits: NDArray[np.bool_]
    early_stop_probability: float
    early_stop_mcse: float
    mean_sample_size: float
    sample_size_mcse: float
    sample_size_quantiles: NDArray[np.int64]
    hit_patterns: tuple[tuple[tuple[int, ...], int], ...]
    monitoring_period: float | None
    accrual_rate: float | None
    response_window: float


def _looks(design: Multc99Design, sizes: ArrayLike | None) -> NDArray[np.int64]:
    if sizes is None:
        return np.arange(design.cohort_size, design.max_subjects, design.cohort_size)
    if isinstance(sizes, (list, tuple)) and len(sizes) > design.max_subjects:
        raise ValueError("look_sizes exceeds max_subjects")
    raw = np.asarray(sizes)
    if raw.ndim != 1 or raw.size > design.max_subjects:
        raise ValueError("look_sizes must be a bounded, strictly increasing integer vector")
    if raw.size == 0:
        return np.empty(0, dtype=np.int64)
    if raw.dtype.kind not in "iu" or np.any(raw < 1) or np.any(raw >= design.max_subjects):
        raise ValueError("interim look_sizes must lie in [1, max_subjects-1]")
    result = np.asarray(raw, dtype=np.int64)
    if np.any(np.diff(result) <= 0):
        raise ValueError("look_sizes must be strictly increasing")
    return result


def run_multc99_trial(
    design: Multc99Design,
    elementary_outcomes: ArrayLike,
    *,
    look_sizes: ArrayLike | None = None,
) -> Multc99Trial:
    """Replay a full potential-outcome tape through the first stop or the cap.

    Outcomes are zero-based elementary-category indices, one per potential
    participant. Only enrolled outcomes enter the returned counts. Monitoring
    defaults to complete cohorts strictly before the cap, with no prior stop.
    Caller-supplied looks permit a recovered period-count monitoring schedule.
    """
    if not isinstance(design, Multc99Design):
        raise TypeError("design must be a Multc99Design")
    if (
        isinstance(elementary_outcomes, (list, tuple))
        and len(elementary_outcomes) != design.max_subjects
    ):
        raise ValueError("elementary_outcomes must contain max_subjects potential outcomes")
    raw = np.asarray(elementary_outcomes)
    if raw.shape != (design.max_subjects,) or raw.dtype.kind not in "iu":
        raise ValueError("elementary_outcomes must contain max_subjects integer category indices")
    categories = design.experimental_prior.size
    if np.any(raw < 0) or np.any(raw >= categories):
        raise ValueError("elementary outcomes must be valid zero-based category indices")
    outcomes = np.asarray(raw, dtype=np.int64)
    looks = _looks(design, look_sizes)
    counts = np.zeros(categories, dtype=np.int64)
    history = [design.monitor_counts(counts)]
    previous = 0
    for n in (*looks, design.max_subjects):
        n = int(n)
        counts += np.bincount(outcomes[previous:n], minlength=categories)
        state = design.monitor_counts(counts)
        history.append(state)
        previous = n
        if state.decision != "continue":
            break
    return Multc99Trial(previous, _immutable(counts), tuple(history), _immutable(looks))


def _positive_poisson(rng: np.random.Generator, mean: float) -> int:
    if mean >= 20:
        for _ in range(100):
            value = int(rng.poisson(mean))
            if value > 0:
                return value
        raise ArithmeticError("positive Poisson sampling failed")
    probability = mean / float(np.expm1(mean))
    cumulative = probability
    draw = float(rng.random())
    k = 1
    while draw > cumulative:
        k += 1
        if k > 512:
            raise ArithmeticError("positive Poisson sampling exceeded its work bound")
        probability *= mean / k
        cumulative += probability
    return k


def _period_looks(
    maximum: int,
    rate: float,
    period: float,
    window: float,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Native period-count law; geometrically skip empty periods.

    No patient-level arrival/follow-up times are implied by this schedule.
    Empty-period durations are not an advertised native numeric output.
    """
    mean = rate * period
    ratio = window / period
    first_period = max(1, int(np.ceil(ratio)))
    first_mean = mean * (first_period - ratio)
    count = int(rng.poisson(first_mean)) if first_mean else 0
    looks: list[int] = []
    if 0 < count < maximum:
        looks.append(count)
    # Native code samples ordinary Poisson increments until nonzero. This is
    # the identical positive-increment distribution without an unbounded loop.
    while count < maximum:
        count += _positive_poisson(rng, mean)
        if count < maximum:
            looks.append(count)
    return np.array(looks, dtype=np.int64)


def simulate_multc99(
    design: Multc99Design,
    elementary_probabilities: ArrayLike,
    *,
    trials: int = 10_000,
    seed: int = 0,
    monitoring_period: float | None = None,
    accrual_rate: float | None = None,
    response_window: float = 0.0,
) -> Multc99Simulation:
    """Serial complete-outcome operating characteristics for a single scenario.

    A full categorical potential tape is drawn before the schedule for each
    replicate. Period-count monitoring requires both period and accrual rate;
    its response window shifts the first observation period as in the source.
    Every enrolled subject has one mutually exclusive category. Sampling is
    through NumPy and does not reproduce the native random stream.
    """
    if not isinstance(design, Multc99Design):
        raise TypeError("design must be a Multc99Design")
    repetitions = _integer(trials, "trials", 1, _MAX_TRIALS)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    raw = np.asarray(elementary_probabilities)
    if raw.shape != design.experimental_prior.shape or raw.dtype.kind not in "iuf":
        raise ValueError("elementary_probabilities must match the prior's categories")
    probabilities = np.asarray(raw, dtype=float)
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or abs(float(probabilities.sum()) - 1) > 1e-12
    ):
        raise ValueError("elementary_probabilities must be nonnegative and sum to one")
    probabilities = probabilities / probabilities.sum()
    window = _real(response_window, "response_window", 0, np.finfo(float).max)
    if (monitoring_period is None) != (accrual_rate is None):
        raise ValueError("period monitoring requires both monitoring_period and accrual_rate")
    period = rate = None
    if monitoring_period is not None and accrual_rate is not None:
        period = _real(
            monitoring_period, "monitoring_period", np.finfo(float).tiny, np.finfo(float).max
        )
        rate = _real(accrual_rate, "accrual_rate", np.finfo(float).tiny, np.finfo(float).max)
        mean = rate * period
        if not 1e-6 <= mean <= 1e6 or not window / period <= 1e9:
            raise ValueError("require 1e-6 <= rate*period <= 1e6 and window/period <= 1e9")
    elif window != 0:
        raise ValueError("response_window requires period-count monitoring")
    work = repetitions * design.max_subjects * (design.experimental_prior.size + len(design.events))
    if work > _MAX_WORK:
        raise ValueError("simulation exceeds the 20,000,000-work budget")
    rng = np.random.default_rng(int(seed))
    sizes = np.empty(repetitions, dtype=np.int64)
    counts = np.empty((repetitions, probabilities.size), dtype=np.int64)
    lower = np.empty((repetitions, len(design.events)), dtype=bool)
    upper = np.empty_like(lower)
    for j in range(repetitions):
        tape = rng.choice(probabilities.size, size=design.max_subjects, p=probabilities)
        looks = (
            None
            if period is None or rate is None
            else _period_looks(
                design.max_subjects,
                rate,
                period,
                window,
                rng,
            )
        )
        trial = run_multc99_trial(design, tape, look_sizes=looks)
        sizes[j] = trial.sample_size
        counts[j] = trial.elementary_counts
        lower[j] = trial.final_state.lower_hits
        upper[j] = trial.final_state.upper_hits
    p_stop = float(np.mean(sizes < design.max_subjects))
    patterns = Counter(
        tuple(
            int(lower_hit) + 2 * int(upper_hit) for lower_hit, upper_hit in zip(lo, hi, strict=True)
        )
        for lo, hi in zip(lower, upper, strict=True)
    )
    quantile_indices = np.minimum(
        (repetitions * np.array([0.1, 0.25, 0.5, 0.75, 0.9])).astype(int),
        repetitions - 1,
    )
    return Multc99Simulation(
        design,
        _freeze(probabilities),
        repetitions,
        int(seed),
        _immutable(sizes),
        _immutable(counts),
        _immutable(lower),
        _immutable(upper),
        p_stop,
        float(np.sqrt(p_stop * (1 - p_stop) / repetitions)),
        float(np.mean(sizes)),
        float(np.std(sizes, ddof=1) / np.sqrt(repetitions)) if repetitions > 1 else float("nan"),
        _immutable(np.sort(sizes)[quantile_indices]),
        tuple(sorted(patterns.items())),
        period,
        rate,
        window,
    )
