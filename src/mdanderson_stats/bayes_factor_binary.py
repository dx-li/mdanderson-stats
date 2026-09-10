"""Single-arm binary trial monitoring using a one-sided nonlocal iMOM prior."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, logit

from ._validation import FloatArray, count, finite
from .bayesian_monitoring import _integer, _prob
from .beta_binomial import _owned
from .imom_binary import _imom_log_bayes_table


def _readonly(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class BayesFactorBinaryState:
    events: FloatArray
    sample_size: FloatArray
    log_bayes_factor: FloatArray
    alternative_probability: FloatArray
    decision: NDArray[np.str_]


@dataclass(frozen=True)
class BayesFactorBinaryOC:
    probability: FloatArray
    looks: NDArray[np.int64]
    stop_inferiority: FloatArray
    stop_superiority: FloatArray
    inconclusive: FloatArray
    sample_size_probability: FloatArray
    expected_sample_size: FloatArray
    sample_size_sd: FloatArray

    @property
    def superiority(self) -> FloatArray:
        return self.stop_superiority.sum(axis=-1)

    @property
    def inferiority(self) -> FloatArray:
        return self.stop_inferiority.sum(axis=-1)

    def sample_size_quantile(self, probability: float) -> FloatArray:
        p = _prob(probability, "probability")
        cumulative = np.cumsum(self.sample_size_probability, axis=-1)
        cumulative[..., -1] = 1
        if p == 0:
            index = np.argmax(self.sample_size_probability > 0, axis=-1)
        elif p == 1:
            index = (
                len(self.looks)
                - 1
                - np.argmax(self.sample_size_probability[..., ::-1] > 0, axis=-1)
            )
        else:
            index = np.argmax(cumulative >= p, axis=-1)
        return _owned(self.looks[index])


@dataclass(frozen=True)
class BayesFactorBinarySimulation:
    probability: FloatArray
    sample_size: NDArray[np.int64]
    events: NDArray[np.int64]
    decision: NDArray[np.str_]
    decision_probability: FloatArray
    monte_carlo_se: FloatArray
    expected_sample_size: FloatArray


@dataclass(frozen=True)
class BayesFactorBinaryDesign:
    """Use bayes_factor_binary_design; probabilities assume equal prior model odds."""

    max_subjects: int
    null_rate: float
    alternative_mode: float
    inferiority_cutoff: float
    superiority_cutoff: float
    looks: NDArray[np.int64]
    inferiority_max: NDArray[np.int64]
    superiority_min: NDArray[np.int64]
    log_bayes_factor: FloatArray

    def monitor(self, events: ArrayLike, sample_size: ArrayLike) -> BayesFactorBinaryState:
        m, n = np.broadcast_arrays(count(events, "events"), count(sample_size, "sample_size"))
        if np.any((m > n) | (n > self.max_subjects)):
            raise ValueError("require 0 <= events <= sample_size <= max_subjects")
        mm, nn = m.astype(np.int64), n.astype(np.int64)
        decision = np.full(n.shape, "continue", dtype="U16")
        decision[nn == self.max_subjects] = "inconclusive"
        for i, look in enumerate(self.looks):
            decision[(nn == look) & (mm <= self.inferiority_max[i])] = "inferiority"
            decision[(nn == look) & (mm >= self.superiority_min[i])] = "superiority"
        log_bf = self.log_bayes_factor[nn, mm]
        return BayesFactorBinaryState(
            _owned(m), _owned(n), _owned(log_bf), _owned(expit(log_bf)), _readonly(decision)
        )

    def monitor_outcomes(self, outcomes: ArrayLike) -> BayesFactorBinaryState:
        y = count(outcomes, "outcomes")
        if y.ndim == 0 or y.shape[-1] > self.max_subjects or np.any(y > 1):
            raise ValueError("outcomes require a final patient axis of zeros and ones, at most N")
        state = self.monitor(np.cumsum(y, axis=-1), np.arange(1, y.shape[-1] + 1))
        decision = state.decision.copy()
        for i in range(1, y.shape[-1]):
            decision[..., i] = np.where(
                decision[..., i - 1] != "continue", decision[..., i - 1], decision[..., i]
            )
        return BayesFactorBinaryState(
            state.events,
            state.sample_size,
            state.log_bayes_factor,
            state.alternative_probability,
            _readonly(decision),
        )

    def operating_characteristics(self, probability: ArrayLike) -> BayesFactorBinaryOC:
        p = finite(probability, "probability")
        if np.any((p < 0) | (p > 1)):
            raise ValueError("probability must lie in [0,1]")
        low = np.zeros((*p.shape, len(self.looks)))
        high = np.zeros_like(low)
        surviving = np.ones((*p.shape, 1))
        index = 0
        for n in range(1, self.max_subjects + 1):
            arriving = np.zeros((*p.shape, n + 1))
            arriving[..., :-1] = surviving * (1 - p[..., None])
            arriving[..., 1:] += surviving * p[..., None]
            if n == self.looks[index]:
                lo, hi = self.inferiority_max[index], self.superiority_min[index]
                low[..., index] = arriving[..., : lo + 1].sum(axis=-1)
                high[..., index] = arriving[..., hi:].sum(axis=-1)
                arriving[..., : lo + 1] = 0
                arriving[..., hi:] = 0
                index += 1
            surviving = arriving
        inconclusive = surviving.sum(axis=-1)
        mass = low + high
        mass[..., -1] += inconclusive
        if np.any(np.abs(mass.sum(axis=-1) - 1) > 1e-12):
            raise ArithmeticError("Bayes-factor trial probabilities do not sum to one")
        mean = mass @ self.looks
        sd = np.sqrt(np.sum(mass * (self.looks - mean[..., None]) ** 2, axis=-1))
        return BayesFactorBinaryOC(
            _owned(p),
            self.looks,
            _owned(low),
            _owned(high),
            _owned(inconclusive),
            _owned(mass),
            _owned(mean),
            _owned(sd),
        )

    def simulate(
        self,
        probability: ArrayLike,
        *,
        n_trials: int = 10000,
        rng: np.random.Generator | int | None = None,
    ) -> BayesFactorBinarySimulation:
        p = finite(probability, "probability")
        trials = _integer(n_trials, "n_trials")
        if np.any((p < 0) | (p > 1)) or not 1 <= trials <= 100000 or p.size * trials > 10000000:
            raise ValueError(
                "require probabilities in [0,1], 1..100000 trials and at most 10M paths"
            )
        generator = np.random.default_rng(rng)
        shape = (*p.shape, trials)
        events = np.zeros(shape, dtype=np.int64)
        size = np.full(shape, self.max_subjects, dtype=np.int64)
        decision = np.full(shape, "continue", dtype="U16")
        previous = 0
        for i, look in enumerate(self.looks):
            alive = decision == "continue"
            new = generator.binomial(int(look - previous), p[..., None], size=shape)
            events += new * alive
            low = alive & (events <= self.inferiority_max[i])
            high = alive & (events >= self.superiority_min[i])
            decision[low], decision[high] = "inferiority", "superiority"
            size[low | high] = look
            previous = int(look)
        decision[decision == "continue"] = "inconclusive"
        probabilities = np.stack(
            [
                (decision == name).mean(axis=-1)
                for name in ("inferiority", "superiority", "inconclusive")
            ],
            axis=-1,
        )
        return BayesFactorBinarySimulation(
            _owned(p),
            _readonly(size),
            _readonly(events),
            _readonly(decision),
            _owned(probabilities),
            _owned(np.sqrt(probabilities * (1 - probabilities) / trials)),
            _owned(size.mean(axis=-1)),
        )


def bayes_factor_binary_design(
    max_subjects: int,
    *,
    null_rate: float = 0.2,
    alternative_mode: float = 0.3,
    inferiority_cutoff: float = 0.1,
    superiority_cutoff: float = 0.9,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BayesFactorBinaryDesign:
    """Guide's k=1, nu=2 iMOM design, with strict cutoffs and three final decisions.

    The alternative is a normalized prior on (null_rate,1), parameterized by
    its mode; it is not a point alternative. Final unresolved trials are inconclusive.
    """
    n, first, step = (
        _integer(v, name)
        for v, name in [
            (max_subjects, "max_subjects"),
            (min_subjects, "min_subjects"),
            (cohort_size, "cohort_size"),
        ]
    )
    if (
        not 1 <= n <= 400
        or not 1 <= first <= n
        or not 1 <= step <= n
        or n % step
        or (first > step and first % step)
    ):
        raise ValueError(
            "require N in 1..400, 1<=minimum<=N, whole cohorts and minimum aligned to cohorts"
        )
    p0, mode = _prob(null_rate, "null_rate"), _prob(alternative_mode, "alternative_mode")
    low, high = (
        _prob(inferiority_cutoff, "inferiority_cutoff"),
        _prob(superiority_cutoff, "superiority_cutoff"),
    )
    if not 0 < p0 < mode < 1 or not low < high:
        raise ValueError(
            "require 0<null_rate<alternative_mode<1 and inferiority_cutoff<superiority_cutoff"
        )
    looks = np.arange(max(first, step), n + 1, step, dtype=np.int64)
    table = _imom_log_bayes_table(n, p0, mode)
    lower, upper = [], []
    for size in looks:
        row = table[size, : size + 1]
        if np.any(np.diff(row) < -1e-10):
            raise ArithmeticError("iMOM Bayes factor failed monotonicity check")
        lo, hi = np.flatnonzero(row < logit(low)), np.flatnonzero(row > logit(high))
        lower.append(int(lo[-1]) if len(lo) else -1)
        upper.append(int(hi[0]) if len(hi) else int(size + 1))
    return BayesFactorBinaryDesign(
        n, p0, mode, low, high, _readonly(looks), _readonly(lower), _readonly(upper), table
    )
