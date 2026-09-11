"""Two-arm block adaptive randomization with patient-wise posterior stopping."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite, scalar
from .beta_binomial import BetaBinomialPosterior
from .beta_comparison import compare_beta_binomial


def _integer(value: int, name: str, low: int, high: int) -> int:
    x = scalar(value, name)
    if x != np.floor(x) or not low <= x <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(x)


@dataclass(frozen=True)
class BlockArandDesign:
    """Jeffreys priors from the shipped config, with explicit library stopping defaults."""

    min_block: int = 2
    max_block: int = 8
    max_patients: int = 200
    burnin: int = 50
    early_cutoff: float = 0.95
    final_cutoff: float = 0.9
    tuning: float = 0.5
    prior: tuple[tuple[float, float], tuple[float, float]] = ((0.5, 0.5), (0.5, 0.5))
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        for name, low, high in (
            ("min_block", 2, 10),
            ("max_block", 2, 10),
            ("max_patients", 1, 10000),
            ("burnin", 0, 10000),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, low, high))
        if self.min_block > self.max_block or self.burnin > self.max_patients:
            raise ValueError("require min_block <= max_block and burnin <= max_patients")
        for name in ("early_cutoff", "final_cutoff", "tuning", "tolerance"):
            object.__setattr__(self, name, scalar(getattr(self, name), name))
        if not 0 <= self.early_cutoff <= 1 or not 0 <= self.final_cutoff <= 1:
            raise ValueError("decision cutoffs must be in [0, 1]")
        if self.tuning < 0 or not 1e-12 <= self.tolerance <= 1e-3:
            raise ValueError("tuning must be nonnegative and tolerance in [1e-12, 1e-3]")
        prior = finite(self.prior, "prior")
        if prior.shape != (2, 2):
            raise ValueError("prior must have two rows of (alpha, beta)")
        BetaBinomialPosterior(prior[:, 0], prior[:, 1])
        object.__setattr__(self, "prior", tuple(tuple(float(v) for v in row) for row in prior))


@dataclass(frozen=True)
class BlockArandPlan:
    target_probability: float
    arm_zero_count: int
    size: int
    allocation_probability: float


@lru_cache(maxsize=45)
def _candidates(minimum: int, maximum: int) -> tuple[FloatArray, FloatArray]:
    pairs = np.array([(k, n) for n in range(minimum, maximum + 1) for k in range(1, n)])
    return _freeze(pairs), _freeze(pairs[:, 0] / pairs[:, 1])


def blockarand_plan(
    probability: float, *, min_block: int = 2, max_block: int = 8
) -> BlockArandPlan:
    """Closest k/n with 1 <= k < n and min_block <= n <= max_block.

    Floating-point ties retain the first candidate: smallest denominator, then
    numerator. This is the strict-improvement scan used by the archived library.
    """
    p = scalar(probability, "probability")
    minimum, maximum = (
        _integer(min_block, "min_block", 2, 10),
        _integer(max_block, "max_block", 2, 10),
    )
    if not 0 <= p <= 1 or minimum > maximum:
        raise ValueError("require probability in [0,1] and min_block <= max_block")
    pairs, fractions = _candidates(minimum, maximum)
    i = int(np.argmin(np.abs(fractions - p)))
    k, n = (int(v) for v in pairs[i])
    return BlockArandPlan(p, k, n, float(fractions[i]))


def blockarand_block(plan: BlockArandPlan, *, rng: np.random.Generator) -> FloatArray:
    """Uniformly permute a full planned block; arm labels are zero and one."""
    if not isinstance(plan, BlockArandPlan):
        raise TypeError("plan must be a BlockArandPlan")
    size = _integer(plan.size, "plan.size", 2, 10)
    k = _integer(plan.arm_zero_count, "plan.arm_zero_count", 1, size - 1)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    return _freeze(rng.permutation(np.concatenate((np.zeros(k), np.ones(size - k)))))


@dataclass(frozen=True)
class BlockArandDecision:
    superiority: tuple[float, float]
    quadrature_error: float
    target_probability: float
    stopped: bool
    selected: int | None
    reason: str


def _comparison(
    successes: tuple[int, int], failures: tuple[int, int], design: BlockArandDesign
) -> tuple[float, float, float]:
    post = [
        BetaBinomialPosterior(*design.prior[i]).update(successes[i], failures[i]) for i in range(2)
    ]
    result = compare_beta_binomial(post[1], post[0], absolute_tolerance=design.tolerance)
    return (
        float(result.treatment_greater),
        float(result.control_greater),
        float(result.absolute_error),
    )


def _decision(
    p: float, q: float, error: float, n: int, design: BlockArandDesign
) -> BlockArandDecision:
    selected = None
    reason = "continue"
    stopped = False
    # Arm zero takes precedence if a user supplies a cutoff below one half.
    if p > design.early_cutoff or q > design.early_cutoff:
        stopped, reason = True, "early superiority"
        selected = 0 if p > design.early_cutoff else 1
    elif n >= design.max_patients:
        stopped, reason = True, "maximum enrollment"
        selected = 0 if p > design.final_cutoff else 1 if q > design.final_cutoff else None
    if n < design.burnin or design.tuning == 0:
        target = 0.5
    elif p == 0 or q == 0:
        target = float(q == 0)
    else:
        with np.errstate(over="ignore"):
            target = float(expit(design.tuning * (np.log(p) - np.log(q))))
    return BlockArandDecision((p, q), error, target, stopped, selected, reason)


def blockarand_decision(
    successes: ArrayLike, failures: ArrayLike, *, design: BlockArandDesign = BlockArandDesign()
) -> BlockArandDecision:
    """Assess complete outcomes, including early stopping during burn-in and at n=0.

    target_probability is for a NEW block only, not the next assignment in a
    partly consumed block. Both ordering tails are integrated directly.
    """
    if not isinstance(design, BlockArandDesign):
        raise TypeError("design must be a BlockArandDesign")
    s, f = count(successes, "successes"), count(failures, "failures")
    if s.shape != (2,) or f.shape != (2,) or (s + f).sum() > design.max_patients:
        raise ValueError("counts must have shape (2,) with total <= max_patients")
    ss, ff = (int(s[0]), int(s[1])), (int(f[0]), int(f[1]))
    return _decision(*_comparison(ss, ff, design), int(s.sum() + f.sum()), design)


@dataclass(frozen=True)
class BlockArandTrial:
    records: FloatArray  # arm, binary response, block index
    plans: tuple[BlockArandPlan, ...]
    block_starts: FloatArray
    successes: FloatArray
    failures: FloatArray
    decision: BlockArandDecision
    design: BlockArandDesign


_Compare = Callable[[tuple[int, int], tuple[int, int]], tuple[float, float, float]]


def _simulate(
    truth: FloatArray, design: BlockArandDesign, rng: np.random.Generator, compare: _Compare
) -> BlockArandTrial:
    successes, failures = [0, 0], [0, 0]
    records: list[tuple[int, int, int]] = []
    plans: list[BlockArandPlan] = []
    starts: list[int] = []
    assignments = np.empty(0)
    position = 0
    while True:
        n = len(records)
        decision = _decision(
            *compare((successes[0], successes[1]), (failures[0], failures[1])), n, design
        )
        if decision.stopped:
            break
        # The library uses separate equal/adaptive randomizers. At the exact
        # burn-in boundary, any unused equal-block assignments are abandoned.
        if position == len(assignments) or n == design.burnin:
            if n < design.burnin:
                size = design.max_block - design.max_block % 2
                plan = blockarand_plan(0.5, min_block=size, max_block=size)
            else:
                plan = blockarand_plan(
                    decision.target_probability,
                    min_block=design.min_block,
                    max_block=design.max_block,
                )
            plans.append(plan)
            starts.append(n)
            assignments = blockarand_block(plan, rng=rng)
            position = 0
        arm = int(assignments[position])
        position += 1
        response = int(rng.random() < truth[arm])
        records.append((arm, response, len(plans) - 1))
        (successes if response else failures)[arm] += 1
    return BlockArandTrial(
        _freeze(np.asarray(records).reshape(-1, 3)),
        tuple(plans),
        _freeze(starts),
        _freeze(successes),
        _freeze(failures),
        decision,
        design,
    )


def _inputs(
    probabilities: ArrayLike, design: BlockArandDesign, rng: np.random.Generator
) -> FloatArray:
    truth = finite(probabilities, "probabilities")
    if truth.shape != (2,) or np.any((truth < 0) | (truth > 1)):
        raise ValueError("probabilities must be two values in [0,1]")
    if not isinstance(design, BlockArandDesign):
        raise TypeError("design must be a BlockArandDesign")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    return truth


def simulate_blockarand(
    probabilities: ArrayLike,
    *,
    design: BlockArandDesign = BlockArandDesign(),
    rng: np.random.Generator,
) -> BlockArandTrial:
    """Simulate one trial with immediately observed independent binary responses."""
    truth = _inputs(probabilities, design, rng)
    return _simulate(truth, design, rng, lambda s, f: _comparison(s, f, design))


@dataclass(frozen=True)
class BlockArandOperatingCharacteristics:
    selection_probability: FloatArray
    selection_mcse: FloatArray
    no_selection_probability: float
    no_selection_mcse: float
    mean_patients: FloatArray
    patients_mcse: FloatArray
    selected: FloatArray  # -1 means no winner
    enrollment: FloatArray
    probabilities: FloatArray
    design: BlockArandDesign
    comparisons: int
    cache_hits: int


def simulate_blockarand_oc(
    probabilities: ArrayLike,
    *,
    repetitions: int = 500,
    design: BlockArandDesign = BlockArandDesign(),
    rng: np.random.Generator,
) -> BlockArandOperatingCharacteristics:
    """Replicate a scenario, sharing a bounded posterior cache across trials.

    Trial histories are released after each replicate. Per-trial selections and
    enrollment are retained; means and Monte Carlo standard errors are returned.
    """
    truth = _inputs(probabilities, design, rng)
    reps = _integer(repetitions, "repetitions", 2, 1000000)
    compare = lru_cache(maxsize=100000)(lambda s, f: _comparison(s, f, design))
    selected, enrollment = np.empty(reps), np.empty((reps, 2))
    for i in range(reps):
        trial = _simulate(truth, design, rng, compare)
        selected[i] = -1 if trial.decision.selected is None else trial.decision.selected
        enrollment[i] = trial.successes + trial.failures
    p = np.array([(selected == arm).mean() for arm in (0, 1)])
    none = float((selected == -1).mean())
    info = compare.cache_info()
    return BlockArandOperatingCharacteristics(
        _freeze(p),
        _freeze(np.sqrt(p * (1 - p) / reps)),
        none,
        float(np.sqrt(none * (1 - none) / reps)),
        _freeze(enrollment.mean(axis=0)),
        _freeze(enrollment.std(axis=0, ddof=1) / np.sqrt(reps)),
        _freeze(selected),
        _freeze(enrollment),
        _freeze(truth),
        design,
        info.misses,
        info.hits,
    )
