"""Four-arm beta-binomial phase I/II workflow from P12Xuelin's C program."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betaincc

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite
from .beta_binomial import BetaBinomialPosterior
from .beta_comparison import compare_beta_binomial


@dataclass(frozen=True)
class ParallelPhase12Result:
    """Records are (zero-based arm, toxicity, response); probabilities are next assignment."""

    records: FloatArray
    treated: FloatArray
    toxicities: FloatArray
    responses: FloatArray
    admissible: np.ndarray
    escalation_cleared: np.ndarray
    probability: FloatArray
    efficacy_probability: FloatArray
    efficacy_target: float | None
    phase: str
    reason: str
    selected: int | None
    phase_one_enrollment: int | None


class _Trial:
    """Mutable state is private to one replay/simulation lifecycle."""

    def __init__(
        self, history: FloatArray | None, truth: FloatArray | None, rng: np.random.Generator | None
    ):
        self.history, self.truth, self.rng = history, truth, rng
        self.records: list[tuple[int, int, int]] = []
        self.n = np.zeros(4)
        self.tox = np.zeros(4)
        self.response = np.zeros(4)
        self.admissible = np.zeros(4, dtype=bool)
        self.cleared = np.full(4, -2, dtype=int)
        self.probability = np.zeros(4)
        self.efficacy_probability = np.full(4, np.nan)
        self.efficacy_target: float | None = None
        self.phase = "phase_i"
        self.reason = "continue enrollment"
        self.selected: int | None = None
        self.start: int | None = None
        self.cache: dict[tuple[float, ...], tuple[float, float]] = {}

    def enroll(self, number: int, arm: int | None = None) -> bool:
        if arm is not None:
            self.probability[:] = 0
            self.probability[arm] = 1
        for _ in range(number):
            if self.history is not None:
                if len(self.records) == len(self.history):
                    return False
                a, t, r = (int(x) for x in self.history[len(self.records)])
                if self.probability[a] <= 0:
                    raise ValueError(f"record {len(self.records) + 1} assigns an ineligible arm")
            else:
                assert self.rng is not None and self.truth is not None
                a = arm if arm is not None else int(self.rng.choice(4, p=self.probability))
                t, r = (int(x) for x in self.rng.random(2) < self.truth[a])
            self.n[a] += 1
            self.tox[a] += t
            self.response[a] += r
            self.records.append((a, t, r))
        return True

    def compare(self, best: int, other: int) -> tuple[float, float]:
        key = (self.response[best], self.n[best], self.response[other], self.n[other])
        if key not in self.cache:
            comparison = compare_beta_binomial(
                BetaBinomialPosterior(
                    0.1 + self.response[other], 1.9 + self.n[other] - self.response[other]
                ),
                BetaBinomialPosterior(
                    0.1 + self.response[best], 1.9 + self.n[best] - self.response[best]
                ),
                absolute_tolerance=1e-12,
            )
            self.cache[key] = (
                float(comparison.treatment_greater),
                float(comparison.absolute_error),
            )
        return self.cache[key]

    def stop(self, reason: str, selected: int | None = None) -> bool:
        self.phase = "complete"
        self.reason = reason
        self.selected = selected
        self.probability[:] = 0
        return True

    def check_stop(self, final: bool = False) -> bool:
        arms = np.flatnonzero(self.admissible)
        if not len(arms):
            return self.stop("no admissible arms")
        # The original >=3-arm gate counts ALL arms with >=5 patients,
        # including arms subsequently excluded for toxicity.
        ready = np.count_nonzero(self.n >= 5) >= 3 if len(arms) >= 3 else np.all(self.n[arms] >= 5)
        if not ready and not final:
            return False
        target = 0.05 if final else 0.20
        self.efficacy_target = target
        self.efficacy_probability = betaincc(
            0.1 + self.response, 1.9 + self.n - self.response, target
        )
        rank = arms[np.argsort(-self.efficacy_probability[arms], kind="stable")]
        best = int(rank[0])
        probability = self.efficacy_probability[best]
        if final:
            return self.stop("maximum sample size", best if probability > 0.95 else None)
        if probability < 0.05:
            return self.stop("futility")
        if len(arms) == 1:
            if probability > 0.95:
                return self.stop("efficacy", best)
        elif probability > 0.9:
            second = int(rank[1])
            superiority, _ = self.compare(best, second)
            if superiority > 0.8:
                return self.stop("efficacy", best)
            if self.efficacy_probability[second] == probability and 1 - superiority > 0.8:
                return self.stop("efficacy", second)
        return False

    def randomize(self) -> None:
        weight = np.zeros(4)
        error = np.zeros(4)
        if self.admissible[0]:
            weight[0] = 0.5
        for arm in range(1, 4):
            if self.admissible[arm]:
                # The reference remains arm zero even after it is closed.
                weight[arm], error[arm] = self.compare(arm, 0)
        if weight.sum() <= 0 or error.sum() > 1e-6 * weight.sum():
            raise ArithmeticError("adaptive weights are too small for reliable normalization")
        weight /= weight.sum()
        weight[weight < 0.01] = 0
        if weight.sum() <= 0:
            raise ArithmeticError("all adaptive weights vanished after pruning")
        self.probability = weight / weight.sum()

    def run(self) -> None:
        self.admissible[0] = True
        if not self.enroll(3, 0):
            return
        if self.tox[0] >= 2:
            self.admissible[:] = False
            self.stop("no admissible arms")
            return
        if self.tox[0] == 1 and not self.enroll(3, 0):
            return
        if self.tox[0] >= 3:
            self.admissible[0] = False
            self.cleared[0] = -1
            self.stop("no admissible arms")
            return
        if self.tox[0] <= 1:
            self.cleared[0] = 1
            self.admissible[1:3] = True
        # Exactly 2/6 at arm zero remains admissible but does not escalate.
        for arm in (1, 2):
            if not self.admissible[arm]:
                continue
            if not self.enroll(3, arm):
                return
            if self.tox[arm] == 1 and not self.enroll(3, arm):
                return
            if self.tox[arm] >= 3 or (self.n[arm] == 3 and self.tox[arm] >= 2):
                self.admissible[arm] = False
                self.cleared[arm] = -1
            elif self.tox[arm] <= 1:
                self.cleared[arm] = 1
            else:
                self.cleared[arm] = -1
        if self.cleared[1] == 1 and self.cleared[2] == 1:
            if not self.enroll(3, 3):
                return
            if self.tox[3] == 1 and not self.enroll(3, 3):
                return
            self.admissible[3] = self.tox[3] <= 1 or (self.n[3] == 6 and self.tox[3] == 2)
        self.start = len(self.records)
        self.phase = "phase_ii"
        changed = False
        while len(self.records) < 100:
            if not np.any(self.admissible):
                self.stop("no admissible arms")
                return
            if (len(self.records) - self.start) % 5 == 0:
                if self.check_stop():
                    return
                self.randomize()
            elif changed:
                self.randomize()
            if not self.enroll(1):
                return
            arm = self.records[-1][0]
            overdose = betaincc(1 + self.tox[arm], 9 + self.n[arm] - self.tox[arm], 0.3333334)
            changed = bool(overdose > 0.8)
            if changed:
                self.admissible[arm] = False
        self.check_stop(final=True)

    def result(self) -> ParallelPhase12Result:
        if self.history is not None and len(self.records) != len(self.history):
            raise ValueError("history contains patients after the trial stopped")
        masks = [np.frombuffer(a.tobytes(), dtype=a.dtype) for a in (self.admissible, self.cleared)]
        return ParallelPhase12Result(
            _freeze(np.asarray(self.records).reshape(-1, 3)),
            _freeze(self.n),
            _freeze(self.tox),
            _freeze(self.response),
            masks[0],
            masks[1],
            _freeze(self.probability),
            _freeze(self.efficacy_probability),
            self.efficacy_target,
            self.phase,
            self.reason,
            self.selected,
            self.start,
        )


def parallel_phase12_replay(records: ArrayLike) -> ParallelPhase12Result:
    """Replay complete binary patient outcomes under the archived four-arm C rules.

    Each row is (arm 0..3, toxicity 0/1, response 0/1). Partial phase-I cohorts
    retain their forced assignment until three new complete outcomes accumulate.
    This C variant completes phase I before beginning adaptive randomization;
    it is not the separate calendar-time C++ variant included in the archive.
    """
    x = count(records, "records")
    if x.size == 0:
        x = np.empty((0, 3))
    if x.ndim != 2 or x.shape[1] != 3 or len(x) > 100:
        raise ValueError("records must have shape (at most 100 patients, 3)")
    if np.any(x[:, 0] > 3) or np.any(x[:, 1:] > 1):
        raise ValueError("arms must be 0..3 and outcomes must be binary")
    trial = _Trial(x, None, None)
    trial.run()
    return trial.result()


def simulate_parallel_phase12(
    toxicity_probability: ArrayLike, efficacy_probability: ArrayLike, *, rng: np.random.Generator
) -> ParallelPhase12Result:
    """Simulate the four-arm C design with independent toxicity and efficacy.

    Fixed source settings: up to 100 patients, phase-II looks every five
    patients, Beta(1,9) toxicity and Beta(.1,1.9) efficacy priors. Supplied
    probabilities each have four entries in native arm order.
    """
    t, e = (
        finite(toxicity_probability, "toxicity_probability"),
        finite(efficacy_probability, "efficacy_probability"),
    )
    if t.shape != (4,) or e.shape != (4,) or np.any((t < 0) | (t > 1)) or np.any((e < 0) | (e > 1)):
        raise ValueError("toxicity and efficacy probabilities must be four-vectors in [0,1]")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    trial = _Trial(None, np.column_stack((t, e)), rng)
    trial.run()
    return trial.result()
