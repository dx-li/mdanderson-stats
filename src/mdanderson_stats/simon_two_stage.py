"""Exact Simon two-stage designs with futility stopping and bounded optimization."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import binom

from ._validation import FloatArray, count, finite, scalar


@dataclass(frozen=True)
class SimonOperatingCharacteristics:
    """Exact probabilities and enrollment moments at the supplied response rates."""

    response_rate: FloatArray
    reject_null: FloatArray
    stop_early: FloatArray
    continue_probability: FloatArray
    expected_sample_size: FloatArray
    sample_size_sd: FloatArray


@dataclass(frozen=True)
class SimonDesign:
    """Stop for futility at X1 <= r1; reject H0 at completion only if X1+X2 > r."""

    n1: int
    n: int
    r1: int
    r: int

    def __post_init__(self) -> None:
        for name in ("n1", "n", "r1", "r"):
            value = count(getattr(self, name), name)
            if value.ndim:
                raise ValueError(f"{name} must be a scalar integer")
            object.__setattr__(self, name, int(value))
        if not (1 <= self.n1 < self.n <= 100_000):
            raise ValueError("require 1 <= n1 < n <= 100000")
        if not (0 <= self.r1 < self.n1 and self.r1 <= self.r < self.n):
            raise ValueError("require 0 <= r1 < n1 and r1 <= r < n")

    def operating_characteristics(self, response_rate: ArrayLike) -> SimonOperatingCharacteristics:
        """Broadcast over response rates; use direct upper tails and positive sums."""
        p = finite(response_rate, "response_rate")
        if np.any((p < 0) | (p > 1)):
            raise ValueError("response_rate must lie in [0, 1]")
        # X1 > r already guarantees final rejection, but accrual still continues.
        rejection = np.asarray(binom.sf(max(self.r, self.r1), self.n1, p))
        for x in range(self.r1 + 1, min(self.n1, self.r) + 1):
            rejection = rejection + binom.pmf(x, self.n1, p) * binom.sf(
                self.r - x, self.n - self.n1, p
            )
        early = binom.cdf(self.r1, self.n1, p)
        continuation = binom.sf(self.r1, self.n1, p)
        expected = self.n1 + (self.n - self.n1) * continuation
        sd = (self.n - self.n1) * np.sqrt(early * continuation)
        arrays = [
            np.array(v, dtype=float, copy=True)
            for v in (p, rejection, early, continuation, expected, sd)
        ]
        for array in arrays:
            array.setflags(write=False)
        return SimonOperatingCharacteristics(*arrays)

    def decision(self, first_stage_responses: int, total_responses: int | None = None) -> str:
        """Classify completed stages; reject contradictory counts after an early stop."""
        x1 = scalar(first_stage_responses, "first_stage_responses")
        if x1 != int(x1) or not 0 <= x1 <= self.n1:
            raise ValueError("first_stage_responses must be an integer in [0, n1]")
        if total_responses is None:
            return "stop_futility" if x1 <= self.r1 else "continue"
        total = scalar(total_responses, "total_responses")
        if total != int(total) or not x1 <= total <= x1 + self.n - self.n1:
            raise ValueError("total_responses must be compatible with both stage sizes")
        if x1 <= self.r1:
            raise ValueError("the trial stopped after stage one; no completed second stage exists")
        return "reject_null" if total > self.r else "do_not_reject"


@dataclass(frozen=True)
class SimonDesignSearch:
    """Both optima over 2 <= n <= max_n, with deterministic tie breaking."""

    null_rate: float
    alternative_rate: float
    alpha: float
    power: float
    max_n: int
    optimal: SimonDesign
    minimax: SimonDesign

    def protocol(self, method: Literal["optimal", "minimax"] = "optimal") -> str:
        """Generate a statistical design paragraph, using achieved error probabilities."""
        if method not in ("optimal", "minimax"):
            raise ValueError("method must be 'optimal' or 'minimax'")
        design = getattr(self, method)
        oc = design.operating_characteristics([self.null_rate, self.alternative_rate])
        return (
            f"Use Simon's {method} two-stage design to test a response rate of "
            f"{self.null_rate:g} against {self.alternative_rate:g}. Enroll {design.n1} "
            f"evaluable participants in stage one. Stop for futility if {design.r1} or "
            f"fewer respond; otherwise enroll {design.n - design.n1} additional participants "
            f"({design.n} total). Reject the null hypothesis if at least {design.r + 1} "
            f"of the {design.n} participants respond. There is no early efficacy stop. "
            f"The achieved one-sided type I error is {oc.reject_null[0]:.6g} "
            f"(limit {self.alpha:g}), and power is {oc.reject_null[1]:.6g} "
            f"(target {self.power:g}). Under the null, expected enrollment is "
            f"{oc.expected_sample_size[0]:.6g} and the early stopping probability is "
            f"{oc.stop_early[0]:.6g}. The design search is bounded by n <= {self.max_n}."
        )


def simon_two_stage(
    null_rate: float = 0.05,
    alternative_rate: float = 0.15,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    max_n: int = 100,
) -> SimonDesignSearch:
    """Find optimal E0[N] and minimax N designs using exact binomial probabilities.

    All nontrivial stages and 0 <= r1 < n1, r1 <= r < n are considered.
    Minimax ties minimize E0[N]; optimal ties minimize N. Remaining ties prefer
    smaller n1, smaller r1, and the largest final r compatible with target power.
    Optimality is within max_n, not a claim about an unbounded search.
    """
    p0, p1 = scalar(null_rate, "null_rate"), scalar(alternative_rate, "alternative_rate")
    alpha, power = scalar(alpha, "alpha"), scalar(power, "power")
    limit = scalar(max_n, "max_n")
    if not 0 < p0 < p1 < 1:
        raise ValueError("require 0 < null_rate < alternative_rate < 1")
    if not 0 < alpha < power < 1:
        raise ValueError("require 0 < alpha < power < 1")
    if limit != int(limit) or not 2 <= limit <= 500:
        raise ValueError("max_n must be an integer in [2, 500]")
    max_n = int(limit)
    sizes = np.arange(max_n + 1)[:, None]
    counts = np.arange(max_n + 1)[None, :]
    pmf = binom.pmf(counts, sizes, np.array([p0, p1])[:, None, None])
    tails = binom.sf(counts, sizes, np.array([p0, p1])[:, None, None])
    optimal = minimax = None
    best_en = np.inf
    minimax_en = np.inf

    for n in range(2, max_n + 1):
        possible_r = np.flatnonzero(tails[1, n, :n] >= power)
        if not len(possible_r):
            continue
        r_max = int(possible_r[-1])
        for n1 in range(1, n):
            # After the first feasible N has been fully searched, every later
            # candidate must improve E0[N]. These bounds prune, never truncate.
            if n1 > best_en:
                break
            r1s = np.arange(min(n1, r_max + 1))
            expected = n1 + (n - n1) * tails[0, n1, r1s]
            r1s = r1s[(tails[1, n1, r1s] >= power) & (expected <= best_en)]
            if not len(r1s):
                continue
            x = np.arange(int(r1s[0]) + 1, n1 + 1)
            rs = np.arange(int(r1s[0]), r_max + 1)
            thresholds = rs[:, None] - x[None, :]
            # Negative thresholds have SF=1. Cache nonnegative thresholds;
            # reverse cumulative sums evaluate every stage-one cutoff at once.
            second_tail = np.where(
                thresholds[None, :, :] < 0,
                1.0,
                tails[:, n - n1, np.maximum(thresholds, 0)],
            )
            joint = second_tail * pmf[:, n1, x][:, None, :]
            rejection = np.cumsum(joint[:, :, ::-1], axis=-1)[:, :, ::-1]
            for r1 in r1s:
                column = int(r1 - r1s[0])
                feasible_power = np.flatnonzero((rs >= r1) & (rejection[1, :, column] >= power))
                if not len(feasible_power):
                    continue
                index = int(feasible_power[-1])
                if rejection[0, index, column] > alpha:
                    continue
                design = SimonDesign(n1, n, int(r1), int(rs[index]))
                en = float(n1 + (n - n1) * tails[0, n1, r1])
                # Ascending n/n1/r1 supplies the documented tie order.
                if en < best_en:
                    best_en, optimal = en, design
                if minimax is None or (n == minimax.n and en < minimax_en):
                    minimax, minimax_en = design, en
    if optimal is None or minimax is None:
        raise ValueError(f"no feasible two-stage design with n <= {max_n}; increase max_n")
    return SimonDesignSearch(p0, p1, alpha, power, max_n, optimal, minimax)
