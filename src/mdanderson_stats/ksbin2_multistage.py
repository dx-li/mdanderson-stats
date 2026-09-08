"""Fixed two-sample multistage binomial designs with cached surviving paths."""

from dataclasses import dataclass, field
from math import comb

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, count, finite
from .ksbin2 import KSBinomialOrdering, ksbin2_ordering
from .kstage_binomial import KStageBinomial


@dataclass(frozen=True)
class KSTwoSampleOperatingCharacteristics:
    rejection: FloatArray
    quitting: FloatArray
    continuation: FloatArray
    rejection_probability: FloatArray
    expected_sample_size: FloatArray


@dataclass(frozen=True, init=False)
class KStageTwoSampleBinomial:
    """Fixed boundaries indexed by reachable tied score group at each stage.

    Cumulative sizes have shape (stages, 2). reject_group has one inclusive
    zero-based group per stage; quit_group has interim entries only. Rejection
    includes groups <= reject_group, quitting includes groups >= quit_group.
    -1 disables either boundary. Every final nonrejection quits. Group indices
    refer to reachable outcomes after earlier stopping, as in the source.
    """

    cumulative_trials: tuple[tuple[int, int], ...]
    reject_group: tuple[int, ...]
    quit_group: tuple[int, ...]
    criteria: tuple[int, ...]
    alternative: str
    orderings: tuple[KSBinomialOrdering, ...]
    _survival_fraction: tuple[FloatArray, ...] = field(repr=False, compare=False)
    _reject: tuple[np.ndarray, ...] = field(repr=False, compare=False)
    _quit: tuple[np.ndarray, ...] = field(repr=False, compare=False)

    def __init__(
        self,
        cumulative_trials: ArrayLike,
        reject_group: ArrayLike,
        quit_group: ArrayLike,
        *,
        criteria: tuple[int, ...] = (1, 2),
        alternative: str = "greater",
    ):
        totals = count(cumulative_trials, "cumulative_trials")
        if (
            totals.ndim != 2
            or totals.shape[1] != 2
            or not 1 <= len(totals) <= 10
            or np.any((totals < 1) | (totals > 100))
            or np.any(np.diff(totals, axis=0) < 0)
            or np.any(np.sum(np.diff(totals, axis=0), axis=1) == 0)
        ):
            raise ValueError(
                "Require 1–10 cumulative size pairs in [1,100], "
                "increasing in at least one group per stage"
            )
        boundaries = []
        for values, size, name in [
            (reject_group, len(totals), "reject_group"),
            (quit_group, len(totals) - 1, "quit_group"),
        ]:
            v = finite(values, name)
            if v.shape != (size,) or np.any(v < -1) or np.any(v != np.floor(v)):
                raise ValueError(f"{name} requires {size} integer group indices, or -1 to disable")
            boundaries.append(tuple(map(int, v)))
        rejects, quits = boundaries
        coefficients = np.ones((1, 1))
        previous = (0, 0)
        orderings = []
        fractions = []
        reject_masks = []
        quit_masks = []
        for stage, (n1, n2) in enumerate(totals.astype(int)):
            increment = (n1 - previous[0], n2 - previous[1])
            for axis, size in enumerate(increment):
                weights = np.array([comb(int(size), k) for k in range(size + 1)], dtype=float)
                coefficients = np.apply_along_axis(
                    lambda x: np.convolve(x, weights), axis, coefficients
                )
            full = ksbin2_ordering(int(n1), int(n2), criteria=criteria, alternative=alternative)
            reachable = coefficients[full.events[:, 0], full.events[:, 1]] > 0
            events, score = full.events[reachable], full.score[reachable]
            relative = np.abs(np.diff(score)) / np.maximum(
                np.abs(score[:-1]) + np.abs(score[1:]), 1e-100
            )
            ends = np.r_[np.flatnonzero(relative >= 1e-12), len(score) - 1]
            ordering = KSBinomialOrdering(
                full.trials, full.criteria, alternative, events, score, ends
            )
            orderings.append(ordering)
            denominator = np.outer(
                [float(comb(int(n1), k)) for k in range(n1 + 1)],
                [float(comb(int(n2), k)) for k in range(n2 + 1)],
            )
            fraction = coefficients / denominator
            fraction.setflags(write=False)
            fractions.append(fraction)
            rg, qg = rejects[stage], quits[stage] if stage < len(quits) else -1
            if rg >= len(ends) or qg >= len(ends):
                raise ValueError(f"Boundary exceeds reachable group count at stage {stage + 1}")
            if rg >= 0 and qg >= 0 and rg >= qg:
                raise ValueError("Rejection and quitting groups overlap")
            reject = np.zeros(coefficients.shape, dtype=bool)
            quit = np.zeros(coefficients.shape, dtype=bool)
            if rg >= 0:
                chosen = events[: ends[rg] + 1]
                reject[chosen[:, 0], chosen[:, 1]] = True
            if stage == len(totals) - 1:
                quit = ~reject
            elif qg >= 0:
                chosen = events[0 if qg == 0 else ends[qg - 1] + 1 :]
                quit[chosen[:, 0], chosen[:, 1]] = True
            reject_masks.append(reject)
            quit_masks.append(quit)
            coefficients = np.where(reject | quit, 0, coefficients)
            if stage < len(totals) - 1 and not np.any(coefficients):
                raise ValueError("Every planned stage must remain reachable")
            previous = (n1, n2)
        for name, value in dict(
            cumulative_trials=tuple(tuple(map(int, row)) for row in totals),
            reject_group=rejects,
            quit_group=quits,
            criteria=full.criteria,
            alternative=alternative,
            orderings=tuple(orderings),
            _survival_fraction=tuple(fractions),
            _reject=tuple(reject_masks),
            _quit=tuple(quit_masks),
        ).items():
            object.__setattr__(self, name, value)

    def stage_distribution(
        self, stage: int, probability1: ArrayLike, probability2: ArrayLike
    ) -> FloatArray:
        """Joint arrival mass; final axes index cumulative event counts in each group."""
        if (
            isinstance(stage, bool)
            or not isinstance(stage, (int, np.integer))
            or not 1 <= stage <= len(self.cumulative_trials)
        ):
            raise ValueError("stage must be a valid one-based integer")
        p1, p2 = np.broadcast_arrays(
            finite(probability1, "probability1"), finite(probability2, "probability2")
        )
        n1, n2 = self.cumulative_trials[stage - 1]
        first = KStageBinomial([n1], [], []).stage_distribution(1, p1)
        second = KStageBinomial([n2], [], []).stage_distribution(1, p2)
        return first[..., :, None] * second[..., None, :] * self._survival_fraction[stage - 1]

    def operating_characteristics(
        self, probability1: ArrayLike, probability2: ArrayLike
    ) -> KSTwoSampleOperatingCharacteristics:
        """Evaluate any probability pairs, including nulls and reversed effects.

        Stage quantities use the last axis. Expected sample size has a final
        two-group axis and is unconditional under the supplied probabilities.
        """
        rejection = []
        quitting = []
        continuation = []
        for stage in range(len(self.cumulative_trials)):
            mass = self.stage_distribution(stage + 1, probability1, probability2)
            rejection.append(np.sum(mass[..., self._reject[stage]], axis=-1))
            quitting.append(np.sum(mass[..., self._quit[stage]], axis=-1))
            continuation.append(
                np.sum(mass[..., ~(self._reject[stage] | self._quit[stage])], axis=-1)
            )
        r, q, c = (np.stack(values, axis=-1) for values in [rejection, quitting, continuation])
        expected = (r + q) @ np.asarray(self.cumulative_trials)
        return KSTwoSampleOperatingCharacteristics(r, q, c, np.sum(r, axis=-1), expected)
