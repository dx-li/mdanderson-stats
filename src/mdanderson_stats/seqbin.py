"""Beta-posterior sequential binomial designs and exact stopping probabilities."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betainc, betaincc

from ._validation import FloatArray, count, finite, scalar


@dataclass(frozen=True)
class SeqBinProperties:
    probability: FloatArray
    quit_low: FloatArray
    quit_high: FloatArray
    complete: FloatArray
    expected_subjects: FloatArray
    expected_subjects_quit_low: FloatArray
    expected_subjects_quit_high: FloatArray

    @property
    def rejection_probability(self) -> FloatArray:
        return self.quit_low.sum(axis=-1) + self.quit_high.sum(axis=-1)


@dataclass(frozen=True, init=False)
class SeqBinDesign:
    """Inclusive continuation boundaries at each scheduled look.

    Stop for fewer than continue_low events or more than continue_high events.
    Boundary values n+1 / -1 permit all outcomes to stop on that side. At the
    maximum sample size, remaining paths complete without rejecting. No final
    analysis is required; an empty look schedule never rejects. Arrays are read-only.
    """

    max_subjects: int
    looks: NDArray[np.int64]
    continue_low: NDArray[np.int64]
    continue_high: NDArray[np.int64]
    prior: tuple[float, float]
    null_probability: float
    alternative: str
    tail_probability: float | tuple[float, float]
    legacy_bounds: bool

    def __init__(
        self,
        max_subjects: int,
        *,
        prior: ArrayLike = (1, 1),
        null_probability: float = 0.2,
        alternative: str = "greater",
        tail_probability: ArrayLike = 0.05,
        looks: ArrayLike | None = None,
        legacy_bounds: bool = False,
    ):
        if (
            isinstance(max_subjects, (bool, np.bool_))
            or not isinstance(max_subjects, (int, np.integer))
            or not 2 <= max_subjects <= 10000
        ):
            raise ValueError("max_subjects must be an integer from 2 to 10000")
        parameters = finite(prior, "prior")
        if parameters.shape != (2,) or np.any(parameters <= 0):
            raise ValueError("prior must contain two positive beta shape parameters")
        p0 = scalar(null_probability, "null_probability")
        cutoffs = finite(tail_probability, "tail_probability")
        if cutoffs.ndim == 0:
            cutoffs = np.repeat(cutoffs, 2)
        elif cutoffs.shape != (2,) or alternative != "two-sided":
            raise ValueError("Two tail_probability values require a two-sided design")
        if not 0 < p0 < 1 or np.any((cutoffs <= 0) | (cutoffs >= 1)):
            raise ValueError("Require null_probability and tail_probability in (0,1)")
        if alternative not in ("less", "greater", "two-sided"):
            raise ValueError("alternative must be less, greater or two-sided")
        if not isinstance(legacy_bounds, (bool, np.bool_)):
            raise ValueError("legacy_bounds must be boolean")
        n = np.arange(1, max_subjects + 1) if looks is None else count(looks, "looks")
        if n.ndim != 1 or np.any(n < 1) or np.any(n > max_subjects) or np.any(np.diff(n) <= 0):
            raise ValueError("looks must increase strictly within [1, max_subjects]")
        n = n.astype(np.int64)
        a, b = map(float, parameters)

        def boundary(low_side: bool) -> NDArray[np.int64]:
            # Find the first count where a monotone predicate becomes true,
            # simultaneously for every look. Sentinels need no beta evaluation.
            cutoff = cutoffs[0 if low_side else 1]
            left, right = np.full(n.size, -1, dtype=np.int64), n + 1
            while np.any(right - left > 1):
                active = right - left > 1
                middle = (left[active] + right[active]) // 2
                aa, bb = a + middle, b + n[active] - middle
                tail = betaincc(aa, bb, p0) if low_side else betainc(aa, bb, p0)
                if not np.all(np.isfinite(tail)):
                    raise ValueError("Beta posterior evaluation failed")
                predicate = tail >= cutoff if low_side else tail < cutoff
                right[active] = np.where(predicate, middle, right[active])
                left[active] = np.where(predicate, left[active], middle)
            result = right if low_side else right - 1
            return np.clip(result, 0, n) if legacy_bounds else result

        lower = boundary(True) if alternative != "greater" else np.zeros_like(n)
        upper = boundary(False) if alternative != "less" else n.copy()
        for name, value in (("looks", n), ("continue_low", lower), ("continue_high", upper)):
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        for name, setting in (
            ("max_subjects", int(max_subjects)),
            ("prior", (a, b)),
            ("null_probability", p0),
            ("alternative", alternative),
            (
                "tail_probability",
                float(cutoffs[0]) if cutoffs[0] == cutoffs[1] else tuple(map(float, cutoffs)),
            ),
            ("legacy_bounds", bool(legacy_bounds)),
        ):
            object.__setattr__(self, name, setting)

    def operating_characteristics(self, probability: ArrayLike) -> SeqBinProperties:
        """Exact forward Bernoulli recursion, broadcasting true event probabilities.

        Quit arrays have a final axis for scheduled looks. Conditional expected
        sample sizes are NaN for events with zero probability. No Monte Carlo
        approximation or negligible-mass pruning is used.
        """
        p = finite(probability, "probability")
        if np.any((p < 0) | (p > 1)):
            raise ValueError("probability must lie in [0,1]")
        low = np.zeros(p.shape + (self.looks.size,))
        high = np.zeros_like(low)
        surviving = np.ones(p.shape + (1,))
        look_index = 0
        # After the final scheduled analysis all surviving paths complete at
        # max_subjects, so further Bernoulli convolution cannot affect outputs.
        last_analysis = int(self.looks[-1]) if self.looks.size else 0
        for n in range(1, last_analysis + 1):
            arriving = np.zeros(p.shape + (n + 1,))
            arriving[..., :-1] = surviving * (1 - p[..., None])
            arriving[..., 1:] += surviving * p[..., None]
            if look_index < self.looks.size and n == self.looks[look_index]:
                lo, hi = self.continue_low[look_index], self.continue_high[look_index]
                low[..., look_index] = arriving[..., :lo].sum(axis=-1)
                arriving[..., :lo] = 0
                high[..., look_index] = arriving[..., hi + 1 :].sum(axis=-1)
                arriving[..., hi + 1 :] = 0
                look_index += 1
            surviving = arriving
        complete = surviving.sum(axis=-1)
        low_total, high_total = low.sum(axis=-1), high.sum(axis=-1)
        low_n, high_n = low @ self.looks, high @ self.looks
        expected = low_n + high_n + complete * self.max_subjects
        low_cond = np.divide(low_n, low_total, out=np.full_like(p, np.nan), where=low_total > 0)
        high_cond = np.divide(high_n, high_total, out=np.full_like(p, np.nan), where=high_total > 0)
        return SeqBinProperties(p.copy(), low, high, complete, expected, low_cond, high_cond)
