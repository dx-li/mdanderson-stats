"""Stage-ordered binomial probabilities and KSB1CI confidence intervals."""

from dataclasses import dataclass, field
from math import comb
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, count, finite


@dataclass(frozen=True, init=False)
class KStageBinomial:
    """A fixed design with inclusive low/high stopping boundaries.

    cumulative_trials contains 1–10 strictly increasing cumulative sample sizes,
    totaling at most 200. low and high have one entry per interim stage; -1
    disables that boundary. Stages are numbered from one. Coefficients counting
    paths that reach each stage are precomputed once for repeated inference.
    """

    cumulative_trials: tuple[int, ...]
    low: tuple[int, ...]
    high: tuple[int, ...]
    _coefficients: tuple[FloatArray, ...] = field(repr=False, compare=False)

    def __init__(self, cumulative_trials: ArrayLike, low: ArrayLike, high: ArrayLike):
        totals = count(cumulative_trials, "cumulative_trials")
        if (
            totals.ndim != 1
            or not 1 <= totals.size <= 10
            or np.any(totals <= 0)
            or np.any(np.diff(totals) <= 0)
            or totals[-1] > 200
        ):
            raise ValueError("Require 1–10 increasing positive cumulative sizes, at most 200")
        boundaries = []
        for value, name in ((low, "low"), (high, "high")):
            a = finite(value, name)
            if (
                a.shape != (totals.size - 1,)
                or np.any(a != np.floor(a))
                or np.any(a < -1)
                or np.any(a > totals[:-1])
            ):
                raise ValueError(
                    f"{name} requires one integer in [-1, stage total] per interim stage"
                )
            boundaries.append(tuple(map(int, a)))
        lows, highs = boundaries
        coefficients = []
        surviving = np.ones(1)
        previous = 0
        for i, total in enumerate(map(int, totals)):
            size = total - previous
            arrivals = np.convolve(
                surviving, np.array([comb(size, k) for k in range(size + 1)], dtype=float)
            )
            arrivals.setflags(write=False)
            coefficients.append(arrivals)
            if i < len(lows):
                k = np.arange(total + 1)
                continuing = (k > lows[i]) & ((highs[i] == -1) | (k < highs[i]))
                if lows[i] >= 0 and highs[i] >= 0 and lows[i] >= highs[i]:
                    raise ValueError("Low and high stopping regions must not overlap")
                surviving = np.where(continuing, arrivals, 0)
                if not np.any(surviving):
                    raise ValueError("Every planned stage must be reachable")
            previous = total
        object.__setattr__(self, "cumulative_trials", tuple(map(int, totals)))
        object.__setattr__(self, "low", lows)
        object.__setattr__(self, "high", highs)
        object.__setattr__(self, "_coefficients", tuple(coefficients))

    def _point(self, stage: int, events: ArrayLike) -> FloatArray:
        if (
            isinstance(stage, bool)
            or not isinstance(stage, (int, np.integer))
            or not 1 <= stage <= len(self.cumulative_trials)
        ):
            raise ValueError("stage must be an integer from 1 to the number of stages")
        k = count(events, "events")
        if np.any(k > self.cumulative_trials[stage - 1]):
            raise ValueError("events exceeds the cumulative sample size")
        if np.any(self._coefficients[stage - 1][k.astype(int)] == 0):
            raise ValueError("Observed count is unreachable under earlier stopping boundaries")
        return k

    def _tails(
        self, stage: int, events: FloatArray, p: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        k, p = np.broadcast_arrays(events, p)
        less, greater = np.zeros(k.shape), np.zeros(k.shape)
        for i in range(stage):
            n = self.cumulative_trials[i]
            indices = np.arange(n + 1)
            # Evaluate complete weighted terms in log space so tiny powers do
            # not underflow before large path counts can scale them back up.
            with np.errstate(divide="ignore", invalid="ignore"):
                successes = np.where(indices == 0, 0, indices * np.log(p[..., None]))
                failures = np.where(n == indices, 0, (n - indices) * np.log1p(-p[..., None]))
                mass = np.exp(np.log(self._coefficients[i]) + successes + failures)
            if i == stage - 1:
                less += np.sum(np.where(indices <= k[..., None], mass, 0), axis=-1)
                greater += np.sum(np.where(indices >= k[..., None], mass, 0), axis=-1)
            else:
                less += np.sum(mass[..., : self.low[i] + 1], axis=-1)
                if self.high[i] >= 0:
                    greater += np.sum(mass[..., self.high[i] :], axis=-1)
        return np.clip(less, 0, 1), np.clip(greater, 0, 1)

    def tails(
        self, stage: int, events: ArrayLike, probability: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        """Return inclusive (less, greater) stage-ordered probabilities.

        Less includes all earlier low stops and paths reaching this stage with
        at most events; greater includes earlier high stops and at least events.
        The observed stage may terminate even inside its continuation region.
        Events and probability broadcast; earlier-stage boundaries still apply.
        """
        k = self._point(stage, events)
        p = finite(probability, "probability")
        if np.any((p < 0) | (p > 1)):
            raise ValueError("probability must lie in [0,1]")
        return self._tails(stage, k, p)

    def interval(
        self, stage: int, events: ArrayLike, confidence: ArrayLike = 0.95
    ) -> tuple[FloatArray, FloatArray]:
        """Invert both inclusive tails for an equal-tailed KSB1CI interval.

        Events and confidence broadcast. Bisection uses 55 iterations on [0,1],
        with exact endpoint bounds for tails that do not cross the target.
        """
        k = self._point(stage, events)
        level = finite(confidence, "confidence")
        if np.any((level <= 0) | (level >= 1)):
            raise ValueError("confidence must lie strictly between zero and one")
        k, level = np.broadcast_arrays(k, level)
        target = (1 - level) / 2
        results = []
        for tail in (1, 0):
            lo, hi = np.zeros(k.shape), np.ones(k.shape)
            start = self._tails(stage, k, lo)[tail]
            end = self._tails(stage, k, hi)[tail]
            for _ in range(55):
                mid = (lo + hi) / 2
                value = self._tails(stage, k, mid)[tail]
                right = value < target if tail == 1 else value > target
                lo, hi = np.where(right, mid, lo), np.where(right, hi, mid)
            bound = (lo + hi) / 2
            at_zero = start >= target if tail == 1 else start <= target
            at_one = end <= target if tail == 1 else end >= target
            results.append(np.where(at_zero, 0, np.where(at_one, 1, bound)))
        return results[0], results[1]

    def report(
        self, stage: int, events: ArrayLike, confidence: ArrayLike = 0.95, *, digits: int = 7
    ) -> str:
        """Compute intervals and return the design and broadcast results as TSV."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 to 17")
        lower, upper = self.interval(stage, events, confidence)
        events, confidence = np.broadcast_arrays(events, confidence)
        rows = [
            "KSB1CI stage-ordered binomial confidence intervals",
            "Stage\tNew trials\tCumulative trials\tLow quit\tHigh quit",
        ]
        previous = 0
        for i, n in enumerate(self.cumulative_trials):
            stops = f"{self.low[i]}\t{self.high[i]}" if i < len(self.low) else "—\t—"
            rows.append(f"{i + 1}\t{n - previous}\t{n}\t{stops}")
            previous = n
        rows += [
            "-1 disables an interim boundary; both stopping boundaries are inclusive.",
            "Observed stage\tEvents\tConfidence\tLower\tUpper",
        ]
        for k, level, lo, hi in zip(
            events.ravel(), confidence.ravel(), lower.ravel(), upper.ravel(), strict=True
        ):
            values = "\t".join(format(float(x), f".{digits}g") for x in (k, level, lo, hi))
            rows.append(f"{stage}\t{values}")
        return "\n".join(rows) + "\n"

    def write_report(
        self,
        path: str | Path,
        stage: int,
        events: ArrayLike,
        confidence: ArrayLike = 0.95,
        *,
        digits: int = 7,
    ) -> Path:
        """Write the computed UTF-8 report, replacing path; propagate I/O errors."""
        content = self.report(stage, events, confidence, digits=digits)
        path = Path(path)
        path.write_text(content, encoding="utf-8")
        return path
