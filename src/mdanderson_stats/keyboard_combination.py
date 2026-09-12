"""Keyboard design decisions for two-agent dose-combination trials.

This module follows the public CRAN ``Keyboard`` implementation of
``next.comb.kb`` and ``select.mtd.comb.kb``.  Dose indices are one-based and
are represented as ``(drug-A, drug-B)`` tuples.  The final estimator uses the
weighted bivariate isotonic regression used by the R package's ``Iso::biviso``.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betainc, betaincc

from ._validation import FloatArray, scalar
from .boin import _owned

DoseCombination = tuple[int, int]
BoolMatrix = NDArray[np.bool_]


@dataclass(frozen=True)
class KeyboardCombBoundaryTable:
    """One-dimensional cutoffs used for the current combination's outcome."""

    patients: NDArray[np.int64]
    escalate_max: NDArray[np.int64]
    deescalate_min: NDArray[np.int64]
    eliminate_min: NDArray[np.int64]
    lowest_stop_min: NDArray[np.int64]


@dataclass(frozen=True)
class KeyboardCombDecision:
    """Dose-combination recommendation and sticky safety state."""

    action: str
    next_dose: DoseCombination | None
    eliminated: BoolMatrix
    overdose_probability: FloatArray


@dataclass(frozen=True)
class KeyboardCombSelection:
    """Final combination MTD estimate and the matrix isotonic fit."""

    dose: DoseCombination | None
    eliminated: BoolMatrix
    isotonic_mean: FloatArray
    report_overdose_probability: FloatArray
    safety_overdose_probability: FloatArray


def _counts(value: ArrayLike, name: str) -> FloatArray:
    """Validate a rectangular matrix of nonnegative integer counts."""

    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[0] < 2 or result.shape[1] < 2:
        raise ValueError(f"{name} must be a matrix with at least 2 rows and 2 columns")
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"{name} must contain finite nonnegative counts")
    if np.any(result != np.floor(result)):
        raise ValueError(f"{name} must contain integer counts")
    return result


def _validate_counts(patients: ArrayLike, toxicities: ArrayLike) -> tuple[FloatArray, FloatArray]:
    n = _counts(patients, "patients")
    y = _counts(toxicities, "toxicities")
    if n.shape[0] > n.shape[1] or y.shape != n.shape:
        raise ValueError(
            "patients and toxicities must have matching dimensions with rows <= columns"
        )
    if np.any(y > n) or n.sum() > 200:
        raise ValueError("require toxicities <= patients and total patients <= 200")
    return n, y


def _validate_dose(dose: Iterable[int], shape: tuple[int, int]) -> tuple[int, int]:
    try:
        values = tuple(dose)
    except TypeError as exc:
        raise ValueError("current_dose must be a two-element one-based tuple") from exc
    if len(values) != 2:
        raise ValueError("current_dose must be a two-element one-based tuple")
    if any(isinstance(v, (bool, np.bool_)) or int(v) != v for v in values):
        raise ValueError("current_dose must contain integer indices")
    result = (int(values[0]), int(values[1]))
    if not (1 <= result[0] <= shape[0] and 1 <= result[1] <= shape[1]):
        raise ValueError("current_dose must be within the dose matrix")
    return result


def _pava(values: FloatArray, weights: FloatArray) -> FloatArray:
    """Weighted increasing PAVA, returning one value per original position."""

    block_values: list[float] = []
    block_weights: list[float] = []
    block_sizes: list[int] = []
    for value, weight in zip(values, weights, strict=True):
        block_values.append(float(value))
        block_weights.append(float(weight))
        block_sizes.append(1)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            weight_sum = block_weights[-2] + block_weights[-1]
            block_values[-2] = (
                block_values[-2] * block_weights[-2] + block_values[-1] * block_weights[-1]
            ) / weight_sum
            block_weights[-2] = weight_sum
            block_sizes[-2] += block_sizes[-1]
            block_values.pop()
            block_weights.pop()
            block_sizes.pop()
    return np.repeat(np.asarray(block_values), block_sizes)


def _biviso(values: FloatArray, weights: FloatArray) -> FloatArray:
    """Weighted bivariate isotonic regression for increasing rows and columns.

    This is the Dykstra--Robertson row/column cycling scheme used by
    Algorithm AS 206 (the algorithm behind ``Iso::biviso``).  The correction
    arrays are essential: repeatedly applying ordinary row and column PAVA
    without them is not, in general, the weighted least-squares solution.
    """

    result = np.asarray(values, dtype=float).copy()
    weight = np.asarray(weights, dtype=float)
    if result.shape != weight.shape or np.any(weight <= 0):
        raise ValueError(
            "bivariate isotonic values and weights must have matching positive entries"
        )
    row_correction = np.zeros_like(result)
    column_correction = np.zeros_like(result)
    # AS 206 uses sqrt(machine epsilon); the tighter tolerance improves the
    # reproducibility of the final dose tie comparisons in Python.
    tolerance = 2e-12
    for _ in range(50_000):
        previous = result.copy()
        for i in range(result.shape[0]):
            before = result[i] - row_correction[i]
            ordered = _pava(before, weight[i])
            row_correction[i] = ordered - before
            result[i] = ordered
        for j in range(result.shape[1]):
            before = result[:, j] - column_correction[:, j]
            ordered = _pava(before, weight[:, j])
            column_correction[:, j] = ordered - before
            result[:, j] = ordered
        row_violation = np.max(result[:, :-1] - result[:, 1:])
        column_violation = np.max(result[:-1, :] - result[1:, :])
        if (
            np.max(np.abs(result - previous)) <= tolerance
            and max(row_violation, column_violation) <= tolerance
        ):
            return result
    raise ArithmeticError("bivariate isotonic regression did not converge")


@dataclass(frozen=True)
class KeyboardCombDesign:
    """Keyboard dose-finding design for a rectangular two-agent grid.

    ``target`` is constrained to the range supported by the public R
    implementation.  ``margin_left`` and ``margin_right`` define the target key.  The
    implementation accepts complete, evaluable binomial outcomes; delayed or
    fractional outcomes require a separate TITE extension.
    """

    target: float = 0.3
    margin_left: float = 0.05
    margin_right: float = 0.05
    cutoff_eli: float = 0.95
    extra_safe: bool = False
    safety_offset: float = 0.05
    early_stop_patients: int | None = 100
    intervals: FloatArray = field(init=False, repr=False)
    target_key: int = field(init=False)
    _cutoff_cache: dict[int, tuple[int, int, int]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        target = scalar(self.target, "target")
        margin_l = scalar(self.margin_left, "margin_left")
        margin_r = scalar(self.margin_right, "margin_right")
        cutoff = scalar(self.cutoff_eli, "cutoff_eli")
        offset = scalar(self.safety_offset, "safety_offset")
        if not 0.05 <= target <= 0.6:
            raise ValueError("target must be in [0.05, 0.6]")
        if not (0 < margin_l and 0 < margin_r):
            raise ValueError("margin_left and margin_right must be positive")
        lower, upper = target - margin_l, target + margin_r
        if not 0 <= lower < target < upper <= 1:
            raise ValueError(
                "target key must satisfy "
                "0 <= target-margin_left < target < target+margin_right <= 1"
            )
        if not 0 < cutoff < 1 or not 0 <= offset < cutoff:
            raise ValueError("require cutoff_eli in (0,1) and 0 <= safety_offset < cutoff_eli")
        if not isinstance(self.extra_safe, (bool, np.bool_)):
            raise ValueError("extra_safe must be boolean")
        if self.early_stop_patients is not None:
            stop = scalar(self.early_stop_patients, "early_stop_patients")
            if stop != int(stop) or stop < 1:
                raise ValueError("early_stop_patients must be a positive integer")

        delta = upper - lower
        if delta < 0.001:
            raise ValueError("target key width must be at least .001")
        left_count = int(np.ceil(lower / delta - 32 * np.finfo(float).eps))
        right_count = int(np.ceil((1 - upper) / delta - 32 * np.finfo(float).eps))
        left = [max(0.0, lower - k * delta) for k in range(left_count, -1, -1)]
        right = [min(1.0, upper + k * delta) for k in range(right_count + 1)]
        edges = np.asarray(left + right, dtype=float)
        # Avoid the duplicate endpoint produced when a target key touches 0 or 1.
        edges = np.unique(np.round(edges, 15))
        target_key = int(np.flatnonzero(np.isclose(edges[:-1], lower, atol=1e-14))[0])
        object.__setattr__(self, "target", float(target))
        object.__setattr__(self, "margin_left", float(margin_l))
        object.__setattr__(self, "margin_right", float(margin_r))
        object.__setattr__(self, "cutoff_eli", float(cutoff))
        object.__setattr__(self, "safety_offset", float(offset))
        object.__setattr__(self, "intervals", _owned(np.column_stack((edges[:-1], edges[1:]))))
        object.__setattr__(self, "target_key", target_key)
        object.__setattr__(self, "_cutoff_cache", {})

    @property
    def keys(self) -> FloatArray:
        """The key edges used by the posterior interval decisions."""

        return _owned(np.r_[self.intervals[:, 0], self.intervals[-1, 1]])

    def _key_scores(self, n: float, y: float) -> FloatArray:
        a, b = y + 1.0, n - y + 1.0
        lo, hi = self.intervals.T
        mass = np.where(
            betainc(a, b, hi) <= 0.5,
            betainc(a, b, hi) - betainc(a, b, lo),
            betaincc(a, b, lo) - betaincc(a, b, hi),
        )
        full_width = self.margin_left + self.margin_right
        edge_width = np.where(np.arange(len(mass)) == 0, hi - lo, full_width)
        edge_width = np.where(np.arange(len(mass)) == len(mass) - 1, hi - lo, edge_width)
        return np.maximum(mass * full_width / edge_width, 0.0)

    def _move(self, n: float, y: float) -> int:
        scores = self._key_scores(n, y)
        strongest = len(scores) - 1 - int(np.argmax(scores[::-1]))
        return int(np.sign(self.target_key - strongest))

    def _cutoffs(self, n: int) -> tuple[int, int, int]:
        cached = self._cutoff_cache.get(n)
        if cached is not None:
            return cached
        scores = np.asarray([self._move(n, y) for y in range(n + 1)])
        posterior = betaincc(np.arange(n + 1) + 1, n - np.arange(n + 1) + 1, self.target)
        # Safety requires at least three evaluated patients, rather than three
        # observed DLTs; this matters for low targets (e.g. n=3, y=1).
        elimination = (
            np.flatnonzero(posterior > self.cutoff_eli) if n >= 3 else np.array([], dtype=int)
        )
        elim = int(elimination[0]) if elimination.size else n + 1
        escalate = np.flatnonzero((scores == 1) & (np.arange(n + 1) < elim))
        deescalate = np.flatnonzero((scores == -1) | (np.arange(n + 1) >= elim))
        result = (
            int(escalate[-1]) if escalate.size else -1,
            int(deescalate[0]) if deescalate.size else n + 1,
            elim,
        )
        self._cutoff_cache[n] = result
        return result

    def boundary_table(self, max_patients: int = 150) -> KeyboardCombBoundaryTable:
        """Return scalar patient/DLT cutoffs for all current-cell sample sizes."""

        maximum = scalar(max_patients, "max_patients")
        if maximum != int(maximum) or not 1 <= maximum <= 150:
            raise ValueError("max_patients must be an integer in [1,150]")
        n = np.arange(1, int(maximum) + 1)
        e = np.empty_like(n)
        d = np.empty_like(n)
        elim = np.empty_like(n)
        for i, total in enumerate(n):
            e[i], d[i], elim[i] = self._cutoffs(int(total))
        stop = np.full_like(n, n + 1)
        for i, total in enumerate(n):
            if total >= 3:
                values = np.arange(1, total + 1)
                posterior = betaincc(values + 1, total - values + 1, self.target)
                unsafe = np.flatnonzero(posterior > self.cutoff_eli - self.safety_offset)
                if unsafe.size:
                    stop[i] = unsafe[0] + 1
        if not self.extra_safe:
            stop = elim.copy()
        return KeyboardCombBoundaryTable(
            _owned(n), _owned(e), _owned(d), _owned(elim), _owned(stop)
        )

    def _elimination_state(
        self,
        n: FloatArray,
        y: FloatArray,
        eliminated: ArrayLike | None,
        *,
        closure: str = "rectangle",
        extra_safe_stop: bool = True,
    ) -> tuple[BoolMatrix, FloatArray]:
        posterior = betaincc(y + 1, n - y + 1, self.target)
        state = np.zeros(n.shape, dtype=bool)
        if eliminated is not None:
            prior = np.asarray(eliminated)
            if prior.shape != n.shape or prior.dtype != np.bool_:
                raise ValueError("eliminated must be a matching boolean matrix")
            state |= prior
        if closure not in ("rectangle", "cross"):
            raise ValueError("closure must be 'rectangle' or 'cross'")
        if closure == "rectangle":
            for i, j in zip(*np.where((n >= 3) & (posterior > self.cutoff_eli)), strict=True):
                # next.comb.kb closes the currently unsafe cell to the
                # southeast, which is the reachable higher-dose region.
                state[i:, j:] = True
        else:
            # select.mtd.comb.kb scans columns within each row and breaks at
            # the first unsafe column, retaining the source's cross closure.
            for i in range(n.shape[0]):
                for j in range(n.shape[1]):
                    if n[i, j] >= 3 and posterior[i, j] > self.cutoff_eli:
                        state[i:, j] = True
                        state[i, j:] = True
                        break
        if (
            self.extra_safe
            and extra_safe_stop
            and n[0, 0] >= 3
            and posterior[0, 0] > self.cutoff_eli - self.safety_offset
        ):
            state[:, :] = True
        return _owned(state), _owned(posterior)

    def _neighbor_scores(
        self, n: FloatArray, y: FloatArray, candidates: list[tuple[int, int]], state: BoolMatrix
    ) -> list[tuple[float, tuple[int, int]]]:
        values: list[tuple[float, tuple[int, int]]] = []
        for i, j in candidates:
            if state[i, j]:
                continue
            a, b = y[i, j] + 0.5, n[i, j] - y[i, j] + 0.5
            cdf_hi = betainc(a, b, self.target + self.margin_right)
            score = float(
                cdf_hi - betainc(a, b, self.target - self.margin_left)
                if cdf_hi <= 0.5
                else betaincc(a, b, self.target - self.margin_left)
                - betaincc(a, b, self.target + self.margin_right)
            )
            # The R implementation adds nn * .0005 before comparing; this
            # deterministic equivalent retains its preference for informative cells.
            values.append((score + n[i, j] * 0.0005, (i, j)))
        return values

    def next_dose(
        self,
        patients: ArrayLike,
        toxicities: ArrayLike,
        current_dose: Iterable[int],
        *,
        eliminated: ArrayLike | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> KeyboardCombDecision:
        """Recommend the next combination after a completed cohort.

        Equal-score neighboring moves are sampled uniformly.  ``rng=None``
        creates a fresh NumPy generator for the call; callers needing a
        reproducible trial should pass one persistent ``Generator``.  This is
        deliberate rather than reproducing the R function's ``set.seed(1)``
        on every invocation, which would reset a simulation's random stream.
        """

        n, y = _validate_counts(patients, toxicities)
        dose = _validate_dose(current_dose, n.shape)
        i, j = dose[0] - 1, dose[1] - 1
        if n[i, j] < 1:
            raise ValueError("current_dose must have evaluated patients")
        state, overdose = self._elimination_state(
            n,
            y,
            eliminated,
            extra_safe_stop=(i == 0 and j == 0),
        )
        if state[0, 0]:
            action, next_dose = "stop_safety", None
        elif self.early_stop_patients is not None and n[i, j] >= self.early_stop_patients:
            action, next_dose = "stop_precision", None
        else:
            escalate, deescalate, eliminate_here = self._cutoffs(int(n[i, j]))
            move = 1 if y[i, j] <= escalate else -1 if y[i, j] >= deescalate else 0
            if state[i, j] or int(y[i, j]) >= eliminate_here:
                move = -1
            candidates = (
                [(i + 1, j), (i, j + 1)]
                if move > 0
                else [(i - 1, j), (i, j - 1)]
                if move < 0
                else []
            )
            candidates = [
                (a, b) for a, b in candidates if 0 <= a < n.shape[0] and 0 <= b < n.shape[1]
            ]
            scored = self._neighbor_scores(n, y, candidates, state)
            if scored:
                generator = np.random.default_rng(rng)
                maximum = max(score for score, _ in scored)
                tied = [candidate for score, candidate in scored if score == maximum]
                next_zero = tied[int(generator.integers(len(tied)))]
                next_dose = (next_zero[0] + 1, next_zero[1] + 1)
                action = "escalate" if move > 0 else "deescalate"
            else:
                if state[i, j]:
                    # The R code can fall through to the current cell when
                    # both lower neighbors are unavailable.  Re-enrolling at
                    # an eliminated cell violates the safety invariant, so
                    # terminate instead.
                    next_dose, action = None, "stop_safety"
                else:
                    next_dose, action = dose, "stay"
        return KeyboardCombDecision(action, next_dose, state, overdose)

    def select_mtd(
        self,
        patients: ArrayLike,
        toxicities: ArrayLike,
        *,
        eliminated: ArrayLike | None = None,
    ) -> KeyboardCombSelection:
        """Select the treated, noneliminated combination closest to the target."""

        n, y = _validate_counts(patients, toxicities)
        state, safety = self._elimination_state(n, y, eliminated, closure="cross")
        if state[0, 0]:
            return KeyboardCombSelection(
                None,
                state,
                _owned(np.full(n.shape, np.nan)),
                _owned(np.full(n.shape, np.nan)),
                safety,
            )
        values = (y + 0.05) / (n + 0.1)
        fitted = _biviso(values, n + 0.1)
        report = betaincc(y + 0.05, n - y + 0.05, self.target)
        report[state] = np.nan
        output = fitted.copy()
        output[n == 0] = np.nan
        admissible = (n > 0) & ~state
        if not np.any(admissible):
            # Untreated and fully excluded grids have no admissible MTD.
            chosen = None
        else:
            tie_break = np.indices(n.shape).sum(axis=0) + 2
            ranking = np.where(
                admissible,
                np.abs(fitted + 1e-5 * tie_break - self.target),
                np.inf,
            )
            # R's which(..., arr.ind=TRUE) scans matrix columns first.
            col, row = np.unravel_index(np.argmin(ranking.T), ranking.T.shape)
            chosen = (int(row + 1), int(col + 1))
        return KeyboardCombSelection(chosen, state, _owned(output), _owned(report), safety)
