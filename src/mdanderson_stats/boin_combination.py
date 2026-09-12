"""Bayesian optimal interval dose finding for two-drug combinations.

The conduct and selection rules follow the public ``BOIN`` R package's
``next.comb`` and ``select.mtd.comb`` functions.  Dose pairs are one-based
``(drug A, drug B)`` labels.  Random ties are resolved by an explicit NumPy
generator so a caller can reproduce a trial without resetting global state.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betainc, betaincc

from ._validation import FloatArray, count
from .boin import BOINDesign, _owned
from .keyboard_combination import _biviso

DoseCombination = tuple[int, int]
BoolMatrix = NDArray[np.bool_]


@dataclass(frozen=True)
class BOINCombDecision:
    """Recommended next dose pair and the sticky safety state."""

    action: str
    next_dose: DoseCombination | None
    eliminated: BoolMatrix
    overdose_probability: FloatArray


@dataclass(frozen=True)
class BOINCombSelection:
    """Final single MTD or MTD contour and isotonic toxicity estimates."""

    dose: DoseCombination | None
    contour: tuple[DoseCombination, ...] | None
    eliminated: BoolMatrix
    isotonic_mean: FloatArray
    safety_overdose_probability: FloatArray


def _counts(value: ArrayLike, name: str) -> FloatArray:
    result = count(value, name)
    if result.ndim != 2 or result.shape[0] < 2 or result.shape[1] < 2:
        raise ValueError(f"{name} must be a matrix with at least 2 rows and 2 columns")
    if result.shape[0] > result.shape[1]:
        raise ValueError("dose matrices must have rows <= columns (rotate the grid first)")
    return result


def _validate_counts(patients: ArrayLike, toxicities: ArrayLike) -> tuple[FloatArray, FloatArray]:
    n = _counts(patients, "patients")
    y = _counts(toxicities, "toxicities")
    if y.shape != n.shape or np.any(y > n) or n.sum() > 200:
        raise ValueError(
            "require matching matrices, toxicities <= patients, and total patients <= 200"
        )
    return n, y


def _validate_dose(dose: Iterable[int], shape: tuple[int, int]) -> DoseCombination:
    try:
        values = tuple(dose)
    except TypeError as exc:
        raise ValueError("current_dose must be a two-element one-based tuple") from exc
    if len(values) != 2 or any(isinstance(v, (bool, np.bool_)) or int(v) != v for v in values):
        raise ValueError("current_dose must contain integer indices")
    result = (int(values[0]), int(values[1]))
    if not (1 <= result[0] <= shape[0] and 1 <= result[1] <= shape[1]):
        raise ValueError("current_dose must be within the dose matrix")
    return result


@dataclass(frozen=True)
class BOINCombDesign:
    """BOIN design for a rectangular two-agent dose grid."""

    target: float = 0.3
    safe_probability: float | None = None
    toxic_probability: float | None = None
    elimination_probability: float = 0.95
    extra_safe: bool = False
    safety_offset: float = 0.05
    early_stop_patients: int | None = 100
    _base: BOINDesign = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        base = BOINDesign(
            self.target,
            self.safe_probability,
            self.toxic_probability,
            elimination_probability=self.elimination_probability,
            extra_safe=self.extra_safe,
            safety_offset=self.safety_offset,
            early_stop_patients=self.early_stop_patients,
        )
        object.__setattr__(self, "target", base.target)
        object.__setattr__(self, "safe_probability", base.safe_probability)
        object.__setattr__(self, "toxic_probability", base.toxic_probability)
        object.__setattr__(self, "elimination_probability", base.elimination_probability)
        object.__setattr__(self, "safety_offset", base.safety_offset)
        object.__setattr__(self, "_base", base)

    @property
    def escalation_boundary(self) -> float:
        return self._base.escalation_boundary

    @property
    def deescalation_boundary(self) -> float:
        return self._base.deescalation_boundary

    def _state(
        self,
        n: FloatArray,
        y: FloatArray,
        inherited: ArrayLike | None,
        *,
        closure: str = "rectangle",
    ) -> tuple[BoolMatrix, FloatArray]:
        posterior = betaincc(y + 1, n - y + 1, self.target)
        state = np.zeros(n.shape, dtype=bool)
        if inherited is not None:
            prior = np.asarray(inherited)
            if prior.shape != n.shape or prior.dtype != np.bool_:
                raise ValueError("eliminated must be a matching boolean matrix")
            state |= prior
        if closure not in ("rectangle", "cross"):
            raise ValueError("closure must be 'rectangle' or 'cross'")
        # Conduct closes an unsafe cell to the reachable southeast rectangle;
        # the source final selector uses its row/column cross closure instead.
        for i, j in zip(
            *np.where((n >= 3) & (posterior > self.elimination_probability)), strict=True
        ):
            if closure == "rectangle":
                state[i:, j:] = True
            else:
                state[i:, j] = True
                state[i, j:] = True
        if (
            self.extra_safe
            and n[0, 0] >= 3
            and posterior[0, 0] > self.elimination_probability - self.safety_offset
        ):
            state[:, :] = True
        return _owned(state), _owned(posterior)

    def next_dose(
        self, patients: ArrayLike, toxicities: ArrayLike, current_dose: Iterable[int], *,
        eliminated: ArrayLike | None = None, rng: np.random.Generator | int | None = None,
    ) -> BOINCombDecision:
        """Choose an adjacent combination after a complete cohort."""
        n, y = _validate_counts(patients, toxicities)
        dose = _validate_dose(current_dose, n.shape)
        i, j = dose[0] - 1, dose[1] - 1
        if n[i, j] < 1:
            raise ValueError("current_dose must have evaluated patients")
        state, posterior = self._state(n, y, eliminated)
        if state[0, 0]:
            return BOINCombDecision("stop_safety", None, state, posterior)
        if self.early_stop_patients is not None and n[i, j] >= self.early_stop_patients:
            return BOINCombDecision("stop_precision", None, state, posterior)
        table = self._base.boundary_table(max(1, int(n[i, j])))
        k = int(n[i, j]) - 1
        if y[i, j] <= table.escalate_max[k]:
            move = ((1, 0), (0, 1))
            # BOIN's source suppresses escalation when a preceding higher-dose
            # cell is already empirically over the toxic alternative.
            candidates: list[tuple[float, tuple[int, int]]] = []
            raw = np.divide(y, np.where(n == 0, 1, n))
            for di, dj in move:
                a, b = i + di, j + dj
                if a >= n.shape[0] or b >= n.shape[1] or state[a, b]:
                    continue
                if (di and np.any(raw[a, : j + 1] >= self.deescalation_boundary)) or (
                    dj and np.any(raw[: i + 1, b] >= self.deescalation_boundary)
                ):
                    continue
                aa, bb = y[a, b] + 0.5, n[a, b] - y[a, b] + 0.5
                score = float(
                    betainc(aa, bb, self.deescalation_boundary)
                    - betainc(aa, bb, self.escalation_boundary)
                )
                candidates.append((score + n[a, b] * 0.0005, (a, b)))
            action = "escalate"
        elif y[i, j] >= table.deescalate_min[k]:
            move = ((-1, 0), (0, -1))
            candidates = []
            for di, dj in move:
                a, b = i + di, j + dj
                if a < 0 or b < 0 or state[a, b]:
                    continue
                aa, bb = y[a, b] + 0.5, n[a, b] - y[a, b] + 0.5
                score = float(
                    betainc(aa, bb, self.deescalation_boundary)
                    - betainc(aa, bb, self.escalation_boundary)
                )
                candidates.append((score + n[a, b] * 0.0005, (a, b)))
            action = "deescalate"
        else:
            return BOINCombDecision("stay", dose, state, posterior)
        if not candidates:
            return BOINCombDecision("stay", dose, state, posterior)
        best = max(score for score, _ in candidates)
        tied = [candidate for score, candidate in candidates if score == best]
        generator = np.random.default_rng(rng)
        chosen = tied[int(generator.integers(len(tied)))]
        return BOINCombDecision(action, (chosen[0] + 1, chosen[1] + 1), state, posterior)

    def select_mtd(
        self, patients: ArrayLike, toxicities: ArrayLike, *, eliminated: ArrayLike | None = None,
        mtd_contour: bool = False, bound_mtd: bool = False,
    ) -> BOINCombSelection:
        """Select a single MTD, or one closest-to-target dose in each contour row."""
        n, y = _validate_counts(patients, toxicities)
        state, safety = self._state(n, y, eliminated, closure="cross")
        if state[0, 0]:
            return BOINCombSelection(
                None,
                tuple() if mtd_contour else None,
                state,
                _owned(np.full(n.shape, np.nan)),
                safety,
            )
        values = (y + 0.05) / (n + 0.1)
        fitted = _biviso(values, n + 0.1)
        fitted[n == 0] = np.nan
        fitted[state] = np.nan
        admissible = (n > 0) & ~state
        if bound_mtd:
            admissible &= fitted <= self.deescalation_boundary
        if not np.any(admissible):
            chosen = None
            contour: tuple[DoseCombination, ...] | None = tuple() if mtd_contour else None
        elif not mtd_contour:
            rank = np.where(
                admissible,
                np.abs(fitted - self.target)
                + 1e-5 * (np.indices(n.shape).sum(0) + 2),
                np.inf,
            )
            row, col = np.unravel_index(np.argmin(rank), rank.shape)
            chosen, contour = (int(row + 1), int(col + 1)), None
        else:
            selected: list[DoseCombination] = []
            previous: int = int(n.shape[1] + 1)
            for a_raw in range(n.shape[0] - 1, -1, -1):
                a = int(a_raw)
                cols = np.flatnonzero(admissible[a])
                if cols.size == 0:
                    continue
                selected_col: int = int(
                    cols[np.argmin(np.abs(fitted[a, cols] - self.target))]
                )
                if previous <= n.shape[1] and selected_col > previous:
                    selected_col = previous
                if not selected or selected_col != selected[-1][1]:
                    selected.append((a + 1, selected_col + 1))
                previous = selected_col
            chosen, contour = None, tuple(reversed(selected))
        return BOINCombSelection(chosen, contour, state, _owned(fitted), safety)
