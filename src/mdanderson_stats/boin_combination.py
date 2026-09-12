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
from .boin import BOINBoundaryTable, BOINDesign, _owned
from .keyboard_combination import _biviso

DoseCombination = tuple[int, int]
BoolMatrix = NDArray[np.bool_]


def _interval_probability(a: float, b: float, lower: float, upper: float) -> float:
    """Stable posterior mass between the BOIN indifference probabilities."""
    cdf_upper = betainc(a, b, upper)
    if cdf_upper <= 0.5:
        return float(cdf_upper - betainc(a, b, lower))
    return float(betaincc(a, b, lower) - betaincc(a, b, upper))


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
    if y.shape != n.shape or np.any(y > n) or n.sum() > 1000:
        raise ValueError(
            "require matching matrices, toxicities <= patients, and total patients <= 1000"
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

    def boundary_table(self, max_patients: int = 150) -> BOINBoundaryTable:
        """Return the scalar BOIN cutoffs shared by every combination cell."""
        return self._base.boundary_table(max_patients)

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
        if closure == "rectangle":
            for i, j in zip(
                *np.where((n >= 3) & (posterior > self.elimination_probability)), strict=True
            ):
                state[i:, j:] = True
        else:
            for i in range(n.shape[0]):
                for j in range(n.shape[1]):
                    if n[i, j] >= 3 and posterior[i, j] > self.elimination_probability:
                        state[i:, j] = True
                        state[i, j:] = True
                        # select.mtd.comb stops scanning a row after its first
                        # unsafe column; later unsafe cells cannot widen it.
                        break
        if (
            self.extra_safe
            and n[0, 0] >= 3
            and posterior[0, 0] > self.elimination_probability - self.safety_offset
        ):
            state[:, :] = True
        return _owned(state), _owned(posterior)

    def next_dose(
        self,
        patients: ArrayLike,
        toxicities: ArrayLike,
        current_dose: Iterable[int],
        *,
        eliminated: ArrayLike | None = None,
        rng: np.random.Generator | int | None = None,
        source_simulation: bool = False,
    ) -> BOINCombDecision:
        """Choose an adjacent combination after a complete cohort.

        ``source_simulation=True`` uses the native simulator's movement rules:
        empirical escalation blockers and the unconditional enrollment stop
        are omitted. The simulation controller applies its own convergence
        stop. Safety monitoring remains active in both modes.
        """
        n, y = _validate_counts(patients, toxicities)
        dose = _validate_dose(current_dose, n.shape)
        i, j = dose[0] - 1, dose[1] - 1
        if n[i, j] < 1:
            raise ValueError("current_dose must have evaluated patients")
        state, posterior = self._state(n, y, eliminated)
        if state[0, 0]:
            return BOINCombDecision("stop_safety", None, state, posterior)
        if (
            not source_simulation
            and self.early_stop_patients is not None
            and n[i, j] >= self.early_stop_patients
        ):
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
                if not source_simulation and (
                    (di and np.any(raw[a, : j + 1] >= self.deescalation_boundary))
                    or (dj and np.any(raw[: i + 1, b] >= self.deescalation_boundary))
                ):
                    continue
                aa, bb = y[a, b] + 0.5, n[a, b] - y[a, b] + 0.5
                score = _interval_probability(
                    aa, bb, self.escalation_boundary, self.deescalation_boundary
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
                score = _interval_probability(
                    aa, bb, self.escalation_boundary, self.deescalation_boundary
                )
                candidates.append((score + n[a, b] * 0.0005, (a, b)))
            action = "deescalate"
        else:
            if state[i, j]:
                return BOINCombDecision("stop_safety", None, state, posterior)
            return BOINCombDecision("stay", dose, state, posterior)
        if not candidates:
            if state[i, j]:
                return BOINCombDecision("stop_safety", None, state, posterior)
            return BOINCombDecision("stay", dose, state, posterior)
        best = max(score for score, _ in candidates)
        tied = [candidate for score, candidate in candidates if score == best]
        generator = np.random.default_rng(rng)
        chosen = tied[int(generator.integers(len(tied)))]
        return BOINCombDecision(action, (chosen[0] + 1, chosen[1] + 1), state, posterior)

    def desirability_table(
        self, patients: ArrayLike, toxicities: ArrayLike, *, eliminated: ArrayLike | None = None
    ) -> FloatArray:
        """Return posterior target-interval scores for every dose cell.

        The BOIN combination guide ranks candidate cells by
        ``Pr(lambda_e < p <= lambda_d | data)`` under a Beta(.5,.5) working
        prior.  Cells in the sticky safety set are assigned ``-inf`` so they
        cannot win a ranking.  The table is useful for inspecting the same
        scores used by adjacent-dose movement. This matrix is not the app's
        enumerated patient-count/y table.
        """
        n, y = _validate_counts(patients, toxicities)
        state, _ = self._state(n, y, eliminated)
        a, b = y + 0.5, n - y + 0.5
        score = np.empty(n.shape, dtype=float)
        for index in np.ndindex(n.shape):
            score[index] = _interval_probability(
                float(a[index]),
                float(b[index]),
                self.escalation_boundary,
                self.deescalation_boundary,
            )
        score[state] = -np.inf
        return _owned(score)

    def select_mtd(
        self,
        patients: ArrayLike,
        toxicities: ArrayLike,
        *,
        eliminated: ArrayLike | None = None,
        mtd_contour: bool = False,
        bound_mtd: bool = False,
        round_selection: bool = True,
    ) -> BOINCombSelection:
        """Select a single MTD or the native row-wise MTD contour.

        Rounded working estimates reproduce the standalone R selector.
        ``round_selection=False`` uses the unrounded estimates used inside
        the native simulator. Returned isotonic means are always unrounded.
        """
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
        source_fit = np.round(fitted, 2) if round_selection else fitted
        admissible = (n > 0) & ~state
        ranked_fit = source_fit + 1e-5 * (np.indices(n.shape).sum(0) + 2)
        if bound_mtd:
            admissible &= ranked_fit <= self.deescalation_boundary
        if not np.any(admissible):
            chosen = None
            contour: tuple[DoseCombination, ...] | None = tuple() if mtd_contour else None
        elif not mtd_contour:
            rank = np.where(
                admissible,
                np.abs(ranked_fit - self.target),
                np.inf,
            )
            col, row = np.unravel_index(np.argmin(rank.T), rank.T.shape)
            chosen, contour = (int(row + 1), int(col + 1)), None
        else:
            selected_by_row: list[int | None] = [None] * n.shape[0]
            for a_raw in range(n.shape[0] - 1, -1, -1):
                a = int(a_raw)
                cols = np.flatnonzero(admissible[a])
                if cols.size == 0:
                    continue
                lower = selected_by_row[a + 1] if a + 1 < n.shape[0] else None
                if lower == n.shape[1] - 1:
                    continue
                selected_col: int = int(cols[np.argmin(np.abs(ranked_fit[a, cols] - self.target))])
                selected_by_row[a] = selected_col
            contour = tuple(
                (a + 1, col + 1) for a, col in enumerate(selected_by_row) if col is not None
            )
            chosen = None
        return BOINCombSelection(chosen, contour, state, _owned(fitted), safety)
