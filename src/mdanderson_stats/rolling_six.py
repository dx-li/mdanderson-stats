"""Rolling Six dose assignment with explicit terminal-dose interpretation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import count, scalar
from .boin import _owned


def _state(
    patients: ArrayLike, toxicities: ArrayLike, pending: ArrayLike, eliminated: ArrayLike | None
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
    n, y, c = (
        count(v, name)
        for v, name in ((patients, "patients"), (toxicities, "toxicities"), (pending, "pending"))
    )
    if n.ndim != 1 or not 2 <= n.size <= 100 or y.shape != n.shape or c.shape != n.shape:
        raise ValueError("require matching vectors for 2..100 doses")
    if np.any((n > 6) | (y + c > n)):
        raise ValueError("require <=6 patients per dose and toxicity+pending <= enrolled")
    excluded = np.zeros(n.size, dtype=bool) if eliminated is None else np.asarray(eliminated)
    if excluded.shape != n.shape or excluded.dtype != np.bool_:
        raise ValueError("eliminated must be a matching boolean vector")
    excluded = np.maximum.accumulate(excluded | (y >= 2))
    return n.astype(np.int64), y.astype(np.int64), c.astype(np.int64), excluded.astype(np.bool_)


@dataclass(frozen=True)
class RollingSixDecision:
    action: str
    next_dose: int | None
    selected_dose: int | None
    eliminated: NDArray[np.bool_]


@dataclass(frozen=True)
class RollingSixSelection:
    dose: int | None
    status: str
    eliminated: NDArray[np.bool_]


@dataclass(frozen=True)
class RollingSixDesign:
    """Allow five DLT-free plus one pending at n=6 unless strict completion is requested."""

    require_complete_before_escalation: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.require_complete_before_escalation, bool):
            raise ValueError("require_complete_before_escalation must be boolean")

    def next_dose(
        self,
        patients: ArrayLike,
        toxicities: ArrayLike,
        pending: ArrayLike,
        current_dose: int,
        *,
        eliminated: ArrayLike | None = None,
    ) -> RollingSixDecision:
        """Recommend one patient's assignment, suspension, or terminal selection.

        Two observed DLTs exclude that dose and all higher doses. Once a dose is
        excluded, accrual at the remaining highest dose fills its six slots and
        waits for complete outcomes before MTD selection. No re-escalation across
        exclusions is permitted. An untreated current dose is valid at trial start.
        """
        n, y, c, excluded = _state(patients, toxicities, pending, eliminated)
        dose = scalar(current_dose, "current_dose")
        if dose != int(dose) or not 1 <= dose <= n.size:
            raise ValueError("current_dose must be a valid one-based dose")
        j = int(dose) - 1
        if excluded[0]:
            return RollingSixDecision("stop_safety", None, None, _owned(excluded))
        highest = int(np.flatnonzero(~excluded)[-1])
        candidate = min(j, highest)
        if candidate == highest:
            if n[candidate] < 6:
                action, next_dose, selected = (
                    ("deescalate" if candidate < j else "stay"),
                    candidate + 1,
                    None,
                )
            elif c[candidate]:
                action, next_dose, selected = "suspend_pending", None, None
            else:
                action = "select_highest" if highest == n.size - 1 else "select_mtd"
                next_dose, selected = None, candidate + 1
        else:
            escalate = (
                (3 <= n[candidate] <= 5 and y[candidate] == 0 and c[candidate] == 0)
                or (n[candidate] == 6 and y[candidate] <= 1 and c[candidate] == 0)
                or (
                    not self.require_complete_before_escalation
                    and n[candidate] == 6
                    and y[candidate] == 0
                    and c[candidate] == 1
                )
            )
            if escalate:
                if n[candidate + 1] == 6:
                    raise ValueError("escalation destination is full; reassess at that dose")
                action, next_dose, selected = "escalate", candidate + 2, None
            elif n[candidate] < 6:
                action, next_dose, selected = "stay", candidate + 1, None
            else:
                action, next_dose, selected = "suspend_pending", None, None
        return RollingSixDecision(action, next_dose, selected, _owned(excluded))

    def select_mtd(
        self, patients: ArrayLike, toxicities: ArrayLike, *, eliminated: ArrayLike | None = None
    ) -> RollingSixSelection:
        """Select from complete data, distinguishing highest-dose recommendation from MTD."""
        n, _, _, excluded = _state(
            patients, toxicities, np.zeros_like(np.asarray(patients)), eliminated
        )
        if excluded[0]:
            return RollingSixSelection(None, "no_safe_dose", _owned(excluded))
        highest = int(np.flatnonzero(~excluded)[-1])
        if n[highest] != 6:
            return RollingSixSelection(None, "inconclusive", _owned(excluded))
        status = "highest_planned_dose" if highest == n.size - 1 else "mtd"
        return RollingSixSelection(highest + 1, status, _owned(excluded))
