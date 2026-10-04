"""Backfill Bayesian optimal interval (BF-BOIN) dose decisions.

This module contains the deterministic decision layer of the MD Anderson
BF-BOIN design.  Calendar-time simulation is deliberately kept separate.
Dose indices are one based; ``patients`` and ``toxicities`` are evaluated
outcomes, while ``assigned`` includes every assignment (including pending
patients) and is used only for the backfill cap.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import count, scalar
from .boin import BOINBoundaryTable, BOINDecision, BOINDesign, BOINSelection, _owned


@dataclass(frozen=True)
class BFBOINBackfill:
    """Backfill allocation and temporary closure state."""

    eligible: NDArray[np.bool_]
    closed: NDArray[np.bool_]
    dose: int | None


@dataclass(frozen=True)
class BFBOINDecision(BOINDecision):
    """BF-BOIN movement, including the doses available for backfill."""

    backfill_eligible: NDArray[np.bool_]
    backfill_closed: NDArray[np.bool_]


def _vectors(*values: ArrayLike) -> tuple[np.ndarray, ...]:
    arrays = tuple(
        count(v, name)
        for v, name in zip(values, ("patients", "toxicities", "assigned"), strict=True)
    )
    if any(a.ndim != 1 for a in arrays) or any(a.shape != arrays[0].shape for a in arrays[1:]):
        raise ValueError("patients, toxicities and assigned must be matching 1D vectors")
    if (
        arrays[0].size < 2
        or arrays[0].size > 100
        or np.any(arrays[1] > arrays[0])
        or np.any(arrays[0] > arrays[2])
        or sum(a.sum() for a in arrays) >= 2**53
    ):
        raise ValueError("require 2..100 doses, toxicities<=patients<=assigned")
    return arrays


@dataclass(frozen=True)
class BFBOINDesign:
    """BF-BOIN design with BOIN safety and final MTD estimation.

    ``n_cap`` closes a lower dose for new backfill at an assigned count of
    ``n_cap`` or more.  ``n_stop`` is an optional precision stop and applies
    only when the current dose action is stay and its assigned count reaches
    the threshold.
    """

    target: float = 0.25
    n_cap: int = 6
    n_stop: int | None = None
    elimination_probability: float = 0.95
    extra_safe: bool = False
    safety_offset: float = 0.05
    bound_mtd: bool = False
    stay_at_one_of_three: bool = False
    deescalate_at_two_of_six: bool = False
    _boin: BOINDesign = field(init=False, repr=False)

    def __post_init__(self) -> None:
        target = scalar(self.target, "target")
        for name in (
            "extra_safe",
            "bound_mtd",
            "stay_at_one_of_three",
            "deescalate_at_two_of_six",
        ):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
        if self.stay_at_one_of_three and not 0.20 <= target <= 0.279:
            raise ValueError("the BF-BOIN 1/3 modification requires target in [0.20,0.279]")
        if self.deescalate_at_two_of_six and not 0.28 <= target <= 0.33:
            raise ValueError("the BF-BOIN 2/6 modification requires target in [0.28,0.33]")
        base = BOINDesign(
            target,
            elimination_probability=self.elimination_probability,
            # The native BF-BOIN guide requires more than three patients for
            # its extra-safe stop.  Apply that BF-specific rule in _state;
            # ordinary BOIN retains its own n>=3 behavior.
            extra_safe=False,
            safety_offset=self.safety_offset,
            bound_mtd=False,
            # Apply the BARD guide modifiers locally so their target ranges do
            # not alter ordinary BOIN's independently documented validation.
            stay_at_one_of_three=False,
            deescalate_at_two_of_six=False,
        )
        cap = scalar(self.n_cap, "n_cap")
        if cap != int(cap) or cap < 1:
            raise ValueError("n_cap must be a positive integer")
        stop = None if self.n_stop is None else scalar(self.n_stop, "n_stop")
        if stop is not None and (stop != int(stop) or stop < 1):
            raise ValueError("n_stop must be a positive integer or None")
        object.__setattr__(self, "n_cap", int(cap))
        object.__setattr__(self, "n_stop", None if stop is None else int(stop))
        object.__setattr__(self, "target", base.target)
        object.__setattr__(self, "extra_safe", bool(self.extra_safe))
        object.__setattr__(self, "bound_mtd", bool(self.bound_mtd))
        object.__setattr__(self, "stay_at_one_of_three", bool(self.stay_at_one_of_three))
        object.__setattr__(self, "deescalate_at_two_of_six", bool(self.deescalate_at_two_of_six))
        object.__setattr__(self, "elimination_probability", base.elimination_probability)
        object.__setattr__(self, "safety_offset", base.safety_offset)
        object.__setattr__(self, "_boin", base)

    def _state(
        self, patients: ArrayLike, toxicities: ArrayLike, eliminated: ArrayLike | None
    ) -> tuple[np.ndarray, np.ndarray, NDArray[np.bool_], np.ndarray]:
        """Return BF-specific cumulative exclusions and safety probabilities."""
        n, y, excluded, posterior = self._boin._state(patients, toxicities, eliminated)
        if (
            self.extra_safe
            and n[0] > 3
            and posterior[0] > self.elimination_probability - self.safety_offset
        ):
            excluded = np.array(excluded, copy=True)
            excluded[0] = True
            excluded = np.maximum.accumulate(excluded)
        return n, y, excluded, posterior

    def boundary_table(self, max_patients: int = 30) -> BOINBoundaryTable:
        """Return BOIN movement/elimination cutoffs and BF extra-safe cutoff."""
        table = self._boin.boundary_table(max_patients)
        escalate = np.array(table.escalate_max, copy=True)
        deescalate = np.array(table.deescalate_min, copy=True)
        if self.stay_at_one_of_three and escalate.size >= 3:
            deescalate[2] = 2
        if self.deescalate_at_two_of_six and escalate.size >= 6:
            escalate[5] = 1
            deescalate[5] = 2
        lowest = np.array(table.lowest_stop_min, copy=True)
        if self.extra_safe:
            extra = self._boin._safety_boundary(
                table.patients, self.elimination_probability - self.safety_offset
            )
            lowest = np.where(
                table.patients > 3,
                np.minimum(table.eliminate_min, extra),
                table.eliminate_min,
            )
        return BOINBoundaryTable(
            table.patients,
            _owned(escalate),
            _owned(deescalate),
            table.eliminate_min,
            _owned(lowest),
        )

    @property
    def escalation_boundary(self) -> float:
        return self._boin.escalation_boundary

    @property
    def deescalation_boundary(self) -> float:
        return self._boin.deescalation_boundary

    def _closed(self, patients: np.ndarray, toxicities: np.ndarray) -> np.ndarray:
        """Current empirical backfill closure; this is intentionally not sticky."""
        rate = np.divide(
            toxicities, patients, out=np.zeros(patients.shape, float), where=patients > 0
        )
        closed: NDArray[np.bool_] = np.zeros(len(patients), dtype=bool)
        # A dose closes when both its own rate and its adjacent pooled rate
        # exceed lambda_d; closure then applies to the upper suffix.  Use
        # completed outcomes only and reopen on later data.
        for j in range(len(patients) - 1):
            total_n = patients[j] + patients[j + 1]
            own_unsafe = patients[j] > 0 and rate[j] > self.deescalation_boundary
            pooled_unsafe = (
                total_n > 0
                and (toxicities[j] + toxicities[j + 1]) / total_n > self.deescalation_boundary
            )
            if own_unsafe and pooled_unsafe:
                closed[j:] = True
        return np.maximum.accumulate(closed)

    def backfill_eligibility(
        self,
        patients: ArrayLike,
        toxicities: ArrayLike,
        assigned: ArrayLike,
        current_dose: int,
        *,
        response_observed: ArrayLike,
        eliminated: ArrayLike | None = None,
    ) -> BFBOINBackfill:
        """Return lower doses eligible for backfill and a highest-dose allocation.

        A lower dose is eligible when its activity is recorded and its
        cumulative evaluated data are not closed. Closure is recomputed from
        evaluated data each call.
        """
        n, y, a = _vectors(patients, toxicities, assigned)
        current_value = scalar(current_dose, "current_dose")
        if current_value != int(current_value):
            raise ValueError("current_dose must be an integer")
        c = int(current_value)
        if not 1 <= c <= len(n):
            raise ValueError("current_dose must be a valid one-based dose")
        observed = np.asarray(response_observed)
        if observed.shape != n.shape or observed.dtype != np.bool_:
            raise ValueError("response_observed must be a matching boolean vector")
        observed = np.maximum.accumulate(observed)
        _, _, safety, _ = self._state(n, y, eliminated)
        closed = self._closed(n, y) | (a >= self.n_cap) | safety
        eligible: NDArray[np.bool_] = np.zeros(len(n), dtype=bool)
        for j in range(c - 1):
            eligible[j] = n[j] > 0 and observed[j] and not closed[j]
        dose = int(np.flatnonzero(eligible)[-1]) + 1 if np.any(eligible) else None
        return BFBOINBackfill(_owned(eligible), _owned(closed), dose)

    def next_dose(
        self,
        patients: ArrayLike,
        toxicities: ArrayLike,
        assigned: ArrayLike,
        current_dose: int,
        *,
        backfilled: ArrayLike | None = None,
        eliminated: ArrayLike | None = None,
        response_observed: ArrayLike | None = None,
    ) -> BFBOINDecision:
        """Apply BOIN movement, pooling conflicts caused by backfill."""
        n, y, a = _vectors(patients, toxicities, assigned)
        current_value = scalar(current_dose, "current_dose")
        if current_value != int(current_value):
            raise ValueError("current_dose must be an integer")
        c = int(current_value)
        if not 1 <= c <= len(n) or n[c - 1] == 0:
            raise ValueError("current_dose must identify a dose with evaluated patients")
        if backfilled is None:
            bf: NDArray[np.bool_] = np.zeros(len(n), dtype=bool)
        else:
            bf = np.asarray(backfilled)
            if bf.shape != n.shape or bf.dtype != np.bool_:
                raise ValueError("backfilled must be a matching boolean vector")
        _, _, excluded, _ = self._state(n, y, eliminated)
        decision = self._boin.next_dose(n, y, c, eliminated=excluded)
        current = c - 1
        action, next_dose = decision.action, decision.next_dose
        if action != "stop_safety" and not decision.eliminated[current]:
            forced_move: int | None = None
            if self.stay_at_one_of_three and n[current] == 3 and y[current] == 1:
                forced_move = 0
            if self.deescalate_at_two_of_six and n[current] == 6:
                forced_move = 1 if y[current] <= 1 else -1
            if forced_move is not None:
                next_index = max(0, min(current + forced_move, len(n) - 1))
                if decision.eliminated[next_index]:
                    next_index = current
                next_dose = next_index + 1
                action = (
                    "escalate"
                    if next_index > current
                    else "deescalate"
                    if next_index < current
                    else "stay"
                )
                decision = BOINDecision(
                    action,
                    next_dose,
                    decision.eliminated,
                    decision.overdose_probability,
                )
        if response_observed is None:
            eligibility = BFBOINBackfill(
                _owned(np.zeros(n.shape, dtype=bool)),
                _owned(self._closed(n, y) | (a >= self.n_cap) | decision.eliminated),
                None,
            )
        else:
            eligibility = self.backfill_eligibility(
                n, y, a, c, response_observed=response_observed, eliminated=eliminated
            )
        action, next_dose = decision.action, decision.next_dose
        current = c - 1
        rates = np.divide(y, n, out=np.zeros(n.shape, float), where=n > 0)
        individual = np.where(
            rates <= self.escalation_boundary,
            1,
            np.where(rates > self.deescalation_boundary, -1, 0),
        )
        if self.stay_at_one_of_three:
            individual[(n == 3) & (y == 1)] = 0
        if self.deescalate_at_two_of_six:
            six = n == 6
            individual[six & (y <= 1)] = 1
            individual[six & (y >= 2)] = -1
        lower_conflict = np.flatnonzero(
            bf[:current]
            & (
                (individual[:current] < 0)
                | ((individual[:current] == 0) & (individual[current] > 0))
            )
        )
        # A safety stop/elimination is never replaced by conflict pooling.
        if lower_conflict.size and action != "stop_safety" and decision.next_dose is not None:
            bstar = int(lower_conflict[-1])
            pool_n: NDArray[np.float64] = np.cumsum(n[bstar : current + 1])
            pool_y: NDArray[np.float64] = np.cumsum(y[bstar : current + 1])
            q_current = pool_y[-1] / pool_n[-1]
            if q_current <= self.escalation_boundary:
                action, next_dose = "escalate", min(c + 1, len(n))
            elif q_current > self.deescalation_boundary:
                safe = np.flatnonzero(pool_y / pool_n <= self.deescalation_boundary)
                k = bstar + int(safe[-1]) if safe.size else bstar - 1
                allowed = (
                    np.flatnonzero(~decision.eliminated[: k + 1]) if k >= 0 else np.empty(0, int)
                )
                destination = int(allowed[-1]) + 1 if allowed.size else 1
                action, next_dose = "deescalate", destination
            else:
                action, next_dose = "stay", c
        if next_dose is not None and decision.eliminated[next_dose - 1]:
            allowed = np.flatnonzero(~decision.eliminated[:next_dose])
            if allowed.size:
                next_dose = int(allowed[-1]) + 1
                action = "stay" if next_dose == c else "deescalate"
            else:
                action, next_dose = "stop_safety", None
        elif action == "escalate" and next_dose is not None and next_dose <= c:
            action, next_dose = "stay", c
        if next_dose == c and action != "stop_safety":
            action = "stay"
        if self.n_stop is not None and action == "stay" and a[current] >= self.n_stop:
            action, next_dose = "stop_precision", None
        return BFBOINDecision(
            action,
            next_dose,
            decision.eliminated,
            decision.overdose_probability,
            eligibility.eligible,
            eligibility.closed,
        )

    def select_mtd(
        self, patients: ArrayLike, toxicities: ArrayLike, *, eliminated: ArrayLike | None = None
    ) -> BOINSelection:
        """Reuse BOIN's safety-filtered isotonic MTD selection."""
        n, y, excluded, _ = self._state(patients, toxicities, eliminated)
        if n.ndim != 1 or n.shape != y.shape:
            raise ValueError("patients and toxicities must be matching 1D vectors")
        result = self._boin.select_mtd(n, y, eliminated=excluded)
        admissible = (n > 0) & ~excluded
        if self.bound_mtd:
            admissible &= result.selection_mean < self.deescalation_boundary
        indices = np.flatnonzero(admissible)
        selected = None
        if indices.size:
            distance = np.abs(result.selection_mean[indices] - self.target)
            tied = indices[np.isclose(distance, distance.min(), rtol=0, atol=1e-14)]
            selected_index = (
                tied[-1] if np.all(result.selection_mean[tied] < self.target) else tied[0]
            )
            selected = int(selected_index) + 1
        return BOINSelection(
            selected,
            _owned(excluded),
            result.isotonic_mean,
            result.selection_mean,
            result.isotonic_interval,
            result.report_overdose_probability,
            result.safety_overdose_probability,
        )
