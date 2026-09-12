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
from .boin import BOINDecision, BOINDesign, _owned


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
    arrays = tuple(count(v, name) for v, name in zip(values, ("patients", "toxicities", "assigned"), strict=True))
    if any(a.ndim != 1 for a in arrays) or any(a.shape != arrays[0].shape for a in arrays[1:]):
        raise ValueError("patients, toxicities and assigned must be matching 1D vectors")
    if arrays[0].size < 2 or arrays[0].size > 100 or np.any(arrays[1] > arrays[0]) or np.any(arrays[0] > arrays[2]):
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
    _boin: BOINDesign = field(init=False, repr=False)

    def __post_init__(self) -> None:
        base = BOINDesign(
            self.target,
            elimination_probability=self.elimination_probability,
            extra_safe=self.extra_safe,
            safety_offset=self.safety_offset,
            bound_mtd=self.bound_mtd,
        )
        cap = scalar(self.n_cap, "n_cap")
        if cap != int(cap) or cap < 1:
            raise ValueError("n_cap must be a positive integer")
        stop = None if self.n_stop is None else scalar(self.n_stop, "n_stop")
        if stop is not None and (stop != int(stop) or stop < 1):
            raise ValueError("n_stop must be a positive integer or None")
        object.__setattr__(self, "n_cap", int(cap))
        object.__setattr__(self, "n_stop", None if stop is None else int(stop))
        object.__setattr__(self, "_boin", base)

    @property
    def escalation_boundary(self) -> float:
        return self._boin.escalation_boundary

    @property
    def deescalation_boundary(self) -> float:
        return self._boin.deescalation_boundary

    def _closed(self, patients: np.ndarray, toxicities: np.ndarray) -> np.ndarray:
        """Current empirical backfill closure; this is intentionally not sticky."""
        rate = np.divide(toxicities, patients, out=np.zeros(patients.shape, float), where=patients > 0)
        closed = np.zeros(len(patients), dtype=bool)
        # A dose closes when both its own rate and its adjacent pooled rate
        # exceed lambda_d; closure then applies to the upper suffix.  Use
        # completed outcomes only and reopen on later data.
        for j in range(len(patients) - 1):
            total_n = patients[j] + patients[j + 1]
            own_unsafe = patients[j] > 0 and rate[j] > self.deescalation_boundary
            pooled_unsafe = total_n > 0 and (toxicities[j] + toxicities[j + 1]) / total_n > self.deescalation_boundary
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
        response_observed: ArrayLike | None = None,
    ) -> BFBOINBackfill:
        """Return lower doses eligible for backfill and a highest-dose allocation.

        A lower dose is eligible when its evaluated empirical rate is at or
        below ``lambda_d``; if outcomes from the current escalation cohort are
        ``response_observed`` gates doses whose activity/response has not yet
        been recorded.  Closure is recomputed from evaluated data each call.
        """
        n, y, a = _vectors(patients, toxicities, assigned)
        current_value = scalar(current_dose, "current_dose")
        if current_value != int(current_value):
            raise ValueError("current_dose must be an integer")
        c = int(current_value)
        if not 1 <= c <= len(n):
            raise ValueError("current_dose must be a valid one-based dose")
        closed = self._closed(n, y) | (a >= self.n_cap)
        eligible = np.zeros(len(n), dtype=bool)
        for j in range(c - 1):
            safe = n[j] > 0 and y[j] / n[j] <= self.deescalation_boundary
            eligible[j] = safe and not closed[j]
        if response_observed is not None:
            observed = np.asarray(response_observed)
            if observed.shape != n.shape or observed.dtype != np.bool_:
                raise ValueError("response_observed must be a matching boolean vector")
            eligible &= observed
        else:
            eligible &= n > 0
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
            bf = np.zeros(len(n), dtype=bool)
        else:
            bf = np.asarray(backfilled)
            if bf.shape != n.shape or bf.dtype != np.bool_:
                raise ValueError("backfilled must be a matching boolean vector")
        decision = self._boin.next_dose(n, y, c, eliminated=eliminated)
        eligibility = self.backfill_eligibility(n, y, a, c)
        action, next_dose = decision.action, decision.next_dose
        current = c - 1
        rates = np.divide(y, n, out=np.zeros(n.shape, float), where=n > 0)
        individual = np.where(rates <= self.escalation_boundary, 1, np.where(rates >= self.deescalation_boundary, -1, 0))
        lower_conflict = np.flatnonzero(bf[:current] & ((individual[:current] < 0) | ((individual[:current] == 0) & (individual[current] > 0))))
        # A safety stop/elimination is never replaced by conflict pooling.
        if lower_conflict.size and action != "stop_safety" and not decision.eliminated[current] and decision.next_dose is not None:
            bstar = int(lower_conflict[-1])
            pool_n = np.cumsum(n[bstar : current + 1])
            pool_y = np.cumsum(y[bstar : current + 1])
            q_current = pool_y[-1] / pool_n[-1]
            if q_current <= self.escalation_boundary:
                action, next_dose = "escalate", min(c + 1, len(n))
            elif q_current > self.deescalation_boundary:
                safe = np.flatnonzero(pool_y / pool_n <= self.deescalation_boundary)
                k = bstar + int(safe[-1]) if safe.size else bstar - 1
                action, next_dose = "deescalate", (k + 1 if k >= 0 else None)
            else:
                action, next_dose = "stay", c
        if self.n_stop is not None and action == "stay" and a[current] >= self.n_stop:
            action, next_dose = "stop_precision", None
        return BFBOINDecision(action, next_dose, decision.eliminated, decision.overdose_probability, eligibility.eligible, eligibility.closed)

    def select_mtd(self, patients: ArrayLike, toxicities: ArrayLike, *, eliminated: ArrayLike | None = None):
        """Reuse BOIN's safety-filtered isotonic MTD selection."""
        n, y = count(patients, "patients"), count(toxicities, "toxicities")
        if n.ndim != 1 or n.shape != y.shape:
            raise ValueError("patients and toxicities must be matching 1D vectors")
        return self._boin.select_mtd(n, y, eliminated=eliminated)
