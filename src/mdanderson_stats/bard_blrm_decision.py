"""Decision rules for the published BARD BF-BLRM stage-one design.

This module consumes posterior summaries produced elsewhere. It implements
the paper's strict overdose eligibility and one-dose movement rules, without
BF-BOIN's empirical conflict pooling. When posterior target probabilities
tie, this Python API chooses the lowest one-based dose index. A one-step move
can itself land on a dose whose POD is not below ``eta``; that fact is exposed
in the result rather than silently skipping an intermediate level.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import count, finite, scalar


def _readonly(value: ArrayLike, dtype: np.dtype | type = np.float64) -> NDArray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _probabilities(
    ptt: ArrayLike, pod: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate dose-aligned posterior probabilities before copying results."""
    ptt_shape, pod_shape = np.shape(ptt), np.shape(pod)
    if len(ptt_shape) != 1 or ptt_shape != pod_shape or not 1 <= ptt_shape[0] <= 100:
        raise ValueError("ptt and pod must be matching vectors for 1..100 doses")
    if np.iscomplexobj(ptt) or np.iscomplexobj(pod):
        raise ValueError("ptt and pod must be real-valued")
    target = finite(ptt, "ptt")
    overdose = finite(pod, "pod")
    if np.any((target < 0) | (target > 1)) or np.any((overdose < 0) | (overdose > 1)):
        raise ValueError("ptt and pod values must lie in [0,1]")
    if np.any(target + overdose > 1 + 32 * np.finfo(float).eps):
        raise ValueError("ptt + pod cannot exceed one, apart from roundoff")
    if np.any(overdose[1:] < overdose[:-1] - 32 * np.finfo(float).eps):
        raise ValueError("pod must be nondecreasing over ordered dose levels")
    return target, overdose


def _eta(value: float) -> float:
    result = scalar(value, "eta")
    if not 0 <= result <= 1:
        raise ValueError("eta must lie in [0,1]")
    return result


@dataclass(frozen=True)
class BARDBLRMDecision:
    """Model-based one-step escalation decision (dose indices are one-based)."""

    ptt: NDArray[np.float64]
    pod: NDArray[np.float64]
    safe: NDArray[np.bool_]
    target_dose: int | None
    tied_target_doses: tuple[int, ...]
    current_dose: int
    next_dose: int | None
    next_dose_safe: bool | None
    action: str


def bard_blrm_next_dose(
    ptt: ArrayLike,
    pod: ArrayLike,
    current_dose: int,
    *,
    eta: float = 0.30,
) -> BARDBLRMDecision:
    """Choose the safe dose with greatest PTT, then move at most one level.

    ``POD < eta`` is strict. If all PODs are strictly greater than ``eta``
    the result is ``stop_all_overdose``. If equality leaves no safe dose but
    not all doses are over the cutoff, the result is
    ``no_eligible_safe_dose``; this is deliberately distinct from stopping
    for all doses being over-toxic.
    """
    target, overdose = _probabilities(ptt, pod)
    cutoff = _eta(eta)
    current_value = scalar(current_dose, "current_dose")
    if current_value != int(current_value) or not 1 <= current_value <= target.size:
        raise ValueError("current_dose must be a valid one-based dose index")
    current = int(current_value)
    safe = overdose < cutoff
    chosen: int | None = None
    tied: tuple[int, ...] = ()
    next_dose: int | None = None
    next_safe: bool | None = None
    if np.any(safe):
        best = float(np.max(target[safe]))
        tied = tuple(int(i) + 1 for i in np.flatnonzero(safe & (target == best)))
        chosen = tied[0]
        next_dose = current + int(np.sign(chosen - current))
        next_safe = bool(safe[next_dose - 1])
        action = (
            "stay"
            if next_dose == current
            else ("escalate" if next_dose > current else "deescalate")
        )
    elif np.all(overdose > cutoff):
        action = "stop_all_overdose"
    else:
        action = "no_eligible_safe_dose"
    return BARDBLRMDecision(
        _readonly(target),
        _readonly(overdose),
        _readonly(safe, bool),
        chosen,
        tied,
        current,
        next_dose,
        next_safe,
        action,
    )


@dataclass(frozen=True)
class BARDBLRMBackfill:
    """Backfill openings from a complete-evaluable snapshot."""

    eligible: NDArray[np.bool_]
    closed: NDArray[np.bool_]
    selected_dose: int | None


def bard_blrm_backfill(
    pod: ArrayLike,
    responses: ArrayLike,
    evaluable: ArrayLike,
    current_dose: int,
    *,
    eta: float = 0.30,
    cap: int,
    assigned: ArrayLike | None = None,
) -> BARDBLRMBackfill:
    """Return lower-dose backfill eligibility and highest eligible dose.

    ``responses`` counts observed responses and may include patients whose
    toxicity assessment is still pending. ``evaluable`` counts completed
    toxicity outcomes and controls the cap; ``assigned`` (defaulting to
    ``evaluable`` for a complete-outcome snapshot) bounds both. A dose is
    closed when its own POD is at least ``eta`` or its evaluable count reaches
    ``cap``. Dose ``b`` is active for backfill only when ``b < current_dose``
    and a response has occurred at ``b`` or any lower dose.
    """
    dose_shape = np.shape(pod)
    if len(dose_shape) != 1 or not 1 <= dose_shape[0] <= 100:
        raise ValueError("pod must be a vector for 1..100 doses")
    overdose = finite(pod, "pod")
    if np.any((overdose < 0) | (overdose > 1)) or np.any(
        overdose[1:] < overdose[:-1] - 32 * np.finfo(float).eps
    ):
        raise ValueError("pod must lie in [0,1] and be nondecreasing")
    response_shape, assessed_shape = np.shape(responses), np.shape(evaluable)
    assigned_shape = assessed_shape if assigned is None else np.shape(assigned)
    if (
        response_shape != overdose.shape
        or assessed_shape != overdose.shape
        or assigned_shape != overdose.shape
    ):
        raise ValueError("responses, evaluable and assigned must match pod as dose vectors")
    if (
        np.iscomplexobj(responses)
        or np.iscomplexobj(evaluable)
        or (assigned is not None and np.iscomplexobj(assigned))
    ):
        raise ValueError("responses, evaluable and assigned must be real-valued")
    response = count(responses, "responses")
    assessed = count(evaluable, "evaluable")
    enrolled = assessed if assigned is None else count(assigned, "assigned")
    if np.any(response > enrolled) or np.any(assessed > enrolled):
        raise ValueError("responses and evaluable counts cannot exceed assigned counts")
    cutoff = _eta(eta)
    cap_value = scalar(cap, "cap")
    if cap_value != int(cap_value) or cap_value < 1:
        raise ValueError("cap must be a positive integer")
    current_value = scalar(current_dose, "current_dose")
    if current_value != int(current_value) or not 1 <= current_value <= overdose.size:
        raise ValueError("current_dose must be a valid one-based dose index")
    closed = (overdose >= cutoff) | (assessed >= cap_value)
    response_seen = np.cumsum(response) > 0
    eligible = np.zeros(overdose.shape, dtype=bool)
    upper = int(current_value) - 1
    eligible[:upper] = response_seen[:upper] & ~closed[:upper]
    selected = int(np.flatnonzero(eligible)[-1]) + 1 if np.any(eligible) else None
    return BARDBLRMBackfill(_readonly(eligible, bool), _readonly(closed, bool), selected)


@dataclass(frozen=True)
class BARDBLRMSelection:
    """Final stage-one MTD selection from all dose-level observations."""

    safe: NDArray[np.bool_]
    eligible: NDArray[np.bool_]
    treated: NDArray[np.int64]
    selected_dose: int | None
    tied_doses: tuple[int, ...]
    status: str


def bard_blrm_select_mtd(
    ptt: ArrayLike,
    pod: ArrayLike,
    treated: ArrayLike,
    *,
    eta: float = 0.30,
    minimum_treated: int = 6,
) -> BARDBLRMSelection:
    """Select highest-PTT safe dose with enough patients treated.

    ``treated`` is the cumulative number assigned to each dose (escalation
    plus backfill); posterior inputs must incorporate all complete-evaluable
    toxicity data at the final look. Ties choose the lowest dose index.
    """
    target, overdose = _probabilities(ptt, pod)
    if np.shape(treated) != target.shape:
        raise ValueError("treated must match the dose vectors with counts in [0,1_000_000]")
    if np.iscomplexobj(treated):
        raise ValueError("treated must be real-valued")
    n = count(treated, "treated")
    if np.any(n > 1_000_000):
        raise ValueError("treated counts must lie in [0,1_000_000]")
    cutoff = _eta(eta)
    minimum_value = scalar(minimum_treated, "minimum_treated")
    if minimum_value != int(minimum_value) or minimum_value < 1:
        raise ValueError("minimum_treated must be a positive integer")
    safe = overdose < cutoff
    eligible = safe & (n >= minimum_value)
    selected: int | None = None
    tied: tuple[int, ...] = ()
    if np.any(eligible):
        best = float(np.max(target[eligible]))
        tied = tuple(int(i) + 1 for i in np.flatnonzero(eligible & (target == best)))
        selected = tied[0]
        status = "selected"
    elif np.all(overdose > cutoff):
        status = "all_doses_overdose"
    elif not np.any(safe):
        status = "no_eligible_safe_dose"
    else:
        status = "no_safe_dose_meets_minimum_treated"
    return BARDBLRMSelection(
        _readonly(safe, bool),
        _readonly(eligible, bool),
        _readonly(n, np.int64),
        selected,
        tied,
        status,
    )
