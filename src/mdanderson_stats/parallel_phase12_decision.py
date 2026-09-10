"""Explicit reproduction of the six-dose C++ posterior decision rules."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite
from .parallel_phase12_model import Phase12ModelFit


def _mask(value: ArrayLike, name: str) -> np.ndarray:
    x = np.asarray(value)
    if x.shape != (6,) or np.any((x != 0) & (x != 1)):
        raise ValueError(f"{name} must have six boolean or 0/1 entries")
    return x.astype(bool)


def _probability(value: ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    x = finite(value, name)
    if x.shape != shape or np.any((x < 0) | (x > 1)):
        raise ValueError(f"{name} must have shape {shape} with values in [0,1]")
    return x


@dataclass(frozen=True)
class Phase12SourceDecision:
    closed: np.ndarray
    suspended: np.ndarray
    probability: FloatArray
    enough_patients: bool
    terminated: bool
    arm_closed: bool
    selected: int | None
    selected_eligible: bool | None
    reason: str


def phase12_source_decision(
    fit: Phase12ModelFit,
    enrolled: ArrayLike,
    *,
    phase_one_admissible: ArrayLike,
    closed: ArrayLike | None = None,
    suspended: ArrayLike = (False,) * 6,
    cohort_size: int = 5,
) -> Phase12SourceDecision:
    """Apply archived EvaluateStoppingRules, including its selection-eligibility quirk.

    Early winners are tested across ALL six doses, including closed/suspended
    candidates and comparators. `selected_eligible` exposes this source behavior.
    This is a source-rule evaluator, not a complete calendar trial controller.
    """
    if not isinstance(fit, Phase12ModelFit):
        raise TypeError("fit must be a Phase12ModelFit")
    initial = _mask(phase_one_admissible, "phase_one_admissible")
    shut = ~initial if closed is None else _mask(closed, "closed")
    if np.any(~initial & ~shut):
        raise ValueError("doses inadmissible after phase I must remain closed")
    paused = _mask(suspended, "suspended")
    n = count(enrolled, "enrolled")
    if n.shape != (6,):
        raise ValueError("enrolled must have six nonnegative integer counts")
    size = count(cohort_size, "cohort_size")
    if size.ndim or size < 1:
        raise ValueError("cohort_size must be a positive integer")
    weight = _probability(fit.reference_superiority, (6,), "reference_superiority")
    efficacy = _probability(fit.efficacy_probability, (6,), "efficacy_probability")
    pairwise = _probability(fit.pairwise_superiority, (6, 6), "pairwise_superiority")
    toxicity = _probability(fit.toxicity_probability, (6,), "toxicity_probability")
    shut |= toxicity > 0.95
    arm_closed = bool(np.all(shut))
    total = float(weight[~shut].sum())
    # The C++ implementation divides by zero with no open weight. Preserve
    # the defined closure result, but do not fabricate undefined Ri values.
    if not arm_closed and total <= 0:
        raise ArithmeticError("open-dose reference weights sum to zero")
    normalized = weight / total if total > 0 else np.zeros(6)
    eligible_for_update = initial & ~shut
    paused[eligible_for_update] = normalized[eligible_for_update] < 0.01
    enough = bool(np.count_nonzero(n >= size) >= 4 or np.all((n >= size) | shut))
    best_efficacy = float(np.max(efficacy[initial])) if np.any(initial) else 0.0
    terminated = False
    selected = None
    reason = "continue enrollment"
    if best_efficacy < 0.05 and enough:
        terminated, arm_closed, reason = True, True, "futility"
    else:
        for dose in range(6):
            competitors = np.arange(6) != dose
            if efficacy[dose] > 0.90 and np.all(pairwise[dose, competitors] > 0.80) and enough:
                selected = dose
                terminated, reason = True, "source efficacy selection"
                break
    available = ~shut & ~paused
    if not np.any(available):
        arm_closed = True
        if not terminated:
            reason = "no open unsuspended doses"
    probability = np.zeros(6)
    if not terminated and not arm_closed:
        denominator = float(weight[available].sum())
        if denominator <= 0:
            raise ArithmeticError("unsuspended reference weights sum to zero")
        probability[available] = weight[available] / denominator
    masks = [np.frombuffer(x.tobytes(), dtype=bool) for x in (shut, paused)]
    return Phase12SourceDecision(
        masks[0],
        masks[1],
        _freeze(probability),
        enough,
        terminated,
        arm_closed,
        selected,
        None if selected is None else bool(available[selected]),
        reason,
    )


def phase12_source_final_selection(
    fit: Phase12ModelFit, *, closed: ArrayLike, suspended: ArrayLike
) -> int | None:
    """Archived PickWinner: open, unsuspended maximum future probability > .90.

    The source uses stopping-rule slot 2 (.90), not its separately configured
    but unused future-study slot 4 (.80). Native first-dose tie order is retained.
    """
    if not isinstance(fit, Phase12ModelFit):
        raise TypeError("fit must be a Phase12ModelFit")
    shut, paused = _mask(closed, "closed"), _mask(suspended, "suspended")
    future = _probability(fit.future_probability, (6,), "future_probability")
    candidates = np.flatnonzero(~shut & ~paused & (future > 0.90))
    return int(candidates[np.argmax(future[candidates])]) if len(candidates) else None
