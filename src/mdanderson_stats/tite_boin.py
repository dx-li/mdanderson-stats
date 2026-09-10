"""Single-mean imputation and interim conduct for TITE-BOIN."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite, scalar
from .boin import BOINDesign, _owned
from .tite_keyboard import toxicity_followup_weights


@dataclass(frozen=True)
class TITEBOINEstimate:
    posterior_mean: FloatArray
    imputed_toxicities: FloatArray
    estimated_rate: FloatArray
    escalate_stft: FloatArray
    deescalate_stft: FloatArray
    move: NDArray[np.int64]


def tite_boin_estimate(
    design: BOINDesign,
    patients: ArrayLike,
    toxicities: ArrayLike,
    pending: ArrayLike,
    stft: ArrayLike,
) -> TITEBOINEstimate:
    """Broadcast TITE-BOIN imputation and coherent ordinary dose transitions.

    stft is the sum of pending patients' conditional time-to-DLT CDF weights.
    The conservative imputed rate can exceed one; it is a decision statistic,
    not a fitted probability. Safety and accrual gates are applied by conduct.
    """
    if not isinstance(design, BOINDesign):
        raise ValueError("design must be a BOINDesign")
    n, y, c, follow = np.broadcast_arrays(
        count(patients, "patients"),
        count(toxicities, "toxicities"),
        count(pending, "pending"),
        finite(stft, "stft"),
    )
    if np.any((n < 1) | (n > 200) | (y + c > n) | (follow < 0) | (follow > c)):
        raise ValueError(
            "require 1..200 enrolled, toxicity+pending <= enrolled and 0<=STFT<=pending"
        )
    observed = n - c
    alpha = design.target / 2
    odds = (y + alpha) / (observed - y + 1 - alpha)
    posterior = (y + alpha) / (observed + 1)
    imputed = y + odds * (c - follow)
    rate = imputed / n
    observed_rate = y / n
    escalation = np.where(
        observed_rate < design.target, c - (n * design.escalation_boundary - y) / odds, np.inf
    )
    deescalation = np.where(
        observed_rate > design.target, c - (n * design.deescalation_boundary - y) / odds, -np.inf
    )
    move = np.where(
        (observed_rate < design.target) & (rate <= design.escalation_boundary),
        1,
        np.where((observed_rate > design.target) & (rate >= design.deescalation_boundary), -1, 0),
    ).astype(np.int64)
    if design.stay_at_one_of_three:
        modified = (n == 3) & (y == 1) & (c == 0)
        move = np.where(modified, 0, move)
        escalation = np.where(modified, np.inf, escalation)
        deescalation = np.where(modified, -np.inf, deescalation)
    if design.deescalate_at_two_of_six:
        modified = (n == 6) & (y == 2)
        move = np.where(modified, -1, move)
        escalation = np.where(modified, np.inf, escalation)
        deescalation = np.where(modified, np.inf, deescalation)
    return TITEBOINEstimate(
        *map(_owned, (posterior, imputed, rate, escalation, deescalation, move))
    )


@dataclass(frozen=True)
class TITEBOINDecision:
    action: str
    next_dose: int | None
    pending_count: NDArray[np.int64]
    standardized_followup: FloatArray
    eliminated: NDArray[np.bool_]
    safety_overdose_probability: FloatArray
    estimate: TITEBOINEstimate


def tite_boin_decision(
    design: BOINDesign,
    patients: ArrayLike,
    toxicities: ArrayLike,
    pending_followup: Sequence[ArrayLike],
    current_dose: int,
    window: float,
    *,
    trimester_probabilities: ArrayLike | None = None,
    minimum_complete_fraction: float = 0.51,
    minimum_pending_followup: float = 0.25,
    eliminated: ArrayLike | None = None,
) -> TITEBOINDecision:
    """Apply imputation, enrolled-count safety, and the app's two suspension rules.

    Provide a one-dimensional pending-follow-up vector per dose. Complete means
    the toxicity outcome is ascertained, including an already observed DLT.
    Final MTD selection uses BOINDesign.select_mtd after pending outcomes resolve.
    """
    if not isinstance(design, BOINDesign):
        raise ValueError("design must be a BOINDesign")
    n, y, excluded, safety = design._state(patients, toxicities, eliminated)
    dose, duration = scalar(current_dose, "current_dose"), scalar(window, "window")
    complete = scalar(minimum_complete_fraction, "minimum_complete_fraction")
    minimum = scalar(minimum_pending_followup, "minimum_pending_followup")
    if not 0.25 <= complete <= 1 or not 0 <= minimum <= 1:
        raise ValueError("require completion fraction in [.25,1] and minimum follow-up in [0,1]")
    if dose != int(dose) or not 1 <= dose <= len(n) or duration <= 0:
        raise ValueError("require a valid one-based current dose and positive window")
    j = int(dose) - 1
    if n.sum() > 200 or n[j] == 0 or len(pending_followup) != len(n):
        raise ValueError(
            "require treated current dose, <=200 patients and a follow-up vector per dose"
        )
    pending = np.zeros(len(n), dtype=np.int64)
    stft = np.zeros(len(n))
    shortest = 1.0
    for level, times in enumerate(pending_followup):
        times = finite(times, "pending_followup")
        if times.ndim != 1 or np.any(times >= duration) or times.size > n[level] - y[level]:
            raise ValueError("invalid pending count or follow-up outside [0,window)")
        weights = toxicity_followup_weights(
            times, duration, trimester_probabilities=trimester_probabilities
        )
        pending[level] = times.size
        stft[level] = weights.sum()
        if level == j and times.size:
            shortest = float(times.min() / duration)
    estimate = tite_boin_estimate(design, n[j], y[j], pending[j], stft[j])
    next_j = max(0, min(j + int(estimate.move), len(n) - 1))
    if excluded[next_j]:
        next_j = j
    modified_deescalation = design.deescalate_at_two_of_six and n[j] == 6 and y[j] == 2
    if excluded[0]:
        action, next_dose = "stop_safety", None
    elif excluded[j]:
        action, next_dose = "deescalate", int(np.flatnonzero(~excluded)[-1]) + 1
    elif (
        (n[j] - pending[j]) / n[j] < complete
        and y[j] / n[j] < design.deescalation_boundary
        and not modified_deescalation
    ):
        action, next_dose = "suspend_pending", None
    elif next_j > j and shortest < minimum:
        action, next_dose = "suspend_followup", None
    elif (
        next_j == j
        and design.early_stop_patients is not None
        and n[j] >= design.early_stop_patients
    ):
        action, next_dose = "stop_enrollment", None
    else:
        action = "escalate" if next_j > j else "deescalate" if next_j < j else "stay"
        next_dose = next_j + 1
    return TITEBOINDecision(
        action, next_dose, _owned(pending), _owned(stft), _owned(excluded), _owned(safety), estimate
    )
