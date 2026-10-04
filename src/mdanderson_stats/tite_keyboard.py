"""Effective follow-up and interim TITE-Keyboard decisions for pending outcomes."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite, scalar
from .boin import _owned
from .keyboard import KeyboardDesign, KeyboardPosterior


def toxicity_followup_weights(
    followup: ArrayLike,
    window: float,
    *,
    trimester_probabilities: ArrayLike | None = None,
) -> FloatArray:
    """Conditional time-to-DLT CDF under uniform or three-piece uniform timing.

    Three prior masses are conditional on eventually having a DLT within the
    window. They are not marginal DLT rates. Follow-up/window use the same units.
    """
    duration = scalar(window, "window")
    if duration <= 0:
        raise ValueError("window must be positive")
    followup = finite(followup, "followup")
    if duration <= 0 or np.any((followup < 0) | (followup > duration)):
        raise ValueError("require window > 0 and followup in [0,window]")
    fraction = followup / duration
    if trimester_probabilities is None:
        return _owned(fraction)
    prior = finite(trimester_probabilities, "trimester_probabilities")
    if (
        prior.shape != (3,)
        or np.any(prior < 0)
        or not np.isclose(prior.sum(), 1, rtol=0, atol=1e-14)
    ):
        raise ValueError(
            "trimester_probabilities must contain three nonnegative masses summing to 1"
        )
    prior = prior / prior.sum()
    covered = np.clip(3 * fraction[..., None] - np.arange(3), 0, 1)
    return _owned(np.sum(covered * prior, axis=-1))


@dataclass(frozen=True)
class TITEEffectiveSampleSize:
    effective_sample_size: FloatArray
    pending_weights: FloatArray


def tite_effective_sample_size(
    nonpending: ArrayLike,
    pending_followup: ArrayLike,
    window: float,
    *,
    trimester_probabilities: ArrayLike | None = None,
) -> TITEEffectiveSampleSize:
    """Observed outcomes plus summed pending weights; pending patients are last axis.

    The observed count includes known DLTs and completed DLT-free assessments.
    Arrays broadcast across scenarios, each having the same number of pending
    patients. An empty last axis represents no pending outcomes.
    """
    weights = toxicity_followup_weights(
        pending_followup, window, trimester_probabilities=trimester_probabilities
    )
    if weights.ndim == 0:
        raise ValueError("pending_followup must have a patient axis")
    observed, total_weight = np.broadcast_arrays(
        count(nonpending, "nonpending"), weights.sum(axis=-1)
    )
    if np.any(observed + weights.shape[-1] > 200):
        raise ValueError("require at most 200 patients per scenario")
    return TITEEffectiveSampleSize(_owned(observed + total_weight), weights)


@dataclass(frozen=True)
class TITEKeyboardDecision:
    action: str
    next_dose: int | None
    effective_sample_size: FloatArray
    pending_count: NDArray[np.int64]
    eliminated: NDArray[np.bool_]
    safety_overdose_probability: FloatArray
    posterior: KeyboardPosterior


def tite_keyboard_decision(
    design: KeyboardDesign,
    patients: ArrayLike,
    toxicities: ArrayLike,
    pending_followup: Sequence[ArrayLike],
    current_dose: int,
    window: float,
    *,
    trimester_probabilities: ArrayLike | None = None,
    pending_weights: Sequence[ArrayLike] | None = None,
    pending_fraction_limit: float | None = 0.5,
    eliminated: ArrayLike | None = None,
) -> TITEKeyboardDecision:
    """Interim dose decision using Beta(y+1, effective_n-y+1).

    Patients include pending outcomes; toxicities count observed DLTs only.
    Supply one 1D pending-follow-up vector per dose, using [] for none. Safety
    uses enrolled n (pending treated as non-DLT), as specified in the paper.
    Final MTD selection requires all pending outcomes to be resolved first.
    """
    if not isinstance(design, KeyboardDesign):
        raise ValueError("design must be a KeyboardDesign")
    n, y, excluded, safety = design._shared._state(patients, toxicities, eliminated)
    if n.sum() > 200 or len(pending_followup) != len(n):
        raise ValueError("require at most 200 patients and one pending-follow-up vector per dose")
    dose = scalar(current_dose, "current_dose")
    if dose != int(dose) or not 1 <= dose <= len(n):
        raise ValueError("current_dose must be a valid one-based dose")
    j = int(dose) - 1
    if n[j] < 1:
        raise ValueError("current dose must have enrolled patients")
    duration = scalar(window, "window")
    if duration <= 0:
        raise ValueError("window must be positive")
    pending = np.zeros(len(n), dtype=np.int64)
    effective = np.zeros(len(n))
    if pending_weights is not None and trimester_probabilities is not None:
        raise ValueError("pending_weights and trimester_probabilities cannot both be supplied")
    weight_rows: list[FloatArray] | None = None
    if pending_weights is not None:
        if not isinstance(pending_weights, Sequence) or isinstance(pending_weights, (str, bytes)):
            raise ValueError("pending_weights must contain one vector per dose")
        if len(pending_weights) != len(n):
            raise ValueError("pending_weights must contain one vector per dose")
        weight_rows = [
            _bounded_decision_vector(value, f"pending_weights[{index}]")
            for index, value in enumerate(pending_weights)
        ]
    for level, times in enumerate(pending_followup):
        times = _bounded_decision_vector(times, f"pending_followup[{level}]")
        if np.any((times < 0) | (times >= duration)):
            raise ValueError("pending follow-up must be 1D and strictly shorter than window")
        pending[level] = times.size
        if pending[level] > n[level] - y[level]:
            raise ValueError(
                "pending patients cannot exceed enrolled patients without observed DLT"
            )
        if pending_weights is None:
            ess = tite_effective_sample_size(
                n[level] - pending[level],
                times,
                duration,
                trimester_probabilities=trimester_probabilities,
            )
            effective[level] = ess.effective_sample_size
        else:
            assert weight_rows is not None
            weights = weight_rows[level]
            if weights.shape != times.shape or np.any((weights < 0) | (weights > 1)):
                raise ValueError("each pending weight vector must match follow-up and lie in [0,1]")
            effective[level] = n[level] - pending[level] + np.sum(weights)
    limit = (
        None
        if pending_fraction_limit is None
        else scalar(pending_fraction_limit, "pending_fraction_limit")
    )
    if limit is not None and not 0 < limit <= 0.65:
        raise ValueError("pending_fraction_limit must be in (0,.65] or None")
    posterior = design._posterior_effective(np.asarray(effective[j]), np.asarray(y[j]))
    move = int(posterior.move)
    next_j = max(0, min(j + move, len(n) - 1))
    if excluded[next_j]:
        next_j = j
    if excluded[0]:
        action, next_dose = "stop_safety", None
    elif excluded[j]:
        action, next_dose = "deescalate", int(np.flatnonzero(~excluded)[-1]) + 1
    elif limit is not None and pending[j] / n[j] > limit:
        action, next_dose = "suspend_pending", None
    elif next_j > j and n[j] - pending[j] < 2:
        action, next_dose = "suspend_escalation", None
    elif design.early_stop_patients is not None and n[j] >= design.early_stop_patients:
        action, next_dose = "stop_enrollment", None
    else:
        action = "escalate" if next_j > j else "deescalate" if next_j < j else "stay"
        next_dose = next_j + 1
    return TITEKeyboardDecision(
        action,
        next_dose,
        _owned(effective),
        _owned(pending),
        _owned(excluded),
        _owned(safety),
        posterior,
    )


def _bounded_decision_vector(value: ArrayLike, name: str) -> FloatArray:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.size > 200 or value.dtype.kind not in "biuf":
            raise ValueError(f"{name} must be a bounded one-dimensional real vector")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) > 200 or any(
            isinstance(item, (list, tuple, np.ndarray)) or np.ndim(item) != 0 for item in value
        ):
            raise ValueError(f"{name} must be a bounded one-dimensional real vector")
    else:
        raise ValueError(f"{name} must be a bounded one-dimensional real vector")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = finite(value, name)
    return np.array(result, dtype=float, copy=True)
