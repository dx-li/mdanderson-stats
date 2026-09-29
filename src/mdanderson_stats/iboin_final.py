"""Final MTD estimation for iBOIN complete-outcome counts."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import isotonic_regression

from ._validation import FloatArray, finite
from .iboin import IBOINDesign
from .iboin_trial import IBOINTrialReplay


@dataclass(frozen=True)
class IBOINSelection:
    """Auditable raw rates, isotonic fit, exclusions and final selected dose."""

    selected_dose: int | None
    raw_rate: FloatArray
    isotonic_rate: FloatArray
    isotonic_weights: FloatArray
    prior_ess_used: NDArray[np.int64]
    fit_mask: NDArray[np.bool_]
    candidate_mask: NDArray[np.bool_]
    eligible: NDArray[np.bool_]
    eliminated: NDArray[np.bool_]
    deescalation_boundary: FloatArray
    safety_stopped: bool
    reason: str
    prior_mode: str
    weight_mode: str


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _validate_selection_options(
    design: IBOINDesign,
    *,
    prior_mode: Literal["none", "original", "effective"],
    isotonic_weights: Literal["patients", "effective", "equal"] | ArrayLike,
    eligible_doses: ArrayLike | None,
    tie_policy: Literal["lowest", "highest"],
    enforce_deescalation_boundary: bool,
    require_all_custom_weights: bool = False,
) -> tuple[str, FloatArray | None, NDArray[np.bool_]]:
    """Validate reusable selector policy before data generation or fitting."""
    if not isinstance(design, IBOINDesign):
        raise TypeError("design must be an IBOINDesign")
    if prior_mode not in ("none", "original", "effective"):
        raise ValueError("prior_mode must be none, original, or effective")
    if tie_policy not in ("lowest", "highest"):
        raise ValueError("tie_policy must be lowest or highest")
    if not isinstance(enforce_deescalation_boundary, (bool, np.bool_)):
        raise ValueError("enforce_deescalation_boundary must be boolean")
    n_doses = design.skeleton.size
    if eligible_doses is None:
        requested = np.ones(n_doses, dtype=bool)
    else:
        requested = np.asarray(eligible_doses)
        if requested.shape != (n_doses,) or requested.dtype != np.bool_:
            raise ValueError("eligible_doses must be a matching boolean vector")
        requested = np.array(requested, copy=True)
    if isinstance(isotonic_weights, str):
        if isotonic_weights not in ("patients", "effective", "equal"):
            raise ValueError("isotonic_weights must be patients, effective, equal, or a vector")
        weight_mode = isotonic_weights
        caller_weights = None
    else:
        weight_mode = "caller"
        if np.iscomplexobj(isotonic_weights):
            raise ValueError("custom isotonic_weights must be real")
        caller_weights = finite(isotonic_weights, "isotonic_weights")
        if caller_weights.shape != (n_doses,) or np.any(caller_weights < 0):
            raise ValueError("custom isotonic_weights must be a nonnegative vector matching doses")
        if require_all_custom_weights and np.any(caller_weights <= 0):
            raise ValueError("simulation custom isotonic_weights must be positive at every dose")
    return weight_mode, caller_weights, requested


def _select(
    design: IBOINDesign,
    patients: ArrayLike,
    toxicities: ArrayLike,
    *,
    prior_mode: Literal["none", "original", "effective"],
    isotonic_weights: Literal["patients", "effective", "equal"] | ArrayLike,
    eliminated: ArrayLike | None,
    eligible_doses: ArrayLike | None,
    tie_policy: Literal["lowest", "highest"],
    enforce_deescalation_boundary: bool,
    terminal_safety_stop: bool,
) -> IBOINSelection:
    weight_mode, caller_weights, requested = _validate_selection_options(
        design,
        prior_mode=prior_mode,
        isotonic_weights=isotonic_weights,
        eligible_doses=eligible_doses,
        tie_policy=tie_policy,
        enforce_deescalation_boundary=enforce_deescalation_boundary,
    )
    if np.iscomplexobj(patients) or np.iscomplexobj(toxicities):
        raise ValueError("patient and toxicity counts must be real")
    if not isinstance(design, IBOINDesign):
        raise TypeError("design must be an IBOINDesign")
    n_float, y_float, _, posterior = design._boin._state(patients, toxicities, eliminated)
    n_doses = design.skeleton.size
    if n_float.shape != (n_doses,):
        raise ValueError("patient/toxicity counts must match the design skeleton")
    n, y = n_float.astype(np.int64), y_float.astype(np.int64)
    if eliminated is None:
        inherited = np.zeros(n_doses, dtype=bool)
    else:
        inherited = np.asarray(eliminated)
        if inherited.shape != (n_doses,) or inherited.dtype != np.bool_:
            raise ValueError("eliminated must be a matching boolean vector")

    if prior_mode == "none":
        used_ess = np.zeros(n_doses, dtype=np.int64)
    elif prior_mode == "original":
        used_ess = np.asarray(design.prior_ess, dtype=np.int64)
    else:
        used_ess = np.asarray(design.effective_prior_ess, dtype=np.int64)
    fit_mask = n > 0
    raw_rate = np.full(n_doses, np.nan)
    raw_rate[fit_mask] = (y[fit_mask] + used_ess[fit_mask] * design.skeleton[fit_mask]) / (
        n[fit_mask] + used_ess[fit_mask]
    )

    if caller_weights is not None:
        weights = caller_weights
        if np.any(weights[fit_mask] <= 0):
            raise ValueError("custom isotonic_weights must be positive at treated doses")
    elif weight_mode == "equal":
        weights = np.ones(n_doses)
    elif weight_mode == "patients":
        weights = n.astype(np.float64)
    else:
        weights = n.astype(np.float64) + used_ess
    if np.any(weights[fit_mask] <= 0) or not np.all(np.isfinite(weights)):
        raise ValueError("all treated-dose isotonic weights must be finite and positive")

    fitted = np.full(n_doses, np.nan)
    if np.any(fit_mask):
        fit_weights = weights[fit_mask] / np.max(weights[fit_mask])
        if np.any(fit_weights <= 0):
            raise ArithmeticError("relative isotonic weights are not representable")
        fitted[fit_mask] = isotonic_regression(
            raw_rate[fit_mask], weights=fit_weights, increasing=True
        ).x

    current_safety = (n >= 3) & (posterior > design.elimination_probability)
    excluded = np.maximum.accumulate(inherited | current_safety)
    extra_safe = bool(
        design.extra_safe
        and n[0] > 3
        and posterior[0] > design.elimination_probability - design.safety_offset
    )
    safety_stopped = bool(terminal_safety_stop or excluded[0] or extra_safe)
    candidates = fit_mask & ~excluded & requested
    eligible = candidates.copy()
    boundary = np.full(n_doses, np.nan)
    if enforce_deescalation_boundary:
        for dose_index in np.flatnonzero(fit_mask):
            table = design.boundaries([int(n[dose_index])])
            boundary[dose_index] = table.deescalation[dose_index, 0]
        eligible &= fitted <= boundary

    chosen: int | None = None
    if safety_stopped:
        eligible[:] = False
        reason = "safety_stop"
    elif not np.any(candidates):
        reason = "no_eligible_dose"
    elif enforce_deescalation_boundary and not np.any(eligible):
        reason = "no_boundary_eligible"
    else:
        index = np.flatnonzero(eligible)
        distance = np.abs(fitted[index] - design.target)
        tied = index[np.isclose(distance, distance.min(), rtol=0, atol=1e-14)]
        chosen = int(tied[0] if tie_policy == "lowest" else tied[-1]) + 1
        reason = "selected"
    return IBOINSelection(
        chosen,
        _readonly(raw_rate),
        _readonly(fitted),
        _readonly(weights),
        _readonly(used_ess),
        _readonly(fit_mask),
        _readonly(candidates),
        _readonly(eligible),
        _readonly(excluded),
        _readonly(boundary),
        safety_stopped,
        reason,
        prior_mode,
        weight_mode,
    )


def select_iboin_mtd(
    design: IBOINDesign,
    patients: ArrayLike,
    toxicities: ArrayLike,
    *,
    prior_mode: Literal["none", "original", "effective"],
    isotonic_weights: Literal["patients", "effective", "equal"] | ArrayLike,
    eliminated: ArrayLike | None = None,
    eligible_doses: ArrayLike | None = None,
    tie_policy: Literal["lowest", "highest"] = "lowest",
    enforce_deescalation_boundary: bool = False,
) -> IBOINSelection:
    """Fit final toxicity rates and select the treated dose closest to target.

    ``prior_mode`` chooses raw rates ``y/n``, original-ESS borrowed rates, or
    robust-effective-ESS borrowed rates. ``isotonic_weights`` is required:
    choose patient counts, n plus the ESS used by ``prior_mode``, equal weights,
    or a custom positive weight vector. The fit includes all treated doses,
    including excluded doses; eligibility is applied only after fitting. Only
    treated, noneliminated doses can be candidates. These unresolved native
    weighting/tie/scope choices are explicit Python behavior. Distances within
    1e-14 are treated as tied.

    The optional final upper bound uses each dose's de-escalation boundary at
    that dose's final patient count. If no candidate passes, the result has no
    selected dose; it never silently falls back to an unconstrained selection.
    """
    return _select(
        design,
        patients,
        toxicities,
        prior_mode=prior_mode,
        isotonic_weights=isotonic_weights,
        eliminated=eliminated,
        eligible_doses=eligible_doses,
        tie_policy=tie_policy,
        enforce_deescalation_boundary=enforce_deescalation_boundary,
        terminal_safety_stop=False,
    )


def select_iboin_trial_mtd(
    design: IBOINDesign,
    trial: IBOINTrialReplay,
    *,
    prior_mode: Literal["none", "original", "effective"],
    isotonic_weights: Literal["patients", "effective", "equal"] | ArrayLike,
    eligible_doses: ArrayLike | None = None,
    tie_policy: Literal["lowest", "highest"] = "lowest",
    enforce_deescalation_boundary: bool = False,
) -> IBOINSelection:
    """Apply final selection to a terminal patient-level iBOIN replay."""
    if not isinstance(trial, IBOINTrialReplay):
        raise TypeError("trial must be an IBOINTrialReplay")
    if trial.stop_reason is None or trial.phase != "stopped":
        raise ValueError("final MTD selection requires a terminal trial replay")
    return _select(
        design,
        trial.patients,
        trial.toxicities,
        prior_mode=prior_mode,
        isotonic_weights=isotonic_weights,
        eliminated=np.asarray(trial.eliminated, dtype=bool),
        eligible_doses=eligible_doses,
        tie_policy=tie_policy,
        enforce_deescalation_boundary=enforce_deescalation_boundary,
        terminal_safety_stop=trial.stop_reason == "stop_safety",
    )
