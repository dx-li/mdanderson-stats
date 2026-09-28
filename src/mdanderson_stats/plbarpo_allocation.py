"""Active-arm posterior allocation for PLBARPO ledgers.

This module updates allocation among currently active arms without defining a
platform scheduler. Arm arrays remain aligned to the full ledger; inactive
arms receive zero best-arm probability and zero allocation. Best-arm
probabilities are recomputed over active arms only. BARN2N's exponent uses all
historical assignments, including inactive arms.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import count, finite, scalar
from .barpo import barpo_allocation, barpo_posterior

_MAX_LEDGER_ARMS = 100
_MAX_ACTIVE_ARMS = 10


def _ledger_counts(value: ArrayLike, name: str, size: int) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.shape != (size,) or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real vector with one entry per ledger arm")
    return count(raw, name)


def _readonly_bool(value: ArrayLike, name: str, size: int) -> NDArray[np.bool_]:
    raw = np.asarray(value)
    if raw.dtype.kind != "b" or raw.shape != (size,):
        raise ValueError(f"{name} must be a boolean vector with one entry per ledger arm")
    result = np.array(raw, dtype=bool, copy=True)
    result.flags.writeable = False
    return result


def _ledger_vector(value: ArrayLike, name: str, size: int) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.shape != (size,) or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real vector with one entry per ledger arm")
    return finite(raw, name)


def _readonly_float(value: ArrayLike) -> NDArray[np.float64]:
    result = np.array(value, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _readonly_int(value: ArrayLike) -> NDArray[np.int64]:
    result = np.array(value, dtype=np.int64, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class PLBarpoActiveAllocation:
    """Posterior and allocation summaries aligned to the complete arm ledger."""

    successes: NDArray[np.float64]
    failures: NDArray[np.float64]
    assigned: NDArray[np.float64]
    active: NDArray[np.bool_]
    active_indices: NDArray[np.int64]
    posterior_alpha: NDArray[np.float64]
    posterior_beta: NDArray[np.float64]
    posterior_variance: NDArray[np.float64]
    best_probability: NDArray[np.float64]
    best_probability_error: NDArray[np.float64]
    allocation_probability: NDArray[np.float64]
    global_enrolled: int
    method: str
    effective_exponent: float


def plbarpo_active_allocation(
    successes: ArrayLike,
    failures: ArrayLike,
    assigned: ArrayLike,
    active: ArrayLike,
    *,
    prior: ArrayLike,
    method: str = "barcp",
    tau: float = 0.5,
    tau1: float = 1.0,
    max_n: int | None = None,
    target_probability: ArrayLike | None = None,
    minimum_probability: ArrayLike | None = None,
    absolute_tolerance: float = 1e-9,
) -> PLBarpoActiveAllocation:
    """Compute posterior and randomization weights for a current active set.

    Count, prior, target, floor, and active arrays use full-ledger order.
    ``target_probability`` is required for DBCD and must be zero on inactive
    arms; active target entries are normalized over the active set. Floors are
    also zero on inactive arms and use BARPO's existing water-filling policy.
    BARN2N requires ``max_n`` and uses total assignments over the entire ledger
    in its exponent, even when some assigned arms have become inactive.
    """
    raw_prior = np.asarray(prior)
    if (
        raw_prior.ndim != 2
        or raw_prior.shape[1] != 2
        or not 1 <= raw_prior.shape[0] <= _MAX_LEDGER_ARMS
        or raw_prior.dtype.kind not in "iuf"
    ):
        raise ValueError(f"prior must have shape (arms, 2), with 1..{_MAX_LEDGER_ARMS} arms")
    ledger_prior = finite(raw_prior, "prior")
    arms = int(raw_prior.shape[0])
    if np.any(ledger_prior <= 0):
        raise ValueError("prior must contain positive beta shapes")
    with np.errstate(over="ignore"):
        if np.any(~np.isfinite(ledger_prior.sum(axis=1))):
            raise ValueError("prior shape sums must be finite")

    success = _ledger_counts(successes, "successes", arms)
    failure = _ledger_counts(failures, "failures", arms)
    enrollment = _ledger_counts(assigned, "assigned", arms)
    is_active = _readonly_bool(active, "active", arms)
    active_indices = np.flatnonzero(is_active)
    if active_indices.size == 0:
        raise ValueError("at least one active arm is required")
    if active_indices.size > _MAX_ACTIVE_ARMS:
        raise ValueError(f"at most {_MAX_ACTIVE_ARMS} active competitors are supported")
    if np.any(success + failure > enrollment):
        raise ValueError("assigned counts cannot be below observed outcomes")
    global_n_value = float(np.sum(enrollment))
    if not np.isfinite(global_n_value) or global_n_value >= 2**53:
        raise ValueError("global assigned total is too large for exact count arithmetic")
    global_n = int(global_n_value)

    if method not in ("barcp", "barn2n", "barmtv", "dbcd"):
        raise ValueError("method must be barcp, barn2n, barmtv, or dbcd")
    tau_value, tau1_value = scalar(tau, "tau"), scalar(tau1, "tau1")
    if tau_value < 0 or tau1_value < 0:
        raise ValueError("tau and tau1 must be nonnegative")
    exponent = tau_value
    max_n_value: float | None = None
    if max_n is not None:
        if isinstance(max_n, (bool, np.bool_)):
            raise ValueError("max_n must be a positive integer")
        max_n_value = scalar(max_n, "max_n")
        if max_n_value <= 0 or int(max_n_value) != max_n_value:
            raise ValueError("max_n must be a positive integer")
        if global_n > max_n_value:
            raise ValueError("global assigned total cannot exceed max_n")
    if method == "barn2n":
        if max_n_value is None:
            raise ValueError("max_n must be a positive integer for barn2n")
        exponent = (global_n / max_n_value) / 2.0

    floors = (
        np.zeros(arms, dtype=float)
        if minimum_probability is None
        else _ledger_vector(minimum_probability, "minimum_probability", arms)
    )
    if np.any(floors < 0) or np.any(floors[~is_active] != 0):
        raise ValueError("minimum_probability must be nonnegative and zero for inactive arms")
    active_floor = floors[active_indices]
    if active_floor.sum() > 1.0:
        raise ValueError("active minimum probabilities must sum to at most one")

    active_target: NDArray[np.float64] | None = None
    if target_probability is not None:
        target = _ledger_vector(target_probability, "target_probability", arms)
        if np.any(target < 0) or np.any(target[~is_active] != 0):
            raise ValueError("target_probability must be nonnegative and zero on inactive arms")
        selected_target = target[active_indices]
        target_sum = float(selected_target.sum())
        if np.any(selected_target <= 0) or not np.isfinite(target_sum) or target_sum <= 0:
            raise ValueError("active target probabilities must be positive")
        active_target = selected_target / target_sum
    if method == "dbcd" and active_target is None:
        raise ValueError("target_probability is required for dbcd")

    active_post = barpo_posterior(
        success[active_indices],
        failure[active_indices],
        prior=ledger_prior[active_indices],
        absolute_tolerance=absolute_tolerance,
    )
    with np.errstate(over="ignore"):
        posterior_alpha = ledger_prior[:, 0] + success
        posterior_beta = ledger_prior[:, 1] + failure
        shape_total = posterior_alpha + posterior_beta
    if np.any(~np.isfinite(shape_total)):
        raise ArithmeticError("posterior beta shape sums overflow")
    posterior_variance = (
        posterior_alpha / shape_total * posterior_beta / shape_total / (shape_total + 1.0)
    )
    best = np.zeros(arms, dtype=float)
    best_error = np.zeros(arms, dtype=float)
    best[active_indices] = active_post.best_probability
    best_error[active_indices] = active_post.best_probability_error

    active_allocation = barpo_allocation(
        active_post,
        enrollment[active_indices],
        method="barcp" if method == "barn2n" else method,
        tau=exponent,
        tau1=tau1_value,
        target_probability=active_target,
        minimum_probability=active_floor,
    )
    allocation = np.zeros(arms, dtype=float)
    allocation[active_indices] = active_allocation
    return PLBarpoActiveAllocation(
        _readonly_float(success),
        _readonly_float(failure),
        _readonly_float(enrollment),
        is_active,
        _readonly_int(active_indices),
        _readonly_float(posterior_alpha),
        _readonly_float(posterior_beta),
        _readonly_float(posterior_variance),
        _readonly_float(best),
        _readonly_float(best_error),
        _readonly_float(allocation),
        global_n,
        method,
        exponent,
    )
