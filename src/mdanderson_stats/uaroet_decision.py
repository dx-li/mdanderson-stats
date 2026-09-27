"""Utility-based admissibility, adaptive randomization and final OBD selection."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray
from .uaroet import _integer, _real_matrix, _scalar
from .uaroet_fit import UAROETFit

_MAX_SUBJECTS = 10_000


def _freeze_bool(values: NDArray[np.bool_]) -> NDArray[np.bool_]:
    return np.frombuffer(np.ascontiguousarray(values).tobytes(), dtype=np.bool_).reshape(
        values.shape
    )


def _count_vector(value: ArrayLike, name: str, size: int) -> NDArray[np.int64]:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size != size or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real vector with {size} entries")
    numeric = np.asarray(raw, dtype=float)
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric < 0)
        or np.any(numeric != np.floor(numeric))
        or np.any(numeric >= 2**53)
        or numeric.sum() > _MAX_SUBJECTS
    ):
        raise ValueError(
            f"{name} must contain nonnegative integer counts totaling <= {_MAX_SUBJECTS}"
        )
    return numeric.astype(np.int64)


@dataclass(frozen=True)
class UAROETAllocation:
    """Posterior allocation weights and summaries; best_dose is mean-utility argmax."""

    action: str
    probabilities: FloatArray
    best_dose: int | None
    mean_utility: FloatArray
    toxicity_risk: FloatArray
    probability_best: FloatArray
    near_optimal: NDArray[np.bool_]
    acceptable: NDArray[np.bool_]
    eligible: NDArray[np.bool_]
    good_outcome_probability: FloatArray
    reason: str


def uaroet_allocation(
    fit: UAROETFit,
    utility: ArrayLike,
    treated: ArrayLike,
    *,
    toxicity_limit: float,
    bad_toxicity_level: int = 1,
    p_L: float = 0.1,
    p_U: float = 0.8,
    utility_tolerance: float = 0.0,
    good_utility_cutoff: float = 0.5,
    starting_dose: int | None = None,
    final: bool = False,
    final_rule: str = "acceptable",
) -> UAROETAllocation:
    """Calculate UAROET posterior acceptability and a deterministic allocation.

    A dose is acceptable when it is near the global maximum posterior mean
    utility, its probability of being best is at least ``p_L``, and its
    posterior probability of exceeding ``toxicity_limit`` at or above
    ``bad_toxicity_level`` does not exceed ``p_U``. During enrollment, doses
    above one level past the highest treated level cannot be skipped. Empty
    trials start at ``starting_dose`` without applying posterior filters.

    The adaptive allocation weight is the posterior predictive probability of
    a ``utility >= good_utility_cutoff`` outcome, normalized over eligible
    doses. At final analysis, ``acceptable`` chooses the highest-mean dose
    within the acceptable set; ``paper_global`` follows the paper's literal
    global-mean-utility argmax whenever the acceptable set is nonempty.
    """
    if not isinstance(fit, UAROETFit):
        raise ValueError("fit must be a UAROETFit")
    joint = np.asarray(fit.joint)
    if (
        joint.ndim != 5
        or joint.shape[0] < 2
        or joint.shape[1] < 1
        or not 1 <= joint.shape[2] <= 5
        or any(not 2 <= size <= 4 for size in joint.shape[3:])
        or joint.size > 2_000_000
        or not np.all(np.isfinite(joint))
        or np.any(joint < 0)
    ):
        raise ValueError("fit joint draws have invalid chain/dose/outcome dimensions")
    if np.any(np.abs(joint.sum(axis=(-2, -1)) - 1) > 1e-11):
        raise ValueError("each fit joint draw and dose must sum to one")
    doses, efficacy_levels, toxicity_levels = joint.shape[2:]
    u = _real_matrix(utility, "utility")
    if (
        u.shape != (efficacy_levels, toxicity_levels)
        or np.any(u < 0)
        or np.any(np.diff(u, axis=0) < 0)
        or np.any(np.diff(u, axis=1) > 0)
    ):
        raise ValueError(
            "utility must be nonnegative and nondecreasing in efficacy, nonincreasing in toxicity"
        )
    n = _count_vector(treated, "treated", doses)
    if not isinstance(final, (bool, np.bool_)):
        raise ValueError("final must be boolean")
    if final_rule not in ("acceptable", "paper_global"):
        raise ValueError("final_rule must be acceptable or paper_global")
    bad_level = _integer(bad_toxicity_level, "bad_toxicity_level", 1, toxicity_levels - 1)
    limit = _scalar(toxicity_limit, "toxicity_limit")
    lower_probability = _scalar(p_L, "p_L")
    upper_probability = _scalar(p_U, "p_U")
    tolerance = _scalar(utility_tolerance, "utility_tolerance")
    good_cutoff = _scalar(good_utility_cutoff, "good_utility_cutoff")
    if (
        not np.isfinite(limit)
        or not 0 <= limit <= 1
        or not np.isfinite(lower_probability)
        or not 0 <= lower_probability <= 1
        or not np.isfinite(upper_probability)
        or not 0 <= upper_probability <= 1
        or not np.isfinite(tolerance)
        or tolerance < 0
        or not np.isfinite(good_cutoff)
    ):
        raise ValueError("invalid toxicity, probability, utility-tolerance or good-utility cutoff")
    start = (
        None if starting_dose is None else _integer(starting_dose, "starting_dose", 0, doses - 1)
    )
    if final and n.sum() == 0:
        raise ValueError("final selection requires at least one treated subject")
    if n.sum() == 0 and start is None:
        raise ValueError("starting_dose is required before any subjects are treated")

    draws = joint.shape[0] * joint.shape[1]
    per_draw_utility = np.einsum("cdjet,et->cdj", joint, u, optimize=True)
    if not np.all(np.isfinite(per_draw_utility)):
        raise ArithmeticError("per-draw UAROET utility is not representable")
    flat_utility = per_draw_utility.reshape(draws, doses)
    maximum = np.max(flat_utility, axis=1, keepdims=True)
    best = np.mean(flat_utility == maximum, axis=0)
    mean_utility = np.mean(flat_utility, axis=0)
    if not np.all(np.isfinite(mean_utility)):
        raise ArithmeticError("posterior mean utility is not representable")
    near = mean_utility >= float(np.max(mean_utility)) - tolerance
    toxicity_per_draw = joint[..., bad_level:].sum(axis=(-2, -1)).reshape(draws, doses)
    risk = np.mean(toxicity_per_draw > limit, axis=0)
    acceptable = near & (best >= lower_probability) & (risk <= upper_probability)
    good_event = u >= good_cutoff
    good = np.einsum("cdjet,et->j", joint, good_event, optimize=True) / draws
    eligible = acceptable.copy()
    if n.sum() > 0 and not final:
        highest = int(np.flatnonzero(n > 0)[-1])
        eligible[np.arange(doses) > highest + 1] = False

    weights = np.zeros(doses, dtype=float)
    selected: int | None = None
    if final:
        if not np.any(acceptable):
            action, reason = "stop", "no_acceptable_dose"
        else:
            candidate = (
                np.arange(doses) if final_rule == "paper_global" else np.flatnonzero(acceptable)
            )
            selected = int(
                candidate[
                    np.flatnonzero(mean_utility[candidate] == np.max(mean_utility[candidate]))[0]
                ]
            )
            weights[selected] = 1
            action, reason = "select_obd", f"final_{final_rule}_utility_maximizer"
    elif n.sum() == 0:
        assert start is not None
        weights[start] = 1
        selected = start
        action, reason = "start", "configured_starting_dose"
    elif not np.any(eligible):
        action, reason = "stop", "no_eligible_acceptable_dose"
    else:
        eligible_good = np.where(eligible, good, 0.0)
        total_good = float(eligible_good.sum())
        if not np.isfinite(total_good) or total_good <= 0:
            raise ValueError("good-outcome randomization is undefined for eligible doses")
        weights = eligible_good / total_good
        candidate = np.flatnonzero(eligible)
        selected = int(
            candidate[np.flatnonzero(mean_utility[candidate] == np.max(mean_utility[candidate]))[0]]
        )
        action, reason = "randomize", "good_outcome_probability_over_eligible_doses"
    return UAROETAllocation(
        action,
        _freeze(weights),
        selected,
        _freeze(mean_utility),
        _freeze(risk),
        _freeze(best),
        _freeze_bool(near),
        _freeze_bool(acceptable),
        _freeze_bool(eligible),
        _freeze(good),
        reason,
    )
