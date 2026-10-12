"""Desktop waterfall conduct, distinct from its simulation subtrial rules."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betaincc

from ._validation import count
from .boin_combination import BOINCombDecision, _validate_counts, _validate_dose
from .boin_desktop_decision import BOINDesktopCombDesign
from .boin_waterfall import _candidate, _space_for_subtrial


def desktop_contour_decision(
    design: BOINDesktopCombDesign,
    patients: ArrayLike,
    toxicities: ArrayLike,
    current_dose: tuple[int, int],
    subtrial_maximums: ArrayLike,
    *,
    compatibility_profile: str = "safe",
) -> BOINCombDecision:
    """Analyze completed counts using current desktop waterfall conduct rules.

    ``subtrial_maximums`` is in execution order, initial staircase first. The
    existing independent weighted isotonic estimator chooses each transition.
    No pending outcomes or RNG are hidden in these aggregate inputs.
    """
    if compatibility_profile not in ("safe", "desktop_legacy"):
        raise ValueError("compatibility_profile must be safe or desktop_legacy")
    n, y = _validate_counts(patients, toxicities)
    dose = _validate_dose(current_dose, n.shape)
    limits = count(subtrial_maximums, "subtrial_maximums")
    if limits.shape != (n.shape[0],) or np.any(limits < 1) or limits.sum() > 1000:
        raise ValueError("require one positive subtrial maximum per row and total <=1000")
    if n[dose[0] - 1, dose[1] - 1] < 1:
        raise ValueError("current dose must contain completed outcomes")
    occupied = [
        index
        for index in range(1, n.shape[0] + 1)
        if any(n[a - 1, b - 1] > 0 for a, b in _space_for_subtrial(index, *n.shape))
    ]
    current = min(occupied)
    space = _space_for_subtrial(current, *n.shape)
    if dose not in space:
        raise ValueError("current dose must lie in the most recently occupied subtrial")
    position = space.index(dose)
    nn = np.array([n[a - 1, b - 1] for a, b in space])
    yy = np.array([y[a - 1, b - 1] for a, b in space])
    excluded = np.maximum.accumulate(
        (nn >= 3) & (betaincc(yy + 1, nn - yy + 1, design.target) > design.elimination_probability)
    )
    state, posterior = design._state(n, y, None)

    def result(action, next_dose=None):
        if (
            compatibility_profile == "safe"
            and next_dose is not None
            and state[next_dose[0] - 1, next_dose[1] - 1]
        ):
            return BOINCombDecision("stop_safety", None, state, posterior)
        return BOINCombDecision(action, next_dose, state, posterior)

    if n.sum() >= limits.sum():
        return result("complete")
    boundary = design.boundary_table(int(max(nn.max(), 1)))
    index = int(nn[position]) - 1
    subtrial_complete = nn.sum() >= limits[n.shape[0] - current]
    lowest_unsafe = position == 0 and (
        excluded[0]
        or (
            design.extra_safe
            and nn[0] >= 3
            and betaincc(yy[0] + 1, nn[0] - yy[0] + 1, design.target)
            > design.elimination_probability - design.safety_offset
        )
    )
    # The native controller resolves capacity before re-testing its current
    # position for toxicity; final selection still applies posterior safety.
    if lowest_unsafe and not subtrial_complete:
        if current == n.shape[0] and excluded[0]:
            return result("stop_safety")
        if current == 1:
            return result("stop_precision")
        next_dose = (dose[0] - 1, dose[1])
        if min(next_dose) < 1:
            return result("error" if compatibility_profile == "desktop_legacy" else "stop_safety")
        return result("next_subtrial", next_dose)
    converged = (
        design.early_stop_patients is not None
        and nn[position] >= design.early_stop_patients
        and (
            yy[position] > boundary.escalate_max[index]
            or position == len(space) - 1
            or excluded[position + 1]
        )
        and (yy[position] < boundary.deescalate_min[index] or position == 0)
    )
    if subtrial_complete or converged:
        if current == 1 or dose == (1, 1):
            return result("stop_precision")
        candidate, _, _, _ = _candidate(
            design._base,
            n,
            y,
            space,
            extra_safe=design.extra_safe,
            cutoff=design.elimination_probability,
            offset=design.safety_offset,
        )
        if candidate is not None:
            return result(
                "next_subtrial", (max(1, candidate[0] - 1), min(n.shape[1], candidate[1] + 1))
            )
        first = next(cell for cell in space if n[cell[0] - 1, cell[1] - 1] > 0)
        next_dose = (first[0] - 1, first[1])
        if state[0, 0]:
            return result("stop_safety")
        if min(next_dose) < 1:
            return result("error" if compatibility_profile == "desktop_legacy" else "stop_safety")
        return result("next_subtrial", next_dose)
    next_position = position
    if yy[position] <= boundary.escalate_max[index]:
        next_position = min(position + 1, len(space) - 1)
        if excluded[next_position]:
            next_position = position
    elif yy[position] >= boundary.deescalate_min[index]:
        next_position = max(position - 1, 0)
    next_dose = space[next_position]
    action = (
        "escalate"
        if next_position > position
        else "deescalate"
        if next_position < position
        else "stay"
    )
    return result(action, next_dose)
