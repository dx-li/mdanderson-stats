"""Desktop final selection and independently computed isotonic report intervals."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betaincinv

from .boin import _owned
from .boin_combination import BOINCombSelection, _validate_counts
from .boin_desktop_decision import BOINDesktopCombDesign
from .keyboard_combination import _biviso


@dataclass(frozen=True)
class BOINDesktopSelection:
    selection: BOINCombSelection
    reported_mean: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    profile: str
    selected_eliminated: bool


def select_boin_desktop_mtd(
    design: BOINDesktopCombDesign,
    patients: ArrayLike,
    toxicities: ArrayLike,
    *,
    contour: bool = False,
    profile: str = "safe",
    intervals: bool = True,
) -> BOINDesktopSelection:
    """Select with safe defaults or explicit current-desktop legacy behavior.

    Legacy single-MTD ranking can select an eliminated cell: the original source
    omits its exclusion. This defect is isolated behind ``desktop_legacy`` and
    explicitly flagged. Legacy also stops if the first observed cell in column
    order is eliminated. The independent default excludes all eliminated cells.

    Report limits are two-decimal isotonic transforms of marginal Beta(.05,.05)
    quantiles, as in the native report, not joint simultaneous credible bands.
    Selection scores within 1e-10 use deterministic native-intended column/row
    ordering; exact native floating-point tie identity is not claimed.
    Set intervals=False for simulation when no per-trial interval report is needed.
    """
    if profile not in ("safe", "desktop_legacy"):
        raise ValueError("profile must be safe or desktop_legacy")
    if not isinstance(contour, (bool, np.bool_)) or not isinstance(intervals, (bool, np.bool_)):
        raise ValueError("contour and intervals must be boolean")
    n, y = _validate_counts(patients, toxicities)
    if n.size > 1000:
        raise ValueError("desktop selection supports at most 1000 dose cells")
    selected = design.select_mtd(n, y, mtd_contour=contour, round_selection=False)
    observed = np.flatnonzero(n.T)
    aborted = not observed.size
    if profile == "desktop_legacy" and observed.size:
        aborted = bool(selected.eliminated.T.flat[observed[0]])
    fitted = _biviso((y + 0.05) / (n + 0.1), n + 0.1)
    reported = fitted.copy()
    reported[n == 0] = np.nan
    if aborted:
        selected = replace(selected, dose=None, contour=() if contour else None)
        reported[:] = np.nan
    else:
        ranked = fitted + 1e-5 * (np.indices(n.shape).sum(0) + 2)
        admissible = (n > 0) & ~selected.eliminated
        if profile == "desktop_legacy" and not contour:
            admissible = n > 0
        # The isotonic solver converges numerically; pooled values can differ
        # by a few e-12. Resolve such equal-score cells in column/row order,
        # not by the direction of an iterative solver's last rounding error.
        tolerance = 1e-10
        if not contour:
            distance = np.where(admissible, np.abs(ranked - design.target), np.inf)
            if np.any(admissible):
                tied = np.flatnonzero(distance.T <= np.min(distance) + tolerance)
                column, row = np.unravel_index(tied[0], distance.T.shape)
                selected = replace(selected, dose=(int(row + 1), int(column + 1)))
            else:
                selected = replace(selected, dose=None)
        else:
            choices: list[int | None] = [None] * n.shape[0]
            for row in range(n.shape[0] - 1, -1, -1):
                if row + 1 < n.shape[0] and choices[row + 1] == n.shape[1] - 1:
                    continue
                columns = np.flatnonzero(admissible[row])
                if columns.size:
                    distance = np.abs(ranked[row, columns] - design.target)
                    choices[row] = int(
                        columns[np.flatnonzero(distance <= distance.min() + tolerance)[0]]
                    )
            selected = replace(
                selected,
                contour=tuple((i + 1, j + 1) for i, j in enumerate(choices) if j is not None),
            )
    lower = upper = None
    if intervals:
        lower = np.round(_biviso(betaincinv(y + 0.05, n - y + 0.05, 0.025), np.ones(n.shape)), 2)
        upper = np.round(_biviso(betaincinv(y + 0.05, n - y + 0.05, 0.975), np.ones(n.shape)), 2)
        for bound in (lower, upper):
            bound[n == 0] = np.nan
            if aborted:
                bound[:] = np.nan
        lower, upper = _owned(lower), _owned(upper)
    dose = selected.dose
    unsafe = bool(dose is not None and selected.eliminated[dose[0] - 1, dose[1] - 1])
    return BOINDesktopSelection(selected, _owned(reported), lower, upper, profile, unsafe)
