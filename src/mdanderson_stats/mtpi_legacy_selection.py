"""Recovered mTPI R final estimation, separate from trial allocation/safety."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import isotonic_regression

from ._validation import FloatArray, count, scalar
from .boin import _owned


@dataclass(frozen=True)
class MTPILegacySelection:
    dose: int | None
    isotonic_mean: FloatArray
    selection_score: FloatArray
    eligible: NDArray[np.bool_]


def select_mtpi_legacy_mtd(
    patients: ArrayLike,
    toxicities: ArrayLike,
    *,
    target: float = 0.3,
    first_excluded_dose: int | None = None,
    no_selection: bool = False,
) -> MTPILegacySelection:
    """Final estimator in the official version-2.1 ``mTPI-middle.R`` archive.

    Use Beta(.005,.005) posterior means and inverse-variance isotonic weights
    from dose 1 through min(highest treated, first excluded minus one), including
    any untried lower doses. Select from lowest treated through that upper bound
    after adding dose*1e-10 to scores at dose 2 and above. Labels are one-based.

    ``first_excluded_dose`` carries the source controller's ``toxdose``;
    ``no_selection`` carries its ``nodose`` (including excessive underdosing).
    Neither is reconstructed from terminal counts. No additional posterior
    overdose filter is applied. This is NOT a native controller or Excel port.
    Empty selection ranges return None; the source's 2:1 indexing defect is
    corrected by perturbing only existing doses. Existing paper APIs are unchanged.
    """
    n, y = count(patients, "patients"), count(toxicities, "toxicities")
    target = scalar(target, "target")
    if (
        n.ndim != 1
        or not 1 <= n.size <= 100
        or y.shape != n.shape
        or np.any(y > n)
        or n.sum() > 200
    ):
        raise ValueError(
            "require matching 1..100 dose vectors, toxicities <= patients, total <= 200"
        )
    if not 0 < target < 1:
        raise ValueError("target must be in (0,1)")
    if not isinstance(no_selection, (bool, np.bool_)):
        raise ValueError("no_selection must be boolean")
    excluded = n.size + 1
    if first_excluded_dose is not None:
        value = scalar(first_excluded_dose, "first_excluded_dose")
        if value != int(value) or not 1 <= value <= n.size + 1:
            raise ValueError("first_excluded_dose must be an integer in [1, dose count + 1]")
        excluded = int(value)
    fitted = np.full(n.shape, np.nan)
    score = np.full(n.shape, np.nan)
    eligible = np.zeros(n.shape, dtype=bool)
    treated = np.flatnonzero(n)
    chosen = None
    if treated.size and not no_selection:
        upper = min(int(treated[-1]) + 1, excluded - 1)
        lower = int(treated[0])
        if lower < upper:
            alpha = y[:upper] + 0.005
            beta = n[:upper] - y[:upper] + 0.005
            total = alpha + beta
            precision = total**2 * (total + 1) / (alpha * beta)
            fitted[:upper] = isotonic_regression(alpha / total, weights=precision).x
            score[:upper] = fitted[:upper]
            score[1:upper] += np.arange(2, upper + 1) * 1e-10
            eligible[lower:upper] = True
            chosen = lower + int(np.argmin(abs(score[lower:upper] - target))) + 1
    return MTPILegacySelection(chosen, _owned(fitted), _owned(score), _owned(eligible))
