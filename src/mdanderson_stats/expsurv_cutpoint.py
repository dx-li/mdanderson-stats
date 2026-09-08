"""Covariate cut-point comparisons for exploratory survival analysis."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite, scalar
from .expsurv import ExploratorySurvival, exploratory_survival


@dataclass(frozen=True)
class CutpointComparison:
    cut: float
    lower: ExploratorySurvival | None
    upper: ExploratorySurvival | None
    lower_indices: NDArray[np.int64]
    upper_indices: NDArray[np.int64]


@dataclass(frozen=True)
class SurvivalCutpoint:
    time: FloatArray
    status: FloatArray
    covariate: FloatArray
    baseline: ExploratorySurvival
    legacy: bool

    def compare(self, cut: float) -> CutpointComparison:
        """Split at covariate<=cut and covariate>cut, preserving original row IDs.

        An empty group has no estimated curve (None), not a fabricated survival
        of one. Any finite cut is permitted, including outside the observed range.
        """
        cut = scalar(cut, "cut")
        lower_indices = np.flatnonzero(self.covariate <= cut)
        upper_indices = np.flatnonzero(self.covariate > cut)
        curves = []
        for indices in (lower_indices, upper_indices):
            curves.append(
                None
                if indices.size == 0
                else exploratory_survival(
                    self.time[indices], self.status[indices], legacy=self.legacy
                )
            )
            indices.flags.writeable = False
        return CutpointComparison(cut, curves[0], curves[1], lower_indices, upper_indices)


def survival_cutpoint(
    time: ArrayLike, status: ArrayLike, covariate: ArrayLike, *, legacy: bool = False
) -> SurvivalCutpoint:
    """Prepare aligned observations for repeated EXPSURV cut-point comparisons."""
    baseline = exploratory_survival(time, status, legacy=legacy)
    t, d, x = (
        finite(time, "time").copy(),
        count(status, "status").copy(),
        finite(covariate, "covariate").copy(),
    )
    if x.shape != t.shape:
        raise ValueError("covariate must be a vector matching time and status")
    for array in (t, d, x):
        array.flags.writeable = False
    return SurvivalCutpoint(t, d, x, baseline, bool(legacy))
