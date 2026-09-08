"""Life-table box geometry for the EXPSURV SCAT-BOX workflow."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray
from .expsurv import ExploratorySurvival, exploratory_survival


@dataclass(frozen=True)
class CensoredBox:
    curve: ExploratorySurvival
    quartiles: FloatArray
    first_failure_time: float | None
    last_failure_time: float | None
    first_failure_survival: float | None
    last_failure_survival: float | None
    segments: FloatArray
    quantile_method: str


def censored_box(
    time: ArrayLike,
    status: ArrayLike,
    *,
    legacy: bool = False,
    quantile_method: Literal["source", "step"] = "source",
) -> CensoredBox:
    """Build SCAT-BOX geometry using survival-derived quartiles, not raw quantiles.

    Quartiles are ascending cumulative levels .25, .5, .75. The default uses
    EXPSURV's interpolated inverse survival; unreached quartiles are NaN.
    All-censored samples have no failure endpoints and no box segments.
    """
    curve = exploratory_survival(time, status, legacy=legacy)
    q = curve.survival_quantile([0.75, 0.5, 0.25], method=quantile_method)
    deaths = np.flatnonzero(curve.events > 0)
    first = last = first_s = last_s = None
    lines = []
    if deaths.size:
        first, last = float(curve.time[deaths[0]]), float(curve.time[deaths[-1]])
        first_s, last_s = float(curve.survival[deaths[0]]), float(curve.survival[deaths[-1]])
        for value in q[np.isfinite(q)]:
            lines.append([[0.667, value], [1.33, value]])
        lower = float(q[0]) if np.isfinite(q[0]) else first
        upper = float(q[2]) if np.isfinite(q[2]) else last
        lines.extend([[[x, lower], [x, upper]] for x in (0.667, 1.33)])
    segments = np.asarray(lines, dtype=float).reshape(-1, 2, 2)
    segments.flags.writeable = False
    return CensoredBox(curve, q, first, last, first_s, last_s, segments, quantile_method)
