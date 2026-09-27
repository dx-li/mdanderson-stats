"""Explicit and paper-moment-matched lognormal priors for dose schedules."""

from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .dose_schedule import _numeric, dose_schedule_parameter_names


def _positive_scalar(value: object, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw)
    if not isfinite(result) or result <= 1:
        raise ValueError(f"{name} must be finite and greater than one")
    return result


@dataclass(frozen=True)
class DoseSchedulePrior:
    """Independent normal prior on per-dose log area, log peak and log tail."""

    mean: FloatArray
    sd: FloatArray
    dose_count: int
    ordered_areas: bool = True

    def __init__(
        self,
        mean: ArrayLike,
        sd: ArrayLike,
        dose_count: int,
        ordered_areas: bool = True,
    ) -> None:
        names = dose_schedule_parameter_names(dose_count, ordered_areas=ordered_areas)
        m = _numeric(mean, "mean", max_size=60)
        s = _numeric(sd, "sd", max_size=60)
        if m.shape != (len(names),) or s.shape != m.shape or np.any(s < 0):
            raise ValueError("prior mean/SD must match per-dose parameters; SDs may be zero")
        object.__setattr__(self, "mean", _freeze(m))
        object.__setattr__(self, "sd", _freeze(s))
        object.__setattr__(self, "dose_count", int(dose_count))
        object.__setattr__(self, "ordered_areas", bool(ordered_areas))


def dose_schedule_moment_prior(
    shortest_schedule_toxicity: ArrayLike,
    administrations: int,
    peak_times: ArrayLike,
    tail_times: ArrayLike,
    *,
    lambda1: float = 1.5,
    lambda2: float = 1.5,
    ordered_areas: bool = True,
) -> DoseSchedulePrior:
    """Apply the paper's approximate moment-matched lognormal elicitation.

    The resulting prior matches elicited hazard-area, peak-time and tail-time
    moments under the shortest schedule; it is not an exact prior-predictive
    calibration. ``administrations`` is the number of administrations in that
    schedule. For ordered areas, toxicity risks must increase by dose.
    """
    p = _numeric(shortest_schedule_toxicity, "shortest_schedule_toxicity", max_size=20)
    b = _numeric(peak_times, "peak_times", max_size=20)
    c = _numeric(tail_times, "tail_times", max_size=20)
    count = np.asarray(administrations)
    if (
        count.ndim != 0
        or count.dtype.kind not in "iu"
        or isinstance(administrations, (bool, np.bool_))
    ):
        raise ValueError("administrations must be a positive integer")
    m1 = int(count)
    if (
        p.ndim != 1
        or not 1 <= p.size <= 20
        or b.shape != p.shape
        or c.shape != p.shape
        or np.any((p <= 0) | (p >= 1))
        or np.any(b <= 0)
        or np.any(c <= 0)
        or m1 < 1
    ):
        raise ValueError("invalid toxicity, schedule size or peak/tail inputs")
    if not isinstance(ordered_areas, (bool, np.bool_)):
        raise ValueError("ordered_areas must be boolean")
    if ordered_areas and np.any(np.diff(p) <= 0):
        raise ValueError("ordered-area elicitation requires strictly increasing toxicity risks")
    l1 = _positive_scalar(lambda1, "lambda1")
    l2 = _positive_scalar(lambda2, "lambda2")
    va = float(np.log1p(1 / (l1 - 1)))
    vb = float(np.log1p(1 / (l2 - 1)))
    if not isfinite(va) or not isfinite(vb) or va <= 0 or vb <= 0:
        raise ValueError("lambda values do not define representable log variances")
    q = -np.log1p(-p)
    area_moments = np.diff(np.r_[0.0, q]) if ordered_areas else q
    if np.any(area_moments <= 0):
        raise ValueError("elicited toxicity increments are not representable")
    means = np.empty(3 * p.size, dtype=float)
    sds = np.empty(3 * p.size, dtype=float)
    area_sd, time_sd = sqrt(va), sqrt(vb)
    for dose in range(p.size):
        index = 3 * dose
        means[index] = np.log(area_moments[dose] / m1) - va / 2
        means[index + 1] = np.log(b[dose]) - vb / 2
        means[index + 2] = np.log(c[dose]) - vb / 2
        sds[index : index + 3] = (area_sd, time_sd, time_sd)
    return DoseSchedulePrior(means, sds, p.size, ordered_areas)
