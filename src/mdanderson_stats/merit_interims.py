"""Independent-arm interim boundaries for MERIT trials."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betainc, betaincc

from ._validation import count, finite
from .boin import _owned
from .merit import _integer


@dataclass(frozen=True)
class MERITInterimBoundaries:
    patients: NDArray[np.int64]
    toxicity_stop_min: NDArray[np.int64]
    efficacy_stop_max: NDArray[np.int64]
    assess_toxicity: NDArray[np.bool_]
    assess_efficacy: NDArray[np.bool_]


@dataclass(frozen=True)
class MERITInterims:
    """Raw-count beta monitoring on explicit per-arm patient counts.

    Empty schedules disable an endpoint. Beta prior shapes have order
    (event, non-event). Equality with a probability cutoff continues enrollment.
    Integer boundaries are for unpooled counts, not fractional isotonic counts.
    """

    toxicity_target: float
    efficacy_target: float
    toxicity_looks: ArrayLike = ()
    efficacy_looks: ArrayLike = ()
    toxicity_cutoff: float = 0.95
    efficacy_cutoff: float = 0.95
    prior: ArrayLike = (0.1, 0.1)

    def __post_init__(self) -> None:
        thresholds = finite(
            [
                self.toxicity_target,
                self.efficacy_target,
                self.toxicity_cutoff,
                self.efficacy_cutoff,
            ],
            "thresholds",
        )
        shapes = finite(self.prior, "prior")
        if thresholds.shape != (4,) or np.any((thresholds <= 0) | (thresholds >= 1)):
            raise ValueError("targets and cutoffs must be scalar probabilities in (0,1)")
        if shapes.shape != (2,) or np.any(shapes <= 0) or shapes.sum() > 1e6:
            raise ValueError("prior requires two positive shapes totaling <=1e6")
        object.__setattr__(self, "prior", _owned(shapes))
        for name in ("toxicity_looks", "efficacy_looks"):
            looks = count(getattr(self, name), name)
            if (
                looks.ndim != 1
                or np.any((looks < 1) | (looks > 499))
                or np.any(np.diff(looks) <= 0)
            ):
                raise ValueError("looks must be strictly increasing patient counts in [1,499]")
            object.__setattr__(self, name, _owned(looks.astype(np.int64)))

    def boundaries(self, patients_per_arm: int) -> MERITInterimBoundaries:
        """Use n+1/-1 for impossible toxicity/futility stops, with endpoint flags."""
        maximum = _integer(patients_per_arm, "patients_per_arm", 1, 500)
        tlooks, elooks = np.asarray(self.toxicity_looks), np.asarray(self.efficacy_looks)
        looks = np.union1d(tlooks, elooks).astype(np.int64)
        if np.any(looks >= maximum):
            raise ValueError("interim looks must precede the final per-arm sample size")
        assess_t, assess_e = np.isin(looks, tlooks), np.isin(looks, elooks)
        tstop, estop = looks + 1, np.full(looks.size, -1, dtype=np.int64)
        a, b = np.asarray(self.prior)
        for i, n in enumerate(looks):
            events = np.arange(n + 1)
            if assess_t[i]:
                crossing = np.flatnonzero(
                    betaincc(a + events, b + n - events, self.toxicity_target)
                    > self.toxicity_cutoff
                )
                if crossing.size:
                    tstop[i] = crossing[0]
            if assess_e[i]:
                crossing = np.flatnonzero(
                    betainc(a + events, b + n - events, self.efficacy_target) > self.efficacy_cutoff
                )
                if crossing.size:
                    estop[i] = crossing[-1]
        return MERITInterimBoundaries(
            _owned(looks), _owned(tstop), _owned(estop), _owned(assess_t), _owned(assess_e)
        )
