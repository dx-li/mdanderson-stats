"""Complete-outcome 1+2+3 rare-disease dose assignment from the public protocol."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betainc, betaincc

from ._validation import FloatArray, count, finite, scalar
from .bayesian_monitoring import _integer
from .boin import BOINDesign, _owned


@dataclass(frozen=True)
class RareDisease123Decision:
    action: str
    next_dose: int | None
    selected_dose: int | None
    cohort_size: int
    admissible: NDArray[np.bool_]
    eliminated: NDArray[np.bool_]
    efficacy_probability: FloatArray


@dataclass(frozen=True)
class RareDisease123Design:
    """1+2+3 with an explicitly supplied efficacy Beta prior; one-based doses.

    The native protocol omits the prior. Beta(1,1) reproduces its default decision
    tables, but it is deliberately not an implicit default. Full outcomes only.
    """

    efficacy_prior: tuple[float, float]
    target_toxicity: float = 0.28
    minimum_efficacy: float = 0.6
    efficacy_cutoff: float = 0.9
    escalation_cutoff: float = 0.5

    def __post_init__(self) -> None:
        prior = finite(self.efficacy_prior, "efficacy_prior")
        if prior.shape != (2,) or np.any(prior < 1e-6) or prior.sum() > 1e6:
            raise ValueError("efficacy_prior must contain two shapes >=1e-6 with sum <=1e6")
        object.__setattr__(self, "efficacy_prior", tuple(map(float, prior)))
        for name in ["target_toxicity", "minimum_efficacy", "efficacy_cutoff", "escalation_cutoff"]:
            object.__setattr__(self, name, scalar(getattr(self, name), name))
        if not 0.05 <= self.target_toxicity <= 0.6 or not 0.1 <= self.minimum_efficacy <= 0.9:
            raise ValueError("target_toxicity must be in [.05,.6], minimum_efficacy in [.1,.9]")
        if (
            not 0 < self.efficacy_cutoff < 1
            or not 1 - self.efficacy_cutoff < self.escalation_cutoff < 1
        ):
            raise ValueError(
                "require 0<efficacy_cutoff<1 and 1-efficacy_cutoff<escalation_cutoff<1"
            )

    @property
    def toxicity_boundaries(self) -> tuple[float, float]:
        """Unrounded BOIN boundaries using the package's standard BOIN alternatives."""
        design = BOINDesign(self.target_toxicity)
        return design.escalation_boundary, design.deescalation_boundary

    def next_dose(
        self,
        patients: ArrayLike,
        toxicities: ArrayLike,
        responses: ArrayLike,
        current_dose: int,
        *,
        eliminated: ArrayLike | None = None,
    ) -> RareDisease123Decision:
        """Apply protocol dose rules and select OBD upon assignment to a full dose.

        Counts per dose must be 0,1,3,6. First assignment treats 1 patient, second
        adds 2 and third adds 3. Assignment to an admissible six-patient dose
        terminates with that dose selected. Carry eliminated between decisions.
        """
        n, t, r = (
            count(patients, "patients"),
            count(toxicities, "toxicities"),
            count(responses, "responses"),
        )
        if (
            n.ndim != 1
            or not 2 <= n.size <= 20
            or t.shape != n.shape
            or r.shape != n.shape
            or not np.isin(n, [0, 1, 3, 6]).all()
            or np.any(t > n)
            or np.any(r > n)
        ):
            raise ValueError(
                "require matching 2..20 dose vectors, n in {0,1,3,6}, and endpoint counts <=n"
            )
        j = _integer(current_dose, "current_dose") - 1
        if not 0 <= j < n.size or n[j] == 0:
            raise ValueError("current_dose must identify a treated dose")
        excluded = np.zeros(n.shape, dtype=bool)
        if eliminated is not None:
            given = np.asarray(eliminated)
            if given.shape != n.shape or given.dtype != bool:
                raise ValueError("eliminated must be a matching boolean vector")
            excluded |= given
        candidates, masks, probabilities = _next_batch(
            self, n[None, :], t[None, :], r[None, :], np.array([j]), excluded[None, :]
        )
        proposed = None if candidates[0] < 0 else int(candidates[0])
        excluded, acceptable = masks[0], probabilities[0]
        admissible = ~excluded
        selected = None
        size = 0
        if proposed is None:
            action, next_dose = "stop_no_obd", None
        elif n[proposed] == 6:
            action, next_dose, selected = "select_obd", None, proposed + 1
        else:
            action = "escalate" if proposed > j else "deescalate" if proposed < j else "stay"
            next_dose = proposed + 1
            size = {0: 1, 1: 2, 3: 3}[int(n[proposed])]
        return RareDisease123Decision(
            action,
            next_dose,
            selected,
            size,
            _owned(admissible),
            _owned(excluded),
            _owned(acceptable),
        )


def _next_batch(
    design: RareDisease123Design,
    n: FloatArray | NDArray[np.int64],
    t: FloatArray | NDArray[np.int64],
    r: FloatArray | NDArray[np.int64],
    current: NDArray[np.int64],
    excluded: NDArray[np.bool_],
) -> tuple[NDArray[np.int64], NDArray[np.bool_], FloatArray]:
    """Apply identical rules to validated trial rows; -1 denotes no next dose."""
    le, ld = design.toxicity_boundaries
    rate = np.divide(t, n, out=np.zeros(n.shape), where=n > 0)
    a, b = design.efficacy_prior
    acceptable = betaincc(r + a, n - r + b, design.minimum_efficacy)
    excluded = excluded | np.maximum.accumulate((n > 1) & (rate >= ld), axis=1)
    excluded |= (n > 1) & (
        betainc(r + a, n - r + b, design.minimum_efficacy) >= design.efficacy_cutoff
    )
    rows = np.arange(n.shape[0])
    lo, hi = np.maximum(current - 1, 0), np.minimum(current + 1, n.shape[1] - 1)
    lower = (current > 0) & ~excluded[rows, lo]
    higher = (current + 1 < n.shape[1]) & ~excluded[rows, hi] & (n[rows, hi] <= 1)
    stay = np.where(~excluded[rows, current], current, -1)
    proposed = np.where(
        rate[rows, current] >= ld,
        np.where(lower, lo, stay),
        np.where(
            rate[rows, current] > le,
            np.where(stay >= 0, stay, np.where(higher, hi, -1)),
            np.where((acceptable[rows, current] < design.escalation_cutoff) & higher, hi, stay),
        ),
    )
    return proposed, excluded, acceptable
