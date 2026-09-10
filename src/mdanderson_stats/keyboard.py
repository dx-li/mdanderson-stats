"""Single-agent Keyboard posterior interval decisions and isotonic MTD selection."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betainc, betaincc, betaincinv

from ._validation import FloatArray, count, scalar
from .boin import BOINBoundaryTable, BOINDecision, BOINDesign, _owned


@dataclass(frozen=True)
class KeyboardPosterior:
    intervals: FloatArray
    probability: FloatArray
    score: FloatArray
    strongest_key: NDArray[np.int64]
    target_key: int
    move: NDArray[np.int64]


@dataclass(frozen=True)
class KeyboardSelection:
    dose: int | None
    eliminated: NDArray[np.bool_]
    isotonic_mean: FloatArray
    marginal_interval: FloatArray
    report_overdose_probability: FloatArray
    safety_overdose_probability: FloatArray


@dataclass(frozen=True)
class KeyboardDesign:
    """Equal-width keys with uniform beta priors for decisions and safety.

    edge_rule='discard' follows the paper; 'rescale' includes shortened endpoint
    keys with probability divided by relative width, as in R Keyboard 0.1.3.
    Indices are one-based. Counts and planned enrollment are limited to 200.
    """

    target: float = 0.3
    lower: float | None = None
    upper: float | None = None
    edge_rule: str = "discard"
    elimination_probability: float = 0.95
    extra_safe: bool = False
    safety_offset: float = 0.05
    early_stop_patients: int | None = None
    intervals: FloatArray = field(init=False, repr=False)
    target_key: int = field(init=False)
    _shared: BOINDesign = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        phi = scalar(self.target, "target")
        lo = phi - 0.05 if self.lower is None else scalar(self.lower, "lower")
        hi = phi + 0.05 if self.upper is None else scalar(self.upper, "upper")
        if not 0.05 <= phi <= 0.6 or not 0 <= lo < phi < hi <= 1:
            raise ValueError("require target in [.05,.6] and 0 <= lower < target < upper <= 1")
        width = hi - lo
        if width < 0.001 or self.edge_rule not in ("discard", "rescale"):
            raise ValueError("key width must be >= .001; edge_rule must be 'discard' or 'rescale'")
        if self.early_stop_patients is not None:
            stop = scalar(self.early_stop_patients, "early_stop_patients")
            if stop != int(stop) or not 3 <= stop <= 200:
                raise ValueError("early_stop_patients must be an integer in [3,200]")
        shared = BOINDesign(
            phi,
            elimination_probability=self.elimination_probability,
            extra_safe=self.extra_safe,
            safety_offset=self.safety_offset,
        )
        # Include only complete grid cells, snapping floating noise at 0 and 1.
        tolerance = 32 * np.finfo(float).eps
        first = int(np.ceil(-lo / width - tolerance))
        last = int(np.floor((1 - lo) / width + tolerance))
        edges = lo + np.arange(first, last + 1) * width
        edges[np.abs(edges) < tolerance] = 0
        edges[np.abs(edges - 1) < tolerance] = 1
        edges[np.arange(first, last + 1) == 0] = lo
        edges[np.arange(first, last + 1) == 1] = hi
        if self.edge_rule == "rescale":
            edges = np.unique(np.r_[0, edges, 1])
        intervals = np.column_stack((edges[:-1], edges[1:]))
        target_key = int(np.flatnonzero(intervals[:, 0] == lo)[0]) + 1
        for name, value in [
            ("target", phi),
            ("lower", lo),
            ("upper", hi),
            ("intervals", _owned(intervals)),
            ("target_key", target_key),
            ("_shared", shared),
        ]:
            object.__setattr__(self, name, value)

    def posterior_keys(self, patients: ArrayLike, toxicities: ArrayLike) -> KeyboardPosterior:
        """Batched key probabilities and decisions; final axis enumerates keys.

        Moves are +1/escalate, 0/stay, -1/de-escalate before safety/dose clipping.
        Probability mass may sum below one when residual endpoint keys are omitted.
        Numerically tied strongest scores choose the highest key, conservatively.
        """
        n, y = np.broadcast_arrays(count(patients, "patients"), count(toxicities, "toxicities"))
        if np.any((n < 1) | (n > 200) | (y > n)):
            raise ValueError("require 1 <= patients <= 200 and toxicities <= patients")
        return self._posterior_effective(n, y)

    def _posterior_effective(self, n: FloatArray, y: FloatArray) -> KeyboardPosterior:
        """Shared kernel for validated observed or effective fractional counts."""
        a, b = y[..., None] + 1, (n - y)[..., None] + 1
        lo, hi = self.intervals.T
        cdf_hi = betainc(a, b, hi)
        mass = np.where(
            cdf_hi <= 0.5, cdf_hi - betainc(a, b, lo), betaincc(a, b, lo) - betaincc(a, b, hi)
        )
        mass = np.maximum(mass, 0)
        score = (
            mass
            if self.edge_rule == "discard"
            else mass
            * (self.intervals[self.target_key - 1, 1] - self.intervals[self.target_key - 1, 0])
            / (hi - lo)
        )
        maximum = score.max(axis=-1, keepdims=True)
        if np.any(maximum <= 0):
            raise ArithmeticError("all key scores underflowed")
        tied = np.abs(score - maximum) <= 32 * np.finfo(float).eps * maximum
        winner = score.shape[-1] - np.argmax(tied[..., ::-1], axis=-1)
        move = np.sign(self.target_key - winner).astype(np.int64)
        return KeyboardPosterior(
            self.intervals,
            _owned(mass),
            _owned(score),
            _owned(winner),
            self.target_key,
            _owned(move),
        )

    def boundary_table(self, max_patients: int = 30) -> BOINBoundaryTable:
        """Inclusive DLT cutoffs, including safety precedence; n+1 means impossible."""
        maximum = scalar(max_patients, "max_patients")
        if maximum != int(maximum) or not 1 <= maximum <= 200:
            raise ValueError("max_patients must be an integer in [1,200]")
        n = np.arange(1, int(maximum) + 1)
        safety = self._shared.boundary_table(int(maximum))
        e, d = np.full(n.shape, -1), n + 1
        for i, total in enumerate(n):
            y = np.arange(total + 1)
            move = self.posterior_keys(total, y).move
            escalate = y[(move == 1) & (y < safety.eliminate_min[i])]
            deescalate = y[(move == -1) | (y >= safety.eliminate_min[i])]
            if escalate.size:
                e[i] = escalate[-1]
            if deescalate.size:
                d[i] = deescalate[0]
        return BOINBoundaryTable(
            _owned(n), _owned(e), _owned(d), safety.eliminate_min, safety.lowest_stop_min
        )

    def next_dose(
        self,
        patients: ArrayLike,
        toxicities: ArrayLike,
        current_dose: int,
        *,
        eliminated: ArrayLike | None = None,
    ) -> BOINDecision:
        n, y, excluded, posterior = self._shared._state(patients, toxicities, eliminated)
        dose = scalar(current_dose, "current_dose")
        if dose != int(dose) or not 1 <= dose <= len(n) or n.sum() > 200:
            raise ValueError("require a valid dose and at most 200 patients")
        j = int(dose) - 1
        move = int(self.posterior_keys(n[j], y[j]).move)
        if excluded[0]:
            action, next_dose = "stop_safety", None
        elif excluded[j]:
            action, next_dose = "deescalate", int(np.flatnonzero(~excluded)[-1]) + 1
        elif self.early_stop_patients is not None and n[j] >= self.early_stop_patients:
            action, next_dose = "stop_precision", None
        else:
            next_j = max(0, min(j + move, len(n) - 1))
            if excluded[next_j]:
                next_j = j
            action = "escalate" if next_j > j else "deescalate" if next_j < j else "stay"
            next_dose = next_j + 1
        return BOINDecision(action, next_dose, _owned(excluded), _owned(posterior))

    def select_mtd(
        self, patients: ArrayLike, toxicities: ArrayLike, *, eliminated: ArrayLike | None = None
    ) -> KeyboardSelection:
        if count(patients, "patients").sum() > 200:
            raise ValueError("require at most 200 patients")
        # Native Keyboard and BOIN use the same weak-prior weighted isotonic MTD.
        result = self._shared.select_mtd(patients, toxicities, eliminated=eliminated)
        n, y = np.asarray(patients, dtype=float), np.asarray(toxicities, dtype=float)
        intervals = np.full((len(n), 2), np.nan)
        treated = n > 0
        intervals[treated] = betaincinv(
            y[treated, None] + 0.05, n[treated, None] - y[treated, None] + 0.05, [0.025, 0.975]
        )
        return KeyboardSelection(
            result.dose,
            result.eliminated,
            result.isotonic_mean,
            _owned(intervals),
            result.report_overdose_probability,
            result.safety_overdose_probability,
        )
