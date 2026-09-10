"""Modified toxicity probability interval decisions under uniform beta priors."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import isotonic_regression
from scipy.special import betainc, betaincc

from ._validation import FloatArray, count, finite, scalar
from .boin import BOINDecision, _owned


@dataclass(frozen=True)
class MTPIPosterior:
    probability: FloatArray
    unit_probability_mass: FloatArray
    move: NDArray[np.int64]
    overdose_probability: FloatArray
    unsafe: NDArray[np.bool_]


@dataclass(frozen=True)
class MTPITable:
    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    action: NDArray[np.str_]


@dataclass(frozen=True)
class MTPISelection:
    dose: int | None
    isotonic_mean: FloatArray
    overdose_probability: FloatArray
    admissible: NDArray[np.bool_]


@dataclass(frozen=True)
class MTPIDesign:
    """mTPI (2010), not original TPI or mTPI-2. Dose indices are one-based.

    Uniform priors govern decisions and inference. Numerical UPM ties prefer
    D over S over E. The paper does not specify a tie policy for UPM scores.
    Complete outcomes only, at most 200 patients in a trial.
    """

    target: float = 0.3
    lower: float = 0.25
    upper: float = 0.35
    elimination_probability: float = 0.95

    def __post_init__(self) -> None:
        for name in ["target", "lower", "upper", "elimination_probability"]:
            object.__setattr__(self, name, scalar(getattr(self, name), name))
        if not 0 < self.lower < self.target < self.upper < 1:
            raise ValueError("require 0 < lower < target < upper < 1")
        if min(self.lower, self.upper - self.lower, 1 - self.upper) < 1e-6:
            raise ValueError("each interval must have width at least 1e-6")
        if not 0 < self.elimination_probability < 1:
            raise ValueError("elimination_probability must be in (0,1)")

    def posterior(self, patients: ArrayLike, toxicities: ArrayLike) -> MTPIPosterior:
        """Vectorized posterior intervals ordered underdose, equivalence, overdose.

        Move +1/0/-1 is the UPM decision before safety. Unsafe uses the strict
        posterior probability cutoff, without a minimum three-patient gate.
        """
        n, y = np.broadcast_arrays(count(patients, "patients"), count(toxicities, "toxicities"))
        if np.any((n < 1) | (n > 200) | (y > n)):
            raise ValueError("require 1 <= patients <= 200 and 0 <= toxicities <= patients")
        a, b = y + 1, n - y + 1
        left, right = betainc(a, b, self.lower), betaincc(a, b, self.upper)
        center = np.where(
            betainc(a, b, self.upper) <= 0.5,
            betainc(a, b, self.upper) - left,
            betaincc(a, b, self.lower) - right,
        )
        mass = np.stack((left, np.maximum(center, 0), right), axis=-1)
        upm = mass / [self.lower, self.upper - self.lower, 1 - self.upper]
        maximum = upm.max(axis=-1, keepdims=True)
        tied = abs(upm - maximum) <= 32 * np.finfo(float).eps * maximum
        move = np.argmax(tied[..., ::-1], axis=-1).astype(np.int64) - 1
        overdose = betaincc(a, b, self.target)
        return MTPIPosterior(
            _owned(mass),
            _owned(upm),
            _owned(move),
            _owned(overdose),
            _owned(overdose > self.elimination_probability),
        )

    def decision_table(self, max_patients: int = 30) -> MTPITable:
        """Rows are DLTs 0..N, columns patients 1..N; invalid cells are empty.

        E/S/D denote the UPM move; U additionally flags posterior overdose above
        the cutoff. Trial-level safety follows next_dose, including exclusion when
        an escalation would return to an unsafe higher dose.
        """
        value = scalar(max_patients, "max_patients")
        if value != int(value) or not 1 <= value <= 200:
            raise ValueError("max_patients must be an integer in [1,200]")
        n = np.arange(1, int(value) + 1)
        y = np.arange(int(value) + 1)
        nn, yy = np.broadcast_arrays(n[None, :], y[:, None])
        valid = yy <= nn
        result = self.posterior(nn[valid], yy[valid])
        action = np.full(valid.shape, "", dtype="U2")
        action[valid] = np.char.add(
            np.array(["D", "S", "E"])[result.move + 1], np.where(result.unsafe, "U", "")
        )
        return MTPITable(_owned(n), _owned(y), _owned(action))

    def _state(
        self, patients: ArrayLike, toxicities: ArrayLike, eliminated: ArrayLike | None
    ) -> tuple[FloatArray, FloatArray, NDArray[np.bool_], FloatArray]:
        n, y = count(patients, "patients"), count(toxicities, "toxicities")
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
        excluded = np.zeros(n.shape, dtype=bool)
        if eliminated is not None:
            given = np.asarray(eliminated)
            if given.dtype != bool or given.shape != n.shape:
                raise ValueError("eliminated must be a matching boolean vector")
            excluded |= given
        overdose = betaincc(y + 1, n - y + 1, self.target)
        excluded = np.maximum.accumulate(excluded)
        return n, y, excluded, overdose

    def next_dose(
        self,
        patients: ArrayLike,
        toxicities: ArrayLike,
        current_dose: int,
        *,
        eliminated: ArrayLike | None = None,
    ) -> BOINDecision:
        """Apply UPM allocation and the paper's two persistent safety rules."""
        n, y, excluded, overdose = self._state(patients, toxicities, eliminated)
        dose = scalar(current_dose, "current_dose")
        if dose != int(dose) or not 1 <= dose <= n.size or n[int(dose) - 1] == 0:
            raise ValueError("current_dose must identify a treated dose")
        j = int(dose) - 1
        if excluded[0] or (n[0] > 0 and overdose[0] > self.elimination_probability):
            excluded[:] = True
            action, next_dose = "stop_safety", None
        else:
            move = int(self.posterior(n[j], y[j]).move)
            proposed = max(0, min(j + move, len(n) - 1))
            if excluded[j]:
                proposed = int(np.flatnonzero(~excluded)[-1])
            elif proposed > j and (
                excluded[proposed]
                or (n[proposed] > 0 and overdose[proposed] > self.elimination_probability)
            ):
                excluded[proposed:] = True
                proposed = j
            action = "escalate" if proposed > j else "deescalate" if proposed < j else "stay"
            next_dose = proposed + 1
        return BOINDecision(action, next_dose, _owned(excluded), _owned(overdose))

    def select_mtd(
        self,
        patients: ArrayLike,
        toxicities: ArrayLike,
        *,
        eliminated: ArrayLike | None = None,
        weights: ArrayLike | None = None,
    ) -> MTPISelection:
        """Isotonic uniform-prior posterior means; equal weights unless supplied.

        Fits all tried doses, then selects among nonexcluded tried doses. On equal
        distances prefer the highest tied dose at/below target, otherwise the lowest.
        Explicit weights permit a chosen isotonic convention; native weights await audit.
        """
        n, y, excluded, overdose = self._state(patients, toxicities, eliminated)
        w = np.ones(n.shape) if weights is None else finite(weights, "weights")
        if w.shape != n.shape or np.any(w <= 0):
            raise ValueError("weights must be a positive matching vector")
        treated = n > 0
        fitted = np.full(n.shape, np.nan)
        if np.any(treated):
            normalized_weights = w[treated] / w[treated].max()
            if np.any(normalized_weights <= 0):
                raise ArithmeticError("relative isotonic weights cannot be represented")
            fitted[treated] = isotonic_regression(
                (y[treated] + 1) / (n[treated] + 2), weights=normalized_weights
            ).x
        admissible = treated & ~excluded & (overdose <= self.elimination_probability)
        if n[0] > 0 and overdose[0] > self.elimination_probability:
            admissible[:] = False
        indices = np.flatnonzero(admissible)
        chosen = None
        if indices.size:
            distance = abs(fitted[indices] - self.target)
            tied = indices[np.isclose(distance, distance.min(), rtol=0, atol=1e-14)]
            below = tied[fitted[tied] <= self.target + 1e-14]
            chosen = int(below[-1] if below.size else tied[0]) + 1
        return MTPISelection(chosen, _owned(fitted), _owned(overdose), _owned(admissible))
