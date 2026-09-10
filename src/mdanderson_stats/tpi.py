"""Original TPI (Ji, Li and Bekele, 2007), using posterior probability mass."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import isotonic_regression
from scipy.special import betainc, betaincc

from ._validation import FloatArray, count, finite, scalar
from .boin import BOINDecision, _owned
from .mtpi import MTPISelection, MTPITable


def _tpi_move(scores: FloatArray) -> NDArray[np.int64]:
    """Scores E/S/D; ties prefer D then S then E; forbidden scores are -1."""
    maximum = scores.max(axis=-1, keepdims=True)
    tied = abs(scores - maximum) <= 32 * np.finfo(float).eps * maximum
    return np.argmax(tied[..., ::-1], axis=-1).astype(np.int64) - 1


@dataclass(frozen=True)
class TPIPosterior:
    mean: FloatArray
    standard_deviation: FloatArray
    lower: FloatArray
    upper: FloatArray
    probability: FloatArray
    move: NDArray[np.int64]
    overdose_probability: FloatArray
    unsafe: NDArray[np.bool_]


@dataclass(frozen=True)
class TPIDesign:
    """TPI with explicit lower/upper SD multipliers to avoid source K-label swaps.

    Defaults reproduce the original Table 1, including Beta(.005,.005) priors.
    Safety requires at least two observed patients. Dose indices are one-based.
    """

    target: float = 0.3
    lower_sd: float = 1.5
    upper_sd: float = 1.0
    elimination_probability: float = 0.95
    prior: tuple[float, float] = (0.005, 0.005)

    def __post_init__(self) -> None:
        for name in ["target", "lower_sd", "upper_sd", "elimination_probability"]:
            object.__setattr__(self, name, scalar(getattr(self, name), name))
        if not 0 < self.target < 1 or not 0 < self.elimination_probability < 1:
            raise ValueError("target and elimination_probability must be in (0,1)")
        if not 1e-6 <= min(self.lower_sd, self.upper_sd) or max(self.lower_sd, self.upper_sd) > 100:
            raise ValueError("SD multipliers must be in [1e-6,100]")
        prior = finite(self.prior, "prior")
        if prior.shape != (2,) or np.any(prior < 1e-6) or prior.sum() > 1e6:
            raise ValueError("prior must contain two shapes >=1e-6 with sum <=1e6")
        object.__setattr__(self, "prior", (float(prior[0]), float(prior[1])))
        if betaincc(*self.prior, self.target) > self.elimination_probability:
            raise ValueError(
                "prior considers untried doses unsafe; revise prior or candidate doses"
            )

    def posterior(self, patients: ArrayLike, toxicities: ArrayLike) -> TPIPosterior:
        """Posterior masses E/S/D with intervals intersected with [0,1].

        Zero patients is allowed for prior summaries; trial actions require data.
        No division by interval widths is performed (unlike mTPI).
        """
        n, y = np.broadcast_arrays(count(patients, "patients"), count(toxicities, "toxicities"))
        if n.size > 2000000 or np.any((n > 200) | (y > n)):
            raise ValueError("require 0<=toxicities<=patients<=200 and <=2 million cells")
        a, b = y + self.prior[0], n - y + self.prior[1]
        total = a + b
        mean = a / total
        sd = np.sqrt((a / total) * (b / total) / (total + 1))
        lower = np.maximum(0, self.target - self.lower_sd * sd)
        upper = np.minimum(1, self.target + self.upper_sd * sd)
        left, right = betainc(a, b, lower), betaincc(a, b, upper)
        center = np.where(
            betainc(a, b, upper) <= 0.5,
            betainc(a, b, upper) - left,
            betaincc(a, b, lower) - right,
        )
        probability = np.stack((left, np.maximum(center, 0), right), axis=-1)
        overdose = betaincc(a, b, self.target)
        return TPIPosterior(
            _owned(mean),
            _owned(sd),
            _owned(lower),
            _owned(upper),
            _owned(probability),
            _owned(_tpi_move(probability)),
            _owned(overdose),
            _owned((n >= 2) & (overdose > self.elimination_probability)),
        )

    def decision_table(self, max_patients: int = 30) -> MTPITable:
        """Rows DLTs 0..N, columns patients 1..N; U flags an unsafe dose."""
        value = scalar(max_patients, "max_patients")
        if value != int(value) or not 1 <= value <= 200:
            raise ValueError("max_patients must be an integer in [1,200]")
        n, y = np.arange(1, int(value) + 1), np.arange(int(value) + 1)
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
    ) -> tuple[FloatArray, TPIPosterior, NDArray[np.bool_]]:
        n, y = count(patients, "patients"), count(toxicities, "toxicities")
        if n.ndim != 1 or not 1 <= n.size <= 100 or y.shape != n.shape or n.sum() > 200:
            raise ValueError("require matching 1..100 dose vectors and total enrollment <=200")
        result = self.posterior(n, y)
        excluded = result.unsafe.copy()
        if eliminated is not None:
            given = np.asarray(eliminated)
            if given.shape != n.shape or given.dtype != bool:
                raise ValueError("eliminated must be a matching boolean vector")
            excluded |= given
        if excluded[0]:
            excluded[:] = True
        return n, result, excluded

    def next_dose(
        self,
        patients: ArrayLike,
        toxicities: ArrayLike,
        current_dose: int,
        *,
        eliminated: ArrayLike | None = None,
    ) -> BOINDecision:
        """Choose adjacent doses, retaining permanent exclusions.

        Bar escalation into an excluded dose and compare remaining probabilities.
        If current dose is excluded, de-escalate; if no adjacent dose is available,
        stop for safety. At dose limits, outward moves stay at the boundary dose.
        """
        n, result, excluded = self._state(patients, toxicities, eliminated)
        value = scalar(current_dose, "current_dose")
        if value != int(value) or not 1 <= value <= n.size or n[int(value) - 1] == 0:
            raise ValueError("current_dose must identify a treated dose")
        j = int(value) - 1
        scores = result.probability[j].copy()
        # Original exclusion rule sets the escalation score to zero, then compares
        # D and S. A -1 sentinel also handles numerically zero remaining tails.
        if j + 1 < n.size and excluded[j + 1]:
            scores[0] = -1
        move = -1 if excluded[j] else int(_tpi_move(scores))
        proposed = int(np.clip(j + move, 0, n.size - 1))
        if excluded[0] or excluded[proposed]:
            action, next_dose = "stop_safety", None
        else:
            action = "escalate" if proposed > j else "deescalate" if proposed < j else "stay"
            next_dose = proposed + 1
        return BOINDecision(action, next_dose, _owned(excluded), result.overdose_probability)

    def select_mtd(
        self,
        patients: ArrayLike,
        toxicities: ArrayLike,
        *,
        eliminated: ArrayLike | None = None,
        weights: ArrayLike | None = None,
    ) -> MTPISelection:
        """Isotonic posterior means of tried doses, equal weights unless supplied.

        Ties prefer highest dose at/below target, otherwise lowest. Native isotonic
        weights are unverified. Previously excluded doses remain inadmissible.
        """
        n, result, excluded = self._state(patients, toxicities, eliminated)
        w = np.ones(n.shape) if weights is None else finite(weights, "weights")
        if w.shape != n.shape or np.any(w <= 0):
            raise ValueError("weights must be a matching positive vector")
        treated = n > 0
        fitted = np.full(n.shape, np.nan)
        if treated.any():
            normalized = w[treated] / w[treated].max()
            if np.any(normalized <= 0):
                raise ArithmeticError("relative isotonic weights cannot be represented")
            fitted[treated] = isotonic_regression(result.mean[treated], weights=normalized).x
        admissible = treated & ~excluded
        indices = np.flatnonzero(admissible)
        chosen = None
        if indices.size:
            distance = abs(fitted[indices] - self.target)
            tied = indices[np.isclose(distance, distance.min(), rtol=0, atol=1e-14)]
            below = tied[fitted[tied] <= self.target + 1e-14]
            chosen = int(below[-1] if below.size else tied[0]) + 1
        return MTPISelection(
            chosen, _owned(fitted), result.overdose_probability, _owned(admissible)
        )
