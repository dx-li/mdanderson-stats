"""Posterior Predictive (PoP) phase-I dose-finding design."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import isotonic_regression
from scipy.special import betaincc

from ._validation import scalar
from .boin import _owned as _freeze


def _integer(value: int, name: str, lower: int, upper: int) -> int:
    number = scalar(value, name)
    if not lower <= number <= upper or number != int(number):
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return int(number)


def _allocation(
    j: int,
    under: np.ndarray,
    over: np.ndarray,
    under_hit: bool,
    over_hit: bool,
    up: bool,
    down: bool,
) -> tuple[str, int | None, np.ndarray, np.ndarray]:
    """Apply transition/exclusion indicators to owned, directionally closed masks."""
    under, over = under.copy(), over.copy()
    if under_hit:
        under[: j + 1] = True
    if over_hit:
        over[j:] = True
    excluded = under | over
    if np.all(excluded):
        return "stop", None, under, over
    candidate = j + (1 if up else -1 if down else 0)
    if not 0 <= candidate < under.size or excluded[candidate]:
        candidate = j
    if excluded[candidate]:
        raise ArithmeticError("allocation cannot remain at an excluded dose")
    action = "escalate" if candidate > j else "deescalate" if candidate < j else "stay"
    return action, candidate + 1, under, over


def predictive_bayes_factor(
    target: float, patients: ArrayLike, toxicities: ArrayLike, *, log: bool = False
) -> NDArray[np.float64]:
    """Return the PoP PrBF in favor of retaining a dose (H0 versus H1)."""
    if not isinstance(log, (bool, np.bool_)):
        raise ValueError("log must be boolean")
    phi = _validate_target(target)
    n, y = np.broadcast_arrays(
        np.asarray(patients, dtype=float), np.asarray(toxicities, dtype=float)
    )
    if n.size > 100000 or np.any(
        ~np.isfinite(n) | ~np.isfinite(y) | (n < 0) | (n > 1000) | (y < 0) | (y > n)
    ):
        raise ValueError("counts must be finite integers with 0 <= toxicities <= patients <= 1000")
    if np.any(n != n.astype(int)) or np.any(y != y.astype(int)):
        raise ValueError("patients and toxicities must be integer counts")
    q = (y + 1.0) / (n + 2.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        logbf = 1.0 + y * np.log(phi / q) + (n - y) * np.log((1.0 - phi) / (1.0 - q))
    return _freeze(logbf if log else np.exp(logbf))


@dataclass(frozen=True)
class PoPBoundaries:
    patients: NDArray[np.int64]
    escalate_max: NDArray[np.int64]
    deescalate_min: NDArray[np.int64]
    exclude_under_max: NDArray[np.int64]
    exclude_over_min: NDArray[np.int64]


@dataclass(frozen=True)
class PoPDecision:
    action: str
    next_dose: int | None
    excluded_under: NDArray[np.bool_]
    excluded_over: NDArray[np.bool_]
    excluded: NDArray[np.bool_]
    predictive_bayes_factor: float


@dataclass(frozen=True)
class PoPSelection:
    dose: int | None
    isotonic_estimate: NDArray[np.float64]
    eligible: NDArray[np.bool_]
    safety_excluded: NDArray[np.bool_]


def _validate_target(value: float) -> float:
    value = scalar(value, "target")
    if not 0.05 <= value <= 0.6:
        raise ValueError("target must be in [0.05, 0.6]")
    return value


@dataclass(frozen=True)
class PoPDesign:
    target: float = 0.25
    cutoff: float = 2.5
    exclusion_cutoff: float = 5.0 / 24.0
    safety_min_patients: int = 3

    def __post_init__(self) -> None:
        phi = _validate_target(self.target)
        c, e = scalar(self.cutoff, "cutoff"), scalar(self.exclusion_cutoff, "exclusion_cutoff")
        if not 1.0 < c < np.e or not 0.0 < e < c:
            raise ValueError("require 1 < cutoff < e and 0 < exclusion_cutoff < cutoff")
        minimum = _integer(self.safety_min_patients, "safety_min_patients", 0, 1000)
        object.__setattr__(self, "target", phi)
        object.__setattr__(self, "cutoff", c)
        object.__setattr__(self, "exclusion_cutoff", e)
        object.__setattr__(self, "safety_min_patients", minimum)

    def boundaries(self, max_patients: int, *, cohort_size: int = 1) -> PoPBoundaries:
        max_patients = _integer(max_patients, "max_patients", 1, 1000)
        cohort_size = _integer(cohort_size, "cohort_size", 1, max_patients)
        if max_patients % cohort_size:
            raise ValueError("max_patients must be divisible by cohort_size")
        ns: NDArray[np.int64] = np.arange(
            cohort_size, max_patients + 1, cohort_size, dtype=np.int64
        )
        rows = []
        for n in ns:
            y = np.arange(n + 1)
            bf = predictive_bayes_factor(self.target, n, y)
            low = y[(bf < self.cutoff) & (y / n < self.target)]
            high = y[(bf < self.cutoff) & (y / n >= self.target)]
            elow = y[(bf < self.exclusion_cutoff) & (y / n < self.target)]
            ehigh = y[(bf < self.exclusion_cutoff) & (y / n >= self.target)]
            rows.append(
                (
                    low[-1] if low.size else -1,
                    high[0] if high.size else n + 1,
                    elow[-1] if elow.size else -1,
                    ehigh[0] if ehigh.size else n + 1,
                )
            )
        a = np.asarray(rows, dtype=np.int64)
        return PoPBoundaries(*[_freeze(v) for v in (ns, a[:, 0], a[:, 1], a[:, 2], a[:, 3])])

    def _state(self, patients: ArrayLike, toxicities: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        n, y = np.broadcast_arrays(
            np.asarray(patients, dtype=float), np.asarray(toxicities, dtype=float)
        )
        if (
            n.ndim != 1
            or n.size < 2
            or n.size > 100
            or np.any(n > 1000)
            or np.any(~np.isfinite(n + y))
            or np.any(n < 0)
            or np.any(y < 0)
            or np.any(y > n)
        ):
            raise ValueError(
                "require matching one-dimensional dose counts, 0 <= toxicities <= patients"
            )
        if np.any(n != n.astype(int)) or np.any(y != y.astype(int)):
            raise ValueError("patients and toxicities must be integer counts")
        return n.astype(np.int64), y.astype(np.int64)

    def decision(
        self,
        current_dose: int,
        patients: ArrayLike,
        toxicities: ArrayLike,
        *,
        excluded_under: ArrayLike | None = None,
        excluded_over: ArrayLike | None = None,
        earlyterm: bool = True,
    ) -> PoPDecision:
        if not isinstance(earlyterm, (bool, np.bool_)):
            raise ValueError("earlyterm must be boolean")
        n, y = self._state(patients, toxicities)
        j = _integer(current_dose, "current_dose", 1, n.size) - 1
        under = (
            np.zeros(n.size, dtype=bool)
            if excluded_under is None
            else np.asarray(excluded_under, dtype=bool).copy()
        )
        over = (
            np.zeros(n.size, dtype=bool)
            if excluded_over is None
            else np.asarray(excluded_over, dtype=bool).copy()
        )
        if under.shape != n.shape or over.shape != n.shape:
            raise ValueError("exclusion masks must match dose counts")
        raw_under = np.asarray(excluded_under) if excluded_under is not None else np.zeros(n.size)
        raw_over = np.asarray(excluded_over) if excluded_over is not None else np.zeros(n.size)
        if np.any((raw_under != 0) & (raw_under != 1)) or np.any((raw_over != 0) & (raw_over != 1)):
            raise ValueError("exclusion masks must contain only boolean or 0/1 values")
        if np.any(under):
            under[: np.flatnonzero(under)[-1] + 1] = True
        if np.any(over):
            over[np.flatnonzero(over)[0] :] = True
        if (under[j] or over[j]) and not np.all(under | over):
            raise ValueError("current dose is excluded while an admissible dose remains")
        bf = float(predictive_bayes_factor(self.target, n[j], y[j]))
        low = bool(n[j] > 0 and y[j] / n[j] < self.target)
        high = bool(n[j] > 0 and not low)
        action, next_dose, under, over = _allocation(
            j,
            under,
            over,
            bool(earlyterm and bf < self.exclusion_cutoff and low),
            bool(earlyterm and bf < self.exclusion_cutoff and high),
            bool(bf < self.cutoff and low),
            bool(bf < self.cutoff and high),
        )
        return PoPDecision(
            action, next_dose, _freeze(under), _freeze(over), _freeze(under | over), bf
        )

    def select_mtd(self, patients: ArrayLike, toxicities: ArrayLike) -> PoPSelection:
        n, y = self._state(patients, toxicities)
        treated = n > 0
        estimate = np.full(n.size, np.nan)
        if np.any(treated):
            mean = (y[treated] + 0.05) / (n[treated] + 0.1)
            var = (
                (y[treated] + 0.05)
                * (n[treated] - y[treated] + 0.05)
                / ((n[treated] + 0.1) ** 2 * (n[treated] + 1.1))
            )
            estimate[treated] = isotonic_regression(mean, weights=1.0 / var).x
            estimate[treated] += (np.arange(treated.sum()) + 1) * 1e-10
        safety_excluded: NDArray[np.bool_] = np.zeros(n.size, dtype=bool)
        exceed = (n >= self.safety_min_patients) & (betaincc(y + 1, n - y + 1, self.target) > 0.95)
        hits = np.flatnonzero(exceed)
        if hits.size:
            safety_excluded[hits[0] :] = True
        eligible = treated & ~safety_excluded
        idx = np.flatnonzero(eligible)
        if not idx.size:
            return PoPSelection(
                None, _freeze(estimate), _freeze(eligible), _freeze(safety_excluded)
            )
        eligible_rank = np.flatnonzero(eligible)
        adjusted = estimate[eligible]
        distance = np.abs(adjusted - self.target)
        dose = int(eligible_rank[np.flatnonzero(distance == distance.min())[-1]]) + 1
        return PoPSelection(dose, _freeze(estimate), _freeze(eligible), _freeze(safety_excluded))
