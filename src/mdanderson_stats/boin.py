"""BOIN single-agent dose decisions, safety boundaries and isotonic MTD selection."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import isotonic_regression
from scipy.special import betaincc, betaincinv

from ._validation import FloatArray, count, scalar


def _owned(value: ArrayLike) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class BOINBoundaryTable:
    patients: NDArray[np.int64]
    escalate_max: NDArray[np.int64]
    deescalate_min: NDArray[np.int64]
    eliminate_min: NDArray[np.int64]
    lowest_stop_min: NDArray[np.int64]


@dataclass(frozen=True)
class BOINDecision:
    action: str
    next_dose: int | None
    eliminated: NDArray[np.bool_]
    overdose_probability: FloatArray


@dataclass(frozen=True)
class BOINSelection:
    dose: int | None
    eliminated: NDArray[np.bool_]
    isotonic_mean: FloatArray
    selection_mean: FloatArray
    isotonic_interval: FloatArray
    report_overdose_probability: FloatArray
    safety_overdose_probability: FloatArray


@dataclass(frozen=True)
class BOINDesign:
    """Local BOIN design. Dose indices are one-based; safety elimination is sticky.

    Uniform Beta(1,1) priors govern safety. Final estimation follows BOIN's
    Beta(.05,.05) working prior with inverse-variance weighted isotonic regression.
    All records accept complete, evaluable outcomes; pending outcomes are not DLTs.
    """

    target: float = 0.25
    safe_probability: float | None = None
    toxic_probability: float | None = None
    elimination_probability: float = 0.95
    extra_safe: bool = False
    safety_offset: float = 0.05
    stay_at_one_of_three: bool = False
    deescalate_at_two_of_six: bool = False
    bound_mtd: bool = False
    early_stop_patients: int | None = None
    escalation_boundary: float = field(init=False)
    deescalation_boundary: float = field(init=False)

    def __post_init__(self) -> None:
        target = scalar(self.target, "target")
        safe = (
            0.6 * target
            if self.safe_probability is None
            else scalar(self.safe_probability, "safe_probability")
        )
        toxic = (
            1.4 * target
            if self.toxic_probability is None
            else scalar(self.toxic_probability, "toxic_probability")
        )
        cutoff = scalar(self.elimination_probability, "elimination_probability")
        offset = scalar(self.safety_offset, "safety_offset")
        if not 0.05 <= target <= 0.6 or not 0 < safe < target < toxic < 1:
            raise ValueError("require target in [.05, .6] and 0 < safe < target < toxic < 1")
        if not 0 < cutoff < 1 or not 0 <= offset < cutoff:
            raise ValueError("require elimination_probability in (0,1) and 0 <= offset < cutoff")
        for name in ("extra_safe", "stay_at_one_of_three", "deescalate_at_two_of_six", "bound_mtd"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
        if self.stay_at_one_of_three and not 0.25 <= target <= 0.279:
            raise ValueError("the 1/3 modification requires target in [.25,.279]")
        if self.deescalate_at_two_of_six and not 0.28 <= target <= 0.33:
            raise ValueError("the 2/6 modification requires target in [.28,.33]")
        if self.early_stop_patients is not None:
            stop = scalar(self.early_stop_patients, "early_stop_patients")
            if stop != int(stop) or not 3 <= stop <= 100_000:
                raise ValueError("early_stop_patients must be an integer in [3,100000]")
        # Likelihood-ratio crossings; log1p preserves close probability differences.
        low = np.log1p((target - safe) / (1 - target))
        high = np.log1p((toxic - target) / (1 - toxic))
        le = low / (low + np.log1p((target - safe) / safe))
        ld = high / (high + np.log1p((toxic - target) / target))
        if not safe < le < target < ld < toxic:
            raise ArithmeticError("BOIN boundaries are not distinguishable at floating precision")
        for name, value in (
            ("target", target),
            ("safe_probability", safe),
            ("toxic_probability", toxic),
            ("elimination_probability", cutoff),
            ("safety_offset", offset),
            ("escalation_boundary", le),
            ("deescalation_boundary", ld),
        ):
            object.__setattr__(self, name, float(value))

    def _safety_boundary(self, n: NDArray[np.int64], cutoff: float) -> NDArray[np.int64]:
        low, high = np.full(n.shape, -1), n + 1
        while np.any(high - low > 1):
            middle = (low + high) // 2
            valid_middle = np.maximum(middle, 0)
            unsafe = betaincc(valid_middle + 1, n - valid_middle + 1, self.target) > cutoff
            active = high - low > 1
            high = np.where(active & unsafe, middle, high)
            low = np.where(active & ~unsafe, middle, low)
        return np.where(n >= 3, high, n + 1)

    def boundary_table(self, max_patients: int = 30) -> BOINBoundaryTable:
        """Rows 1..max_patients. A safety cutoff n+1 means no possible elimination."""
        maximum = scalar(max_patients, "max_patients")
        if maximum != int(maximum) or not 1 <= maximum <= 100_000:
            raise ValueError("max_patients must be an integer in [1,100000]")
        n = np.arange(1, int(maximum) + 1)
        escalate = np.floor(n * self.escalation_boundary).astype(np.int64)
        deescalate = np.ceil(n * self.deescalation_boundary).astype(np.int64)
        if self.stay_at_one_of_three:
            deescalate = np.where(n == 3, 2, deescalate)
        if self.deescalate_at_two_of_six:
            deescalate = np.where(n == 6, 2, deescalate)
        eliminate = self._safety_boundary(n, self.elimination_probability)
        stop = (
            self._safety_boundary(n, self.elimination_probability - self.safety_offset)
            if self.extra_safe
            else eliminate
        )
        return BOINBoundaryTable(*[_owned(v) for v in (n, escalate, deescalate, eliminate, stop)])

    def _state(
        self, patients: ArrayLike, toxicities: ArrayLike, eliminated: ArrayLike | None
    ) -> tuple[FloatArray, FloatArray, NDArray[np.bool_], FloatArray]:
        n, y = count(patients, "patients"), count(toxicities, "toxicities")
        if (
            n.ndim != 1
            or n.shape != y.shape
            or not 2 <= n.size <= 100
            or np.any(y > n)
            or n.sum() >= 2**53
        ):
            raise ValueError("require matching 1D counts for 2..100 doses, y<=n and total<2**53")
        excluded = np.zeros(n.shape, dtype=bool)
        if eliminated is not None:
            prior = np.asarray(eliminated)
            if prior.shape != n.shape or prior.dtype != np.bool_:
                raise ValueError("eliminated must be a matching boolean vector")
            excluded |= prior
        posterior = betaincc(y + 1, n - y + 1, self.target)
        excluded |= (n >= 3) & (posterior > self.elimination_probability)
        if (
            self.extra_safe
            and n[0] >= 3
            and posterior[0] > self.elimination_probability - self.safety_offset
        ):
            excluded[0] = True
        excluded = np.maximum.accumulate(excluded)
        return n, y, excluded, np.asarray(posterior)

    def next_dose(
        self,
        patients: ArrayLike,
        toxicities: ArrayLike,
        current_dose: int,
        *,
        eliminated: ArrayLike | None = None,
    ) -> BOINDecision:
        """Decision after an evaluated cohort, including inherited safety exclusions.

        At a dose-range boundary, an unavailable escalation/de-escalation becomes
        stay. Precision stopping requires the underlying BOIN rate rule to say
        stay, not merely clipping an escalation to the highest dose.
        """
        n, y, excluded, posterior = self._state(patients, toxicities, eliminated)
        dose = scalar(current_dose, "current_dose")
        if dose != int(dose) or not 1 <= dose <= len(n):
            raise ValueError("current_dose must be a valid one-based dose index")
        j = int(dose) - 1
        if n[j] == 0:
            raise ValueError("the current dose must have evaluated patients")
        if excluded[0]:
            action, next_dose = "stop_safety", None
        else:
            rate = y[j] / n[j]
            move = (
                1
                if rate <= self.escalation_boundary
                else -1
                if rate >= self.deescalation_boundary
                else 0
            )
            if self.stay_at_one_of_three and n[j] == 3 and y[j] == 1:
                move = 0
            if self.deescalate_at_two_of_six and n[j] == 6 and y[j] == 2:
                move = -1
            if excluded[j]:
                next_j = int(np.flatnonzero(~excluded)[-1])
                action, next_dose = "deescalate", next_j + 1
            elif (
                move == 0
                and self.early_stop_patients is not None
                and n[j] >= self.early_stop_patients
            ):
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
    ) -> BOINSelection:
        """Select only treated, noneliminated doses; None means no admissible MTD.

        Reporting fits include all treated doses, as in BOIN; selection refits only
        admissible doses. The isotonic interval transforms marginal beta quantiles
        and is a source reporting convention, not a joint posterior credible band.
        """
        n, y, excluded, safety = self._state(patients, toxicities, eliminated)
        treated = n > 0
        mean = (y + 0.05) / (n + 0.1)
        variance = (y + 0.05) * (n - y + 0.05) / ((n + 0.1) ** 2 * (n + 1.1))
        fitted = np.full(n.shape, np.nan)
        selection = fitted.copy()
        interval = np.full((len(n), 2), np.nan)
        report_overdose = fitted.copy()
        if np.any(treated):
            weights = 1 / variance[treated]
            fitted[treated] = isotonic_regression(mean[treated], weights=weights).x
            for side, probability in enumerate((0.025, 0.975)):
                quantiles = betaincinv(
                    y[treated] + 0.05, n[treated] - y[treated] + 0.05, probability
                )
                interval[treated, side] = isotonic_regression(quantiles, weights=weights).x
            report_overdose[treated] = isotonic_regression(
                betaincc(y[treated] + 0.05, n[treated] - y[treated] + 0.05, self.target)
            ).x
        admissible = treated & ~excluded
        chosen = None
        if np.any(admissible):
            selection[admissible] = isotonic_regression(
                mean[admissible], weights=1 / variance[admissible]
            ).x
            if self.bound_mtd:
                admissible &= selection <= self.deescalation_boundary
            indices = np.flatnonzero(admissible)
            if indices.size:
                distance = np.abs(selection[indices] - self.target)
                tied = indices[np.isclose(distance, distance.min(), rtol=0, atol=1e-14)]
                chosen = int(tied[-1] if np.all(selection[tied] < self.target) else tied[0]) + 1
        return BOINSelection(
            chosen,
            _owned(excluded),
            _owned(fitted),
            _owned(selection),
            _owned(interval),
            _owned(report_overdose),
            _owned(safety),
        )
