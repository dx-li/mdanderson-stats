"""Posterior calculations and dose rules for the BOIN12 design.

BOIN12 combines BOIN toxicity boundaries with a quasi-beta-binomial posterior
for a clinician-specified toxicity/efficacy utility.  This module implements
the binary-endpoint rules described by Lin et al. (2020); time-to-event and
multilevel endpoint extensions are intentionally outside its scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import isotonic_regression
from scipy.special import betainc, betaincc
from scipy.stats import rankdata

from ._validation import count, finite, scalar
from .boin import BOINDesign, _owned

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _vector(value: ArrayLike, name: str) -> FloatArray:
    result = count(value, name)
    if result.ndim != 1 or result.size < 1:
        raise ValueError(f"{name} must be a nonempty one-dimensional count vector")
    return result


def _counts(
    patients: ArrayLike, toxicities: ArrayLike, efficacies: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    n, t, e = (
        _vector(value, name)
        for value, name in (
            (patients, "patients"),
            (toxicities, "toxicities"),
            (efficacies, "efficacies"),
        )
    )
    if n.shape != t.shape or n.shape != e.shape:
        raise ValueError("patients, toxicities, and efficacies must have matching shapes")
    if np.any(t > n) or np.any(e > n):
        raise ValueError("toxicities and efficacies cannot exceed patients")
    if np.any(n > 1000):
        raise ValueError("patients per dose must not exceed 1000")
    return n, t, e


def _utilities(value: ArrayLike) -> FloatArray:
    utilities = finite(value, "utilities")
    if utilities.shape != (4,) or np.any((utilities < 0) | (utilities > 100)):
        raise ValueError("utilities must contain four scores in [0,100]")
    if not utilities[0] > utilities[3]:
        raise ValueError("utilities must score no-toxicity/efficacy above toxicity/no-efficacy")
    return utilities


@dataclass(frozen=True)
class BOIN12Posterior:
    """Marginal endpoint and quasi-beta-binomial utility posterior summaries.

    ``utility_probability`` is a probability in ``[0, 1]``.  RDS display
    adapters may convert it to the source application's percentage scale.
    """

    toxicity_overdose_probability: FloatArray
    efficacy_futility_probability: FloatArray
    utility_mean: FloatArray
    utility_probability: FloatArray
    utility_events: FloatArray


@dataclass(frozen=True)
class BOIN12Decision:
    """Next-dose recommendation and the state used to obtain it."""

    action: str
    next_dose: int | None
    admissible: NDArray[np.bool_]
    posterior: BOIN12Posterior


@dataclass(frozen=True)
class BOIN12Selection:
    """Final isotonic toxicity MTD and utility-based OBD."""

    obd: int | None
    mtd: int | None
    admissible: NDArray[np.bool_]
    isotonic_toxicity: FloatArray
    posterior: BOIN12Posterior


@dataclass(frozen=True)
class BOIN12RDSTable:
    """Rank-based desirability scores for all binary outcomes at each sample size."""

    patients: IntArray
    toxicities: IntArray
    efficacies: IntArray
    admissible: NDArray[np.bool_]
    rds: FloatArray


def _additive_utilities(utilities: FloatArray) -> bool:
    return abs(float(utilities[0] + utilities[3] - utilities[1] - utilities[2])) <= 1e-12


def _joint_counts(
    n: FloatArray,
    t: FloatArray,
    e: FloatArray,
    efficacy_without_toxicity: ArrayLike | None,
    utilities: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return counts in (noT/E, noT/noE, T/E, T/noE) order."""

    additive = _additive_utilities(utilities)
    if efficacy_without_toxicity is None:
        if not additive:
            raise ValueError(
                "efficacy_without_toxicity is required unless utility scores are additive"
            )
        # The joint count cancels algebraically from the utility when the
        # cross-sum condition holds; any valid value gives the same events.
        n01 = np.zeros_like(n)
    else:
        n01 = _vector(efficacy_without_toxicity, "efficacy_without_toxicity")
        if n01.shape != n.shape or np.any(n01 > e) or np.any(n01 > n - t):
            raise ValueError(
                "efficacy_without_toxicity must be a valid subset of efficacy and non-toxicity"
            )
    n11 = e - n01
    n00 = n - t - e + n11
    n10 = t - n11
    if np.any(n00 < 0) or np.any(n10 < 0):
        raise ValueError("endpoint counts imply invalid joint toxicity/efficacy cells")
    return n01, n00, n11, n10


def posterior(
    patients: ArrayLike,
    toxicities: ArrayLike,
    efficacies: ArrayLike,
    *,
    toxicity_limit: float,
    efficacy_limit: float,
    utilities: ArrayLike = (100.0, 40.0, 60.0, 0.0),
    efficacy_without_toxicity: ArrayLike | None = None,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    utility_benchmark: float | None = None,
) -> BOIN12Posterior:
    """Compute BOIN12 endpoint and utility posterior summaries.

    Utility order is ``(no toxicity/efficacy, no toxicity/no efficacy,
    toxicity/efficacy, toxicity/no efficacy)``.  With the default utility
    scores, only marginal toxicity and efficacy counts are needed.
    """

    n, t, e = _counts(patients, toxicities, efficacies)
    phi_t, phi_e = (
        scalar(toxicity_limit, "toxicity_limit"),
        scalar(efficacy_limit, "efficacy_limit"),
    )
    if not 0 < phi_t < 1 or not 0 < phi_e < 1:
        raise ValueError("toxicity_limit and efficacy_limit must lie in (0,1)")
    utilities_array = _utilities(utilities)
    alpha, beta = scalar(prior_alpha, "prior_alpha"), scalar(prior_beta, "prior_beta")
    if alpha <= 0 or beta <= 0:
        raise ValueError("prior_alpha and prior_beta must be positive")
    additive = _additive_utilities(utilities_array)
    if efficacy_without_toxicity is None and additive:
        # The joint cell coefficient is zero, so marginal counts suffice.
        utility_sum = (
            utilities_array[1] * n
            + (utilities_array[0] - utilities_array[1]) * e
            + (utilities_array[3] - utilities_array[1]) * t
        )
    else:
        n01, n00, n11, n10 = _joint_counts(n, t, e, efficacy_without_toxicity, utilities_array)
        utility_sum = (
            utilities_array[0] * n01
            + utilities_array[1] * n00
            + utilities_array[2] * n11
            + utilities_array[3] * n10
        )
    utility_events = utility_sum / 100.0
    shape_a, shape_b = alpha + utility_events, beta + n - utility_events
    if np.any(shape_b <= 0):
        raise ValueError("utility posterior has nonpositive beta shape")
    benchmark = (
        float(utility_benchmark)
        if utility_benchmark is not None
        else _expected_utility(phi_t, phi_e, utilities_array)
        + (100.0 - _expected_utility(phi_t, phi_e, utilities_array)) / 2.0
    )
    benchmark = scalar(benchmark, "utility_benchmark")
    if not 0 < benchmark < 100:
        raise ValueError("utility_benchmark must lie in (0,100)")
    return BOIN12Posterior(
        _owned(betaincc(t + 1, n - t + 1, phi_t)),
        _owned(betainc(e + 1, n - e + 1, phi_e)),
        _owned(100.0 * shape_a / (shape_a + shape_b)),
        _owned(betaincc(shape_a, shape_b, benchmark / 100.0)),
        _owned(utility_events),
    )


def _expected_utility(toxicity: float, efficacy: float, utilities: FloatArray) -> float:
    probabilities = np.asarray(
        [
            (1 - toxicity) * efficacy,
            (1 - toxicity) * (1 - efficacy),
            toxicity * efficacy,
            toxicity * (1 - efficacy),
        ]
    )
    return float(probabilities @ utilities)


def admissibility(
    posterior_result: BOIN12Posterior,
    *,
    toxicity_cutoff: float = 0.95,
    efficacy_cutoff: float = 0.90,
    patients: ArrayLike | None = None,
) -> NDArray[np.bool_]:
    """Return doses meeting the BOIN12 safety and efficacy criteria."""

    ct, ce = scalar(toxicity_cutoff, "toxicity_cutoff"), scalar(efficacy_cutoff, "efficacy_cutoff")
    if not 0 < ct < 1 or not 0 < ce < 1:
        raise ValueError("toxicity_cutoff and efficacy_cutoff must lie in (0,1)")
    mask = (posterior_result.toxicity_overdose_probability < ct) & (
        posterior_result.efficacy_futility_probability < ce
    )
    if patients is not None:
        n = _vector(patients, "patients")
        if n.shape != mask.shape:
            raise ValueError("patients must match posterior vector")
        mask &= n > 0
    return _owned(mask)


def rank_desirability(
    sample_sizes: ArrayLike,
    *,
    toxicity_limit: float,
    efficacy_limit: float,
    utilities: ArrayLike = (100.0, 40.0, 60.0, 0.0),
    toxicity_cutoff: float = 0.95,
    efficacy_cutoff: float = 0.90,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    efficacy_without_toxicity: ArrayLike | None = None,
) -> BOIN12RDSTable:
    """Generate global RDS ranks, averaging ties among admissible outcomes."""

    sizes = _vector(sample_sizes, "sample_sizes")
    if np.any(sizes > 1000):
        raise ValueError("sample_sizes must not exceed 1000")
    row_count = int(np.sum((sizes.astype(np.int64) + 1) ** 2))
    if row_count > 100_000:
        raise ValueError("RDS enumeration exceeds the 100000-case safety limit")
    rows: list[tuple[int, int, int]] = []
    for size in sizes.astype(int):
        rows.extend(
            (int(size), tox, eff) for tox in range(int(size) + 1) for eff in range(int(size) + 1)
        )
    n = np.asarray([x[0] for x in rows], dtype=float)
    t = np.asarray([x[1] for x in rows], dtype=float)
    e = np.asarray([x[2] for x in rows], dtype=float)
    result = posterior(
        n,
        t,
        e,
        toxicity_limit=toxicity_limit,
        efficacy_limit=efficacy_limit,
        utilities=utilities,
        efficacy_without_toxicity=efficacy_without_toxicity,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
    )
    allowed = admissibility(
        result, toxicity_cutoff=toxicity_cutoff, efficacy_cutoff=efficacy_cutoff
    )
    rds = np.full(n.shape, np.nan)
    eligible = np.flatnonzero(allowed)
    if eligible.size:
        rds[eligible] = rankdata(result.utility_probability[eligible], method="average")
    return BOIN12RDSTable(
        np.asarray(n, dtype=np.int64),
        np.asarray(t, dtype=np.int64),
        np.asarray(e, dtype=np.int64),
        _owned(allowed),
        _owned(rds),
    )


@dataclass(frozen=True)
class BOIN12Design:
    """Binary-endpoint BOIN12 design with explicit clinical limits."""

    toxicity_limit: float
    efficacy_limit: float
    utilities: tuple[float, float, float, float] = (100.0, 40.0, 60.0, 0.0)
    toxicity_cutoff: float = 0.95
    efficacy_cutoff: float = 0.90
    exploration_patients: int = 9
    stay_patients: int = 6
    early_stop_patients: int | None = None
    _boin: BOINDesign = field(init=False, repr=False)

    def __post_init__(self) -> None:
        limit = scalar(self.toxicity_limit, "toxicity_limit")
        efficacy = scalar(self.efficacy_limit, "efficacy_limit")
        if not 0.05 <= limit <= 0.6:
            raise ValueError("toxicity_limit must lie in [.05,.6]")
        _utilities(self.utilities)
        if not 0 < efficacy < 1:
            raise ValueError("efficacy_limit must lie in (0,1)")
        if (
            not isinstance(self.exploration_patients, (int, np.integer))
            or self.exploration_patients < 0
        ):
            raise ValueError("exploration_patients must be a nonnegative integer")
        for name, value in (
            ("stay_patients", self.stay_patients),
            ("early_stop_patients", self.early_stop_patients),
        ):
            if value is not None and (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        object.__setattr__(self, "toxicity_limit", limit)
        object.__setattr__(self, "efficacy_limit", efficacy)
        object.__setattr__(self, "utilities", tuple(float(x) for x in self.utilities))
        object.__setattr__(self, "_boin", BOINDesign(target=limit))

    def posterior(
        self,
        patients: ArrayLike,
        toxicities: ArrayLike,
        efficacies: ArrayLike,
        *,
        efficacy_without_toxicity: ArrayLike | None = None,
    ) -> BOIN12Posterior:
        return posterior(
            patients,
            toxicities,
            efficacies,
            toxicity_limit=self.toxicity_limit,
            efficacy_limit=self.efficacy_limit,
            utilities=self.utilities,
            efficacy_without_toxicity=efficacy_without_toxicity,
        )

    def next_dose(
        self,
        patients: ArrayLike,
        toxicities: ArrayLike,
        efficacies: ArrayLike,
        current_dose: int,
        *,
        efficacy_without_toxicity: ArrayLike | None = None,
        eliminated: ArrayLike | None = None,
    ) -> BOIN12Decision:
        n, t, e = _counts(patients, toxicities, efficacies)
        if isinstance(current_dose, (bool, np.bool_)) or int(current_dose) != current_dose:
            raise ValueError("current_dose must be an integer dose index")
        current = int(current_dose)
        if not 1 <= current <= n.size or n[current - 1] == 0:
            raise ValueError("current_dose must identify a treated dose")
        result = self.posterior(n, t, e, efficacy_without_toxicity=efficacy_without_toxicity)
        all_allowed = admissibility(
            result, toxicity_cutoff=self.toxicity_cutoff, efficacy_cutoff=self.efficacy_cutoff
        ).copy()
        excluded = (
            np.zeros(n.size, dtype=bool)
            if eliminated is None
            else np.asarray(eliminated, dtype=bool)
        )
        if excluded.shape != n.shape:
            raise ValueError("eliminated must match the dose vector")
        all_allowed &= ~excluded
        index = current - 1
        rate = t[index] / n[index]
        local = np.zeros(n.size, dtype=bool)
        local[max(0, index - 1) : min(n.size, index + 2)] = True
        local &= all_allowed
        if excluded[index] or not np.any(all_allowed):
            return BOIN12Decision("stop_safety", None, local, result)
        if self.early_stop_patients is not None and n[index] >= self.early_stop_patients:
            return BOIN12Decision("stop_precision", None, local, result)
        if (
            n[index] >= self.exploration_patients
            and rate < self._boin.deescalation_boundary
            and index + 1 < n.size
            and n[index + 1] == 0
            and all_allowed[index + 1]
        ):
            return BOIN12Decision("explore_escalate", current + 1, local, result)
        if rate >= self._boin.deescalation_boundary:
            # The app's admissibility rule remains a safety guard at the
            # destination; do not force assignment to an inadmissible lower dose.
            lower = max(current - 1, 1)
            if all_allowed[lower - 1]:
                return BOIN12Decision("deescalate", lower, local, result)
            return BOIN12Decision("stop_no_admissible_neighbor", None, local, result)
        candidates = np.arange(max(0, index - 1), min(n.size, index + 2))
        if rate > self._boin.escalation_boundary and n[index] >= self.stay_patients:
            candidates = candidates[candidates <= index]
        candidates = candidates[local[candidates]]
        if candidates.size == 0:
            return BOIN12Decision("stop_no_admissible_neighbor", None, local, result)
        values = result.utility_probability[candidates]
        best = int(candidates[np.flatnonzero(values == values.max())[-1]])
        action = "stay" if best == index else "escalate" if best > index else "deescalate"
        return BOIN12Decision(action, best + 1, local, result)

    def select_obd(
        self,
        patients: ArrayLike,
        toxicities: ArrayLike,
        efficacies: ArrayLike,
        *,
        efficacy_without_toxicity: ArrayLike | None = None,
        eliminated: ArrayLike | None = None,
    ) -> BOIN12Selection:
        n, t, e = _counts(patients, toxicities, efficacies)
        result = self.posterior(n, t, e, efficacy_without_toxicity=efficacy_without_toxicity)
        allowed = admissibility(
            result,
            toxicity_cutoff=self.toxicity_cutoff,
            efficacy_cutoff=self.efficacy_cutoff,
            patients=n,
        ).copy()
        if eliminated is not None:
            excluded = np.asarray(eliminated, dtype=bool)
            if excluded.shape != n.shape:
                raise ValueError("eliminated must match the dose vector")
            allowed &= ~excluded
        fitted = np.full(n.shape, np.nan)
        treated = n > 0
        if np.any(treated):
            fitted[treated] = isotonic_regression(t[treated] / n[treated]).x
        if not np.any(allowed):
            return BOIN12Selection(None, None, allowed, _owned(fitted), result)
        observed = np.flatnonzero(treated)
        distances = np.abs(fitted[observed] - self.toxicity_limit)
        mtd = int(observed[np.flatnonzero(distances == distances.min())[-1]])
        eligible = np.flatnonzero(allowed & (np.arange(n.size) <= mtd))
        obd = (
            None
            if eligible.size == 0
            else int(eligible[np.argmax(result.utility_probability[eligible])])
        )
        return BOIN12Selection(
            None if obd is None else obd + 1, mtd + 1, allowed, _owned(fitted), result
        )
