"""Stage-two allocation and dose selection rules for BARD.

The functions here implement the mathematical stage-two rules independently of
the Shiny application.  Counts from stage one may be passed by the caller;
this module deliberately does not infer eligibility or impose a cap rule.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betainc, betaincc

from ._validation import FloatArray, count, finite, scalar


def _readonly(value: ArrayLike, dtype: np.dtype | type = np.float64) -> NDArray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _validate_categorical(
    value: ArrayLike, name: str, *, rows: int | None = None
) -> NDArray[np.int64]:
    array = count(value, name)
    if array.ndim != 2 or (rows is not None and array.shape[0] != rows):
        raise ValueError(f"{name} must be a two-dimensional array with {rows} rows")
    if array.shape[1] < 1 or array.shape[1] > 5 or np.any(array < 1):
        raise ValueError(f"{name} must contain 1..5 positive categorical factors")
    return array.astype(np.int64)


@dataclass(frozen=True)
class BARDMinimizationResult:
    scores: FloatArray
    probabilities: FloatArray
    assigned_arm: int

    @property
    def score1(self) -> float:
        return float(self.scores[0])

    @property
    def score2(self) -> float:
        return float(self.scores[1])

    @property
    def probability1(self) -> float:
        return float(self.probabilities[0])

    @property
    def probability2(self) -> float:
        return float(self.probabilities[1])


def bard_minimization(
    history_arms: ArrayLike,
    history_factors: ArrayLike,
    new_factors: ArrayLike,
    *,
    probability: float = 0.95,
    tie_probability: float = 0.5,
    seed: int | None = None,
) -> BARDMinimizationResult:
    """Assign one patient using BARD's covariate-adaptive minimization.

    The score for a candidate arm is the sum over factors of the absolute
    imbalance at the new patient's observed level after that candidate arm is
    added.  ``history_*`` are the combined eligible stage-one/stage-two rows
    supplied by the caller.
    """
    arms = count(history_arms, "history_arms")
    if arms.ndim != 1 or np.any((arms != 1) & (arms != 2)):
        raise ValueError("history_arms must be a one-dimensional array of arm codes 1 or 2")
    factors = _validate_categorical(history_factors, "history_factors", rows=arms.size)
    new = count(new_factors, "new_factors")
    if new.ndim != 1 or new.size != factors.shape[1] or np.any(new < 1):
        raise ValueError("new_factors must have one positive categorical value per factor")
    if arms.size * factors.shape[1] > 100000:
        raise ValueError("history contains more than 100000 factor cells")
    p = scalar(probability, "probability")
    tie = scalar(tie_probability, "tie_probability")
    if not 0 <= p <= 1 or not 0 <= tie <= 1:
        raise ValueError("probability and tie_probability must lie in [0,1]")

    scores: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
    for candidate in (1, 2):
        for factor in range(factors.shape[1]):
            matching = factors[:, factor] == new[factor]
            arm1 = np.count_nonzero((arms == 1)[matching]) + (candidate == 1)
            arm2 = np.count_nonzero((arms == 2)[matching]) + (candidate == 2)
            scores[candidate - 1] += abs(arm1 - arm2)
    probability1 = tie if scores[0] == scores[1] else p if scores[0] < scores[1] else 1 - p
    probabilities = np.array([probability1, 1 - probability1])
    assigned = 1 if np.random.default_rng(seed).random() < probability1 else 2
    return BARDMinimizationResult(_readonly(scores), _readonly(probabilities), assigned)


@dataclass(frozen=True)
class BARDSelectionResult:
    posterior_shape: FloatArray
    mean_utility: FloatArray
    observed_rates: FloatArray
    overdose_probability: FloatArray
    adjusted_overdose_probability: FloatArray
    low_efficacy_probability: FloatArray
    admissible: NDArray[np.bool_]
    selected_arm: int | None

    @property
    def posterior(self) -> FloatArray:
        return self.posterior_shape

    @property
    def adjusted_overdose(self) -> FloatArray:
        return self.adjusted_overdose_probability

    @property
    def raw_overdose(self) -> FloatArray:
        return self.overdose_probability

    @property
    def low_efficacy(self) -> FloatArray:
        return self.low_efficacy_probability


def _validate_prior(prior: ArrayLike) -> FloatArray:
    value = finite(prior, "prior")
    if value.shape not in ((4,), (2, 4)) or np.any(value <= 0):
        raise ValueError("prior must have shape (4,) or (2,4), with positive finite entries")
    return value


def bard_select_obd(
    counts: ArrayLike,
    *,
    prior: ArrayLike,
    safety_weights: ArrayLike,
    toxicity_limit: float = 0.3,
    efficacy_limit: float = 0.2,
    safety_cutoff: float = 0.95,
    efficacy_cutoff: float = 0.95,
    method: str = "utility",
    utilities: ArrayLike = (0, 30, 50, 100),
    margin: float = 0.05,
    tie_arm: int = 1,
) -> BARDSelectionResult:
    """Select the stage-two optimal biological dose from joint outcome counts.

    Joint columns are ``(T/noE, noT/noE, T/E, noT/E)``.  ``safety_weights``
    controls the explicitly caller-chosen two-arm isotonic safety pooling when
    the lower arm has the larger raw overdose probability.
    """
    n = count(counts, "counts")
    if n.shape != (2, 4) or np.any(n > 1_000_000):
        raise ValueError("counts must have shape (2,4), with entries in [0,1_000_000]")
    a = _validate_prior(prior)
    prior2 = np.broadcast_to(a, (2, 4)).astype(np.float64, copy=True)
    weights = finite(safety_weights, "safety_weights")
    if weights.shape != (2,) or np.any(weights <= 0):
        raise ValueError("safety_weights must contain two positive finite values")
    tox_limit, eff_limit = (
        scalar(toxicity_limit, "toxicity_limit"),
        scalar(efficacy_limit, "efficacy_limit"),
    )
    safe_cut, eff_cut = (
        scalar(safety_cutoff, "safety_cutoff"),
        scalar(efficacy_cutoff, "efficacy_cutoff"),
    )
    delta = scalar(margin, "margin")
    if (
        not 0 <= tox_limit <= 1
        or not 0 <= eff_limit <= 1
        or not 0 <= safe_cut <= 1
        or not 0 <= eff_cut <= 1
    ):
        raise ValueError("limits and cutoffs must lie in [0,1]")
    if (
        delta < 0
        or method not in ("utility", "noninferiority", "noninferior")
        or tie_arm not in (1, 2)
    ):
        raise ValueError("invalid method, margin, or tie_arm")
    u = finite(utilities, "utilities")
    if u.shape != (4,):
        raise ValueError("utilities must contain four finite values")
    posterior = prior2 + n
    totals = posterior.sum(axis=1)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
        raise ArithmeticError("posterior sums are not finite")
    mean_utility = posterior @ u / totals
    tox_a, tox_b = posterior[:, 0] + posterior[:, 2], posterior[:, 1] + posterior[:, 3]
    eff_a, eff_b = posterior[:, 2] + posterior[:, 3], posterior[:, 0] + posterior[:, 1]
    overdose = betaincc(tox_a, tox_b, tox_limit)
    low_eff = betainc(eff_a, eff_b, eff_limit)
    adjusted = overdose.copy()
    if overdose[0] > overdose[1]:
        adjusted[:] = np.average(overdose, weights=weights)
    admissible = (adjusted <= safe_cut) & (low_eff <= eff_cut)
    selected: int | None
    eligible = np.flatnonzero(admissible)
    if eligible.size == 0:
        selected = None
    elif eligible.size == 1:
        selected = int(eligible[0]) + 1
    elif method in ("noninferiority", "noninferior"):
        totals_observed = n[:, 0] + n[:, 1] + n[:, 2] + n[:, 3]
        if np.any(totals_observed < 1):
            raise ValueError(
                "noninferiority selection requires at least one observed outcome per arm"
            )
        margin_fraction = Fraction(str(delta))
        left = int(n[0, 2] + n[0, 3]) * int(totals_observed[1])
        right = int(n[1, 2] + n[1, 3]) * int(totals_observed[0])
        selected = (
            1
            if left - right >= -margin_fraction * int(totals_observed[0]) * int(totals_observed[1])
            else 2
        )
    else:
        best = max(mean_utility[eligible])
        tied = eligible[
            np.isclose(mean_utility[eligible], best, rtol=0, atol=32 * np.finfo(float).eps)
        ]
        selected = tie_arm if tie_arm - 1 in tied else int(tied[0]) + 1
    observed_total = n.sum(axis=1)
    observed_rates = (n[:, 2] + n[:, 3]) / np.where(observed_total == 0, 1, observed_total)
    return BARDSelectionResult(
        _readonly(posterior),
        _readonly(mean_utility),
        _readonly(observed_rates),
        _readonly(overdose),
        _readonly(adjusted),
        _readonly(low_eff),
        _readonly(admissible, bool),
        selected,
    )
