"""EffTox desirability contour, posterior admissibility, and dose decisions."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import logsumexp

from ._validation import FloatArray, finite
from .efftox_model import EffToxFit, _owned


def _mask(value: ArrayLike) -> NDArray[np.bool_]:
    result = np.array(value, dtype=np.bool_, copy=True)
    result.flags.writeable = False
    return result


def _probability(value: float, name: str) -> float:
    candidate = finite(value, name)
    if candidate.ndim != 0:
        raise ValueError(f"{name} must be a scalar probability")
    result = float(candidate)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return result


@dataclass(frozen=True)
class EffToxContour:
    """Current EffTox Lp desirability contour through three equally valued points."""

    efficacy_intercept: float
    toxicity_intercept: float
    middle_efficacy: float
    middle_toxicity: float
    shape: float

    def __post_init__(self) -> None:
        e0 = _probability(self.efficacy_intercept, "efficacy_intercept")
        t0 = _probability(self.toxicity_intercept, "toxicity_intercept")
        e1 = _probability(self.middle_efficacy, "middle_efficacy")
        t1 = _probability(self.middle_toxicity, "middle_toxicity")
        p = float(finite(self.shape, "shape"))
        if not (0 < e0 < e1 < 1 and 0 < t1 < t0 < 1 and np.isfinite(p) and p > 0):
            raise ValueError("contour requires 0<e*<e1<1, 0<t1<t*<1 and shape>0")
        a = (1.0 - e1) / (1.0 - e0)
        b = t1 / t0
        if abs(float(logsumexp([p * np.log(a), p * np.log(b)]))) > 1e-8:
            raise ValueError("shape does not make the three contour points equally desirable")
        object.__setattr__(self, "efficacy_intercept", e0)
        object.__setattr__(self, "toxicity_intercept", t0)
        object.__setattr__(self, "middle_efficacy", e1)
        object.__setattr__(self, "middle_toxicity", t1)
        object.__setattr__(self, "shape", p)

    @classmethod
    def from_points(
        cls,
        efficacy_intercept: float,
        toxicity_intercept: float,
        middle_efficacy: float,
        middle_toxicity: float,
    ) -> EffToxContour:
        """Solve the unique positive Lp shape matching the three elicited points."""
        e0 = _probability(efficacy_intercept, "efficacy_intercept")
        t0 = _probability(toxicity_intercept, "toxicity_intercept")
        e1 = _probability(middle_efficacy, "middle_efficacy")
        t1 = _probability(middle_toxicity, "middle_toxicity")
        if not (0 < e0 < e1 < 1 and 0 < t1 < t0 < 1):
            raise ValueError("contour points require 0<e*<e1<1 and 0<t1<t*<1")
        a = (1.0 - e1) / (1.0 - e0)
        b = t1 / t0
        log_a, log_b = np.log(a), np.log(b)

        def objective(p: float) -> float:
            return float(logsumexp([p * log_a, p * log_b]))

        upper = 1.0
        while objective(upper) > 0:
            upper *= 2.0
            if upper > 1e16:
                raise ArithmeticError("could not bracket EffTox contour shape")
        shape = float(brentq(objective, np.nextafter(0.0, 1.0), upper, xtol=1e-13))
        return cls(e0, t0, e1, t1, shape)

    def utility(self, efficacy: ArrayLike, toxicity: ArrayLike) -> FloatArray:
        """Evaluate Lp desirability at marginal efficacy/toxicity probability pairs."""
        e, t = finite(efficacy, "efficacy"), finite(toxicity, "toxicity")
        if np.any((e < 0) | (e > 1)) or np.any((t < 0) | (t > 1)):
            raise ValueError("efficacy and toxicity must lie in [0,1]")
        with np.errstate(divide="ignore", invalid="ignore"):
            a = self.shape * (np.log1p(-e) - np.log1p(-self.efficacy_intercept))
            b = self.shape * (np.log(t) - np.log(self.toxicity_intercept))
        try:
            output_shape = np.broadcast_shapes(np.shape(a), np.shape(b))
        except ValueError as exc:
            raise ValueError("efficacy and toxicity inputs are not broadcast-compatible") from exc
        if prod(output_shape) > 200_000:
            raise ValueError("contour utility output exceeds 200000 cells")
        a, b = np.broadcast_arrays(a, b)
        log_norm = logsumexp(np.stack((a, b)), axis=0) / self.shape
        utility = -np.expm1(log_norm)
        if np.any(~np.isfinite(utility)):
            raise ArithmeticError("EffTox desirability calculation is nonfinite")
        return _owned(utility)


@dataclass(frozen=True)
class EffToxDecision:
    """Dose decision plus posterior admissibility probabilities and utilities."""

    action: Literal["start", "assign", "select", "stop_no_admissible", "stop_no_reachable"]
    dose: int | None
    admissible: NDArray[np.bool_]
    efficacy_tail_probability: FloatArray
    toxicity_tail_probability: FloatArray
    utility: FloatArray
    phase: Literal["interim", "final"]
    skip_policy: Literal["both", "escalation"]


def _tail_probability(draws: FloatArray, limit: float, *, upper: bool) -> FloatArray:
    if upper:
        if limit == 0:
            return _owned(np.ones(draws.shape[2]))
        if limit == 1:
            return _owned(np.zeros(draws.shape[2]))
        threshold = np.log(limit) - np.log1p(-limit)
        return _owned(np.mean(draws > threshold, axis=(0, 1)))
    if limit == 0:
        return _owned(np.zeros(draws.shape[2]))
    if limit == 1:
        return _owned(np.ones(draws.shape[2]))
    threshold = np.log(limit) - np.log1p(-limit)
    return _owned(np.mean(draws < threshold, axis=(0, 1)))


def efftox_decision(
    fit: EffToxFit,
    contour: EffToxContour,
    *,
    efficacy_limit: float,
    toxicity_limit: float,
    efficacy_probability: float,
    toxicity_probability: float,
    starting_dose: int,
    last_dose: int | None = None,
    phase: Literal["interim", "final"] = "interim",
    allow_untried_exploration: bool = True,
    skip_policy: Literal["both", "escalation"] = "both",
) -> EffToxDecision:
    """Choose a dose using EffTox posterior threshold rules and utility ordering.

    Dose indices in this function are one-based. Exploration admits the lowest
    untried dose above the starting dose on the toxicity criterion alone; the
    flag applies to interim and final decisions. Final selection maximizes
    utility over the resulting admissible set without a transition constraint.
    """
    if not isinstance(fit, EffToxFit) or not isinstance(contour, EffToxContour):
        raise ValueError("fit and contour must be EffToxFit and EffToxContour instances")
    if phase not in ("interim", "final"):
        raise ValueError("phase must be interim or final")
    if skip_policy not in ("both", "escalation"):
        raise ValueError("skip_policy must be both or escalation")
    if not isinstance(allow_untried_exploration, (bool, np.bool_)):
        raise ValueError("allow_untried_exploration must be boolean")
    e_limit = _probability(efficacy_limit, "efficacy_limit")
    t_limit = _probability(toxicity_limit, "toxicity_limit")
    e_cut = _probability(efficacy_probability, "efficacy_probability")
    t_cut = _probability(toxicity_probability, "toxicity_probability")
    dose_count = fit.doses.size
    if isinstance(starting_dose, (bool, np.bool_)) or not isinstance(
        starting_dose, (int, np.integer)
    ):
        raise ValueError("starting_dose must be a one-based dose index")
    start = int(starting_dose)
    if not 1 <= start <= dose_count:
        raise ValueError("starting_dose is outside the dose range")
    if last_dose is not None and (
        isinstance(last_dose, (bool, np.bool_))
        or not isinstance(last_dose, (int, np.integer))
        or not 1 <= int(last_dose) <= dose_count
    ):
        raise ValueError("last_dose must be None or a one-based dose index")
    if not isinstance(fit.counts, np.ndarray) or fit.counts.shape != (dose_count, 2, 2):
        raise ValueError("fit must retain dose-by-efficacy-by-toxicity counts")
    tried = fit.counts.sum(axis=(1, 2)) > 0
    if phase == "final" and not np.any(tried):
        raise ValueError("final selection requires at least one observed patient")
    if phase == "interim" and np.any(tried):
        if last_dose is None or not tried[int(last_dose) - 1]:
            raise ValueError("last_dose must identify a dose with observed patients")
    p_e = _tail_probability(fit.efficacy_logits, e_limit, upper=True)
    p_t = _tail_probability(fit.toxicity_logits, t_limit, upper=False)
    acceptable = (p_e > e_cut) & (p_t > t_cut)
    if allow_untried_exploration:
        untried_above = np.flatnonzero(~tried & (np.arange(dose_count) + 1 > start))
        if untried_above.size:
            exploration_index = int(untried_above[0])
            acceptable[exploration_index] |= p_t[exploration_index] > t_cut
    mean_e = fit.efficacy_probabilities.mean(axis=(0, 1))
    mean_t = fit.toxicity_probabilities.mean(axis=(0, 1))
    utility = contour.utility(mean_e, mean_t)
    if phase == "interim" and not np.any(tried):
        return EffToxDecision(
            "start", start, _mask(acceptable), p_e, p_t, utility, phase, skip_policy
        )
    eligible = np.flatnonzero(acceptable)
    if eligible.size == 0:
        return EffToxDecision(
            "stop_no_admissible", None, _mask(acceptable), p_e, p_t, utility, phase, skip_policy
        )
    if phase == "interim":
        assert last_dose is not None
        current = int(last_dose) - 1
        untried = ~tried
        reachable = np.ones(dose_count, dtype=bool)
        for candidate in eligible:
            low, high = sorted((current, int(candidate)))
            gap = untried[low + 1 : high]
            if candidate > current or skip_policy == "both":
                reachable[candidate] = not bool(np.any(gap))
        eligible = eligible[reachable[eligible]]
        if eligible.size == 0:
            if acceptable[current]:
                eligible = np.array([current])
            else:
                return EffToxDecision(
                    "stop_no_reachable",
                    None,
                    _mask(acceptable),
                    p_e,
                    p_t,
                    utility,
                    phase,
                    skip_policy,
                )
    best = int(eligible[np.argmax(utility[eligible])])
    return EffToxDecision(
        "assign" if phase == "interim" else "select",
        best + 1,
        _mask(acceptable),
        p_e,
        p_t,
        utility,
        phase,
        skip_policy,
    )
