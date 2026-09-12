"""Complete-outcome conduct rules for U-BOIN.

The paper leaves the scope of the admissible set (tried doses versus all
configured doses) and several boundary tie conventions implicit.  This
module therefore requires ``candidate_scope`` explicitly and documents the
remaining conduct choices in the design docstrings.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betaincc

from ._validation import FloatArray, count, finite, scalar
from .boin import BOINDesign
from .uboin import UBOINPosterior, uboin_allocation, uboin_posterior


def _readonly(value: ArrayLike, dtype: np.dtype | type = np.float64) -> NDArray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _int_scalar(value: int, name: str, low: int, high: int) -> int:
    candidate = scalar(value, name)
    if candidate != np.floor(candidate) or not low <= candidate <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(candidate)


def _probability(value: float, name: str) -> float:
    result = scalar(value, name)
    if not 0 < result < 1:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return result


@dataclass(frozen=True)
class UBOINDecision:
    stage: int
    action: str
    allocation_probabilities: FloatArray
    next_dose: int | None
    selected_dose: int | None
    eliminated: NDArray[np.bool_]
    posterior: UBOINPosterior


@dataclass(frozen=True)
class UBOINSelection:
    posterior: UBOINPosterior
    eligible: NDArray[np.bool_]
    dose: int | None


@dataclass(frozen=True)
class UBOINDesign:
    """Immutable U-BOIN conduct settings for complete categorical outcomes.

    ``candidate_scope`` is required because the paper does not specify
    whether B2 and final OBD selection include untried doses.  ``"tried"``
    uses only doses with observed patients; ``"all"`` permits prior-only
    doses.  Stage-I safety uses the paper's uniform Beta(1,1) overdose rule.
    Stage-II admissibility is dose-wise and does not create new suffix
    eliminations; any suffix mask returned from Stage I remains sticky.
    """

    prior: ArrayLike
    utilities: ArrayLike
    candidate_scope: str
    toxicity_limit: float = 0.30
    efficacy_limit: float = 0.20
    delta: float = 0.05
    safety_cutoff: float = 0.95
    efficacy_cutoff: float = 0.90
    s1: int = 12
    s2: int = 24
    max_patients: int = 54
    starting_dose: int = 1
    method: str = "winner"
    run_in_3plus3: bool = False
    dlt_level: int = 1
    response_level: int = 1
    _boin: BOINDesign = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.candidate_scope not in ("tried", "all"):
            raise ValueError("candidate_scope must be 'tried' or 'all'")
        if self.method not in ("winner", "proportional", "equal"):
            raise ValueError("method must be 'winner', 'proportional', or 'equal'")
        tox = _probability(self.toxicity_limit, "toxicity_limit")
        eff = _probability(self.efficacy_limit, "efficacy_limit")
        delta = scalar(self.delta, "delta")
        if not 0 < delta < tox:
            raise ValueError("delta must lie strictly between 0 and toxicity_limit")
        if not 0.05 <= tox - delta <= 0.6:
            raise ValueError("toxicity_limit - delta must be a valid BOIN target in [0.05,0.6]")
        if not isinstance(self.run_in_3plus3, (bool, np.bool_)):
            raise ValueError("run_in_3plus3 must be boolean")
        if self.run_in_3plus3 and not np.isclose(tox - delta, 0.25, rtol=0, atol=1e-12):
            raise ValueError("run_in_3plus3 requires toxicity_limit - delta=0.25")
        safety = _probability(self.safety_cutoff, "safety_cutoff")
        efficacy = _probability(self.efficacy_cutoff, "efficacy_cutoff")
        s1 = _int_scalar(self.s1, "s1", 1, 1000)
        s2 = _int_scalar(self.s2, "s2", 1, 1000)
        maximum = _int_scalar(self.max_patients, "max_patients", 1, 1000)
        if s2 <= s1:
            raise ValueError("s2 must be greater than s1")
        if s1 > maximum:
            raise ValueError("s1 must not exceed max_patients")
        prior_shape = np.shape(self.prior)
        if len(prior_shape) not in (2, 3) or prior_shape[-2:] not in (
            (2, 2), (2, 3), (3, 2), (3, 3)
        ):
            raise ValueError("prior must have shape (E,T) or (D,E,T), with E,T in {2,3}")
        if len(prior_shape) == 3 and not 1 <= prior_shape[0] <= 100:
            raise ValueError("prior dose dimension must lie in [1,100]")
        prior = finite(self.prior, "prior")
        if np.any(prior <= 0) or np.any(prior > 1e300):
            raise ValueError("prior must contain positive finite values no larger than 1e300")
        if np.shape(self.utilities) != prior_shape[-2:]:
            raise ValueError("utilities must match prior categories and lie in [0,100]")
        utilities = finite(self.utilities, "utilities")
        if np.any((utilities < 0) | (utilities > 100)):
            raise ValueError("utilities must match prior categories and lie in [0,100]")
        starting = _int_scalar(self.starting_dose, "starting_dose", 1, 100)
        dlt = _int_scalar(self.dlt_level, "dlt_level", 1, prior_shape[-1] - 1)
        response = _int_scalar(self.response_level, "response_level", 1, prior_shape[-2] - 1)
        object.__setattr__(self, "prior", _readonly(prior))
        object.__setattr__(self, "utilities", _readonly(utilities))
        for name, value in (("toxicity_limit", tox), ("efficacy_limit", eff), ("delta", delta),
                            ("safety_cutoff", safety), ("efficacy_cutoff", efficacy),
                            ("s1", s1), ("s2", s2), ("max_patients", maximum)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "starting_dose", starting)
        object.__setattr__(self, "dlt_level", dlt)
        object.__setattr__(self, "response_level", response)
        object.__setattr__(self, "_boin", BOINDesign(target=tox - delta))

    def _validate_counts(self, counts: ArrayLike) -> tuple[NDArray[np.float64], int, int, int]:
        shape = np.shape(counts)
        if len(shape) != 3:
            raise ValueError("counts must have shape (D,E,T)")
        d, e, t = shape
        if not 1 <= d <= 100 or self.prior.shape[-2:] != (e, t):
            raise ValueError("counts dose/category dimensions do not match design")
        observed = count(counts, "counts")
        if observed.shape != shape or np.any(observed > 1_000_000):
            raise ValueError("counts entries must be at most 1,000,000")
        total = observed.sum()
        if total > 1000:
            raise ValueError("total count must be at most 1,000")
        if self.prior.ndim == 3 and self.prior.shape[0] != d:
            raise ValueError("prior dose dimension must match counts")
        _int_scalar(self.starting_dose, "starting_dose", 1, d)
        return observed, d, e, t

    def _posterior(self, counts: NDArray[np.float64]) -> UBOINPosterior:
        return uboin_posterior(
            counts,
            prior=self.prior,
            utilities=self.utilities,
            toxicity_limit=self.toxicity_limit,
            efficacy_limit=self.efficacy_limit,
            safety_cutoff=self.safety_cutoff,
            efficacy_cutoff=self.efficacy_cutoff,
            dlt_level=self.dlt_level,
            response_level=self.response_level,
        )

    def _selection(
        self, counts: NDArray[np.float64], eliminated: NDArray[np.bool_]
    ) -> UBOINSelection:
        posterior = self._posterior(counts)
        tried = counts.sum(axis=(1, 2)) > 0
        eligible = tried if self.candidate_scope == "tried" else np.ones(tried.size, dtype=bool)
        eligible &= ~eliminated & posterior.admissible
        probabilities = uboin_allocation(posterior, eligible=eligible, method="winner")
        dose = (
            int(np.flatnonzero(probabilities == 1)[0] + 1)
            if np.count_nonzero(probabilities)
            else None
        )
        return UBOINSelection(posterior, _readonly(eligible, dtype=np.bool_), dose)

    def select_obd(
        self, counts: ArrayLike, *, eliminated: ArrayLike | None = None
    ) -> UBOINSelection:
        """Select the posterior-mean OBD; caller supplies any Stage-I mask."""
        observed, d, _, _ = self._validate_counts(counts)
        mask = self._mask(eliminated, d)
        return self._selection(observed, mask)

    @staticmethod
    def _mask(eliminated: ArrayLike | None, d: int) -> NDArray[np.bool_]:
        if eliminated is None:
            return np.zeros(d, dtype=bool)
        mask = np.asarray(eliminated)
        if mask.shape != (d,) or mask.dtype.kind != "b":
            raise ValueError("eliminated must be a boolean vector matching dose count")
        return np.array(mask, dtype=bool, copy=True)

    def decision(
        self,
        counts: ArrayLike,
        *,
        current_dose: int,
        stage: int = 1,
        eliminated: ArrayLike | None = None,
    ) -> UBOINDecision:
        """Determine the next complete-outcome cohort assignment.

        The returned stage is the effective stage after the ``s1`` transition.
        Boundary clamping and lowest-index utility ties are explicit Python
        choices because the paper does not specify those conventions.
        """
        observed, d, _, _ = self._validate_counts(counts)
        current = _int_scalar(current_dose, "current_dose", 1, d)
        stage = _int_scalar(stage, "stage", 1, 2)
        mask = self._mask(eliminated, d)
        n = observed.sum(axis=(1, 2))
        total = int(n.sum())
        if total and n[current - 1] == 0:
            raise ValueError("current_dose must have treated patients once data exist")
        posterior = self._posterior(observed)

        # Stage-I uniform-Beta overdose control at the configured upper limit.
        if stage == 1:
            dlt = observed[:, :, self.dlt_level :].sum(axis=(1, 2))
            safety_n = n >= 3
            overdose = betaincc(dlt + 1, n - dlt + 1, self.toxicity_limit)
            new_suffix = safety_n & (overdose > 0.95)
            if np.any(new_suffix):
                first = int(np.flatnonzero(new_suffix)[0])
                mask[first:] = True
            if mask[0]:
                return self._result(1, "stop_safety", posterior, mask, d)

        effective_stage = 2 if stage == 2 or np.any(n >= self.s1) else 1
        if mask[0]:
            return self._result(effective_stage, "stop_safety", posterior, mask, d)
        if effective_stage == 1 and mask[current - 1]:
            target = self._available_target(current - 1, current, mask)
            if target is None:
                return self._result(effective_stage, "stop_safety", posterior, mask, d)
            return self._result(
                effective_stage,
                "deescalate",
                posterior,
                mask,
                d,
                self._onehot(d, target),
                target,
            )
        if total == 0:
            if stage == 2:
                raise ValueError("stage 2 requires observed patients")
            if mask[self.starting_dose - 1]:
                return self._result(effective_stage, "stop_safety", posterior, mask, d)
            probabilities = np.zeros(d)
            probabilities[self.starting_dose - 1] = 1
            return self._result(1, "start", posterior, mask, d, probabilities, self.starting_dose)
        if total >= self.max_patients:
            selection = self._selection(observed, mask)
            return self._result(
                effective_stage,
                "stop_max_patients",
                selection.posterior,
                mask,
                d,
                selected=selection.dose,
            )
        if np.any(n >= self.s2):
            selection = self._selection(observed, mask)
            return self._result(
                effective_stage,
                "stop_s2",
                selection.posterior,
                mask,
                d,
                selected=selection.dose,
            )

        if effective_stage == 1:
            rate = float(observed[current - 1, :, self.dlt_level :].sum() / n[current - 1])
            boin = self._boin
            if self.run_in_3plus3 and n[current - 1] in (3, 6):
                events = int(observed[current - 1, :, self.dlt_level :].sum())
                if n[current - 1] == 3 and events == 1:
                    desired = current
                elif n[current - 1] == 6 and events >= 2:
                    desired = current - 1
                elif n[current - 1] == 3 and events >= 2:
                    desired = current - 1
                else:
                    desired = current + 1
            elif rate <= boin.escalation_boundary:
                desired = current + 1
            elif rate >= boin.deescalation_boundary:
                desired = current - 1
            else:
                desired = current
            desired = min(max(desired, 1), d)
            target = self._available_target(desired, current, mask)
            if target is None:
                return self._result(1, "stop_safety", posterior, mask, d)
            if target > current:
                action = "escalate"
            elif target < current:
                action = "deescalate"
            else:
                action = "stay"
            return self._result(1, action, posterior, mask, d, self._onehot(d, target), target)

        tried = np.flatnonzero(n > 0)
        highest = int(tried[-1])
        rate = float(observed[highest, :, self.dlt_level :].sum() / n[highest])
        boin = self._boin
        if rate <= boin.escalation_boundary and highest + 1 < d and not mask[highest + 1]:
            target = highest + 2
            return self._result(2, "escalate", posterior, mask, d, self._onehot(d, target), target)
        selection = self._selection(observed, mask)
        if selection.dose is None:
            return self._result(2, "stop_no_admissible", selection.posterior, mask, d)
        probabilities = uboin_allocation(
            selection.posterior, eligible=selection.eligible, method=self.method
        )
        next_dose = (
            int(np.flatnonzero(probabilities == 1)[0] + 1)
            if np.count_nonzero(probabilities) == 1
            else None
        )
        return self._result(2, "assign", selection.posterior, mask, d, probabilities, next_dose)

    @staticmethod
    def _onehot(d: int, target: int) -> FloatArray:
        result = np.zeros(d)
        result[target - 1] = 1
        return _readonly(result)

    @staticmethod
    def _available_target(desired: int, current: int, mask: NDArray[np.bool_]) -> int | None:
        if not mask[desired - 1]:
            return desired
        if desired > current:
            return current if not mask[current - 1] else None
        lower = np.flatnonzero(~mask[:current - 1])
        return int(lower[-1] + 1) if lower.size else None

    @staticmethod
    def _result(
        stage: int,
        action: str,
        posterior: UBOINPosterior,
        eliminated: NDArray[np.bool_],
        d: int,
        probabilities: ArrayLike | None = None,
        next_dose: int | None = None,
        selected: int | None = None,
    ) -> UBOINDecision:
        return UBOINDecision(
            stage,
            action,
            _readonly(np.zeros(d) if probabilities is None else probabilities),
            next_dose,
            selected,
            _readonly(eliminated, dtype=np.bool_),
            posterior,
        )
