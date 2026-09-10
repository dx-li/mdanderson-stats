"""Informative BOIN prior elicitation and complete-outcome dose decisions."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln, logsumexp, xlog1py, xlogy

from ._validation import FloatArray, count, finite
from .bayesian_monitoring import _integer
from .boin import BOINDecision, BOINDesign, _owned


@dataclass(frozen=True)
class IBOINBoundaries:
    patients: NDArray[np.int64]
    escalation: FloatArray
    deescalation: FloatArray
    escalate_max: NDArray[np.int64]
    deescalate_min: NDArray[np.int64]
    eliminate_min: NDArray[np.int64]


@dataclass(frozen=True)
class IBOINDesign:
    """Historical skeleton and integer prior ESS; doses are one-based.

    Hypothesis columns are [target, safe, toxic]. The historical prior affects
    interval decisions only; safety uses the standard uniform Beta(1,1) prior.
    """

    skeleton: ArrayLike
    prior_ess: ArrayLike
    target: float = 0.25
    safe_probability: float | None = None
    toxic_probability: float | None = None
    elimination_probability: float = 0.95
    robust_prior: bool = False
    effective_prior_ess: NDArray[np.int64] = field(init=False)
    log_hypothesis_probability: FloatArray = field(init=False, repr=False)
    _boin: BOINDesign = field(init=False, repr=False)

    def __post_init__(self) -> None:
        base = BOINDesign(
            self.target,
            self.safe_probability,
            self.toxic_probability,
            elimination_probability=self.elimination_probability,
        )
        q, ess = finite(self.skeleton, "skeleton"), count(self.prior_ess, "prior_ess")
        if (
            q.ndim != 1
            or not 2 <= q.size <= 100
            or ess.shape != q.shape
            or np.any((q < 0) | (q > 1))
            or np.any(np.diff(q) < 0)
            or np.any(ess > 10000)
        ):
            raise ValueError(
                "require 2..100 ordered probabilities and matching integer ESS 0..10000"
            )
        if not isinstance(self.robust_prior, (bool, np.bool_)):
            raise ValueError("robust_prior must be boolean")
        effective = ess.astype(np.int64)
        if self.robust_prior:
            matches = np.flatnonzero(q == base.target)
            if matches.size != 1:
                raise ValueError("robust prior requires exactly one skeleton value equal to target")
            prior_mtd = int(matches[0]) + 1
            if 2 * prior_mtd >= q.size:
                effective[prior_mtd:] = 0
        object.__setattr__(self, "effective_prior_ess", _owned(effective))
        phi = np.array([base.target, base.safe_probability, base.toxic_probability], dtype=float)
        log_prior = np.empty((q.size, 3))
        # Vectorize historical counts and hypotheses, keeping memory O(max ESS).
        for j, (probability, size) in enumerate(zip(q, effective, strict=True)):
            x = np.arange(int(size) + 1)
            likelihood = x[:, None] * np.log(phi) + (size - x[:, None]) * np.log1p(-phi)
            hypothesis = likelihood - logsumexp(likelihood, axis=1, keepdims=True)
            binomial = (
                gammaln(size + 1)
                - gammaln(x + 1)
                - gammaln(size - x + 1)
                + xlogy(x, probability)
                + xlog1py(size - x, -probability)
            )
            log_prior[j] = logsumexp(hypothesis + binomial[:, None], axis=0)
        log_prior -= logsumexp(log_prior, axis=1, keepdims=True)
        object.__setattr__(self, "skeleton", _owned(q))
        object.__setattr__(self, "prior_ess", _owned(ess.astype(np.int64)))
        object.__setattr__(self, "log_hypothesis_probability", _owned(log_prior))
        object.__setattr__(self, "_boin", base)
        for name in ["target", "safe_probability", "toxic_probability", "elimination_probability"]:
            object.__setattr__(self, name, getattr(base, name))

    @property
    def hypothesis_probability(self) -> FloatArray:
        """Dose-by-hypothesis probabilities; use logs for extremely small values."""
        return _owned(np.exp(self.log_hypothesis_probability))

    def boundaries(self, patients: ArrayLike) -> IBOINBoundaries:
        """Dose-by-count boundaries for a vector of positive evaluated sample sizes.

        Uses untruncated equation (2.3) crossings, matching native NA cells. Overlapping
        boundaries raise instead of inventing an undocumented conflict rule.
        """
        n = count(patients, "patients")
        if n.ndim != 1 or not 1 <= n.size <= 10000 or np.any((n < 1) | (n > 100000)):
            raise ValueError("patients must contain 1..10000 counts in [1,100000]")
        base = self._boin
        assert base.safe_probability is not None and base.toxic_probability is not None
        phi, safe, toxic = base.target, base.safe_probability, base.toxic_probability
        de = (np.log(phi) - np.log(safe)) + np.log1p((phi - safe) / (1 - phi))
        dd = (np.log(toxic) - np.log(phi)) + np.log1p((toxic - phi) / (1 - toxic))
        prior = self.log_hypothesis_probability
        le = base.escalation_boundary + (prior[:, 1] - prior[:, 0])[:, None] / n / de
        ld = base.deescalation_boundary + (prior[:, 0] - prior[:, 2])[:, None] / n / dd
        if np.any(le >= ld):
            raise ArithmeticError(
                "informative prior gives overlapping boundaries at requested counts"
            )
        e = np.floor(n * np.clip(le, 0, 1)).astype(np.int64)
        d = np.ceil(n * np.clip(ld, 0, 1)).astype(np.int64)
        # Match inclusive conduct comparisons at exactly representable cutoffs.
        e += (e + 1) / n <= le
        e -= e / n > le
        d -= (d - 1) / n >= ld
        d += d / n < ld
        return IBOINBoundaries(
            _owned(n.astype(np.int64)),
            _owned(le),
            _owned(ld),
            _owned(np.clip(e, -1, n).astype(np.int64)),
            _owned(np.clip(d, 0, n + 1).astype(np.int64)),
            _owned(base._safety_boundary(n.astype(np.int64), base.elimination_probability)),
        )

    def next_dose(
        self,
        patients: ArrayLike,
        toxicities: ArrayLike,
        current_dose: int,
        *,
        eliminated: ArrayLike | None = None,
    ) -> BOINDecision:
        """Assign a dose after complete outcomes, retaining previous exclusions."""
        n, y, excluded, posterior = self._boin._state(patients, toxicities, eliminated)
        if n.size != self.log_hypothesis_probability.shape[0]:
            raise ValueError("counts must match the skeleton's dose count")
        j = _integer(current_dose, "current_dose") - 1
        if not 0 <= j < n.size or not 1 <= n[j] <= 100000:
            raise ValueError("current_dose must identify a dose with 1..100000 evaluated patients")
        if excluded[0]:
            action, next_dose = "stop_safety", None
        elif excluded[j]:
            action, next_dose = "deescalate", int(np.flatnonzero(~excluded)[-1]) + 1
        else:
            boundaries = self.boundaries([n[j]])
            rate = y[j] / n[j]
            move = (
                1
                if rate <= boundaries.escalation[j, 0]
                else -1
                if rate >= boundaries.deescalation[j, 0]
                else 0
            )
            proposed = max(0, min(j + move, n.size - 1))
            if excluded[proposed]:
                proposed = j
            action = "escalate" if proposed > j else "deescalate" if proposed < j else "stay"
            next_dose = proposed + 1
        return BOINDecision(action, next_dose, _owned(excluded), _owned(posterior))
