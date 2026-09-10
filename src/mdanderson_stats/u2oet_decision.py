"""Posterior acceptability and cohort allocation for the U2OET design."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .u2oet import _real


def _integer(value: int | float, name: str, minimum: int, maximum: int) -> int:
    a = _real(value, name)
    if a.ndim or a != np.floor(a) or not minimum <= a <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    return int(a)


@dataclass(frozen=True)
class U2OETCriteria:
    """Paper's example cutoffs; specify suitable values for other designs."""

    efficacy_level: int = 2
    toxicity_level: int = 2
    min_efficacy: float = 0.40
    max_toxicity: float = 0.45
    inefficacy_cutoff: float = 0.90
    toxicity_cutoff: float = 0.90

    def __post_init__(self) -> None:
        for name in ("efficacy_level", "toxicity_level"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1, 3))
        for name in ("min_efficacy", "max_toxicity", "inefficacy_cutoff", "toxicity_cutoff"):
            value = _real(getattr(self, name), name)
            if value.ndim or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a probability")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class U2OETPosterior:
    """Equal-weight posterior Monte Carlo summaries; all arrays are dose grids."""

    mean_utility: FloatArray
    inefficacy_probability: FloatArray
    toxicity_probability: FloatArray
    acceptable: np.ndarray
    draws: int


def u2oet_posterior(
    joint_draws: ArrayLike, utility: ArrayLike, *, criteria: U2OETCriteria = U2OETCriteria()
) -> U2OETPosterior:
    """Summarize supplied posterior joint-probability draws, without fitting.

    Axes: draw, agent 1, agent 2, efficacy, toxicity. Every draw/dose slice must
    sum to one. Strict inequalities follow paper equations (6) and (7).
    """
    p = _real(joint_draws, "joint_draws")
    if (
        p.ndim != 5
        or not 1 <= p.shape[0]
        or p.size > 20_000_000
        or any(not 2 <= n <= 5 for n in p.shape[1:3])
        or any(not 2 <= n <= 4 for n in p.shape[3:])
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError(
            "require posterior draws on 2–5 doses and 2–4 categories; at most 20M cells"
        )
    if not np.all(np.abs(p.sum(axis=(-2, -1)) - 1) <= 1e-12):
        raise ValueError("each posterior draw and dose must have unit joint probability")
    if not isinstance(criteria, U2OETCriteria):
        raise ValueError("criteria must be U2OETCriteria")
    if criteria.efficacy_level >= p.shape[-2] or criteria.toxicity_level >= p.shape[-1]:
        raise ValueError("event thresholds must be below the number of outcome categories")
    u = _real(utility, "utility")
    if u.shape != p.shape[-2:] or np.any(u < 0):
        raise ValueError("utility must be a nonnegative efficacy-by-toxicity matrix")
    # Average probabilities first: avoids a second draw-sized utility array.
    mean = np.sum(p.mean(axis=0) * u, axis=(-2, -1))
    if not np.all(np.isfinite(mean)):
        raise ArithmeticError("posterior utility exceeds floating-point range")
    efficacy = p[..., criteria.efficacy_level :, :].sum(axis=(-2, -1))
    toxicity = p[..., criteria.toxicity_level :].sum(axis=(-2, -1))
    pe = np.mean(efficacy < criteria.min_efficacy, axis=0)
    pt = np.mean(toxicity > criteria.max_toxicity, axis=0)
    acceptable = (pe <= criteria.inefficacy_cutoff) & (pt <= criteria.toxicity_cutoff)
    mask = np.frombuffer(acceptable.tobytes(), dtype=bool).reshape(acceptable.shape)
    return U2OETPosterior(_freeze(mean), _freeze(pe), _freeze(pt), mask, p.shape[0])


@dataclass(frozen=True)
class U2OETAllocation:
    """Next-cohort assignment distribution, not an RNG draw or final selection."""

    probabilities: FloatArray
    candidate_mask: np.ndarray
    best: tuple[int, int] | None
    randomized: bool
    reason: str


def u2oet_allocation(
    posterior: U2OETPosterior,
    treated: ArrayLike,
    *,
    surplus: int = 3,
    top: int | None = 2,
    initial: tuple[int, int] | None = None,
    greedy: bool = False,
) -> U2OETAllocation:
    """Paper section 3.3 and guide section 1.6, at a new-cohort boundary.

    Counts include all assigned patients, not only completed outcomes. Untried
    agent levels cannot be skipped above the highest previously tried level.
    Ties use ascending agent-1 then agent-2 indices. If no patients were treated,
    an explicit initial pair is required; the protocol starting pair overrides
    prior acceptability. This function does not manage an open cohort.
    """
    n = _real(treated, "treated")
    if not isinstance(posterior, U2OETPosterior):
        raise ValueError("posterior must be U2OETPosterior")
    u = _real(posterior.mean_utility, "mean_utility")
    mask = np.asarray(posterior.acceptable)
    if (
        n.ndim != 2
        or any(not 2 <= m <= 5 for m in n.shape)
        or n.shape != u.shape
        or mask.shape != n.shape
        or mask.dtype != bool
        or np.any(n < 0)
        or np.any(n != np.floor(n))
        or n.sum() >= 2**53
        or np.any(u < 0)
    ):
        raise ValueError(
            "require matching dose grids, nonnegative utilities and integer treated counts"
        )
    surplus = _integer(surplus, "surplus", 0, 2**53 - 1)
    if top is not None:
        top = _integer(top, "top", 1, n.size)
    if not isinstance(greedy, (bool, np.bool_)):
        raise ValueError("greedy must be boolean")
    weights = np.zeros(n.shape)
    candidates = mask.copy()
    best = None
    randomized = False
    if n.sum() == 0:
        if initial is None or len(initial) != 2:
            raise ValueError("the first cohort requires an explicit initial dose pair")
        i, j = (_integer(v, "initial index", 0, n.shape[a] - 1) for a, v in enumerate(initial))
        weights[i, j] = 1
        candidates[:] = False
        candidates[i, j] = True
        best = (i, j)
        reason = "initial cohort"
    else:
        # Both agents may advance one level; an untried combination of already
        # tried individual levels is eligible, even if the pair is far away.
        tried1, tried2 = np.nonzero(n > 0)
        candidates[int(tried1.max()) + 2 :, :] = False
        candidates[:, int(tried2.max()) + 2 :] = False
        indices = np.flatnonzero(candidates)
        if indices.size == 0:
            reason = "no acceptable eligible dose pair"
        else:
            ranked = indices[np.argsort(-u.ravel()[indices], kind="stable")]
            b = int(ranked[0])
            best = (b // n.shape[1], b % n.shape[1])
            use_ar = (
                not greedy
                and top != 1
                and ranked.size > 1
                and n.ravel()[b] - n.ravel()[ranked[1:]].max() >= surplus
            )
            if use_ar:
                selected = ranked if top is None else ranked[:top]
                values = u.ravel()[selected]
                scale = float(values.max())
                if scale == 0:
                    raise ValueError(
                        "utility-proportional randomization is undefined for all-zero utilities"
                    )
                values = values / scale
                weights.ravel()[selected] = values / values.sum()
                randomized = bool(np.count_nonzero(weights) > 1)
                reason = "utility-proportional allocation after patient surplus"
            else:
                weights.ravel()[b] = 1
                reason = "single candidate" if ranked.size == 1 else "greedy allocation"
    immutable_mask = np.frombuffer(candidates.tobytes(), dtype=bool).reshape(candidates.shape)
    return U2OETAllocation(_freeze(weights), immutable_mask, best, randomized, reason)
