"""Dirichlet posterior and allocation primitives for U-BOIN.

This module implements the finite categorical posterior used by U-BOIN.  It
does not decide which doses are tried, eliminated, or in which stage a trial
is operating; those conduct rules remain caller inputs.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betainc, betaincc

from ._validation import FloatArray, count, finite, scalar


def _readonly(value: ArrayLike, dtype: np.dtype | type = np.float64) -> NDArray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _integer_level(value: int, name: str, upper: int) -> int:
    candidate = scalar(value, name)
    if candidate != np.floor(candidate) or not 1 <= candidate < upper:
        raise ValueError(f"{name} must be an integer in [1, {upper - 1}]")
    return int(candidate)


def _probability(value: float, name: str) -> float:
    result = scalar(value, name)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


@dataclass(frozen=True)
class UBOINPosterior:
    """Per-dose posterior summaries for a categorical U-BOIN model."""

    posterior_shape: FloatArray
    mean_utility: FloatArray
    utility_variance: FloatArray
    overdose_probability: FloatArray
    low_efficacy_probability: FloatArray
    admissible: NDArray[np.bool_]
    n: FloatArray


def uboin_posterior(
    counts: ArrayLike,
    *,
    prior: ArrayLike,
    utilities: ArrayLike,
    toxicity_limit: float = 0.30,
    efficacy_limit: float = 0.20,
    safety_cutoff: float = 0.95,
    efficacy_cutoff: float = 0.90,
    dlt_level: int = 1,
    response_level: int = 1,
) -> UBOINPosterior:
    """Compute the exact Dirichlet posterior summaries used by U-BOIN.

    ``counts`` has shape ``(D, E, T)`` with efficacy and toxicity categories
    in ascending order.  ``prior`` may have shape ``(E,T)`` or ``(D,E,T)``;
    its entries are positive and its total mass is caller-selected.  The
    paper requires total mass one but does not specify equal cell masses.
    """
    try:
        count_shape = np.shape(counts)
    except (TypeError, ValueError) as exc:
        raise ValueError("counts must have shape (D,E,T)") from exc
    if len(count_shape) != 3:
        raise ValueError("counts must have shape (D,E,T)")
    d, e, t = count_shape
    if not 1 <= d <= 100 or not 2 <= e <= 3 or not 2 <= t <= 3:
        raise ValueError("counts must have shape D,E,T with D in 1..100 and E,T in 2..3")
    observed = count(counts, "counts")
    if np.any(observed > 1_000_000):
        raise ValueError("counts entries must be at most 1,000,000")
    try:
        prior_shape = np.shape(prior)
    except (TypeError, ValueError) as exc:
        raise ValueError("prior must have shape (E,T) or (D,E,T)") from exc
    if prior_shape not in ((e, t), (d, e, t)):
        raise ValueError("prior must have shape (E,T) or (D,E,T), with positive entries")
    prior_array = finite(prior, "prior")
    if np.any(prior_array <= 0):
        raise ValueError("prior must have shape (E,T) or (D,E,T), with positive entries")
    if np.any(prior_array > 1e300):
        raise ValueError("prior entries are too large for stable posterior evaluation")
    prior_by_dose = np.broadcast_to(prior_array, (d, e, t)).astype(np.float64, copy=True)
    posterior = prior_by_dose + observed
    if not np.all(np.isfinite(posterior)):
        raise ArithmeticError("posterior shape is not finite")
    totals = posterior.sum(axis=(1, 2))
    if not np.all(np.isfinite(totals)) or np.any(totals <= 0):
        raise ArithmeticError("posterior mass is not finite")

    if np.shape(utilities) != (e, t):
        raise ValueError("utilities must have shape (E,T), with finite values in [0,100]")
    utility = finite(utilities, "utilities")
    if np.any((utility < 0) | (utility > 100)):
        raise ValueError("utilities must have shape (E,T), with finite values in [0,100]")
    tox_level = _integer_level(dlt_level, "dlt_level", t)
    response = _integer_level(response_level, "response_level", e)
    _probability(toxicity_limit, "toxicity_limit")
    _probability(efficacy_limit, "efficacy_limit")
    safety_cutoff = _probability(safety_cutoff, "safety_cutoff")
    efficacy_cutoff = _probability(efficacy_cutoff, "efficacy_cutoff")

    # Normalize before multiplying by utilities: this also avoids overflow if
    # callers deliberately use a large but finite prior mass.
    probabilities = posterior / totals[:, None, None]
    flat_utility = utility.reshape(-1)
    flat_probability = probabilities.reshape(d, -1)
    mean_utility = flat_probability @ flat_utility
    centered = flat_utility[None, :] - mean_utility[:, None]
    utility_variance = (flat_probability * np.square(centered)).sum(axis=1) / (totals + 1)

    tox = posterior[:, :, tox_level:].sum(axis=(1, 2))
    no_tox = posterior[:, :, :tox_level].sum(axis=(1, 2))
    response_mass = posterior[:, response:, :].sum(axis=(1, 2))
    no_response = posterior[:, :response, :].sum(axis=(1, 2))
    overdose = betaincc(tox, no_tox, toxicity_limit)
    low_efficacy = betainc(response_mass, no_response, efficacy_limit)
    if not all(
        np.all(np.isfinite(x)) for x in (mean_utility, utility_variance, overdose, low_efficacy)
    ):
        raise ArithmeticError("posterior summary is not finite")
    admissible = (overdose <= safety_cutoff) & (low_efficacy <= efficacy_cutoff)
    return UBOINPosterior(
        _readonly(posterior),
        _readonly(mean_utility),
        _readonly(utility_variance),
        _readonly(overdose),
        _readonly(low_efficacy),
        _readonly(admissible, dtype=np.bool_),
        _readonly(observed.sum(axis=(1, 2))),
    )


def uboin_allocation(
    posterior: UBOINPosterior,
    *,
    eligible: ArrayLike,
    method: str = "winner",
) -> FloatArray:
    """Return deterministic or randomized allocation probabilities.

    ``eligible`` is caller-supplied and is intersected with posterior
    admissibility.  No tried-dose or elimination rule is inferred here.
    """
    if not isinstance(posterior, UBOINPosterior):
        raise TypeError("posterior must be a UBOINPosterior")
    means = finite(posterior.mean_utility, "posterior.mean_utility")
    admissible = np.asarray(posterior.admissible, dtype=bool)
    candidate = np.asarray(eligible)
    if candidate.shape != means.shape or candidate.dtype.kind != "b":
        raise ValueError("eligible must be a boolean vector matching the number of doses")
    if method not in ("winner", "proportional", "equal"):
        raise ValueError("method must be 'winner', 'proportional', or 'equal'")
    allowed = candidate & admissible
    probabilities = np.zeros(means.size, dtype=np.float64)
    indices = np.flatnonzero(allowed)
    if indices.size == 0:
        return _readonly(probabilities)
    if method == "winner":
        probabilities[indices[np.argmax(means[indices])]] = 1.0
    elif method == "equal":
        probabilities[indices] = 1.0 / indices.size
    else:
        selected = means[indices]
        if np.any(selected < 0) or not np.any(selected > 0):
            raise ValueError("proportional allocation requires a positive eligible utility")
        probabilities[indices] = selected / selected.sum()
    return _readonly(probabilities)
