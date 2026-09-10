"""Response-category/exponential-survival posterior of Huang et al. (2009)."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite


@dataclass(frozen=True)
class ResponseSurvivalPosterior:
    response_concentration: FloatArray
    survival_shape: FloatArray
    survival_scale: FloatArray
    probability_b_superior: float
    monte_carlo_standard_error: float
    draws: int


def _prior(value: ArrayLike, categories: int, name: str) -> FloatArray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    a = finite(value, name)
    if a.shape not in ((categories,), (2, categories)) or np.any(a <= 0):
        raise ValueError(f"{name} must be positive with shape (K,) or (2,K)")
    return np.broadcast_to(a, (2, categories)).copy()


def _draw_settings(draws: int, seed: int | None) -> None:
    if isinstance(draws, bool) or not isinstance(draws, int) or not 1 <= draws <= 1_000_000:
        raise ValueError("draws must be an integer in [1,1000000]")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")


def _comparison(
    concentration: FloatArray,
    shape: FloatArray,
    scale: FloatArray,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    wins = 0
    # Bounded memory, including many categories. Every draw compares independent
    # Dirichlet-weighted inverse-gamma means, not hazards or category rankings.
    batch = max(1, min(4096, 1_000_000 // concentration.size))
    for start in range(0, draws, batch):
        size = min(batch, draws - start)
        means = np.empty((size, 2))
        for arm in (0, 1):
            weights = rng.dirichlet(concentration[arm], size=size)
            gamma = rng.gamma(shape[arm], size=(size, shape.shape[1]))
            if np.any(gamma <= 0) or not np.all(np.isfinite(gamma)):
                raise ArithmeticError("inverse-gamma sampling exceeds floating-point range")
            with np.errstate(divide="ignore"):
                means[:, arm] = logsumexp(
                    np.log(weights) + np.log(scale[arm]) - np.log(gamma), axis=1
                )
            if not np.all(np.isfinite(means[:, arm])):
                raise ArithmeticError("mixture mean sampling is not finite")
        wins += int(np.sum(means[:, 1] > means[:, 0]))
    p = wins / draws
    return p, float(np.sqrt(p * (1 - p) / draws))


def response_survival_posterior(
    category_counts: ArrayLike,
    events: ArrayLike,
    exposure: ArrayLike,
    *,
    response_prior: ArrayLike,
    survival_shape: ArrayLike,
    survival_scale: ArrayLike,
    draws: int = 10000,
    seed: int | None = None,
) -> ResponseSurvivalPosterior:
    """Compare arms A=0/B=1 using (2,K) sufficient-statistic arrays.

    Response categories are observed immediately. Independent category means
    have inverse-gamma(shape, scale) priors; their expectation is scale/(shape-1)
    when shape>1. Events update shape and total censored/event time updates scale.
    The result estimates P(sum(p_B*mean_B)>sum(p_A*mean_A)).
    """
    if any(np.iscomplexobj(a) for a in (category_counts, events, exposure)):
        raise ValueError("sufficient statistics must be real")
    n, d, t = (
        count(category_counts, "category_counts"),
        count(events, "events"),
        finite(exposure, "exposure"),
    )
    if (
        n.ndim != 2
        or n.shape[0] != 2
        or not 1 <= n.shape[1] <= 100
        or d.shape != n.shape
        or t.shape != n.shape
    ):
        raise ValueError("sufficient statistics must have equal shape (2,K), 1<=K<=100")
    if np.any(d > n) or np.any(t < 0) or np.any((n == 0) & (t != 0)) or np.any((d > 0) & (t == 0)):
        raise ValueError(
            "require events<=counts and valid nonnegative exposure, positive for events"
        )
    _draw_settings(draws, seed)
    a = _prior(response_prior, n.shape[1], "response_prior") + n
    shape = _prior(survival_shape, n.shape[1], "survival_shape") + d
    scale = _prior(survival_scale, n.shape[1], "survival_scale") + t
    if not all(np.all(np.isfinite(x)) for x in (a, shape, scale)):
        raise ArithmeticError("posterior parameters overflow")
    p, se = _comparison(a, shape, scale, draws, np.random.default_rng(seed))
    return ResponseSurvivalPosterior(_freeze(a), _freeze(shape), _freeze(scale), p, se, draws)
