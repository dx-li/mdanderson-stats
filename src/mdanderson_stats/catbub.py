"""Categorical-outcome Bayesian utility comparisons (CATBUB)."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite, scalar
from .beta_binomial import BetaBinomialPosterior
from .beta_comparison import BetaComparison, compare_beta_binomial


def _integer(value: int, name: str, maximum: int = 1_000_000) -> int:
    x = scalar(value, name)
    if x != np.floor(x) or not 1 <= x <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return int(x)


def _probabilities(value: ArrayLike, name: str) -> FloatArray:
    p = finite(value, name)
    if p.ndim < 2 or p.shape[-2] != 2 or p.shape[-1] < 2 or np.any(p < 0):
        raise ValueError(f"{name} must end in (2, K), K >= 2, with nonnegative entries")
    # Source gen.data normalizes row weights. Scale first to avoid overflow.
    scale = p.max(axis=-1, keepdims=True)
    if np.any(scale <= 0):
        raise ValueError(f"{name} rows must have positive mass")
    p = p / scale
    return p / p.sum(axis=-1, keepdims=True)


def _inputs(
    counts: ArrayLike, utility: ArrayLike, prior_ess: float, prior_probability: ArrayLike | None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    x, u = count(counts, "counts"), finite(utility, "utility")
    if x.ndim < 2 or x.shape[-2] != 2 or u.ndim != 1 or x.shape[-1] != len(u) or len(u) < 2:
        raise ValueError("counts must end in (2, K) and utility must have length K >= 2")
    if np.any(x.sum(axis=-1) >= 2**53):
        raise ValueError("arm totals must be smaller than 2**53")
    ess = scalar(prior_ess, "prior_ess")
    if not 0 < ess < 2**53:
        raise ValueError("prior_ess must be positive and smaller than 2**53")
    p = (
        np.full((2, len(u)), 1 / len(u))
        if prior_probability is None
        else _probabilities(prior_probability, "prior_probability")
    )
    if p.shape != (2, len(u)) or np.any(p <= 0):
        raise ValueError("prior_probability must be a strictly positive (2, K) matrix")
    a = x + ess * p
    if np.any(a <= 0) or np.any(a.sum(axis=-1) >= 2**53):
        raise ValueError("posterior shapes must be positive with arm totals smaller than 2**53")
    return x, u, a


@dataclass(frozen=True)
class CatbubComparison:
    """Ordering is (..., B>A/A>B). Errors are quadrature estimates or MCSEs."""

    probability: FloatArray
    error: FloatArray
    scaled_mean: FloatArray
    scaled_variance: FloatArray
    beta_shapes: FloatArray
    method: str
    draws: int


def catbub_compare(
    counts: ArrayLike,
    utility: ArrayLike,
    *,
    prior_ess: float = 1,
    prior_probability: ArrayLike | None = None,
    method: str = "beta",
    draws: int = 100_000,
    rng: np.random.Generator | None = None,
    absolute_tolerance: float = 1e-8,
) -> CatbubComparison:
    """Compare independent Dirichlet posterior mean utilities in two arms.

    Supports batched count tables. Beta mode matches the first two moments of
    utility scaled to [0, 1]. MC mode samples the actual Dirichlet posterior.
    Constant utilities give zero probability of either strict ordering.
    """
    x, u, a = _inputs(counts, utility, prior_ess, prior_probability)
    if method not in ("beta", "mc"):
        raise ValueError("method must be beta or mc")
    if method == "mc":
        draws = _integer(draws, "draws", 10_000_000)
        if not isinstance(rng, np.random.Generator):
            raise TypeError("MC comparison requires a numpy Generator")
    scale = float(np.max(np.abs(u)))
    v = u / scale if scale else u.copy()
    width = float(v.max() - v.min())
    v = (v - v.min()) / width if width else np.zeros_like(v)
    total = a.sum(axis=-1)
    weights = a / total[..., None]
    mean = np.sum(weights * v, axis=-1)
    complement = np.sum(weights * (1 - v), axis=-1)
    variance = np.sum(weights * (v - mean[..., None]) ** 2, axis=-1) / (total + 1)
    shape = x.shape[:-2]
    probability = np.zeros((*shape, 2))
    error = np.zeros_like(probability)
    beta_shapes = np.full((*shape, 2, 2), np.nan)
    if width and method == "beta":
        concentration = mean * complement / variance - 1
        beta_shapes = np.stack((mean * concentration, complement * concentration), axis=-1)
        if np.any(beta_shapes <= 0) or not np.all(np.isfinite(beta_shapes)):
            raise ArithmeticError("scaled-beta moments cannot be resolved")
        comparison = compare_beta_binomial(
            BetaBinomialPosterior(beta_shapes[..., 0, 0], beta_shapes[..., 0, 1]),
            BetaBinomialPosterior(beta_shapes[..., 1, 0], beta_shapes[..., 1, 1]),
            absolute_tolerance=absolute_tolerance,
        )
        probability = np.stack((comparison.treatment_greater, comparison.control_greater), axis=-1)
        error = np.broadcast_to(comparison.absolute_error[..., None], probability.shape)
    elif width:
        assert rng is not None
        for index in np.ndindex(shape):
            wins = np.zeros(2)
            # Bound temporary draws, including large numbers of outcome categories.
            batch = max(1, min(4096, 1_000_000 // len(u)))
            for start in range(0, draws, batch):
                size = min(batch, draws - start)
                ua = rng.dirichlet(a[index][0], size=size) @ v
                ub = rng.dirichlet(a[index][1], size=size) @ v
                if not np.all(np.isfinite(ua)) or not np.all(np.isfinite(ub)):
                    raise ArithmeticError("Dirichlet sampling returned nonfinite utilities")
                wins += [np.count_nonzero(ub > ua), np.count_nonzero(ua > ub)]
            probability[index] = wins / draws
        error = np.sqrt(probability * (1 - probability) / draws)
    return CatbubComparison(
        _freeze(probability),
        _freeze(error),
        _freeze(mean),
        _freeze(variance),
        _freeze(beta_shapes),
        method,
        draws if method == "mc" else 0,
    )


def catbub_binary_compare(
    counts: ArrayLike, *, prior: ArrayLike = ((0.5, 0.5), (0.5, 0.5))
) -> BetaComparison:
    """Source beta-binomial comparator: only the first category is success."""
    x = count(counts, "counts")
    a = finite(prior, "prior")
    if x.ndim < 2 or x.shape[-2] != 2 or x.shape[-1] < 2:
        raise ValueError("counts must end in (2, K), K >= 2")
    if a.shape != (2, 2) or np.any(a <= 0):
        raise ValueError("prior must be a positive (2, 2) matrix")
    posterior = np.stack((x[..., 0], x[..., 1:].sum(axis=-1)), axis=-1) + a
    return compare_beta_binomial(
        BetaBinomialPosterior(posterior[..., 0, 0], posterior[..., 0, 1]),
        BetaBinomialPosterior(posterior[..., 1, 0], posterior[..., 1, 1]),
    )


def catbub_simulate_counts(
    sample_sizes: ArrayLike,
    probabilities: ArrayLike,
    *,
    trials: int = 1,
    rng: np.random.Generator,
) -> FloatArray:
    """Return cumulative counts (trial, look, arm, category).

    sample_sizes is a look vector (equal arms) or (2, looks), in patients per arm.
    Nondecreasing sizes allow repeated looks with no new observations.
    """
    p = _probabilities(probabilities, "probabilities")
    if p.ndim != 2:
        raise ValueError("probabilities must be a (2, K) matrix")
    ns = count(sample_sizes, "sample_sizes")
    if ns.ndim == 0:
        ns = ns.reshape(1)
    if ns.ndim == 1:
        ns = np.broadcast_to(ns, (2, len(ns)))
    if ns.ndim != 2 or ns.shape[0] != 2 or ns.shape[1] == 0:
        raise ValueError("sample_sizes must be a nonempty vector or (2, looks) matrix")
    if np.any(ns <= 0) or np.any(ns > 1_000_000) or np.any(np.diff(ns, axis=1) < 0):
        raise ValueError("sample_sizes must be nondecreasing positive integers <= 1000000")
    trials = _integer(trials, "trials")
    if trials * ns.shape[1] * p.size > 20_000_000:
        raise ValueError("simulated counts exceed 20 million cells")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    increments = np.diff(ns, prepend=0, axis=1).astype(int)
    counts = np.empty((trials, ns.shape[1], 2, p.shape[1]))
    for arm in range(2):
        for look in range(ns.shape[1]):
            counts[:, look, arm] = rng.multinomial(increments[arm, look], p[arm], size=trials)
    return _freeze(np.cumsum(counts, axis=1))
