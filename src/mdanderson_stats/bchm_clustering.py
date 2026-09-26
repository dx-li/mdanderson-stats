"""Weighted Chinese-restaurant clustering used by BCHM."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from ._validation import FloatArray, scalar


def _owned[T: np.generic](x: NDArray[T]) -> NDArray[T]:
    a = np.asarray(x).copy()
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class BCHMClusterResult:
    allocations: NDArray[np.int16]
    raw_similarity: FloatArray
    similarity: FloatArray
    representative: NDArray[np.int16]
    representative_score: float


def _assignment_probabilities(
    x: float,
    w: float,
    counts: FloatArray,
    sums: FloatArray,
    mu: float,
    var0: float,
    vard: float,
    alpha: float,
) -> FloatArray:
    """Existing patient-weighted clusters followed by one new-cluster option."""
    counts = np.asarray(counts, dtype=float)
    sums = np.asarray(sums, dtype=float)
    if counts.ndim != 1 or sums.shape != counts.shape:
        raise ValueError("counts and sums must match")
    pv = 1 / (1 / var0 + counts / vard)
    pm = pv * (mu / var0 + sums / vard)
    variance: FloatArray = np.append(pv, var0)
    mean: FloatArray = np.append(pm, mu)
    increment = -0.5 * np.log1p(w * variance / vard)
    increment -= 0.5 * w * (x - mean) ** 2 / (vard + w * variance)
    # The common CRP denominator cancels in normalization.
    log_weights = np.log(np.append(counts, alpha)) + increment
    probability = np.exp(log_weights - logsumexp(log_weights))
    if not np.all(np.isfinite(probability)):
        raise ArithmeticError("BCHM assignment probabilities are non-finite")
    return probability


def _silhouette(row: NDArray[np.int16], values: FloatArray) -> float:
    labels = np.unique(row)
    if len(labels) <= 1 or len(labels) >= len(row):
        return -0.1
    dist = np.abs(values[:, None] - values[None, :])
    scores = []
    for i in range(len(row)):
        own = row == row[i]
        own[i] = False
        if not np.any(own):
            scores.append(0.0)
            continue
        a = np.mean(dist[i, own])
        b = min(np.mean(dist[i, row == lab]) for lab in labels if lab != row[i])
        scores.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return float(np.mean(scores))


def weighted_crp(
    rates: ArrayLike,
    weights: ArrayLike,
    *,
    mu: float = 0.2,
    sigma02: float = 20.0,
    sigmaD2: float = 0.01,
    alpha: float = 0.001,
    burn_in: int = 1000,
    iterations: int = 2000,
    rng: np.random.Generator | None = None,
) -> BCHMClusterResult:
    """Bounded native allocation sweeps with streaming co-clustering counts."""
    shape = np.shape(rates)
    if len(shape) != 1 or not 1 <= shape[0] <= 20 or np.shape(weights) != shape:
        raise ValueError("rates and weights must match, <=20 groups")
    x = np.asarray(rates, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.ndim != 1 or w.shape != x.shape or not 1 <= x.size <= 20:
        raise ValueError("rates and weights must match, <=20 groups")
    mu, sigma02, sigmaD2, alpha = (
        scalar(value, name)
        for value, name in (
            (mu, "mu"),
            (sigma02, "sigma02"),
            (sigmaD2, "sigmaD2"),
            (alpha, "alpha"),
        )
    )
    if abs(mu) > 1e4 or sigma02 > 1e8 or sigmaD2 > 1e8 or alpha > 1e100 or alpha < 1e-300:
        raise ValueError("clustering hyperparameters outside stable range")
    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(w))
        or np.any((x < 0) | (x > 1))
        or np.any(w <= 0)
        or np.any(w > 10000)
        or np.any(w != np.floor(w))
    ):
        raise ValueError("invalid rates or weights")
    if (
        not all(np.isfinite(v) for v in (mu, sigma02, sigmaD2, alpha))
        or sigma02 < 1e-8
        or sigmaD2 < 1e-8
        or alpha <= 0
    ):
        raise ValueError("invalid clustering hyperparameters")
    if any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
        for v in (burn_in, iterations)
    ):
        raise ValueError("MCMC counts must be integers")
    if (
        burn_in < 0
        or iterations < 1
        or (burn_in + iterations) * x.size * x.size > 2_000_000
        or iterations * x.size > 200_000
    ):
        raise ValueError("allocation MCMC budget too large")
    rng = np.random.default_rng() if rng is None else rng
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    k = x.size
    z = np.zeros(k, dtype=np.int16)
    tables = np.empty((iterations, k), dtype=np.int16)
    co = np.zeros((k, k), dtype=np.int64)
    best = -np.inf
    rep = z.copy()
    for it in range(burn_in + iterations):
        for i in range(k):
            old = int(z[i])
            z[i] = 0
            if old and not np.any(z == old):
                z[z > old] -= 1
            labels = np.unique(z[z > 0]).tolist()
            counts = np.bincount(z, weights=w, minlength=len(labels) + 1)[1:]
            sums = np.bincount(z, weights=w * x, minlength=len(labels) + 1)[1:]
            p = _assignment_probabilities(x[i], w[i], counts, sums, mu, sigma02, sigmaD2, alpha)
            choice = int(rng.choice(len(p), p=p))
            z[i] = (
                (max(labels) + 1)
                if choice == len(labels) and labels
                else (1 if not labels else labels[choice])
            )
        if it >= burn_in:
            j = it - burn_in
            tables[j] = z
            co += z[:, None] == z[None, :]
            score = _silhouette(z, x)
            if score > best:
                best, rep = score, z.copy()
    raw = co / iterations
    return BCHMClusterResult(_owned(tables), _owned(raw), _owned(raw), _owned(rep), float(best))
