"""Weighted Chinese-restaurant clustering used by BCHM."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


def _owned(x):
    a = np.asarray(x).copy()
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class BCHMClusterResult:
    allocations: np.ndarray
    raw_similarity: np.ndarray
    similarity: np.ndarray
    representative: np.ndarray
    representative_score: float


def _assignment_probabilities(x, w, counts, sums, mu, var0, vard, alpha):
    counts = np.asarray(counts, dtype=float)
    sums = np.asarray(sums, dtype=float)
    if counts.ndim != 1 or sums.shape != counts.shape:
        raise ValueError("counts and sums must match")
    denom = np.sum(counts) + alpha
    out = []
    for count, total in zip(counts, sums):
        pv = 1 / (1 / var0 + count / vard)
        pm = pv * (mu / var0 + total / vard)
        lp = -0.5 * np.log1p(w * pv / vard) - 0.5 * w * (x - pm) ** 2 / (vard + w * pv)
        out.append(np.log(count) - np.log(denom) + lp)
    lp = -0.5 * np.log1p(w * var0 / vard) - 0.5 * w * (x - mu) ** 2 / (vard + w * var0)
    out.append(np.log(alpha) - np.log(denom) + lp)
    return np.exp(np.asarray(out) - logsumexp(out))


def _silhouette(row, values):
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
    rates: object,
    weights: object,
    *,
    mu=0.2,
    sigma02=20.0,
    sigmaD2=0.01,
    alpha=0.001,
    burn_in=1000,
    iterations=2000,
    rng=None,
):
    x = np.asarray(rates, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.ndim != 1 or w.shape != x.shape or not 1 <= x.size <= 20:
        raise ValueError("rates and weights must match, <=20 groups")
    if abs(mu) > 1e4 or sigma02 > 1e8 or sigmaD2 > 1e8 or alpha > 1e100 or alpha < 1e-300:
        raise ValueError("clustering hyperparameters outside stable range")
    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(w))
        or np.any((x < 0) | (x > 1))
        or np.any(w <= 0)
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
            counts = np.array([np.sum(w[z == lab]) for lab in labels])
            sums = np.array([np.sum(w[z == lab] * x[z == lab]) for lab in labels])
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
