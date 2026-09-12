"""The weighted Chinese-restaurant clustering used by BCHM."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp


def _owned(x):
    a = np.asarray(x).copy()
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class BCHMClusterResult:
    allocations: np.ndarray
    similarity: np.ndarray
    representative: np.ndarray
    representative_score: float


def _log_marginal(sum_x, sum_x2, weight, mu, var0, vard):
    """Integrated normal likelihood, up to constants common to assignments."""
    post_var = 1.0 / (1.0 / var0 + weight / vard)
    post_mean = post_var * (mu / var0 + sum_x / vard)
    return 0.5 * (
        -sum_x2 / vard
        + post_mean * post_mean / post_var
        + np.log(post_var)
        - mu * mu / var0
        - np.log(var0)
    )


def weighted_crp(
    rates,
    weights,
    *,
    mu=0.2,
    sigma02=20.0,
    sigmaD2=0.01,
    alpha=0.001,
    burn_in=1000,
    iterations=2000,
    rng=None,
):
    """Sample BCHM allocations; weights are patient counts, as in native R."""
    x = np.asarray(rates, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.ndim != 1 or w.shape != x.shape or not 1 <= x.size <= 20:
        raise ValueError("rates and weights must be matching vectors of <=20 groups")
    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(w))
        or np.any((x < 0) | (x > 1))
        or np.any(w <= 0)
    ):
        raise ValueError("rates must be in [0,1] and weights positive finite")
    for value, name in ((mu, "mu"), (sigma02, "sigma02"), (sigmaD2, "sigmaD2"), (alpha, "alpha")):
        if not np.isfinite(value) or value <= 0 and name != "mu":
            raise ValueError(f"{name} must be finite and positive (except mu)")
    if (
        isinstance(burn_in, (bool, np.bool_))
        or isinstance(iterations, (bool, np.bool_))
        or burn_in < 0
        or iterations < 1
    ):
        raise ValueError("burn_in >= 0 and iterations >= 1 are required")
    if burn_in + iterations > 2_000_000:
        raise ValueError("MCMC allocation budget is too large")
    rng = np.random.default_rng() if rng is None else rng
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    k = x.size
    z = np.zeros(k, dtype=int)
    tables = np.empty((iterations, k), dtype=int)
    for it in range(burn_in + iterations):
        for i in range(k):
            old = z[i]
            if old:
                members = z == old
                members[i] = False
                if not np.any(members):
                    z[z > old] -= 1
                    z[i] = 0
            labels = np.unique(z[z > 0])
            labels = labels.tolist()
            logp = []
            for lab in labels:
                mask = z == lab
                sx = np.sum(w[mask] * x[mask])
                sx2 = np.sum(w[mask] * x[mask] ** 2)
                sw = np.sum(w[mask])
                old_l = _log_marginal(sx, sx2, sw, mu, sigma02, sigmaD2)
                new_l = _log_marginal(
                    sx + w[i] * x[i], sx2 + w[i] * x[i] ** 2, sw + w[i], mu, sigma02, sigmaD2
                )
                logp.append(np.log(sw) - np.log(np.sum(w[z > 0]) + alpha) + new_l - old_l)
            logp.append(
                np.log(alpha)
                - np.log(np.sum(w[z > 0]) + alpha)
                + _log_marginal(w[i] * x[i], w[i] * x[i] ** 2, w[i], mu, sigma02, sigmaD2)
            )
            prob = np.exp(np.asarray(logp) - logsumexp(logp))
            choice = int(rng.choice(len(prob), p=prob))
            if choice == len(labels):
                z[i] = (max(labels) + 1) if labels else 1
            else:
                z[i] = labels[choice]
        if it >= burn_in:
            tables[it - burn_in] = z
    sim = np.mean(tables[:, :, None] == tables[:, None, :], axis=0)
    best = -0.1
    rep = tables[0].copy()
    if k > 1:
        dist = cdist(x[:, None], x[:, None], metric="euclidean")
        for row in tables:
            ncl = len(np.unique(row))
            if ncl <= 1 or ncl >= k:
                score = -0.1
            else:
                vals = []
                for i in range(k):
                    own = row == row[i]
                    own[i] = False
                    a = np.mean(dist[i, own]) if np.any(own) else 0.0
                    b = min(np.mean(dist[i, row == lab]) for lab in np.unique(row) if lab != row[i])
                    vals.append((b - a) / max(a, b) if max(a, b) else 0.0)
                score = float(np.mean(vals))
            if score > best:
                best, rep = score, row.copy()
    return BCHMClusterResult(_owned(tables), _owned(sim), _owned(rep), float(best))
