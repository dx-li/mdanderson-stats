"""Bayesian Cluster Hierarchical Model (BCHM), with bounded NumPy sampling."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit

from .bchm_clustering import weighted_crp
from .hierarchical_binomial import ChainSummary, summarize_chains


def _owned(x):
    a = np.asarray(x).copy()
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class BCHMCluster:
    rates: np.ndarray
    trials: np.ndarray
    result: object


@dataclass(frozen=True)
class BCHMBorrowResult:
    target: int
    samples: np.ndarray
    posterior_mean: float
    probability: float
    similarity: np.ndarray
    summary: ChainSummary


@dataclass(frozen=True)
class BCHMFit:
    cluster: BCHMCluster
    borrowing: tuple
    posterior_mean: np.ndarray
    raw_probability: np.ndarray
    native_probability: np.ndarray
    decision: np.ndarray
    similarity: np.ndarray
    allocations: np.ndarray
    summaries: tuple


def _validate(successes, trials, *, require_prior=True):
    y = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    if y.ndim != 1 or n.shape != y.shape or not 1 <= y.size <= 20:
        raise ValueError("successes and trials must be matching vectors of 1..20 groups")
    if np.any(y != np.floor(y)) or np.any(n != np.floor(n)):
        raise ValueError("successes and trials must be integers")
    if (
        not np.all(np.isfinite(y))
        or not np.all(np.isfinite(n))
        or np.any(n <= 0)
        or np.any(y < 0)
        or np.any(y > n)
    ):
        raise ValueError("require finite 0 <= successes <= positive trials")
    if np.any(n > 10000):
        raise ValueError("each trial count must be <=10000")
    if require_prior and (np.all(y == 0) or np.all(y == n)):
        raise ValueError("empirical prior mean is undefined at all-zero or all-one data")
    return y, n


def bchm_cluster(
    successes: object,
    trials: object,
    *,
    mu=0.2,
    sigma02=20.0,
    sigmaD2=0.01,
    alpha=0.001,
    d0=0.0,
    burn_in=1000,
    iterations=2000,
    seed=None,
):
    y, n = _validate(successes, trials)
    for v, name in ((sigma02, "sigma02"), (sigmaD2, "sigmaD2"), (alpha, "alpha")):
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be positive finite")
    if not np.isfinite(d0) or not 0 <= d0 <= 1:
        raise ValueError("d0 must be in [0,1]")
    res = weighted_crp(
        y / n,
        n,
        mu=mu,
        sigma02=sigma02,
        sigmaD2=sigmaD2,
        alpha=alpha,
        burn_in=burn_in,
        iterations=iterations,
        rng=np.random.default_rng(seed),
    )
    raw = res.similarity
    floored = np.maximum(raw, d0)
    return BCHMCluster(
        _owned(y / n),
        _owned(n),
        res.__class__(
            res.allocations,
            res.raw_similarity,
            _owned(floored),
            res.representative,
            res.representative_score,
        ),
    )


def _elliptical(theta, mu, tau, m, y, n, rng):
    center = theta - mu
    direction = rng.normal(size=theta.size) / np.sqrt(tau * m)
    ll = -y * np.logaddexp(0, -theta) - (n - y) * np.logaddexp(0, theta)
    height = ll + np.log1p(-rng.random(theta.size))
    angle = rng.uniform(0, 2 * np.pi, theta.size)
    lo = angle - 2 * np.pi
    hi = angle.copy()
    active = np.ones(theta.size, bool)
    out = theta.copy()
    for _ in range(1000):
        prop = mu + center * np.cos(angle) + direction * np.sin(angle)
        pll = -y * np.logaddexp(0, -prop) - (n - y) * np.logaddexp(0, prop)
        ok = active & (pll >= height)
        out[ok] = prop[ok]
        active &= ~ok
        if not np.any(active):
            return out
        lo = np.where(active & (angle < 0), angle, lo)
        hi = np.where(active & (angle >= 0), angle, hi)
        angle[active] = rng.uniform(lo[active], hi[active])
    raise ArithmeticError("BCHM elliptical update failed")


def bchm_borrow(
    successes: object,
    trials: object,
    similarity: object,
    *,
    target=0,
    prior_mean=None,
    alpha1=30.0,
    beta1=6.0,
    tau2=0.1,
    phi1=0.2,
    deltaT=0.15,
    thetaT=0.5,
    draws=1000,
    warmup=500,
    chains=2,
    seed=None,
):
    y, n = _validate(successes, trials, require_prior=prior_mean is None)
    if isinstance(target, (bool, np.bool_)) or not isinstance(target, (int, np.integer)):
        raise ValueError("target must be an integer")
    m = np.asarray(similarity, dtype=float)
    if m.ndim == 2:
        if m.shape != (y.size, y.size):
            raise ValueError("similarity matrix must be square")
        if np.any(~np.isfinite(m)) or np.any((m < 0) | (m > 1)):
            raise ValueError("similarity matrix must be finite in [0,1]")
        m = m[int(target)]
    if m.shape != y.shape or np.any(~np.isfinite(m)) or np.any((m < 0) | (m > 1)):
        raise ValueError("similarity must be positive and match groups")
    m = np.maximum(m, 0.001)
    if not 0 <= target < y.size:
        raise ValueError("target out of range")
    if prior_mean is None:
        prior_mean = float(logit(np.mean(y / n)))
    if not np.isfinite(prior_mean):
        raise ValueError("prior_mean must be finite")
    for v, name in ((alpha1, "alpha1"), (beta1, "beta1"), (tau2, "tau2")):
        if not np.isfinite(v) or v <= 0 or v > 1e12:
            raise ValueError(f"{name} must be positive finite")
    if (
        any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
            for v in (chains, draws, warmup)
        )
        or chains < 2
        or chains > 4
        or draws < 8
        or draws > 10000
        or warmup < 0
        or warmup > 10000
        or chains * (draws + warmup) * y.size > 1_000_000
    ):
        raise ValueError("require chains>=2 and draws>=8")
    rng = np.random.default_rng(seed)
    k = y.size
    th = np.broadcast_to(logit((y + 0.5) / (n + 1)), (chains, k)).copy() + rng.normal(
        size=(chains, k)
    )
    mu = np.full(chains, prior_mean)
    tau = np.ones(chains)
    out = np.empty((chains, draws))
    for it in range(warmup + draws):
        for c in range(chains):
            th[c] = _elliptical(th[c], mu[c], tau[c], m, y, n, rng)
            prec = tau[c] * np.sum(m) + tau2
            mu[c] = (
                (tau[c] * np.sum(m * th[c]) + tau2 * prior_mean) / prec
            ) + rng.normal() / np.sqrt(prec)
            rate = beta1 + 0.5 * np.sum(m * (th[c] - mu[c]) ** 2)
            tau[c] = rng.gamma(alpha1 + k / 2, 1 / rate)
        if it >= warmup:
            out[:, it - warmup] = expit(th[:, target])
    summary = summarize_chains(out[:, :, None])
    if not (
        np.isfinite(phi1)
        and 0 <= phi1 <= 1
        and np.isfinite(deltaT)
        and deltaT >= 0
        and phi1 + deltaT <= 1
        and np.isfinite(thetaT)
        and 0 <= thetaT <= 1
    ):
        raise ValueError("invalid efficacy thresholds")
    prob = float(np.mean(out > phi1 + deltaT))
    return BCHMBorrowResult(target, _owned(out), float(np.mean(out)), prob, _owned(m), summary)


def bchm_fit(
    successes: object,
    trials: object,
    *,
    mu=0.2,
    sigma02=20.0,
    sigmaD2=0.01,
    alpha=0.001,
    d0=0.0,
    alpha1=30.0,
    beta1=6.0,
    tau2=0.1,
    phi1=0.2,
    deltaT=0.15,
    thetaT=0.5,
    burn_in=1000,
    iterations=2000,
    draws=1000,
    warmup=500,
    chains=2,
    seed=None,
):
    y, n = _validate(successes, trials)
    if (
        chains < 2
        or chains > 4
        or draws < 8
        or draws > 10000
        or warmup < 0
        or warmup > 10000
        or chains * (draws + warmup) * y.size * y.size > 1_000_000
    ):
        raise ValueError("fit MCMC budget is too large")
    cl = bchm_cluster(
        y,
        n,
        mu=mu,
        sigma02=sigma02,
        sigmaD2=sigmaD2,
        alpha=alpha,
        d0=d0,
        burn_in=burn_in,
        iterations=iterations,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    bor = []
    for i in range(y.size):
        bor.append(
            bchm_borrow(
                y,
                n,
                np.maximum(cl.result.similarity, 0.001),
                target=i,
                alpha1=alpha1,
                beta1=beta1,
                tau2=tau2,
                phi1=phi1,
                deltaT=deltaT,
                thetaT=thetaT,
                draws=draws,
                warmup=warmup,
                chains=chains,
                seed=int(rng.integers(2**31)),
            )
        )
    raw = np.array([b.probability for b in bor])
    native = np.round(raw, 3)
    dec = native > thetaT
    return BCHMFit(
        cl,
        tuple(bor),
        _owned(np.array([b.posterior_mean for b in bor])),
        _owned(raw),
        _owned(native),
        _owned(dec),
        np.maximum(cl.result.similarity, 0.001),
        cl.result.allocations,
        tuple(b.summary for b in bor),
    )
