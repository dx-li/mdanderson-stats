"""Bayesian Cluster Hierarchical Model (BCHM), with bounded NumPy sampling."""

from dataclasses import dataclass
from math import floor

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, logit

from ._validation import FloatArray, scalar
from .bchm_clustering import BCHMClusterResult, weighted_crp
from .hierarchical_binomial import ChainSummary, summarize_chains


def _owned[T: np.generic](x: NDArray[T]) -> NDArray[T]:
    a = np.asarray(x).copy()
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class BCHMCluster:
    rates: FloatArray
    trials: FloatArray
    result: BCHMClusterResult


@dataclass(frozen=True)
class BCHMBorrowResult:
    target: int
    samples: FloatArray
    posterior_mean: float
    probability: float
    native_probability: float
    decision: bool
    similarity: FloatArray
    summary: ChainSummary


@dataclass(frozen=True)
class BCHMFit:
    cluster: BCHMCluster
    borrowing: tuple[BCHMBorrowResult, ...]
    posterior_mean: FloatArray
    raw_probability: FloatArray
    native_probability: FloatArray
    decision: NDArray[np.bool_]
    raw_similarity: FloatArray
    similarity: FloatArray
    borrowing_similarity: FloatArray
    allocations: NDArray[np.int16]
    summaries: tuple[ChainSummary, ...]


def _validate(
    successes: ArrayLike, trials: ArrayLike, *, require_prior: bool = True
) -> tuple[FloatArray, FloatArray]:
    shape = np.shape(successes)
    if len(shape) != 1 or not 1 <= shape[0] <= 20 or np.shape(trials) != shape:
        raise ValueError("successes and trials must be matching vectors of 1..20 groups")
    if np.asarray(successes).dtype == np.dtype(bool) or np.asarray(trials).dtype == np.dtype(bool):
        raise ValueError("successes and trials must be integer counts")
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


def _sampling(draws: int, warmup: int, chains: int, groups: int, *, all_targets: bool) -> None:
    if any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
        for v in (draws, warmup, chains)
    ):
        raise ValueError("draws, warmup and chains must be integers")
    if not (2 <= chains <= 4 and 8 <= draws <= 10000 and 0 <= warmup <= 10000):
        raise ValueError("require 2..4 chains, 8..10000 draws and 0..10000 warmup")
    targets = groups if all_targets else 1
    if chains * (draws + warmup) * groups * targets > 1_000_000:
        raise ValueError("borrowing MCMC work budget exceeds 1000000")
    if chains * draws * targets > 200_000:
        raise ValueError("retained borrowing sample budget exceeds 200000")


def _borrow_parameters(
    alpha1: float, beta1: float, tau2: float, phi1: float, deltaT: float, thetaT: float
) -> tuple[float, float, float, float, float]:
    a, b, t = (
        scalar(v, name) for v, name in ((alpha1, "alpha1"), (beta1, "beta1"), (tau2, "tau2"))
    )
    if not all(0 < v <= 1e12 for v in (a, b, t)):
        raise ValueError("alpha1, beta1 and tau2 must be positive and <=1e12")
    phi, delta, cutoff = (
        scalar(v, name) for v, name in ((phi1, "phi1"), (deltaT, "deltaT"), (thetaT, "thetaT"))
    )
    if not (0 <= phi <= 1 and delta >= 0 and phi + delta <= 1 and 0 <= cutoff <= 1):
        raise ValueError("invalid efficacy thresholds")
    return a, b, t, phi + delta, cutoff


def _native_probability(probability: float) -> float:
    """Nearest floating-point thousandth, resolving equal distances to even.

    Matches the R 4.4.1 reference. Python's scalar round and NumPy's scaled
    rounding can both disagree with this rule at decimal-looking ties.
    """
    lower = floor(1000 * probability)
    selected = min(
        (lower, lower + 1),
        key=lambda index: (abs(index / 1000 - probability), index % 2),
    )
    return selected / 1000


def bchm_cluster(
    successes: ArrayLike,
    trials: ArrayLike,
    *,
    mu: float = 0.2,
    sigma02: float = 20.0,
    sigmaD2: float = 0.01,
    alpha: float = 0.001,
    d0: float = 0.0,
    burn_in: int = 1000,
    iterations: int = 2000,
    seed: int | None = None,
) -> BCHMCluster:
    """Estimate native patient-weighted CRP similarity; allocation labels are one-based."""
    y, n = _validate(successes, trials, require_prior=False)
    for v, name in ((sigma02, "sigma02"), (sigmaD2, "sigmaD2"), (alpha, "alpha")):
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be positive finite")
    d0 = scalar(d0, "d0")
    if not 0 <= d0 <= 1:
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


def _elliptical(
    theta: FloatArray,
    mu: float,
    tau: float,
    m: FloatArray,
    y: FloatArray,
    n: FloatArray,
    rng: np.random.Generator,
) -> FloatArray:
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
    successes: ArrayLike,
    trials: ArrayLike,
    similarity: ArrayLike,
    *,
    target: int = 0,
    prior_mean: float | None = None,
    alpha1: float = 30.0,
    beta1: float = 6.0,
    tau2: float = 0.1,
    phi1: float = 0.2,
    deltaT: float = 0.15,
    thetaT: float = 0.5,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 2,
    seed: int | None = None,
) -> BCHMBorrowResult:
    """Fit one target-specific hierarchy from a supplied similarity row or matrix.

    ``target`` is zero-based. Zero similarities are floored at 0.001. A supplied
    ``prior_mean`` is on the logit scale and overrides the native empirical center.
    """
    y, n = _validate(successes, trials, require_prior=prior_mean is None)
    if isinstance(target, (bool, np.bool_)) or not isinstance(target, (int, np.integer)):
        raise ValueError("target must be an integer")
    if not 0 <= target < y.size:
        raise ValueError("target out of range")
    if np.shape(similarity) not in ((y.size,), (y.size, y.size)):
        raise ValueError("similarity must be a subgroup vector or square matrix")
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
    if prior_mean is None:
        prior_mean = float(logit(np.mean(y / n)))
    prior_mean = scalar(prior_mean, "prior_mean")
    if abs(prior_mean) > 1e4:
        raise ValueError("prior_mean is outside the stable range")
    _sampling(draws, warmup, chains, y.size, all_targets=False)
    alpha1, beta1, tau2, response_target, thetaT = _borrow_parameters(
        alpha1, beta1, tau2, phi1, deltaT, thetaT
    )
    rng = np.random.default_rng(seed)
    k = y.size
    th = np.broadcast_to(logit((y + 0.5) / (n + 1)), (chains, k)).copy() + rng.normal(
        size=(chains, k)
    )
    mu = np.full(chains, prior_mean)
    tau = np.ones(chains)
    out = np.empty((chains, draws))
    logit_target = float(logit(response_target))
    exceedances = 0
    for it in range(warmup + draws):
        for c in range(chains):
            th[c] = _elliptical(th[c], float(mu[c]), float(tau[c]), m, y, n, rng)
            prec = tau[c] * np.sum(m) + tau2
            mu[c] = (
                (tau[c] * np.sum(m * th[c]) + tau2 * prior_mean) / prec
            ) + rng.normal() / np.sqrt(prec)
            rate = beta1 + 0.5 * np.sum(m * (th[c] - mu[c]) ** 2)
            tau[c] = rng.gamma(alpha1 + k / 2, 1 / rate)
            if (
                not np.all(np.isfinite(th[c]))
                or not np.isfinite(mu[c])
                or not np.isfinite(tau[c])
                or tau[c] <= 0
            ):
                raise ArithmeticError("BCHM sampler reached non-finite state")
        if it >= warmup:
            out[:, it - warmup] = expit(th[:, target])
            exceedances += int(np.count_nonzero(th[:, target] > logit_target))
    summary = summarize_chains(out[:, :, None])
    prob = exceedances / (chains * draws)
    native = _native_probability(prob)
    return BCHMBorrowResult(
        int(target),
        _owned(out),
        float(np.mean(out)),
        prob,
        native,
        native > thetaT,
        _owned(m),
        summary,
    )


def bchm_fit(
    successes: ArrayLike,
    trials: ArrayLike,
    *,
    mu: float = 0.2,
    sigma02: float = 20.0,
    sigmaD2: float = 0.01,
    alpha: float = 0.001,
    d0: float = 0.0,
    alpha1: float = 30.0,
    beta1: float = 6.0,
    tau2: float = 0.1,
    phi1: float = 0.2,
    deltaT: float = 0.15,
    thetaT: float = 0.5,
    burn_in: int = 1000,
    iterations: int = 2000,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 2,
    seed: int | None = None,
) -> BCHMFit:
    """Run BCHM clustering followed by each target-specific borrowing model."""
    y, n = _validate(successes, trials)
    _sampling(draws, warmup, chains, y.size, all_targets=True)
    _borrow_parameters(alpha1, beta1, tau2, phi1, deltaT, thetaT)
    master = np.random.default_rng(seed)
    cluster_seed = int(master.integers(2**31))
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
        seed=cluster_seed,
    )
    rng = master
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
    native = np.array([b.native_probability for b in bor])
    dec = native > thetaT
    return BCHMFit(
        cl,
        tuple(bor),
        _owned(np.array([b.posterior_mean for b in bor])),
        _owned(raw),
        _owned(native),
        _owned(dec),
        _owned(cl.result.raw_similarity),
        _owned(cl.result.similarity),
        _owned(np.maximum(cl.result.similarity, 0.001)),
        cl.result.allocations,
        tuple(b.summary for b in bor),
    )
