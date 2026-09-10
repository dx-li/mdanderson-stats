"""BHM-BLN logistic-normal hierarchy, with independent and pooled beta comparisons."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit

from ._validation import FloatArray, count, finite, scalar
from .beta_binomial import BetaBinomialPosterior


def _owned(x: ArrayLike) -> FloatArray:
    value = np.array(x, dtype=float, copy=True)
    value.flags.writeable = False
    return value


@dataclass(frozen=True)
class ChainSummary:
    mean: FloatArray
    median: FloatArray
    standard_deviation: FloatArray
    interval: FloatArray
    split_rhat: FloatArray
    batch_mean_mcse: FloatArray


def summarize_chains(samples: ArrayLike, *, probability: float = 0.8) -> ChainSummary:
    """Empirical shortest intervals, classical split R-hat and batch-means MCSE.

    Axes are (chain, draw, ...). Diagnostics are estimates, not convergence
    guarantees; split R-hat here is not rank-normalized. Constant chains receive
    NaN R-hat if all values agree, infinity if chain means disagree.
    """
    x = finite(samples, "samples")
    p = scalar(probability, "probability")
    if x.ndim < 2 or x.shape[0] < 2 or x.shape[1] < 8 or not 0 < p < 1:
        raise ValueError("require >=2 chains, >=8 draws per chain and probability in (0,1)")
    chains, draws = x.shape[:2]
    pooled = x.reshape((-1, *x.shape[2:]))
    ordered = np.sort(pooled, axis=0)
    included = max(1, int(np.ceil(p * pooled.shape[0])))
    widths = ordered[included - 1 :] - ordered[: ordered.shape[0] - included + 1]
    index = np.argmin(widths, axis=0)
    low = np.take_along_axis(ordered, index[None], axis=0)[0]
    high = np.take_along_axis(ordered, (index + included - 1)[None], axis=0)[0]
    half = draws // 2
    split = np.concatenate((x[:, :half], x[:, -half:]), axis=0)
    within = np.var(split, axis=1, ddof=1).mean(axis=0)
    between = half * np.var(split.mean(axis=1), axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(((half - 1) / half * within + between / half) / within)
    rhat = np.where((within == 0) & (between > 0), np.inf, rhat)
    batch_size = max(2, int(np.sqrt(draws)))
    batches = draws // batch_size
    batch_means = (
        x[:, : batches * batch_size]
        .reshape((chains * batches, batch_size, *x.shape[2:]))
        .mean(axis=1)
    )
    mcse = np.std(batch_means, axis=0, ddof=1) / np.sqrt(chains * batches)
    return ChainSummary(
        *[
            _owned(v)
            for v in (
                pooled.mean(axis=0),
                np.median(pooled, axis=0),
                np.std(pooled, axis=0, ddof=1),
                np.stack((low, high), axis=-1),
                rhat,
                mcse,
            )
        ]
    )


@dataclass(frozen=True)
class HierarchicalBinomialFit:
    successes: FloatArray
    trials: FloatArray
    group_logit: FloatArray
    global_logit: FloatArray
    precision: FloatArray
    independent: BetaBinomialPosterior
    pooled: BetaBinomialPosterior
    prior_mean: float
    prior_mean_precision: float
    prior_precision_shape: float
    prior_precision_rate: float
    warmup: int

    @property
    def group_probability(self) -> FloatArray:
        return expit(self.group_logit)

    @property
    def overall_probability(self) -> FloatArray:
        """Source target expit(global logit), not a size-weighted group average."""
        return expit(self.global_logit)


def _log_likelihood(theta: FloatArray, successes: FloatArray, trials: FloatArray) -> FloatArray:
    return -successes * np.logaddexp(0, -theta) - (trials - successes) * np.logaddexp(0, theta)


def _elliptical_step(
    theta: FloatArray,
    mu: FloatArray,
    tau: FloatArray,
    successes: FloatArray,
    trials: FloatArray,
    rng: np.random.Generator,
) -> FloatArray:
    centered = theta - mu[:, None]
    direction = rng.normal(size=theta.shape) / np.sqrt(tau[:, None])
    # 1-U is in (0,1], so the slice height is never log(0).
    height = _log_likelihood(theta, successes, trials) + np.log1p(-rng.random(theta.shape))
    angle = rng.uniform(0, 2 * np.pi, size=theta.shape)
    lower, upper = angle - 2 * np.pi, angle.copy()
    result = theta.copy()
    active = np.ones(theta.shape, dtype=bool)
    for _ in range(1000):
        proposal = mu[:, None] + centered * np.cos(angle) + direction * np.sin(angle)
        accept = active & (_log_likelihood(proposal, successes, trials) >= height)
        result[accept] = proposal[accept]
        active &= ~accept
        if not np.any(active):
            return result
        lower = np.where(active & (angle < 0), angle, lower)
        upper = np.where(active & (angle >= 0), angle, upper)
        angle[active] = rng.uniform(lower[active], upper[active])
    raise ArithmeticError("elliptical slice bracket failed to find an acceptable draw")


def hierarchical_binomial(
    successes: ArrayLike,
    trials: ArrayLike,
    *,
    prior_mean: float = 0,
    prior_mean_precision: float = 0.000001,
    precision_shape: float = 0.01,
    precision_rate: float = 0.01,
    independent_prior: ArrayLike = (0.5, 0.5),
    draws: int = 4000,
    warmup: int = 1000,
    chains: int = 4,
    rng: np.random.Generator,
) -> HierarchicalBinomialFit:
    """Sample BHM-BLN using elliptical slice group updates and Gibbs hyperupdates.

    Gamma precision priors use shape/rate, not shape/scale. Samples retain chain
    and draw axes. draws excludes warmup. No convergence or effective sample size
    is implied by the requested number of iterations.
    """
    y, n = count(successes, "successes"), count(trials, "trials")
    if y.ndim != 1 or y.shape != n.shape or not 2 <= y.size <= 100 or np.any(y > n):
        raise ValueError("require matching vectors of 2..100 groups with successes <= trials")
    if n.sum() >= 2**53:
        raise ValueError("total trial size must be smaller than 2**53")
    mu0 = scalar(prior_mean, "prior_mean")
    p0, a0, b0 = (
        scalar(v, name)
        for v, name in (
            (prior_mean_precision, "prior_mean_precision"),
            (precision_shape, "precision_shape"),
            (precision_rate, "precision_rate"),
        )
    )
    if min(p0, a0, b0) <= 0:
        raise ValueError("precision and gamma hyperparameters must be positive")
    prior = finite(independent_prior, "independent_prior")
    if prior.shape != (2,):
        raise ValueError("independent_prior must contain two beta shapes")
    beta_prior = BetaBinomialPosterior(*prior)
    independent, pooled = beta_prior.update(y, n - y), beta_prior.update(y.sum(), (n - y).sum())
    for v, name, minimum in ((draws, "draws", 8), (warmup, "warmup", 0), (chains, "chains", 2)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    theta = np.broadcast_to(np.log(y + 0.5) - np.log(n - y + 0.5), (chains, y.size)).copy()
    theta += rng.normal(size=theta.shape)
    mu = theta.mean(axis=1)
    tau = np.exp(np.linspace(-1, 1, chains))
    group = np.empty((chains, draws, y.size))
    global_mean, precision = np.empty((chains, draws)), np.empty((chains, draws))
    for iteration in range(warmup + draws):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            theta = _elliptical_step(theta, mu, tau, y, n, rng)
            total_precision = p0 + y.size * tau
            center = (p0 / total_precision) * mu0 + (y.size * tau / total_precision) * theta.mean(
                axis=1
            )
            mu = center + rng.normal(size=chains) / np.sqrt(total_precision)
            rate = b0 + 0.5 * np.sum((theta - mu[:, None]) ** 2, axis=1)
            tau = rng.gamma(a0 + y.size / 2, scale=1 / rate)
        if not all(np.all(np.isfinite(v)) for v in (theta, mu, tau)) or np.any(tau <= 0):
            raise ArithmeticError("hierarchical sampler reached unrepresentable state")
        if iteration >= warmup:
            j = iteration - warmup
            group[:, j], global_mean[:, j], precision[:, j] = theta, mu, tau
    return HierarchicalBinomialFit(
        _owned(y),
        _owned(n),
        _owned(group),
        _owned(global_mean),
        _owned(precision),
        independent,
        pooled,
        mu0,
        p0,
        a0,
        b0,
        warmup,
    )
