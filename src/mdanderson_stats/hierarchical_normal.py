"""BHM-NN: normal groups with unknown group and between-group precisions."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from ._validation import FloatArray, scalar
from .normal_updating import NormalSample


def _owned(value: ArrayLike) -> FloatArray:
    x = np.array(value, dtype=float, copy=True)
    x.flags.writeable = False
    return x


@dataclass(frozen=True)
class HierarchicalNormalFit:
    sample: NormalSample
    pooled_sample: NormalSample
    group_mean: FloatArray
    observation_precision: FloatArray
    global_mean: FloatArray
    between_precision: FloatArray
    prior_mean: float
    prior_mean_precision: float
    observation_shape: float
    observation_rate: float
    between_shape: float
    between_rate: float
    warmup: int

    def empirical_density(self, values: ArrayLike, *, pooled: bool = False) -> FloatArray:
        """Descriptive normal fits using sample SD, not posterior mean distributions.

        An evaluation grid occupies the final axis; group densities have shape
        (groups, *grid_shape). Positive sample variance is required.
        """
        sample = self.pooled_sample if pooled else self.sample
        sd = np.sqrt(sample.sum_squares) / np.sqrt(sample.size - 1)
        if np.any(~np.isfinite(sd) | (sd <= 0)):
            raise ValueError("empirical normal density requires positive sample variance")
        x = np.asarray(values, dtype=float)
        shape = (*sample.mean.shape, *((1,) * x.ndim))
        return norm.pdf(x, loc=sample.mean.reshape(shape), scale=sd.reshape(shape))


def _sample(groups: Sequence[ArrayLike] | NormalSample) -> NormalSample:
    if isinstance(groups, NormalSample):
        result = groups
    else:
        stats = [NormalSample.from_data(group) for group in groups]
        if not 2 <= len(stats) <= 100 or any(s.mean.ndim != 0 for s in stats):
            raise ValueError("groups must contain 2..100 one-dimensional data vectors")
        means = np.array([float(s.mean) for s in stats])
        sizes = np.array([float(s.size) for s in stats])
        sd = np.array(
            [
                np.sqrt(float(s.sum_squares)) / np.sqrt(float(s.size - 1)) if s.size > 1 else 0
                for s in stats
            ]
        )
        result = NormalSample(means, sizes, sample_sd=sd)
    if result.mean.ndim != 1 or not 2 <= result.mean.size <= 100:
        raise ValueError("summary statistics must describe 2..100 groups")
    if np.any(~np.isfinite(result.sum_squares)) or result.size.sum() >= 2**53:
        raise ValueError("require known sample spreads and total size smaller than 2**53")
    return result


def _pooled(sample: NormalSample) -> NormalSample:
    n = sample.size.sum()
    weights = sample.size / n
    scale = max(float(np.max(np.abs(sample.mean))), 1)
    with np.errstate(over="ignore", invalid="ignore"):
        delta = sample.mean - sample.mean[0]
        mean = sample.mean[0] + weights @ delta
        if not np.isfinite(mean):
            mean = (weights @ (sample.mean / scale)) * scale
        ss = sample.sum_squares.sum() + sample.size @ (sample.mean - mean) ** 2
    if not np.isfinite(ss):
        raise ArithmeticError("pooled sample spread is not representable")
    return NormalSample(mean, n, sample_sd=np.sqrt(ss) / np.sqrt(n - 1))


def hierarchical_normal(
    groups: Sequence[ArrayLike] | NormalSample,
    *,
    prior_mean: float = 0,
    prior_mean_precision: float = 0.0001,
    observation_shape: float = 0.0001,
    observation_rate: float = 0.0001,
    between_shape: float = 0.0001,
    between_rate: float = 0.0001,
    draws: int = 4000,
    warmup: int = 1000,
    chains: int = 4,
    rng: np.random.Generator,
) -> HierarchicalNormalFit:
    """Sample the normal-normal hierarchy with conjugate Gibbs updates.

    All Gamma distributions use shape/rate. draws is retained draws per chain,
    excluding warmup. Use summarize_chains on retained parameters to assess
    Monte Carlo behavior; a completed run is not a convergence guarantee.
    """
    sample = _sample(groups)
    pooled = _pooled(sample)
    mu0 = scalar(prior_mean, "prior_mean")
    p0, ap, bp, at, bt = [
        scalar(v, name)
        for v, name in (
            (prior_mean_precision, "prior_mean_precision"),
            (observation_shape, "observation_shape"),
            (observation_rate, "observation_rate"),
            (between_shape, "between_shape"),
            (between_rate, "between_rate"),
        )
    ]
    if min(p0, ap, bp, at, bt) <= 0:
        raise ValueError("prior precisions, Gamma shapes and rates must be positive")
    for v, name, minimum in ((draws, "draws", 8), (warmup, "warmup", 0), (chains, "chains", 2)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    m = sample.mean.size
    theta = np.broadcast_to(sample.mean, (chains, m)).copy()
    phi = np.broadcast_to(1 / (sample.sum_squares / sample.size + 1), (chains, m)).copy()
    theta += rng.normal(size=theta.shape) / np.sqrt(sample.size * phi)
    mu = theta.mean(axis=1)
    tau = np.exp(np.linspace(-1, 1, chains)) / (float(np.var(sample.mean)) + 1)
    group_draws, observation_draws = np.empty((chains, draws, m)), np.empty((chains, draws, m))
    global_draws, between_draws = np.empty((chains, draws)), np.empty((chains, draws))
    for iteration in range(warmup + draws):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            data_precision = sample.size * phi
            total = data_precision + tau[:, None]
            weight = data_precision / total
            center = mu[:, None] + weight * (sample.mean - mu[:, None])
            theta = center + rng.normal(size=(chains, m)) / np.sqrt(total)
            rate = bp + 0.5 * (sample.sum_squares + sample.size * (sample.mean - theta) ** 2)
            phi = rng.gamma(ap + sample.size / 2, scale=1 / rate)
            total_global = p0 + m * tau
            global_center = (p0 / total_global) * mu0 + (m * tau / total_global) * theta.mean(
                axis=1
            )
            mu = global_center + rng.normal(size=chains) / np.sqrt(total_global)
            rate_between = bt + 0.5 * np.sum((theta - mu[:, None]) ** 2, axis=1)
            tau = rng.gamma(at + m / 2, scale=1 / rate_between)
        if (
            not all(np.all(np.isfinite(x)) for x in (theta, phi, mu, tau))
            or np.any(phi <= 0)
            or np.any(tau <= 0)
        ):
            raise ArithmeticError("normal hierarchical sampler reached unrepresentable state")
        if iteration >= warmup:
            j = iteration - warmup
            group_draws[:, j], observation_draws[:, j] = theta, phi
            global_draws[:, j], between_draws[:, j] = mu, tau
    return HierarchicalNormalFit(
        sample,
        pooled,
        _owned(group_draws),
        _owned(observation_draws),
        _owned(global_draws),
        _owned(between_draws),
        mu0,
        p0,
        ap,
        bp,
        at,
        bt,
        warmup,
    )
