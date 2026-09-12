"""Bayesian classification and information sharing for binary subgroups.

BaCIS first classifies subgroups around two fixed response-rate centers and
then fits an independent logistic-normal hierarchy within each cluster.
"""

from dataclasses import dataclass
from math import log

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import betaincc, expit, gammaln

from ._validation import FloatArray, count, scalar
from .hierarchical_binomial import (
    ChainSummary,
    HierarchicalBinomialFit,
    hierarchical_binomial,
    summarize_chains,
)


def _readonly(value: ArrayLike, dtype: np.dtype | type = np.float64) -> NDArray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _probability(value: float, name: str) -> float:
    result = scalar(value, name)
    if not 0 < result < 1:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return result


def _integer(value: int, name: str, low: int, high: int) -> int:
    candidate = scalar(value, name)
    if candidate != np.floor(candidate) or not low <= candidate <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(candidate)


def _validate_data(successes: ArrayLike, trials: ArrayLike) -> tuple[FloatArray, FloatArray]:
    if np.shape(successes) != np.shape(trials):
        raise ValueError("successes and trials must have the same one-dimensional shape")
    shape = np.shape(successes)
    if len(shape) != 1 or not 1 <= shape[0] <= 100:
        raise ValueError("successes and trials must contain 1..100 subgroups")
    y, n = count(successes, "successes"), count(trials, "trials")
    if np.any(n < 1) or np.any(n > 10_000) or np.any(y > n):
        raise ValueError("trials must lie in [1,10000] and successes must not exceed trials")
    return y, n


def _adaptive_cutoff(
    y: FloatArray,
    n: FloatArray,
    low: float,
    high: float,
    weighting: str,
) -> float:
    if weighting == "subgroup":
        observed = float(np.mean(y / n))
    elif weighting == "patient":
        observed = float(y.sum() / n.sum())
    else:
        raise ValueError("adaptive_weighting must be 'subgroup' or 'patient'")
    delta = observed - (low + high) / 2
    return float(expit(-2 * delta / (high - low)))


def _log_evidence(y: float, n: float, gamma: float, precision: float) -> tuple[float, float]:
    """Integrate one logistic-normal binomial evidence term stably."""

    def score(eta: float) -> float:
        return y - n * expit(eta) - precision * (eta - gamma)

    left = gamma + (y - n) / precision
    right = gamma + y / precision
    mode = brentq(score, left, right, xtol=1e-12, rtol=1e-14)
    p_mode = float(expit(mode))
    log_kernel_mode = -y * np.logaddexp(0.0, -mode) - (n - y) * np.logaddexp(0.0, mode)
    log_kernel_mode -= 0.5 * precision * (mode - gamma) ** 2
    scale = 1 / np.sqrt(precision + n * p_mode * (1 - p_mode))

    def scaled(z: float) -> float:
        eta = mode + scale * z
        value = -y * np.logaddexp(0.0, -eta) - (n - y) * np.logaddexp(0.0, eta)
        value -= 0.5 * precision * (eta - gamma) ** 2
        return float(np.exp(value - log_kernel_mode))

    quadrature = quad(
        scaled,
        -np.inf,
        np.inf,
        epsabs=1e-11,
        epsrel=1e-11,
        full_output=1,
        limit=200,
    )
    if len(quadrature) == 3:
        integral, error, _ = quadrature
    elif len(quadrature) == 4:
        integral, error, _, message = quadrature
        raise ArithmeticError(f"classification quadrature failed: {message}")
    else:
        raise ArithmeticError("classification quadrature returned an invalid result")
    if not np.isfinite(integral) or integral <= 0 or not np.isfinite(error):
        raise ArithmeticError("classification quadrature failed")
    relative_error = float(error / integral)
    if relative_error > 1e-7:
        raise ArithmeticError("classification quadrature did not converge")
    log_binomial = gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)
    log_normal = 0.5 * (log(precision) - log(2 * np.pi))
    log_value = log_binomial + log_kernel_mode + log_normal + log(scale) + log(integral)
    return float(log_value), relative_error


@dataclass(frozen=True)
class BaCISClassification:
    high_probability: FloatArray
    low_probability: FloatArray
    cluster: NDArray[np.int64]
    cutoff: float
    precision: float
    log_evidence: FloatArray
    quadrature_error: FloatArray


def bacis_classify(
    successes: ArrayLike,
    trials: ArrayLike,
    *,
    phi_low: float = 0.1,
    phi_high: float = 0.3,
    classification_precision: float | None = None,
    classification_cutoff: float | None = None,
    adaptive_weighting: str = "subgroup",
) -> BaCISClassification:
    """Classify subgroups using the BaCIS latent-sign model.

    ``adaptive_weighting='subgroup'`` matches the CRAN implementation's
    unweighted mean of subgroup response rates.  ``'patient'`` matches the
    official help slide's pooled-response formula.  The native paper/app
    backend does not resolve this discrepancy, so the choice is explicit.
    """
    y, n = _validate_data(successes, trials)
    low = _probability(phi_low, "phi_low")
    high = _probability(phi_high, "phi_high")
    if low >= high:
        raise ValueError("phi_low must be less than phi_high")
    if classification_precision is None:
        separation = ((np.log(high) - np.log1p(-high)) - (np.log(low) - np.log1p(-low))) / 6
        precision = 1 / separation**2
    else:
        precision = scalar(classification_precision, "classification_precision")
    if not 1e-6 <= precision <= 1e6:
        raise ValueError("classification_precision must lie in [1e-6,1e6]")
    if adaptive_weighting not in ("subgroup", "patient"):
        raise ValueError("adaptive_weighting must be 'subgroup' or 'patient'")
    if classification_cutoff is None:
        cutoff = _adaptive_cutoff(y, n, low, high, adaptive_weighting)
    else:
        cutoff = _probability(classification_cutoff, "classification_cutoff")
    gamma = (
        np.log(low) - np.log1p(-low),
        np.log(high) - np.log1p(-high),
    )
    evidence = np.empty((y.size, 2), dtype=float)
    errors = np.empty((y.size, 2), dtype=float)
    for i, (success, total) in enumerate(zip(y, n, strict=True)):
        for j in range(2):
            evidence[i, j], errors[i, j] = _log_evidence(success, total, gamma[j], precision)
    high_probability = expit(evidence[:, 1] - evidence[:, 0])
    low_probability = expit(evidence[:, 0] - evidence[:, 1])
    cluster = np.where(high_probability > cutoff, 2, 1)
    return BaCISClassification(
        _readonly(high_probability),
        _readonly(low_probability),
        _readonly(cluster, dtype=np.int64),
        cutoff,
        precision,
        _readonly(evidence),
        _readonly(errors),
    )


@dataclass(frozen=True)
class BaCISFit:
    classification: BaCISClassification
    probability_samples: FloatArray
    summary: ChainSummary
    posterior_mean: FloatArray
    posterior_sd: FloatArray
    efficacy_probability: FloatArray
    high_response_probability: FloatArray
    efficacious: NDArray[np.bool_]
    cluster_fits: tuple[HierarchicalBinomialFit | None, HierarchicalBinomialFit | None]


def bacis_fit(
    successes: ArrayLike,
    trials: ArrayLike,
    *,
    phi_low: float = 0.1,
    phi_high: float = 0.3,
    classification_precision: float | None = None,
    classification_cutoff: float | None = None,
    adaptive_weighting: str = "subgroup",
    mean_precision: float = 0.1,
    precision_shape: float = 50,
    precision_rate: float = 10,
    efficacy_cutoff: float = 0.92,
    draws: int = 2000,
    warmup: int = 1000,
    chains: int = 2,
    seed: int | None = None,
) -> BaCISFit:
    """Fit both BaCIS stages, reusing the repository logistic-normal sampler."""
    y, n = _validate_data(successes, trials)
    low = _probability(phi_low, "phi_low")
    high = _probability(phi_high, "phi_high")
    if low >= high:
        raise ValueError("phi_low must be less than phi_high")
    repetitions, burnin, chain_count = (
        _integer(draws, "draws", 8, 10_000),
        _integer(warmup, "warmup", 0, 10_000),
        _integer(chains, "chains", 2, 8),
    )
    if chain_count * repetitions * y.size > 500_000:
        raise ValueError("chains * draws * groups must be at most 500,000")
    if chain_count * (repetitions + burnin) * y.size > 1_000_000:
        raise ValueError("chains * (draws + warmup) * groups must be at most 1,000,000")
    mean_precision = scalar(mean_precision, "mean_precision")
    precision_shape = scalar(precision_shape, "precision_shape")
    precision_rate = scalar(precision_rate, "precision_rate")
    if (
        min(mean_precision, precision_shape, precision_rate) <= 0
        or max(mean_precision, precision_shape, precision_rate) > 1e12
    ):
        raise ValueError("hierarchical precision hyperparameters must lie in (0,1e12]")
    efficacy_cutoff = _probability(efficacy_cutoff, "efficacy_cutoff")
    classification = bacis_classify(
        y,
        n,
        phi_low=low,
        phi_high=high,
        classification_precision=classification_precision,
        classification_cutoff=classification_cutoff,
        adaptive_weighting=adaptive_weighting,
    )
    rng = np.random.default_rng(seed)
    samples = np.empty((chain_count, repetitions, y.size), dtype=float)
    fits: list[HierarchicalBinomialFit | None] = [None, None]
    for cluster_index, center in enumerate((low, high), start=1):
        members = np.flatnonzero(classification.cluster == cluster_index)
        if members.size == 0:
            continue
        if members.size == 1:
            member = int(members[0])
            samples[:, :, member] = rng.beta(
                1 + y[member],
                1 + n[member] - y[member],
                size=(chain_count, repetitions),
            )
            continue
        fit = hierarchical_binomial(
            y[members],
            n[members],
            prior_mean=float(np.log(center) - np.log1p(-center)),
            prior_mean_precision=mean_precision,
            precision_shape=precision_shape,
            precision_rate=precision_rate,
            independent_prior=(1, 1),
            draws=repetitions,
            warmup=burnin,
            chains=chain_count,
            rng=rng,
        )
        fits[cluster_index - 1] = fit
        samples[:, :, members] = expit(fit.group_logit)
    summary = summarize_chains(samples)
    posterior_mean = summary.mean.copy()
    posterior_sd = summary.standard_deviation.copy()
    efficacy_probability = np.mean(samples > low, axis=(0, 1))
    high_probability = np.mean(samples > high, axis=(0, 1))
    for i, cluster in enumerate(classification.cluster):
        if np.count_nonzero(classification.cluster == cluster) == 1:
            a = 1 + y[i]
            b = 1 + n[i] - y[i]
            total = a + b
            posterior_mean[i] = a / total
            posterior_sd[i] = np.sqrt(a * b / (total**2 * (total + 1)))
            efficacy_probability[i] = betaincc(a, b, low)
            high_probability[i] = betaincc(a, b, high)
    efficacious = efficacy_probability > efficacy_cutoff
    return BaCISFit(
        classification,
        _readonly(samples),
        summary,
        _readonly(posterior_mean),
        _readonly(posterior_sd),
        _readonly(efficacy_probability),
        _readonly(high_probability),
        _readonly(efficacious, dtype=np.bool_),
        (fits[0], fits[1]),
    )
