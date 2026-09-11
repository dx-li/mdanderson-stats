"""PRT random-walk probit posterior and covariance-weighted isotonic transformation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve
from scipy.special import log_ndtr

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite
from .hierarchical_binomial import ChainSummary, summarize_chains
from .prt import prt_conditional_toxicity


@dataclass(frozen=True)
class PRTModelFit:
    beta: FloatArray
    conditional_toxicity: FloatArray
    beta_summary: ChainSummary
    risk_summary: ChainSummary
    prior_mean: FloatArray
    prior_variance: FloatArray
    likelihood_evaluations: int
    warmup: int


def fit_prt_model(
    survived: ArrayLike,
    events: ArrayLike,
    *,
    prior_mean: ArrayLike = -14,
    prior_variance: ArrayLike = 28,
    draws: int = 2000,
    warmup: int = 1000,
    chains: int = 4,
    rng: np.random.Generator,
) -> PRTModelFit:
    """Fit equation (2.1) under independent interval-specific Gaussian dose random walks.

    Counts have axes (assessment interval, dose). Each dose increment has the
    supplied prior variance, including the first increment from prior_mean.
    Independent interval/chain blocks update together by elliptical slice sampling;
    this is not the archived latent-variable Gibbs implementation or its RNG.
    Returned risks are raw and require isotonic transformation before PRT decisions.
    """
    s, e = count(survived, "survived"), count(events, "events")
    if s.ndim != 2 or e.shape != s.shape or not 1 <= s.shape[0] <= 10 or not 2 <= s.shape[1] <= 10:
        raise ValueError("counts must match with 1..10 intervals and 2..10 doses")
    if np.any(s[1:] + e[1:] > s[:-1]):
        raise ValueError("later-interval event/survival counts exceed preceding survivors")
    h, k = s.shape
    mu = np.broadcast_to(finite(prior_mean, "prior_mean"), (h,)).copy()
    variance = np.broadcast_to(finite(prior_variance, "prior_variance"), (h,)).copy()
    if np.any(variance <= 0):
        raise ValueError("prior_variance must be positive for each interval")
    settings = count([draws, warmup, chains], "sampler settings")
    if not (8 <= settings[0] <= 100000 and settings[1] <= 100000 and 2 <= settings[2] <= 16):
        raise ValueError("require draws 8..100000, warmup 0..100000, chains 2..16")
    draws, warmup, chains = (int(v) for v in settings)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    mean = np.tile(mu, chains)[:, None]
    sd = np.tile(np.sqrt(variance), chains)[:, None]
    failures, successes = np.tile(s, (chains, 1)), np.tile(e, (chains, 1))
    informed = (failures + successes).sum(axis=1) > 0
    active = np.flatnonzero(informed)
    beta = mean + sd * rng.normal(size=(chains * h, k)).cumsum(axis=1)
    samples = np.empty((chains, draws, h, k))
    evaluations = 0

    def loglik(values: FloatArray, index: np.ndarray) -> FloatArray:
        nonlocal evaluations
        evaluations += len(index)
        result = (successes[index] * log_ndtr(values) + failures[index] * log_ndtr(-values)).sum(
            axis=1
        )
        if np.any(~np.isfinite(result)):
            raise ArithmeticError("PRT likelihood evaluation is not finite")
        return result

    current_ll = np.zeros(chains * h)
    current_ll[active] = loglik(beta[active], active)
    for iteration in range(warmup + draws):
        beta[~informed] = mean[~informed] + sd[~informed] * rng.normal(
            size=(int((~informed).sum()), k)
        ).cumsum(axis=1)
        if len(active):
            centered = beta[active] - mean[active]
            direction = sd[active] * rng.normal(size=(len(active), k)).cumsum(axis=1)
            threshold = current_ll[active] + np.log(
                np.maximum(rng.random(len(active)), np.finfo(float).tiny)
            )
            angle = rng.uniform(0, 2 * np.pi, len(active))
            lower, upper = angle - 2 * np.pi, angle.copy()
            waiting = np.arange(len(active))
            for _ in range(1000):
                index = active[waiting]
                proposal = (
                    mean[index]
                    + centered[waiting] * np.cos(angle[waiting, None])
                    + direction[waiting] * np.sin(angle[waiting, None])
                )
                ll = loglik(proposal, index)
                accepted = ll >= threshold[waiting]
                beta[index[accepted]] = proposal[accepted]
                current_ll[index[accepted]] = ll[accepted]
                waiting = waiting[~accepted]
                if not len(waiting):
                    break
                left = angle[waiting] < 0
                lower[waiting[left]] = angle[waiting[left]]
                upper[waiting[~left]] = angle[waiting[~left]]
                angle[waiting] = rng.uniform(lower[waiting], upper[waiting])
            else:
                raise ArithmeticError("PRT elliptical-slice bracket failed to accept")
        if iteration >= warmup:
            samples[:, iteration - warmup] = beta.reshape(chains, h, k)
    if np.any(~np.isfinite(samples)):
        raise ArithmeticError("PRT posterior draws exceed floating-point range")
    risks = prt_conditional_toxicity(samples)
    return PRTModelFit(
        _freeze(samples),
        risks,
        summarize_chains(samples),
        summarize_chains(risks[..., :-1, :]),
        _freeze(mu),
        _freeze(variance),
        evaluations,
        warmup,
    )


@dataclass(frozen=True)
class PRTIsotonicProjection:
    probability: FloatArray
    covariance: FloatArray
    maximum_condition_number: float


def prt_isotonic_projection(probability: ArrayLike) -> PRTIsotonicProjection:
    """Apply Section 3's covariance-weighted min-max transformation to raw risk draws.

    Axes are (draw, interval, dose); flatten chain and retained-draw axes first.
    Include only nonterminal intervals: the all-zero completed-window row has
    singular covariance and must be appended separately if needed. No ridge,
    pseudoinverse, diagonal substitute or probability clipping is introduced.
    """
    p = finite(probability, "probability")
    if p.ndim != 3 or not 2 <= p.shape[2] <= 10 or p.shape[1] < 1 or len(p) <= p.shape[2]:
        raise ValueError("require (draws > doses, intervals, 2..10 doses)")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("risk draws must lie in [0,1]")
    _, h, k = p.shape
    covariance = np.empty((h, k, k))
    transformed = np.empty_like(p)
    maximum_condition = 0.0
    for interval in range(h):
        x = p[:, interval]
        cov = np.cov(x, rowvar=False, ddof=1)
        covariance[interval] = cov
        condition = float(np.linalg.cond(cov))
        if not np.isfinite(condition) or condition > 1e12:
            raise ArithmeticError(
                "PRT covariance is singular or ill-conditioned; no regularization applied"
            )
        maximum_condition = max(maximum_condition, condition)
        means: dict[tuple[int, int], FloatArray] = {}
        for lo in range(k):
            means[lo, lo] = x[:, lo]
            for hi in range(lo + 1, k):
                sub = cov[lo : hi + 1, lo : hi + 1]
                try:
                    weights = cho_solve(cho_factor(sub), np.ones(hi - lo + 1))
                except np.linalg.LinAlgError as exc:
                    raise ArithmeticError(
                        "PRT covariance submatrix is not positive definite"
                    ) from exc
                denominator = float(weights.sum())
                if not np.isfinite(denominator) or denominator <= 0:
                    raise ArithmeticError("PRT covariance weights cannot be normalized")
                means[lo, hi] = x[:, lo : hi + 1] @ (weights / denominator)
        for dose in range(k):
            transformed[:, interval, dose] = np.minimum.reduce(
                [
                    np.maximum.reduce([means[lo, hi] for lo in range(dose + 1)])
                    for hi in range(dose, k)
                ]
            )
    if np.any(~np.isfinite(transformed)) or np.any((transformed < 0) | (transformed > 1)):
        raise ArithmeticError(
            f"published covariance-weighted projection leaves [0,1]: "
            f"range [{transformed.min():.6g}, {transformed.max():.6g}]; no clipping applied"
        )
    if np.any(np.diff(transformed, axis=-1) < 0):
        raise ArithmeticError("PRT isotonic transformation failed its dose-order check")
    return PRTIsotonicProjection(_freeze(transformed), _freeze(covariance), maximum_condition)
