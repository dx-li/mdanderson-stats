"""SMO expected-log-likelihood power from original ASYPOW S-plus."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from ._validation import FloatArray, finite, scalar
from .asypow import AsymptoticPower, _probability
from .asypow_constraints import _group_null
from .boin import _owned
from .cdflib_elementary import rlog1


@dataclass(frozen=True)
class SMOPower:
    divergence_per_observation: float
    degrees_of_freedom: int
    null_parameters: FloatArray
    subtract_df: bool

    def _unit(self) -> AsymptoticPower:
        return AsymptoticPower(1, self.degrees_of_freedom, self.null_parameters)

    def _noncentrality(self, sample_size: ArrayLike) -> FloatArray:
        n = finite(sample_size, "sample_size")
        if np.any(n < 0):
            raise ValueError("sample_size must be nonnegative")
        with np.errstate(over="ignore"):
            nu = n * self.divergence_per_observation - (
                self.degrees_of_freedom if self.subtract_df else 0
            )
        if np.any(nu <= 0):
            raise ValueError(
                "SMO requires positive noncentrality; increase sample size or review the null"
            )
        return nu

    def power(self, sample_size: ArrayLike, significance: ArrayLike = 0.05) -> FloatArray:
        """Power at nu=n*w-df (or n*w); nonpositive nu is rejected as in ASYPOW."""
        return self._unit().power(self._noncentrality(sample_size), significance)

    def significance(self, sample_size: ArrayLike, power: ArrayLike = 0.8) -> FloatArray:
        """Invert SMO power using the actual constraint degrees of freedom."""
        return self._unit().significance(self._noncentrality(sample_size), power)

    def sample_size(self, power: float = 0.8, significance: float = 0.05) -> float:
        """Continuous sample size; requested power must exceed significance."""
        requested = float(_probability(scalar(power, "power"), "power"))
        alpha = float(_probability(scalar(significance, "significance"), "significance"))
        if requested <= alpha or self.divergence_per_observation == 0:
            raise ValueError(
                "SMO inversion requires power above significance and a non-null alternative"
            )
        nu = self._unit().sample_size(requested, alpha)
        n = (
            nu + (self.degrees_of_freedom if self.subtract_df else 0)
        ) / self.divergence_per_observation
        if not np.isfinite(n):
            raise ArithmeticError("required sample size is not representable")
        return float(n)


def asypow_smo_binomial(
    probabilities: ArrayLike,
    *,
    null_probabilities: ArrayLike | None = None,
    constraints: ArrayLike | None = None,
    group_size: ArrayLike = 1,
    subtract_df: bool = True,
) -> SMOPower:
    """SMO for all-equal binomial groups or a fully specified null vector.

    Omit null_probabilities to test equality of G>=2 probabilities (df=G-1).
    Supply a scalar/vector to fix all G null probabilities (df=G). The null for
    equality is the allocation-weighted mean, maximizing expected likelihood.
    Alternatively supply original ASYPOW constraints with one-based indices.
    """
    p = _probability(probabilities, "probabilities")
    if p.ndim != 1 or not 1 <= p.size <= 500:
        raise ValueError("probabilities must be a vector of length 1..500")
    if not isinstance(subtract_df, (bool, np.bool_)):
        raise ValueError("subtract_df must be boolean")
    weights = np.broadcast_to(finite(group_size, "group_size"), p.shape)
    if np.any(weights <= 0):
        raise ValueError("group_size must be positive")
    log_weights = np.log(weights) - logsumexp(np.log(weights))
    q, df = _group_null(p, log_weights, null_probabilities, constraints)
    _probability(q, "null probabilities")
    delta = q - p
    near = np.abs(delta) <= 0.125 * np.minimum(p, 1 - p)
    kl = np.empty_like(p)
    kl[near] = p[near] * rlog1(delta[near] / p[near]) + (1 - p[near]) * rlog1(
        -delta[near] / (1 - p[near])
    )
    kl[~near] = p[~near] * (np.log(p[~near]) - np.log(q[~near])) + (1 - p[~near]) * (
        np.log1p(-p[~near]) - np.log1p(-q[~near])
    )
    with np.errstate(divide="ignore", under="ignore"):
        w = float(np.exp(np.log(2) + logsumexp(log_weights + np.log(kl))))
    if not np.isfinite(w) or np.any(kl < 0):
        raise ArithmeticError("binomial divergence calculation failed")
    return SMOPower(w, df, _owned(q), bool(subtract_df))


def asypow_smo_poisson(
    means: ArrayLike,
    *,
    null_means: ArrayLike | None = None,
    constraints: ArrayLike | None = None,
    group_size: ArrayLike = 1,
    subtract_df: bool = True,
) -> SMOPower:
    """SMO for equal Poisson means or a fully specified positive null vector."""
    p = finite(means, "means")
    if p.ndim != 1 or not 1 <= p.size <= 500 or np.any(p <= 0):
        raise ValueError("means must be a positive vector of length 1..500")
    if not isinstance(subtract_df, (bool, np.bool_)):
        raise ValueError("subtract_df must be boolean")
    weights = np.broadcast_to(finite(group_size, "group_size"), p.shape)
    if np.any(weights <= 0):
        raise ValueError("group_size must be positive")
    log_weights = np.log(weights) - logsumexp(np.log(weights))
    q, df = _group_null(p, log_weights, null_means, constraints)
    if np.any(q <= 0):
        raise ValueError("null means must be positive")
    with np.errstate(over="ignore", under="ignore"):
        w = float(np.exp(np.log(2) + logsumexp(log_weights + _poisson_log_kl(p, q))))
    if not np.isfinite(w):
        raise ArithmeticError("Poisson divergence is not representable")
    return SMOPower(w, df, _owned(q), bool(subtract_df))


def _poisson_log_kl(p: FloatArray, q: FloatArray) -> FloatArray:
    """Log of p*log(p/q)-p+q, also a nonnegative categorical KL summand."""
    delta = q - p
    near = np.abs(delta) <= 0.125 * p
    # KL(Pois(p), Pois(q)) = p * [q/p - 1 - log(q/p)].
    # Keep large divergences in logs until allocation weighting is applied.
    with np.errstate(divide="ignore", under="ignore"):
        log_kl = np.empty_like(p)
        log_kl[near] = np.log(p[near]) + np.log(rlog1(delta[near] / p[near]))
        ratio = np.log(q[~near]) - np.log(p[~near])
        positive = ratio >= 0
        remainder = np.empty_like(ratio)
        r = ratio[positive]
        remainder[positive] = np.log(q[~near][positive]) + np.log(-np.expm1(-r) - r * np.exp(-r))
        r = ratio[~positive]
        remainder[~positive] = np.log(p[~near][~positive]) + np.log(np.expm1(r) - r)
        log_kl[~near] = remainder
    return log_kl
