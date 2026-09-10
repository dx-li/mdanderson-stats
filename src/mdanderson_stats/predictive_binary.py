"""Binary two-arm posterior prediction of final frequentist or Bayesian conclusions."""

from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp, ndtri

from ._validation import FloatArray, count, finite, scalar
from .beta_binomial import BetaBinomialPosterior, _owned
from .beta_comparison import compare_beta_binomial


@dataclass(frozen=True)
class BinaryPredictiveProbabilities:
    arm_a_superior: float
    arm_b_superior: float
    inconclusive: float
    future_success_probability_a: FloatArray
    future_success_probability_b: FloatArray
    final_decision: NDArray[np.str_]


@dataclass(frozen=True)
class BinaryPredictivePlan:
    """Matrices indexed by interim successes on A (rows) and B (columns)."""

    arm_a_superior: FloatArray
    arm_b_superior: FloatArray
    inconclusive: FloatArray
    successes_a: FloatArray
    successes_b: FloatArray


def _pair(value: ArrayLike, name: str) -> NDArray[np.int64]:
    values = count(value, name)
    if values.shape != (2,) or np.any(values > 200):
        raise ValueError(f"{name} must contain two integer counts in [0,200]")
    return values.astype(np.int64)


def _prior(value: ArrayLike) -> BetaBinomialPosterior:
    prior = finite(value, "prior")
    if prior.shape != (2, 2):
        raise ValueError("prior must be [[a_A,b_A],[a_B,b_B]]")
    return BetaBinomialPosterior(prior[:, 0], prior[:, 1])


def _mass(n: int, a: float, b: float) -> FloatArray:
    # Relative adjacent weights avoid subtracting nearly equal large log-beta values.
    k = np.arange(n, dtype=float)
    increments = np.log(n - k) - np.log(k + 1) + np.log(a + k) - np.log(b + (n - k - 1))
    logs = np.r_[0.0, np.cumsum(increments)]
    return np.exp(logs - logsumexp(logs))


def _decisions(
    totals: NDArray[np.int64],
    prior: BetaBinomialPosterior,
    a_counts: NDArray[np.int64],
    b_counts: NDArray[np.int64],
    method: str,
    significance_level: float,
    posterior_cutoff: float,
) -> NDArray[np.int64]:
    if method not in ("frequentist", "bayesian"):
        raise ValueError("method must be frequentist or bayesian")
    alpha = scalar(significance_level, "significance_level")
    cutoff = scalar(posterior_cutoff, "posterior_cutoff")
    if not 0 < alpha < 1 or not 0.5 <= cutoff < 1:
        raise ValueError("require significance_level in (0,1), posterior_cutoff in [0.5,1)")
    result = np.zeros((a_counts.size, b_counts.size), dtype=np.int64)
    if method == "frequentist":
        if np.any(totals == 0):
            raise ValueError("frequentist comparison requires positive planned sizes on both arms")
        a, b = a_counts[:, None] / totals[0], b_counts[None, :] / totals[1]
        pooled = (a_counts[:, None] + b_counts[None, :]) / totals.sum()
        variance = pooled * (1 - pooled) * (1 / totals[0] + 1 / totals[1])
        z = np.divide(a - b, np.sqrt(variance), out=np.zeros_like(variance), where=variance > 0)
        critical = -ndtri(alpha / 2)
        result[z > critical] = 1
        result[z < -critical] = 2
        return result
    # Posterior A superiority decreases monotonically as final B successes increase.
    for i, a in enumerate(a_counts):
        arm_a = BetaBinomialPosterior(prior.alpha[0] + a, prior.beta[0] + totals[0] - a)
        cache: dict[int, tuple[float, float, float]] = {}

        def wins(j: int, arm: int) -> bool:
            if j not in cache:
                b = b_counts[j]
                arm_b = BetaBinomialPosterior(prior.alpha[1] + b, prior.beta[1] + totals[1] - b)
                comparison = compare_beta_binomial(arm_b, arm_a, absolute_tolerance=1e-11)
                cache[j] = (
                    float(comparison.treatment_greater),
                    float(comparison.control_greater),
                    float(comparison.absolute_error),
                )
            p, error = cache[j][arm], cache[j][2]
            if error > 0 and abs(p - cutoff) <= error:
                shapes = (
                    float(arm_a.alpha),
                    float(arm_a.beta),
                    float(prior.alpha[1] + b_counts[j]),
                    float(prior.beta[1] + totals[1] - b_counts[j]),
                )
                if all(v.is_integer() and v <= 1000 for v in shapes):
                    aa, bb, cc, dd = map(int, shapes)

                    def beta(x: int, y: int) -> Fraction:
                        return Fraction(factorial(x - 1) * factorial(y - 1), factorial(x + y - 1))

                    order = sum(
                        (
                            comb(cc + dd - 1, k) * beta(aa + k, bb + cc + dd - 1 - k) / beta(aa, bb)
                            for k in range(cc, cc + dd)
                        ),
                        Fraction(0),
                    )
                    return (order if arm == 0 else 1 - order) > Fraction(str(cutoff))
                raise ArithmeticError("posterior superiority is too close to cutoff to resolve")
            return p > cutoff

        lo, hi = 0, b_counts.size
        while lo < hi:
            mid = (lo + hi) // 2
            if wins(mid, 0):
                lo = mid + 1
            else:
                hi = mid
        result[i, :lo] = 1
        first_b = lo
        lo, hi = first_b, b_counts.size
        while lo < hi:
            mid = (lo + hi) // 2
            if wins(mid, 1):
                hi = mid
            else:
                lo = mid + 1
        result[i, lo:] = 2
    return result


def predictive_binary(
    successes: ArrayLike,
    failures: ArrayLike,
    planned_sizes: ArrayLike,
    *,
    prior: ArrayLike = ((1, 1), (1, 1)),
    method: str = "bayesian",
    significance_level: float = 0.05,
    posterior_cutoff: float = 0.95,
) -> BinaryPredictiveProbabilities:
    """Predict A superior, B superior or inconclusive at the planned final sample sizes.

    Inputs are length-two arm counts; planned sizes include already observed patients.
    Independent beta priors are updated with current data before predicting future counts.
    Final decisions use strict inequalities. No additional interim stopping is assumed.
    """
    s, f, totals = (
        _pair(successes, "successes"),
        _pair(failures, "failures"),
        _pair(planned_sizes, "planned_sizes"),
    )
    if np.any(s + f > totals):
        raise ValueError("planned_sizes must be at least current successes plus failures")
    original = _prior(prior)
    posterior = original.update(s, f)
    future = totals - s - f
    pa, pb = [
        _mass(int(n), float(a), float(b))
        for n, a, b in zip(future, posterior.alpha, posterior.beta, strict=True)
    ]
    decision = _decisions(
        totals,
        original,
        np.arange(s[0], s[0] + future[0] + 1),
        np.arange(s[1], s[1] + future[1] + 1),
        method,
        significance_level,
        posterior_cutoff,
    )
    weights = pa[:, None] * pb[None, :]
    probabilities = [float(weights[decision == code].sum()) for code in (1, 2, 0)]
    labels = np.array(["inconclusive", "arm_a_superior", "arm_b_superior"])[decision]
    labels.flags.writeable = False
    return BinaryPredictiveProbabilities(
        probabilities[0], probabilities[1], probabilities[2], _owned(pa), _owned(pb), labels
    )


def plan_predictive_binary(
    interim_sizes: ArrayLike,
    planned_sizes: ArrayLike,
    *,
    prior: ArrayLike = ((1, 1), (1, 1)),
    method: str = "bayesian",
    significance_level: float = 0.05,
    posterior_cutoff: float = 0.95,
) -> BinaryPredictivePlan:
    """All first-stage outcomes; reuse final decisions and multiply predictive kernels.

    Returns conditional predictive probabilities, not probabilities of reaching the
    interim counts and not an optimized sample size. Each arm supports up to 200 patients.
    """
    interim, totals = _pair(interim_sizes, "interim_sizes"), _pair(planned_sizes, "planned_sizes")
    if np.any(interim > totals):
        raise ValueError("interim_sizes cannot exceed planned_sizes")
    original = _prior(prior)
    decision = _decisions(
        totals,
        original,
        np.arange(totals[0] + 1),
        np.arange(totals[1] + 1),
        method,
        significance_level,
        posterior_cutoff,
    )
    kernels = []
    for m, total, a, b in zip(interim, totals, original.alpha, original.beta, strict=True):
        kernel = np.zeros((m + 1, total + 1))
        for s in range(m + 1):
            kernel[s, s : s + total - m + 1] = _mass(int(total - m), float(a + s), float(b + m - s))
        kernels.append(kernel)
    probabilities = [_owned(kernels[0] @ (decision == code) @ kernels[1].T) for code in (1, 2, 0)]
    return BinaryPredictivePlan(
        probabilities[0],
        probabilities[1],
        probabilities[2],
        _owned(np.arange(interim[0] + 1)),
        _owned(np.arange(interim[1] + 1)),
    )
