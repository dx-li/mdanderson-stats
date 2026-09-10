"""Likelihood-weighted CID2BP tails using polynomial Gaussian quadrature."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp, roots_legendre, xlogy
from scipy.stats import binom


def _weighted(n1: int, x1: int, n2: int, x2: int, tail: float, mid_p: bool) -> tuple[float, float]:
    if n1 + n2 > 200:
        raise ValueError("weighted CID2BP methods require total trials <=200")
    # Product of observed likelihood and replicate tail has degree <=2*(n1+n2).
    nodes, weights = roots_legendre(n1 + n2 + 1)
    u = (nodes + 1) / 2
    log_weights = np.log(weights)
    counts = np.arange(n1 + 1)[:, None]
    observed = x1 * n2 - x2 * n1
    numerator = counts * n2 - observed
    cutoff = numerator // n1
    tied = numerator % n1 == 0
    # Inclusive upper tail in sample difference is a lower tail in sample-2 X.
    lower_cutoff = cutoff - tied  # ceil(numerator/n1)-1 for opposite tail

    def probability(delta: float, upper: bool) -> float:
        if abs(delta) == 1:
            point = n1 * n2 if delta == 1 else -n1 * n2
            return float(
                0.5
                if mid_p and point == observed
                else point >= observed
                if upper
                else point <= observed
            )
        width = 1 - abs(delta)
        # Form all four probabilities directly, avoiding subtraction near +/-1.
        if delta >= 0:
            q = width * u
            p = delta + q
            one_p = width * (1 - u)
            one_q = 1 - q
        else:
            p = width * u
            q = -delta + p
            one_q = width * (1 - u)
            one_p = 1 - p
        likelihood = xlogy(x1, p) + xlogy(n1 - x1, one_p) + xlogy(x2, q) + xlogy(n2 - x2, one_q)
        posterior_weights = np.exp(log_weights + likelihood - logsumexp(log_weights + likelihood))
        mass1 = np.exp(binom.logpmf(counts, n1, p))
        probabilities = binom.cdf(cutoff, n2, q) if upper else binom.sf(lower_cutoff, n2, q)
        if mid_p:
            probabilities -= 0.5 * binom.pmf(cutoff, n2, q) * tied
        averaged = (mass1 * probabilities).sum(axis=0) @ posterior_weights
        return float(averaged)

    lower = (
        -1.0
        if observed == -n1 * n2
        else brentq(lambda d: probability(d, True) - tail, -1, 1, xtol=1e-11)
    )
    upper = (
        1.0
        if observed == n1 * n2
        else brentq(lambda d: probability(d, False) - tail, -1, 1, xtol=1e-11)
    )
    if lower > upper:
        raise ArithmeticError("weighted tail inversion yields an empty interval")
    return float(lower), float(upper)
