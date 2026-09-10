"""Normalized one-sided iMOM binary Bayes factors (k=1, nu=2)."""

import numpy as np
from scipy.special import logsumexp, roots_legendre

from ._validation import FloatArray


def _log_marginal(n: int, p0: float, mode: float, order: int) -> FloatArray:
    logc = np.log(1.5) + 2 * (np.log(mode - p0) - np.log1p(-p0))
    c = float(np.exp(logc))
    # t=log((theta-p0)/sqrt(tau)). Its prior density is
    # 2 exp(-2t-exp(-2t)+c), t < -log(c)/2.
    # Excluded prior mass is exp(-cutoff), far below the likelihood scale.
    cutoff = n * max(-np.log(p0), -np.log1p(-p0)) + 80
    lower, upper = -0.5 * np.log(c + cutoff), -0.5 * logc
    edges = np.linspace(lower, upper, int(np.ceil(upper - lower)) + 1)
    nodes, weights = roots_legendre(order)
    half = np.diff(edges) / 2
    t = ((edges[:-1] + half)[:, None] + half[:, None] * nodes).ravel()
    log_weights = (np.log(half)[:, None] + np.log(weights)).ravel()
    log_weights += np.log(2) - 2 * t - np.exp(-2 * t) + c
    logv = t + 0.5 * logc
    success = np.logaddexp(np.log(p0), np.log1p(-p0) + logv)
    failure = np.log1p(-p0) + np.log(-np.expm1(logv))
    result = np.empty(n + 1)
    for start in range(0, n + 1, 32):
        m = np.arange(start, min(start + 32, n + 1))[:, None]
        result[start : start + len(m)] = logsumexp(
            m * success + (n - m) * failure + log_weights, axis=1
        )
    return result


def _imom_log_bayes_table(n: int, p0: float, mode: float) -> FloatArray:
    previous = _log_marginal(n, p0, mode, 8)
    for order in (16, 32, 64, 128, 256):
        current = _log_marginal(n, p0, mode, order)
        if np.all(np.isfinite(current)) and np.max(np.abs(current - previous)) < 2e-10:
            break
        previous = current
    else:
        raise ArithmeticError("iMOM marginal likelihood quadrature did not converge")
    table = np.full((n + 1, n + 1), np.nan)
    table[n] = current
    # Ordered Bernoulli likelihoods obey L_n(m)=L_(n+1)(m)+L_(n+1)(m+1).
    for size in range(n - 1, -1, -1):
        table[size, : size + 1] = np.logaddexp(
            table[size + 1, : size + 1], table[size + 1, 1 : size + 2]
        )
    if abs(table[0, 0]) > 2e-9:
        raise ArithmeticError("iMOM prior failed its normalization check")
    for size in range(n + 1):
        m = np.arange(size + 1)
        table[size, : size + 1] -= m * np.log(p0) + (size - m) * np.log1p(-p0)
    table[0, 0] = 0  # With no observations, the Bayes factor is exactly one.
    table.flags.writeable = False
    return table
