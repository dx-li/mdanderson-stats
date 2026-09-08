"""Grouped-binomial STUKEL likelihood and analytic derivatives for six families."""

from dataclasses import dataclass
from math import factorial

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit

from ._validation import FloatArray, count, finite
from .stukel import stukel_log_odds


@dataclass(frozen=True)
class StukelObjective:
    negative_log_likelihood: float
    gradient: FloatArray
    hessian: FloatArray
    log_odds: FloatArray
    probabilities: FloatArray


def _family_shapes(
    coef: FloatArray, p: int, family: int, fixed: FloatArray
) -> tuple[FloatArray, FloatArray]:
    if isinstance(family, bool) or not isinstance(family, int) or not 0 <= family <= 5:
        raise ValueError("family must be an integer from zero to five")
    mappings = (
        np.empty((2, 0)),
        np.array([[1.0], [0.0]]),
        np.array([[0.0], [1.0]]),
        np.array([[1.0], [1.0]]),
        np.array([[1.0], [-1.0]]),
        np.eye(2),
    )
    mapping = mappings[family]
    if coef.size != p + mapping.shape[1]:
        raise ValueError("Coefficient count does not match design and family")
    shapes = fixed.copy()
    active = np.any(mapping != 0, axis=1)
    shapes[active] = (mapping @ coef[p:])[active]
    return shapes, mapping


def _link_derivatives(
    eta: FloatArray, shapes: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    sign = np.where(eta >= 0, 1.0, -1.0)
    a = np.where(eta >= 0, shapes[0], shapes[1])
    m = np.abs(eta)
    z = a * m
    positive = a >= 0
    de = np.where(positive, np.exp(z), 1 / (1 - z))
    dee = sign * np.where(positive, a * de, a * de**2)
    dea = np.where(positive, m * de, m * de**2)
    f1, f2 = np.zeros_like(z), np.zeros_like(z)
    small = np.abs(z) < 0.1
    zs = z[small]
    # Differentiated series for (exp(z)-1)/z and -log(1-z)/z.
    # Sixteen terms avoid cancellation in the shape Hessian near zero.
    v1, v2 = np.zeros_like(zs), np.zeros_like(zs)
    for k in range(16, 0, -1):
        c = np.where(positive[small], 1 / factorial(k + 1), 1 / (k + 1))
        v1 = v1 * zs + k * c
        if k >= 2:
            v2 = v2 * zs + k * (k - 1) * c
    f1[small], f2[small] = v1, v2
    upper = ~small & positive
    zu = z[upper]
    f1[upper] = ((zu - 1) * de[upper] + 1) / zu**2
    f2[upper] = ((zu**2 - 2 * zu + 2) * de[upper] - 2) / zu**3
    lower = ~small & ~positive
    zl = z[lower]
    t = zl / (1 - zl)
    f1[lower] = (t + np.log1p(-zl)) / zl**2
    f2[lower] = (t**2 - 2 * t - 2 * np.log1p(-zl)) / zl**3
    return de, sign * m**2 * f1, dee, dea, sign * m**3 * f2


def stukel_objective(
    x: ArrayLike,
    successes: ArrayLike,
    trials: ArrayLike,
    coefficients: ArrayLike,
    *,
    family: int = 0,
    fixed_alpha: ArrayLike = (0, 0),
) -> StukelObjective:
    """Evaluate FGH's negative log likelihood, gradient and observed Hessian.

    x is an explicit design matrix, including an intercept column if desired.
    Coefficients contain beta, then free shapes. Families 0..5 mean fixed shapes,
    alpha1 free, alpha2 free, equal shapes, opposite shapes, both free.
    Binomial combinatorial constants are omitted, matching FGH. At shape zero
    the source's nonnegative-side second derivative is used; the two-sided
    shape Hessian need not exist there. Family 4 applies the correct chain rule,
    fixing the archived FGH's omitted negative-tail signs.
    """
    x, coef, fixed = (
        finite(x, "x"),
        finite(coefficients, "coefficients"),
        finite(fixed_alpha, "fixed_alpha"),
    )
    y, n = count(successes, "successes"), count(trials, "trials")
    if x.ndim != 2 or min(x.shape) < 1 or coef.ndim != 1 or fixed.shape != (2,):
        raise ValueError(
            "Expected a nonempty design matrix, coefficient vector and two fixed shapes"
        )
    if y.shape != (x.shape[0],) or n.shape != y.shape or np.any((n == 0) | (y > n)):
        raise ValueError("Counts must match design rows, with 0<=successes<=positive trials")
    p = x.shape[1]
    shapes, mapping = _family_shapes(coef, p, family, fixed)
    eta = x @ coef[:p]
    h = stukel_log_odds(eta, *shapes)
    if not np.all(np.isfinite(h)):
        raise ValueError("STUKEL objective exceeds numerical range")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        de, da, dee, dea, daa = _link_derivatives(eta, shapes)
        rows = np.where(eta >= 0, 0, 1)
        weights = mapping[rows]
        jac = np.concatenate((x * de[:, None], da[:, None] * weights), axis=1)
        prob, complement = expit(h), expit(-h)
        residual = (n - y) * prob - y * complement
        curvature = n * prob * complement
        gradient = jac.T @ residual
        hessian = jac.T @ (curvature[:, None] * jac)
        hessian[:p, :p] += x.T @ ((residual * dee)[:, None] * x)
        cross = x.T @ ((residual * dea)[:, None] * weights)
        hessian[:p, p:] += cross
        hessian[p:, :p] += cross.T
        hessian[p:, p:] += weights.T @ ((residual * daa)[:, None] * weights)
        objective = float(np.sum((n - y) * np.logaddexp(0, h) + y * np.logaddexp(0, -h)))
    if (
        not np.isfinite(objective)
        or not np.all(np.isfinite(gradient))
        or not np.all(np.isfinite(hessian))
    ):
        raise ValueError("STUKEL objective derivatives exceed numerical range")
    return StukelObjective(objective, gradient, hessian, h, prob)
