"""Adaptive conditional quadrature for the explicitly configured ComPAS hierarchy."""

import math
import warnings
from collections.abc import Callable

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, gammaln, logit, ndtr


def _quad(f: Callable[[float], float], a: float, b: float, tol: float) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        try:
            value, error = quad(f, a, b, epsabs=tol, epsrel=tol, limit=160)
        except IntegrationWarning as exc:
            raise ArithmeticError("ComPAS quadrature did not converge") from exc
    if not math.isfinite(value) or not math.isfinite(error) or value < 0:
        raise ArithmeticError("ComPAS quadrature returned invalid mass")
    return float(value), float(error)


def _arm(
    mu: float, variance: float, n: int, y: int, threshold: float, tol: float
) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, float(ndtr((mu - threshold) / math.sqrt(variance))), 0.0
    mode = float(
        brentq(
            lambda t: y - n * expit(t) - (t - mu) / variance,
            mu - variance * (n - y) - 1,
            mu + variance * y + 1,
            xtol=1e-12,
        )
    )
    p = float(expit(mode))
    scale = 1 / math.sqrt(1 / variance + n * p * (1 - p))
    const = float(gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)) - 0.5 * math.log(
        2 * math.pi * variance
    )

    def logf(theta: float) -> float:
        return float(
            -y * np.logaddexp(0.0, -theta)
            - (n - y) * np.logaddexp(0.0, theta)
            - 0.5 * ((theta - mu) / math.sqrt(variance)) ** 2
        )

    peak = logf(mode)

    gradient = y - n * p - (mode - mu) / variance

    def f(t: float) -> float:
        delta = scale * t
        theta = mode + delta
        # Evaluate the log-density ratio without subtracting two potentially
        # huge log likelihoods in the diffuse group-prior tails.
        if mode >= 0 and theta >= 0:
            remainder = (
                float(expit(-mode)) * delta
                + math.log1p(math.exp(-theta))
                - math.log1p(math.exp(-mode))
            )
        elif mode <= 0 and theta <= 0:
            remainder = math.log1p(math.exp(theta)) - math.log1p(math.exp(mode)) - p * delta
        else:
            remainder = float(np.logaddexp(0.0, theta) - np.logaddexp(0.0, mode)) - p * delta
        return math.exp(-n * remainder - 0.5 * delta * delta / variance + gradient * delta)

    # The log likelihood is concave, so about the posterior mode the
    # integrand is bounded by peak*exp(-delta**2/(2*variance)). The omitted
    # +/-12 prior-SD mass is bounded by 2*Phi(-12)*sqrt(1+n*variance/4)
    # relative to the denominator (global upper curvature 1/variance+n/4).
    bound = 12 * math.sqrt(variance) / scale
    cut = max(-bound, min(bound, (threshold - mode) / scale))
    points = sorted(set([-bound, 0.0, cut, bound]))
    results = [_quad(f, a, b, tol) for a, b in zip(points, points[1:])]
    den = sum(v for v, _ in results)
    if den <= 0:
        raise ArithmeticError("ComPAS conditional posterior has no representable mass")
    tail = sum(v for a, (v, _) in zip(points, results) if a >= cut) / den
    relative_error = sum(e for _, e in results) / den + 2 * float(ndtr(-12)) * math.sqrt(
        1 + n * variance / 4
    )
    return const + peak + math.log(scale * den), tail, relative_error


def compas_group_integral(
    completed: tuple[int, ...],
    successes: tuple[int, ...],
    variance: float,
    null_response: float,
    tolerance: float,
) -> tuple[float, tuple[float, ...], float, tuple[float, ...], int]:
    """Return log evidence, arm tails and quadrature error estimates for one group.

    Gaussian normalizers and binomial coefficients are included. Error values
    are numerical estimates, not certified posterior intervals or clinical CIs.
    """
    threshold = float(logit(null_response))
    if not any(completed):
        tail = float(ndtr(-threshold / math.sqrt(1e6 + variance)))
        return 0.0, (tail,) * len(completed), 0.0, (0.0,) * len(completed), 0
    cache: dict[float, tuple[float, tuple[float, ...]]] = {}
    max_inner_error = 0.0

    def calc(mu: float) -> tuple[float, tuple[float, ...]]:
        nonlocal max_inner_error
        if mu not in cache:
            if len(cache) >= 20000:
                raise ArithmeticError("ComPAS group quadrature exceeded 20000 evaluations")
            results = [
                _arm(mu, variance, n, y, threshold, tolerance * 0.001)
                for n, y in zip(completed, successes)
            ]
            max_inner_error = max(max_inner_error, sum(r[2] for r in results))
            cache[mu] = (
                sum(r[0] for r in results) - 0.5 * mu**2 / 1e6,
                tuple(r[1] for r in results),
            )
        return cache[mu]

    bound = max(100.0, 12 * math.sqrt(variance))
    optimum = minimize_scalar(
        lambda mu: -calc(float(mu))[0],
        bounds=(-bound, bound),
        method="bounded",
        options={"xatol": 1e-7},
    )
    if not optimum.success:
        raise ArithmeticError("ComPAS integration centering failed")
    mode = float(optimum.x)
    peak = calc(mode)[0]
    step = 0.01
    curvature = -(calc(mode + step)[0] + calc(mode - step)[0] - 2 * peak) / step**2
    scale = 1 / math.sqrt(max(curvature, 1e-6))

    def f(z: float, index: int | None = None) -> float:
        logval, tails = calc(mode + scale * z)
        return math.exp(logval - peak) * (1.0 if index is None else tails[index])

    # Explicit transition splits prevent a diffuse group prior from hiding a
    # narrow arm-tail boundary. The mode split also protects concentrated data.
    points = sorted(
        set(
            [0.0]
            + [
                (threshold + variance * (n * null_response - y) + k * math.sqrt(variance) - mode)
                / scale
                for n, y in zip(completed, successes)
                for k in (-3, 0, 3)
            ]
        )
    )
    bounds = [-math.inf, *points, math.inf]
    budget = tolerance / (10 * len(bounds))
    denominator = [_quad(f, a, b, budget) for a, b in zip(bounds, bounds[1:])]
    den = sum(v for v, _ in denominator)
    if den <= 0:
        raise ArithmeticError("ComPAS marginal likelihood has no representable mass")
    den_error = sum(e for _, e in denominator) / den
    tails, errors = [], []
    for j in range(len(completed)):
        numerator = [_quad(lambda z: f(z, j), a, b, budget) for a, b in zip(bounds, bounds[1:])]
        tail = sum(v for v, _ in numerator) / den
        error = sum(e for _, e in numerator) / den + tail * den_error
        if not 0 <= tail <= 1 + 1e-12:
            raise ArithmeticError("ComPAS posterior tail is outside probability bounds")
        tails.append(min(1.0, tail))
        errors.append(error)
    # Conditional error affects both the product likelihood and its tail ratio.
    relative_error = den_error + 2 * max_inner_error
    errors = [e + 4 * max_inner_error for e in errors]
    if max(relative_error, *errors) > tolerance:
        raise ArithmeticError("ComPAS integration precision was not achieved")
    return (
        peak + math.log(scale * den) - 0.5 * math.log(2 * math.pi * 1e6),
        tuple(tails),
        relative_error,
        tuple(errors),
        len(cache),
    )
