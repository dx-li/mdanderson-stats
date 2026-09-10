"""Independent same-family stochastic inequalities, including additive shifts."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.special import expit, ndtr

from ._validation import scalar
from .beta_binomial import BetaBinomialPosterior
from .beta_comparison import _cdf_from_logs, compare_beta_binomial
from .parameter_distribution import ParameterDistribution


@dataclass(frozen=True)
class InequalityProbability:
    """P(X > Y + delta), its reverse, and estimated absolute numerical error.

    Quadrature error includes an explicit bound for discarded probability tails.
    It is not a rigorous floating-point error bound. Analytic routes report zero
    quadrature error. Both directions are evaluated without subtraction from one.
    """

    x_greater: float
    shifted_y_greater: float
    absolute_error: float
    method: str


def _normal_z(mx: float, my: float, sx: float, sy: float, delta: float) -> float:
    scale = max(sx, sy)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        difference = float(np.float64(mx) - my - delta)
        if np.isfinite(difference):
            z = difference / scale / np.hypot(sx / scale, sy / scale)
        else:
            magnitude = max(abs(mx), abs(my), abs(delta), scale)
            difference = mx / magnitude - my / magnitude - delta / magnitude
            z = difference / (scale / magnitude) / np.hypot(sx / scale, sy / scale)
    if np.isnan(z):
        raise ArithmeticError("normal comparison cannot be resolved")
    return float(z)


def _integrate_ordering(
    x: ParameterDistribution, y: ParameterDistribution, delta: float, tolerance: float
) -> tuple[float, float]:
    # U=F_X(X) is uniform, so E[F_Y(X-delta)] has a bounded integrand.
    # Quantile-derived breakpoints expose narrow CDF transitions to QUADPACK.
    tail = tolerance / 16
    dx, dy = x._distribution(), y._distribution()
    marks = np.array([tail, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1 - tail])
    x.quantile([tail, 1 - tail])  # Fail explicitly if integration bounds are unresolved.
    quantiles = y.quantile(marks)
    with np.errstate(over="ignore"):
        locations = np.concatenate(([delta], quantiles + delta))
    if x.family == "beta":
        locations = np.append(locations, 1 + delta)
    points = np.asarray(dx.cdf(locations), dtype=float)
    if np.any(~np.isfinite(points)):
        raise ArithmeticError("inequality integration breakpoints cannot be resolved")
    points = np.unique(points[(points > tail) & (points < 1 - tail)])
    # Adjacent rounded support/quantile breakpoints can make QUADPACK unable to
    # subdivide an interval. Coalescing points does not discard integration mass.
    separated: list[float] = []
    previous = tail
    for point in points:
        if (
            point - previous > 32 * np.finfo(float).eps
            and 1 - tail - point > 32 * np.finfo(float).eps
        ):
            separated.append(float(point))
            previous = float(point)

    def integrand(u: float) -> float:
        value = float(dx.ppf(u))
        if not np.isfinite(value) or value <= 0 or (x.family == "beta" and value >= 1):
            raise ArithmeticError("inequality integration quantile cannot be resolved")
        with np.errstate(over="ignore"):
            shifted = np.float64(value) - delta
        probability = float(dy.cdf(shifted))
        if not 0 <= probability <= 1:
            raise ArithmeticError("inequality integration CDF cannot be resolved")
        return probability

    result = quad(
        integrand,
        tail,
        1 - tail,
        points=separated,
        epsabs=tolerance / 4,
        epsrel=tolerance / 4,
        limit=300,
        full_output=1,
    )
    value, error = result[:2]
    if len(result) != 3 or not np.isfinite(error) or error > tolerance / 2 or not 0 <= value <= 1:
        raise ArithmeticError("inequality quadrature failed to converge")
    return float(value), float(error + 2 * tail)


def inequality_probability(
    x: ParameterDistribution,
    y: ParameterDistribution,
    *,
    delta: float = 0,
    absolute_tolerance: float = 1e-9,
) -> InequalityProbability:
    """Compare independent variables in the same family, P(X > Y + delta).

    Normal uses mean/variance; lognormal uses log-mean/log-SD. Other conventions
    follow ParameterDistribution. Absolute tolerance is not a relative guarantee
    for arbitrarily rare events. Scalar parameters are required.
    """
    if not isinstance(x, ParameterDistribution) or not isinstance(y, ParameterDistribution):
        raise TypeError("x and y must be ParameterDistribution objects")
    if x.family != y.family:
        raise ValueError("x and y must belong to the same distribution family")
    delta = scalar(delta, "delta")
    tol = scalar(absolute_tolerance, "absolute_tolerance")
    if not 1e-12 <= tol <= 1e-3:
        raise ValueError("absolute_tolerance must be in [1e-12, 1e-3]")
    family = x.family
    a, b, c, d = x.parameter1, x.parameter2, y.parameter1, y.parameter2
    if family == "normal" or (family == "lognormal" and delta == 0):
        sx, sy = (float(np.sqrt(b)), float(np.sqrt(d))) if family == "normal" else (b, d)
        z = _normal_z(a, c, sx, sy, delta)
        return InequalityProbability(float(ndtr(z)), float(ndtr(-z)), 0, "normal identity")
    if family == "beta" and abs(delta) >= 1:
        return InequalityProbability(float(delta < 0), float(delta > 0), 0, "disjoint supports")
    if delta == 0:
        if x == y:
            return InequalityProbability(0.5, 0.5, 0, "exchangeability")
        if family == "beta":
            result = compare_beta_binomial(
                BetaBinomialPosterior(c, d), BetaBinomialPosterior(a, b), absolute_tolerance=tol
            )
            return InequalityProbability(
                float(result.treatment_greater),
                float(result.control_greater),
                float(result.absolute_error),
                "beta logit quadrature",
            )
        if family in ("gamma", "inverse_gamma"):
            lb, ld = float(np.log(b)), float(np.log(d))
            lr, lc = -float(np.logaddexp(0, ld - lb)), -float(np.logaddexp(0, lb - ld))
            first, second = (c, a) if family == "gamma" else (a, c)
            p = _cdf_from_logs(first, second, lr, lc)
            q = _cdf_from_logs(second, first, lc, lr)
            if not 0 <= p <= 1 or not 0 <= q <= 1 or abs(p + q - 1) > 2 * tol:
                raise ArithmeticError("gamma inequality identity failed its probability check")
            return InequalityProbability(p, q, 0, "beta identity")
        if family == "weibull" and a == c:
            with np.errstate(over="ignore"):
                log_odds = a * (np.log(b) - np.log(d))
            return InequalityProbability(
                float(expit(log_odds)), float(expit(-log_odds)), 0, "equal-shape Weibull identity"
            )
    p, pe = _integrate_ordering(x, y, delta, tol)
    q, qe = _integrate_ordering(y, x, -delta, tol)
    if abs(p + q - 1) > 2 * tol:
        raise ArithmeticError("inequality probabilities failed their complement check")
    return InequalityProbability(p, q, max(pe, qe), "probability-domain quadrature")
