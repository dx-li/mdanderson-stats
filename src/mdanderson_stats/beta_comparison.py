"""Deterministic comparison of two independent beta response probabilities."""

from dataclasses import dataclass
from math import exp, isfinite, log

import numpy as np
from scipy.integrate import quad
from scipy.special import betainc, betaincc, betainccinv, betaincinv, betaln

from ._validation import FloatArray, scalar
from .beta_binomial import BetaBinomialPosterior, _owned


@dataclass(frozen=True)
class BetaComparison:
    """Posterior ordering probabilities and absolute quadrature error estimates.

    Error estimates include an explicit bound on discarded integration tails;
    the quadrature portion is an estimate, not a rigorous mathematical bound.
    """

    treatment_greater: FloatArray
    control_greater: FloatArray
    absolute_error: FloatArray


def _logit_quantile(a: float, b: float, p: float) -> float:
    x, xc = float(betaincinv(a, b, p)), float(betainccinv(b, a, p))
    # Near zero, I_x(a,b) = x**a / (a*B(a,b)) * (1 + O(b*x)).
    # Recover logarithms when even the quantile itself underflows.
    lx = log(x) if x > np.finfo(float).tiny else (log(p) + log(a) + float(betaln(a, b))) / a
    lc = log(xc) if xc > np.finfo(float).tiny else (log(1 - p) + log(b) + float(betaln(a, b))) / b
    if (x <= np.finfo(float).tiny and lx + log(b) > -25) or (
        xc <= np.finfo(float).tiny and lc + log(a) > -25
    ):
        raise ArithmeticError("beta comparison quantile cannot be resolved")
    return lx - lc


def _cdf_from_logs(a: float, b: float, lx: float, lc: float) -> float:
    if lx < -700:
        if lx + log(b) > -25:
            raise ArithmeticError("beta comparison tail cannot be resolved")
        return exp(a * lx - log(a) - float(betaln(a, b)))
    if lc < -700:
        if lc + log(a) > -25:
            raise ArithmeticError("beta comparison tail cannot be resolved")
        return -float(np.expm1(b * lc - log(b) - float(betaln(a, b))))
    return float(betainc(a, b, exp(lx)) if lx < lc else betaincc(b, a, exp(lc)))


def _ordering(a: float, b: float, c: float, d: float, tolerance: float) -> tuple[float, float]:
    # Integrate F_control(x) f_treatment(x) in logit coordinates. This removes
    # endpoint singularities and resolves concentrated distributions on their
    # own scale. Truncation drops at most two tail masses from a bounded CDF.
    tail = tolerance / 16
    lower, upper = _logit_quantile(c, d, tail), _logit_quantile(c, d, 1 - tail)
    center = log(c) - log(d)
    scale = (1 / c + 1 / d) ** 0.5
    if not all(map(isfinite, (lower, upper, center, scale))) or scale <= 0 or lower >= upper:
        raise ArithmeticError("beta comparison integration scale cannot be resolved")
    normalization = float(betaln(c, d))

    def integrand(z: float) -> float:
        t = center + scale * z
        lx, lc = -float(np.logaddexp(0, -t)), -float(np.logaddexp(0, t))
        return _cdf_from_logs(a, b, lx, lc) * exp(c * lx + d * lc - normalization) * scale

    result = quad(
        integrand,
        (lower - center) / scale,
        (upper - center) / scale,
        epsabs=tolerance / 4,
        epsrel=tolerance / 4,
        limit=300,
        full_output=1,
    )
    value, error = result[:2]
    if len(result) != 3 or not isfinite(value) or not isfinite(error) or error > tolerance / 2:
        raise ArithmeticError("beta ordering quadrature failed to converge")
    if not 0 <= value <= 1:
        raise ArithmeticError("beta ordering quadrature returned an invalid probability")
    return float(value), float(error + 2 * tail)


def compare_beta_binomial(
    control: BetaBinomialPosterior,
    treatment: BetaBinomialPosterior,
    *,
    absolute_tolerance: float = 1e-9,
) -> BetaComparison:
    """Compute P(treatment > control) for independent beta distributions.

    Inputs may represent priors or posteriors and broadcast. Both directions
    are integrated directly, preserving small probabilities without subtraction.
    Tolerance is absolute, not relative accuracy for arbitrarily rare events.
    """
    if not isinstance(control, BetaBinomialPosterior) or not isinstance(
        treatment, BetaBinomialPosterior
    ):
        raise TypeError("control and treatment must be BetaBinomialPosterior objects")
    tol = scalar(absolute_tolerance, "absolute_tolerance")
    if not 1e-12 <= tol <= 1e-3:
        raise ValueError("absolute_tolerance must be in [1e-12, 1e-3]")
    a, b, c, d = np.broadcast_arrays(control.alpha, control.beta, treatment.alpha, treatment.beta)
    greater, less, errors = np.empty(a.shape), np.empty(a.shape), np.empty(a.shape)
    for index in np.ndindex(a.shape):
        aa, bb, cc, dd = (float(v[index]) for v in (a, b, c, d))
        if aa == cc and bb == dd:
            greater[index], less[index], errors[index] = 0.5, 0.5, 0
            continue
        p, pe = _ordering(aa, bb, cc, dd, tol)
        q, qe = _ordering(cc, dd, aa, bb, tol)
        if abs(p + q - 1) > 2 * tol:
            raise ArithmeticError("beta ordering probabilities failed their complement check")
        greater[index], less[index], errors[index] = p, q, max(pe, qe)
    return BetaComparison(_owned(greater), _owned(less), _owned(errors))
