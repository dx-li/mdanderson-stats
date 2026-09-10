"""Moment and two-quantile elicitation for Parameter Solver's six families."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.special import betainc, gammainccinv, gammaincinv, gammaln, ndtri

from ._validation import finite, scalar
from .parameter_distribution import ParameterDistribution, _positive_exp, _weibull_log_cv2


def _root(function: Callable[[float], float], low: float = -20, high: float = 30) -> float:
    def checked(x: float) -> float:
        value = float(function(x))
        if not np.isfinite(value):
            raise ArithmeticError("nonfinite parameter-solver residual")
        return value

    left, right = checked(low), checked(high)
    if left == 0:
        return low
    if right == 0:
        return high
    if np.signbit(left) == np.signbit(right):
        raise ArithmeticError("solution is outside the numerically supported search bracket")
    value, info = brentq(
        checked, low, high, xtol=2e-12, rtol=2e-12, maxiter=200, full_output=True, disp=False
    )
    if not info.converged:
        raise ArithmeticError("parameter solver did not converge")
    return float(value)


def solve_distribution_moments(
    family: str,
    mean: float,
    variance: float | None = None,
    *,
    standard_deviation: float | None = None,
) -> ParameterDistribution:
    """Elicit parameters from a mean and exactly one positive variance or SD."""
    m = scalar(mean, "mean")
    if (variance is None) == (standard_deviation is None):
        raise ValueError("provide exactly one of variance and standard_deviation")
    spread_value = variance if variance is not None else standard_deviation
    assert spread_value is not None
    spread = scalar(spread_value, "spread")
    if spread <= 0:
        raise ValueError("variance or standard deviation must be positive")
    logv = float(np.log(spread) * (1 if variance is not None else 2))
    v = _positive_exp(logv)
    if family == "normal":
        result = ParameterDistribution(family, m, v)
    else:
        if m <= 0:
            raise ValueError("mean must be positive for this family")
        lm = float(np.log(m))
        if family == "beta":
            if not m < 1 or not v < m * (1 - m):
                raise ValueError("beta requires 0<mean<1 and 0<variance<mean*(1-mean)")
            concentration = (m * (1 - m) - v) / v
            result = ParameterDistribution(family, m * concentration, (1 - m) * concentration)
        elif family == "gamma":
            result = ParameterDistribution(
                family, _positive_exp(2 * lm - logv), _positive_exp(logv - lm)
            )
        elif family == "inverse_gamma":
            a = 2 + _positive_exp(2 * lm - logv)
            result = ParameterDistribution(family, a, _positive_exp(lm + float(np.log(a - 1))))
        elif family == "lognormal":
            log_variance = float(np.logaddexp(0, logv - 2 * lm))
            result = ParameterDistribution(
                family, lm - log_variance / 2, float(np.sqrt(log_variance))
            )
        elif family == "weibull":
            a = _positive_exp(_root(lambda loga: _weibull_log_cv2(np.exp(loga)) - (logv - 2 * lm)))
            result = ParameterDistribution(family, a, _positive_exp(lm - float(gammaln(1 + 1 / a))))
        else:
            raise ValueError("unsupported distribution family")
    actual_mean, actual_variance = result.mean, result.variance
    if (
        not np.isfinite(actual_mean)
        or not np.isfinite(actual_variance)
        or not np.isclose(actual_mean, m, rtol=2e-8, atol=0)
        or not np.isclose(actual_variance, v, rtol=2e-8, atol=0)
    ):
        raise ArithmeticError("fitted parameters do not reproduce the requested moments")
    return result


def _log_gamma_quantile(shape: float, probability: float, *, upper: bool = False) -> float:
    value = float(gammainccinv(shape, probability) if upper else gammaincinv(shape, probability))
    if value > 0 and np.isfinite(value):
        return float(np.log(value))
    if value == 0:
        lower = 1 - probability if upper else probability
        logq = (np.log(lower) + gammaln(1 + shape)) / shape
        if logq < -25:
            return float(logq)
    raise ArithmeticError("gamma quantile cannot be resolved")


def solve_distribution_quantiles(
    family: str,
    values: ArrayLike,
    probabilities: ArrayLike,
) -> ParameterDistribution:
    """Solve two ordered interior quantiles and verify both resulting probability tails.

    Shape searches use log-parameters in [-20,30]. Requests outside that bracket,
    unrepresentable parameters or unresolved quantiles raise ArithmeticError.
    """
    x, p = finite(values, "values"), finite(probabilities, "probabilities")
    if x.shape != (2,) or p.shape != (2,) or not x[0] < x[1] or not 0 < p[0] < p[1] < 1:
        raise ValueError("require two increasing values and 0<p1<p2<1")
    if family == "normal":
        z = ndtri(p)
        sd = float((x[1] - x[0]) / (z[1] - z[0]))
        result = ParameterDistribution(
            family, float(x[0] - sd * z[0]), _positive_exp(2 * float(np.log(sd)))
        )
    else:
        if x[0] <= 0:
            raise ValueError("quantile values must be positive for this family")
        logs = np.log(x)
        with np.errstate(over="ignore"):
            ratio = (x[1] - x[0]) / x[0]
        log_ratio = float(np.log1p(ratio) if np.isfinite(ratio) else logs[1] - logs[0])
        if family == "lognormal":
            z = ndtri(p)
            sd = float(log_ratio / (z[1] - z[0]))
            result = ParameterDistribution(family, float(logs[0] - sd * z[0]), sd)
        elif family == "weibull":
            logh = np.log(-np.log1p(-p))
            a = float((logh[1] - logh[0]) / log_ratio)
            result = ParameterDistribution(family, a, _positive_exp(float(logs[0] - logh[0] / a)))
        elif family in ("gamma", "inverse_gamma"):
            inverse = family == "inverse_gamma"

            def unit_quantile(a: float, probability: float) -> float:
                q = _log_gamma_quantile(a, probability, upper=inverse)
                return -q if inverse else q

            a = _positive_exp(
                _root(
                    lambda loga: (
                        unit_quantile(np.exp(loga), float(p[1]))
                        - unit_quantile(np.exp(loga), float(p[0]))
                        - log_ratio
                    )
                )
            )
            b = _positive_exp(float(logs[0]) - unit_quantile(a, float(p[0])))
            result = ParameterDistribution(family, a, b)
        elif family == "beta":
            if x[1] >= 1:
                raise ValueError("beta quantiles must lie strictly inside (0,1)")

            def shapes(log_total: float) -> tuple[float, float]:
                def at(logit: float) -> tuple[float, float]:
                    return (
                        float(np.exp(log_total - np.logaddexp(0, -logit))),
                        float(np.exp(log_total - np.logaddexp(0, logit))),
                    )

                logit = _root(lambda z: betainc(*at(z), x[0]) - p[0], -700, 700)
                return at(logit)

            total = _root(lambda logtotal: betainc(*shapes(logtotal), x[1]) - p[1])
            result = ParameterDistribution(family, *shapes(total))
        else:
            raise ValueError("unsupported distribution family")
    # Direct upper tails prevent subtraction from one obscuring high-quantile residuals.
    achieved = np.where(p <= 0.5, result.cdf(x), result.sf(x))
    target = np.minimum(p, 1 - p)
    if np.any(np.abs(achieved - target) > 2e-8 * target):
        raise ArithmeticError(
            "fitted parameters do not reproduce the requested quantile probabilities"
        )
    return result
