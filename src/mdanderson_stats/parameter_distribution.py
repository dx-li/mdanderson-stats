"""Parameter Solver's six distribution conventions and forward calculations."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.special import gammaln, zeta

from ._validation import FloatArray, finite, scalar

_FAMILIES = ("beta", "gamma", "inverse_gamma", "normal", "lognormal", "weibull")


def _positive_exp(log_value: float) -> float:
    with np.errstate(over="ignore", under="ignore"):
        result = float(np.exp(log_value))
    if not np.isfinite(result) or result <= 0:
        raise ArithmeticError("positive quantity is not representable in float64")
    return result


def _weibull_log_cv2(shape: float) -> float:
    t = 1 / shape
    if t < 0.001:
        n = np.arange(2, 13)
        difference = float(np.sum((-1.0) ** n * zeta(n, 1) / n * (2.0**n - 2) * t**n))
    else:
        difference = float(gammaln(1 + 2 * t) - 2 * gammaln(1 + t))
    if difference <= 0 or not np.isfinite(difference):
        raise ArithmeticError("Weibull variance cannot be resolved")
    return (
        difference + float(np.log1p(-np.exp(-difference)))
        if difference > 50
        else float(np.log(np.expm1(difference)))
    )


@dataclass(frozen=True)
class ParameterDistribution:
    """Scalar distribution parameters; density/tail/quantile evaluations accept arrays.

    Normal parameters are mean and variance. Lognormal parameters are log-mean
    and log-standard-deviation. Gamma/inverse-gamma/Weibull use shape and scale.
    """

    family: str
    parameter1: float
    parameter2: float

    def __post_init__(self) -> None:
        a, b = scalar(self.parameter1, "parameter1"), scalar(self.parameter2, "parameter2")
        if self.family not in _FAMILIES:
            raise ValueError(f"family must be one of {_FAMILIES}")
        if b <= 0 or (self.family not in ("normal", "lognormal") and a <= 0):
            raise ValueError("scale/variance and shape parameters must be positive")
        if self.family == "beta" and not np.isfinite(a + b):
            raise ValueError("beta shape sum must be finite")
        object.__setattr__(self, "parameter1", a)
        object.__setattr__(self, "parameter2", b)

    @property
    def parameter_names(self) -> tuple[str, str]:
        return {
            "beta": ("alpha", "beta"),
            "normal": ("mean", "variance"),
            "lognormal": ("log_mean", "log_sd"),
        }.get(self.family, ("shape", "scale"))

    @property
    def mean(self) -> float:
        a, b = self.parameter1, self.parameter2
        if self.family == "normal":
            return a
        if self.family == "beta":
            return a / (a + b)
        if self.family == "inverse_gamma" and a <= 1:
            return np.inf
        log_mean = {
            "gamma": lambda: np.log(a) + np.log(b),
            "inverse_gamma": lambda: np.log(b) - np.log(a - 1),
            "lognormal": lambda: a + b * b / 2,
            "weibull": lambda: np.log(b) + gammaln(1 + 1 / a),
        }[self.family]()
        return _positive_exp(float(log_mean))

    @property
    def variance(self) -> float:
        a, b = self.parameter1, self.parameter2
        if self.family == "normal":
            return b
        if self.family == "beta":
            total = a + b
            return _positive_exp(float(np.log(a / total) + np.log(b / total) - np.log1p(total)))
        if self.family == "inverse_gamma" and a <= 2:
            return np.inf
        if self.family == "gamma":
            log_var = np.log(a) + 2 * np.log(b)
        elif self.family == "inverse_gamma":
            log_var = 2 * np.log(b) - 2 * np.log(a - 1) - np.log(a - 2)
        elif self.family == "lognormal":
            s2 = b * b
            log_extra = s2 + np.log1p(-np.exp(-s2)) if s2 > 50 else np.log(np.expm1(s2))
            log_var = 2 * a + s2 + log_extra
        else:
            log_var = 2 * (np.log(b) + gammaln(1 + 1 / a)) + _weibull_log_cv2(a)
        return _positive_exp(float(log_var))

    def _distribution(self):
        a, b = self.parameter1, self.parameter2
        if self.family == "beta":
            return stats.beta(a, b)
        if self.family == "gamma":
            return stats.gamma(a, scale=b)
        if self.family == "inverse_gamma":
            return stats.invgamma(a, scale=b)
        if self.family == "normal":
            return stats.norm(loc=a, scale=np.sqrt(b))
        if self.family == "lognormal":
            return stats.lognorm(b, scale=_positive_exp(a))
        return stats.weibull_min(a, scale=b)

    def cdf(self, values: ArrayLike) -> FloatArray:
        result = np.asarray(self._distribution().cdf(finite(values, "values")), dtype=float)
        if np.any(~np.isfinite(result)):
            raise ArithmeticError("CDF evaluation failed")
        return result

    def sf(self, values: ArrayLike) -> FloatArray:
        """Evaluate the upper tail directly."""
        result = np.asarray(self._distribution().sf(finite(values, "values")), dtype=float)
        if np.any(~np.isfinite(result)):
            raise ArithmeticError("survival function evaluation failed")
        return result

    def pdf(self, values: ArrayLike) -> FloatArray:
        result = np.asarray(self._distribution().pdf(finite(values, "values")), dtype=float)
        if np.any(np.isnan(result)):
            raise ArithmeticError("density evaluation failed")
        return result

    def quantile(self, probabilities: ArrayLike) -> FloatArray:
        p = finite(probabilities, "probabilities")
        if np.any((p < 0) | (p > 1)):
            raise ValueError("probabilities must lie in [0,1]")
        result = np.asarray(self._distribution().ppf(p), dtype=float)
        interior = (p > 0) & (p < 1)
        invalid = ~np.isfinite(result)
        if self.family not in ("normal",):
            invalid |= result <= 0
        if self.family == "beta":
            invalid |= result >= 1
        if np.any(np.isnan(result)) or np.any(interior & invalid):
            raise ArithmeticError("interior quantile is not representable")
        return result
