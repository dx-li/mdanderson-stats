"""One-sided normalized inverse-moment priors for binary response probabilities."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite, scalar
from .parameter_distribution import _positive_exp


@dataclass(frozen=True)
class IMOMBinaryPrior:
    """iMOM prior on (null_rate,1), with nu=2*shape and a specified mode.

    Numerical support is 0.1<=shape<=100. This is explicit parameterization,
    not a conversion from effective sample size.
    """

    null_rate: float
    mode: float
    shape: float = 1

    def __post_init__(self) -> None:
        p0, mode, k = (
            scalar(v, name)
            for v, name in [
                (self.null_rate, "null_rate"),
                (self.mode, "mode"),
                (self.shape, "shape"),
            ]
        )
        if not 0 < p0 < mode < 1 or not 0.1 <= k <= 100:
            raise ValueError("require 0<null_rate<mode<1 and 0.1<=shape<=100")
        for name, value in [("null_rate", p0), ("mode", mode), ("shape", k)]:
            object.__setattr__(self, name, value)

    @property
    def log_tau(self) -> float:
        return float(
            2 * np.log(self.mode - self.null_rate) + np.log1p(1 / (2 * self.shape)) / self.shape
        )

    @property
    def tau(self) -> float:
        return _positive_exp(self.log_tau)

    def _constants(self) -> tuple[float, float]:
        logc = self.log_tau - 2 * np.log1p(-self.null_rate)
        return float(logc), float(np.exp(self.shape * logc))

    def pdf(self, probability: ArrayLike) -> FloatArray:
        p = finite(probability, "probability")
        active = (p > self.null_rate) & (p < 1)
        result = np.zeros(p.shape)
        gap = p[active] - self.null_rate
        _, c = self._constants()
        with np.errstate(over="ignore", under="ignore"):
            z = np.exp(self.shape * (self.log_tau - 2 * np.log(gap)))
            logpdf = (
                np.log(2 * self.shape)
                + self.shape * self.log_tau
                - (2 * self.shape + 1) * np.log(gap)
                - z
                + c
            )
            result[active] = np.exp(logpdf)
        if np.any(~np.isfinite(result)):
            raise ArithmeticError("iMOM prior density cannot be represented")
        return result

    def cdf(self, probability: ArrayLike) -> FloatArray:
        p = finite(probability, "probability")
        active = (p > self.null_rate) & (p < 1)
        result = np.where(p >= 1, 1.0, 0.0)
        _, c = self._constants()
        with np.errstate(over="ignore", under="ignore"):
            z = np.exp(self.shape * (self.log_tau - 2 * np.log(p[active] - self.null_rate)))
            result[active] = np.exp(c - z)
        return result

    def quantile(self, probability: ArrayLike) -> FloatArray:
        p = finite(probability, "probability")
        if np.any((p < 0) | (p > 1)):
            raise ValueError("probability must lie in [0,1]")
        active = (p > 0) & (p < 1)
        result = np.where(p == 1, 1.0, self.null_rate)
        _, c = self._constants()
        result[active] = self.null_rate + np.exp(
            self.log_tau / 2 - np.log(c - np.log(p[active])) / (2 * self.shape)
        )
        if np.any(active & ((result <= self.null_rate) | (result >= 1))):
            raise ArithmeticError("interior iMOM prior quantile cannot be represented")
        return result
