"""Median-effect regression for independently measured drug dose-response curves."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit

from ._validation import FloatArray, finite, scalar
from .boin import _owned


@dataclass(frozen=True)
class MedianEffectFit:
    intercept: float
    slope: float
    covariance: FloatArray
    observations: int
    residual_variance: float

    def __post_init__(self) -> None:
        for name in ["intercept", "slope", "residual_variance"]:
            object.__setattr__(self, name, scalar(getattr(self, name), name))
        n = scalar(self.observations, "observations")
        if n != int(n) or n < 3 or self.slope == 0 or self.residual_variance < 0:
            raise ValueError(
                "require >=3 observations, nonzero slope and nonnegative residual variance"
            )
        object.__setattr__(self, "observations", int(n))
        covariance = finite(self.covariance, "covariance")
        if covariance.shape != (2, 2) or not np.allclose(
            covariance, covariance.T, rtol=1e-12, atol=0
        ):
            raise ValueError("covariance must be a symmetric 2x2 matrix in intercept/slope order")
        scale = np.max(abs(covariance))
        if scale > 0 and np.linalg.eigvalsh(covariance / scale).min() < -1e-12:
            raise ValueError("covariance must be positive semidefinite")
        object.__setattr__(self, "covariance", _owned(covariance))

    def effect(self, dose: ArrayLike) -> FloatArray:
        d = finite(dose, "dose")
        if np.any(d <= 0):
            raise ValueError("dose must be positive")
        return _owned(expit(self.intercept + self.slope * np.log(d)))

    def log_dose(self, effect: ArrayLike) -> FloatArray:
        y = finite(effect, "effect")
        if np.any((y <= 0) | (y >= 1)):
            raise ValueError("effect must be strictly between zero and one")
        value = (np.log(y) - np.log1p(-y) - self.intercept) / self.slope
        if np.any(~np.isfinite(value)):
            raise ArithmeticError("inverse median-effect dose is not representable in log space")
        return _owned(value)


def fit_median_effect(dose: ArrayLike, effect: ArrayLike) -> MedianEffectFit:
    """OLS logit(effect) on log(dose), with iid homoscedastic transformed errors.

    Replicates are individual rows; no clipping or pseudocounts are introduced.
    Both increasing and decreasing curves are supported. Covariance uses SSE/(n-2).
    """
    d, y = finite(dose, "dose"), finite(effect, "effect")
    if (
        d.ndim != 1
        or d.size < 3
        or y.shape != d.shape
        or np.any(d <= 0)
        or np.any((y <= 0) | (y >= 1))
    ):
        raise ValueError("require >=3 matching doses >0 and effects strictly in (0,1)")
    x, z = np.log(d), np.log(y) - np.log1p(-y)
    center = x.mean()
    dx = x - center
    sxx = float(dx @ dx)
    if sxx <= np.finfo(float).eps ** 2 * max(1.0, float(x @ x)):
        raise ValueError("log doses have insufficient variation to fit a slope")
    slope = float(dx @ (z - z.mean()) / sxx)
    intercept = float(z.mean() - slope * center)
    residual = z - (z.mean() + slope * dx)
    variance = float(residual @ residual / (d.size - 2))
    covariance = variance * np.array(
        [[1 / d.size + center**2 / sxx, -center / sxx], [-center / sxx, 1 / sxx]]
    )
    return MedianEffectFit(intercept, slope, covariance, d.size, variance)
