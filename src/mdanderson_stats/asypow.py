"""ASYPOW local information-matrix power calculations for linear nulls."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.stats import chi2, ncx2

from ._validation import FloatArray, finite, scalar
from .boin import _owned


def _probability(value: ArrayLike, name: str) -> FloatArray:
    p = finite(value, name)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError(f"{name} must lie strictly between zero and one")
    return p


@dataclass(frozen=True)
class AsymptoticPower:
    noncentrality_per_observation: float
    degrees_of_freedom: int
    null_parameters: FloatArray

    def power(self, sample_size: ArrayLike, significance: ArrayLike = 0.05) -> FloatArray:
        """Noncentral chi-square power; sample size may be continuous or zero."""
        n, alpha = np.broadcast_arrays(
            finite(sample_size, "sample_size"), _probability(significance, "significance")
        )
        if np.any(n < 0):
            raise ValueError("sample_size must be nonnegative")
        with np.errstate(over="ignore"):
            noncentrality = n * self.noncentrality_per_observation
        if np.any(~np.isfinite(noncentrality)) or np.any(noncentrality > 1e8):
            raise ValueError("total noncentrality must not exceed 1e8")
        return _owned(
            ncx2.sf(
                chi2.isf(alpha, self.degrees_of_freedom), self.degrees_of_freedom, noncentrality
            )
        )

    def sample_size(self, power: float = 0.8, significance: float = 0.05) -> float:
        """Continuous sample size reaching requested power (ceil for integer planning)."""
        requested = float(_probability(scalar(power, "power"), "power"))
        alpha = float(_probability(scalar(significance, "significance"), "significance"))
        if requested <= alpha:
            return 0.0
        if self.noncentrality_per_observation == 0:
            raise ValueError("power above significance is unattainable under the null")
        critical = chi2.isf(alpha, self.degrees_of_freedom)

        def difference(ncp: float) -> float:
            return float(ncx2.sf(critical, self.degrees_of_freedom, ncp)) - requested

        high = 1.0
        while difference(high) < 0 and high < 1e8:
            high = min(2 * high, 1e8)
        if difference(high) < 0:
            raise ValueError("requested power requires noncentrality above 1e8")
        ncp = brentq(difference, 0, high, xtol=1e-12)
        result = ncp / self.noncentrality_per_observation
        if not np.isfinite(result):
            raise ArithmeticError("required sample size is not representable")
        return float(result)

    def significance(self, sample_size: ArrayLike, power: ArrayLike = 0.8) -> FloatArray:
        """Significance needed for requested power, using the model's actual df."""
        n, requested = np.broadcast_arrays(
            finite(sample_size, "sample_size"), _probability(power, "power")
        )
        if np.any(n < 0):
            raise ValueError("sample_size must be nonnegative")
        with np.errstate(over="ignore"):
            ncp = n * self.noncentrality_per_observation
        if np.any(~np.isfinite(ncp)) or np.any(ncp > 1e8):
            raise ValueError("total noncentrality must not exceed 1e8")
        critical = ncx2.isf(requested, self.degrees_of_freedom, ncp)
        return _owned(chi2.sf(critical, self.degrees_of_freedom))


def asypow_information(
    parameters: ArrayLike,
    information: ArrayLike,
    contrasts: ArrayLike,
    null_values: ArrayLike = 0,
    *,
    information_observations: float = 1,
) -> AsymptoticPower:
    """Construct local asymptotic power for H0: contrasts @ parameters = null_values.

    information is positive definite Fisher information at the alternative,
    for information_observations observations. Contrasts must have independent
    rows. Fixed-parameter and equality nulls are special cases of this interface.
    The returned null vector is the information-metric projection, not an exact
    constrained model MLE for a nonlinear likelihood.
    """
    theta, info, c = (
        finite(parameters, "parameters"),
        finite(information, "information"),
        finite(contrasts, "contrasts"),
    )
    if theta.ndim != 1 or not 1 <= theta.size <= 500 or info.shape != (theta.size, theta.size):
        raise ValueError(
            "parameters must have length 1..500 and information must be square of that size"
        )
    if c.ndim == 1:
        c = c[None, :]
    if c.ndim != 2 or c.shape[1] != theta.size or not 1 <= c.shape[0] <= theta.size:
        raise ValueError("contrasts must have 1..p rows and p columns")
    target = np.broadcast_to(finite(null_values, "null_values"), (c.shape[0],))
    observations = scalar(information_observations, "information_observations")
    if observations <= 0:
        raise ValueError("information_observations must be positive")
    if not np.allclose(info, info.T, rtol=1e-12, atol=0):
        raise ValueError("information must be symmetric")
    scale = np.max(np.abs(c), axis=1)
    if np.any(scale == 0):
        raise ValueError("contrast rows must be nonzero and independent")
    c, target = c / scale[:, None], target / scale
    try:
        chol = np.linalg.cholesky(info / 2 + info.T / 2)
    except np.linalg.LinAlgError as error:
        raise ValueError("information must be positive definite") from error
    whitened = np.linalg.solve(chol, c.T).T
    u, singular, vt = np.linalg.svd(whitened, full_matrices=False)
    if singular[-1] <= np.finfo(float).eps * max(c.shape) * singular[0]:
        raise ValueError("contrasts are dependent or numerically unresolved")
    standardized = (u.T @ (c @ theta - target)) / singular
    per_observation = standardized / np.sqrt(observations)
    ncp = float(per_observation @ per_observation)
    null = theta - np.linalg.solve(chol.T, vt.T @ standardized)
    if not np.isfinite(ncp) or not np.all(np.isfinite(null)):
        raise ArithmeticError("information calculation exceeds floating-point range")
    return AsymptoticPower(ncp, c.shape[0], _owned(null))
