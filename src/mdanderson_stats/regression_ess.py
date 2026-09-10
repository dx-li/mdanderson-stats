"""Morita–Thall–Müller prior ESS for independent-prior regression models."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from ._validation import FloatArray, count, finite, scalar
from .boin import _owned
from .parameter_distribution import ParameterDistribution


def _gain(priors: Sequence[ParameterDistribution], inflation: float) -> FloatArray:
    c = scalar(inflation, "variance_inflation")
    if c <= 1:
        raise ValueError("variance_inflation must exceed one")
    if not 1 <= len(priors) <= 12 or any(p.family not in ("normal", "gamma") for p in priors):
        raise ValueError("require 1..12 independent normal or gamma priors")
    log_factor = np.log(c - 1) - np.log(c)
    # Direct differences avoid cancellation of negative gamma curvatures when
    # shape<1: D_p-D_q0 = (1-1/c)/Var(theta), for both supported families.
    return np.array(
        [
            log_factor - np.log(p.parameter2)
            if p.family == "normal"
            else log_factor - np.log(p.parameter1) - 2 * np.log(p.parameter2)
            for p in priors
        ]
    )


def _ordinary(log_values: ArrayLike) -> FloatArray:
    values = np.asarray(log_values)
    with np.errstate(over="ignore", under="ignore"):
        result = np.exp(values)
    if np.any(np.isfinite(values) & (~np.isfinite(result) | (result == 0))):
        raise ArithmeticError("ESS is outside float64 range; use log ESS")
    return _owned(result)


@dataclass(frozen=True)
class RegressionESS:
    log_prior_information_gain: FloatArray
    log_observation_information: FloatArray
    variance_inflation: float
    parameter_names: tuple[str, ...]

    @property
    def log_ess(self) -> float:
        return float(
            logsumexp(self.log_prior_information_gain) - logsumexp(self.log_observation_information)
        )

    @property
    def ess(self) -> float:
        return float(_ordinary(self.log_ess))

    @property
    def log_component_ess(self) -> FloatArray:
        return _owned(self.log_prior_information_gain - self.log_observation_information)

    @property
    def component_ess(self) -> FloatArray:
        return _ordinary(self.log_component_ess)

    def log_subvector_ess(self, indices: ArrayLike) -> float:
        """Trace-information ratio for selected zero-based parameter indices."""
        subset = count(indices, "indices")
        if (
            subset.ndim != 1
            or subset.size == 0
            or np.any(subset >= len(self.parameter_names))
            or np.unique(subset).size != subset.size
        ):
            raise ValueError("indices must be a nonempty vector of distinct parameter indices")
        selected = subset.astype(np.int64)
        return float(
            logsumexp(self.log_prior_information_gain[selected])
            - logsumexp(self.log_observation_information[selected])
        )

    def subvector_ess(self, indices: ArrayLike) -> float:
        return float(_ordinary(self.log_subvector_ess(indices)))


def logistic_regression_ess(
    coefficient_priors: Sequence[ParameterDistribution],
    covariate_support: ArrayLike,
    *,
    probabilities: ArrayLike | None = None,
    variance_inflation: float = 10000,
) -> RegressionESS:
    """Exact expectation over a finite covariate distribution; intercept is explicit.

    Rows are support points, columns are regression coefficients (at most 11).
    Omitted probabilities mean uniform support, not a fixed total sample size.
    Curvature is evaluated at prior means, not averaged over parameter draws.
    A continuous covariate distribution can be approximated by supplied samples
    or quadrature points/weights; that approximation remains the caller's choice.
    """
    priors = tuple(coefficient_priors)
    gain = _gain(priors, variance_inflation)
    x = finite(covariate_support, "covariate_support")
    if (
        x.ndim != 2
        or not 1 <= x.shape[0]
        or x.shape[1] != len(priors)
        or len(priors) > 11
        or x.size > 20000000
    ):
        raise ValueError(
            "support requires nonempty rows, 1..11 prior-matched columns and <=20 million cells"
        )
    w = np.ones(x.shape[0]) if probabilities is None else finite(probabilities, "probabilities")
    if w.shape != (x.shape[0],) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("probabilities require nonnegative weights with positive total")
    positive = w > 0
    x = x[positive]
    logw = np.log(w[positive])
    logw -= logsumexp(logw)
    means = np.array([p.mean for p in priors])
    with np.errstate(over="ignore", invalid="ignore"):
        eta = x @ means
    if np.any(~np.isfinite(eta)):
        raise ArithmeticError("linear predictor at prior means is not representable")
    log_variance = -np.logaddexp(0, eta) - np.logaddexp(0, -eta)
    with np.errstate(divide="ignore"):
        information = logsumexp(
            logw[:, None] + log_variance[:, None] + 2 * np.log(np.abs(x)), axis=0
        )
    return RegressionESS(
        _owned(gain),
        _owned(information),
        float(variance_inflation),
        tuple(f"coefficient_{j}" for j in range(len(priors))),
    )


def normal_regression_ess(
    coefficient_priors: Sequence[ParameterDistribution],
    covariate_second_moments: ArrayLike,
    precision_prior: ParameterDistribution,
    *,
    variance_inflation: float = 10000,
) -> RegressionESS:
    """Normal linear regression with independent coefficients and gamma precision.

    Supply E[X_j**2] per design column, including 1 for an intercept. Coefficient
    priors are normal(mean,variance) or gamma(shape,scale); the precision prior
    uses gamma(shape,scale). Precision is the final component of the result.
    """
    coefficients = tuple(coefficient_priors)
    if not 1 <= len(coefficients) <= 11 or precision_prior.family != "gamma":
        raise ValueError("require 1..11 coefficient priors and a gamma precision prior")
    gain = _gain((*coefficients, precision_prior), variance_inflation)
    moment = finite(covariate_second_moments, "covariate_second_moments")
    if moment.shape != (len(coefficients),) or np.any(moment < 0):
        raise ValueError("require one nonnegative second moment per coefficient")
    log_precision = np.log(precision_prior.parameter1) + np.log(precision_prior.parameter2)
    with np.errstate(divide="ignore"):
        information = np.r_[log_precision + np.log(moment), -np.log(2) - 2 * log_precision]
    return RegressionESS(
        _owned(gain),
        _owned(information),
        float(variance_inflation),
        tuple(f"coefficient_{j}" for j in range(len(coefficients))) + ("precision",),
    )
