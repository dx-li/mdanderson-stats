"""SURVAN binary logistic regression through the shared logistic fit kernel."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit, gammaincc, ndtr

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite
from .stukel_fit import fit_stukel


@dataclass(frozen=True)
class SurvanLogistic:
    coefficients: FloatArray
    covariance: FloatArray
    standard_errors: FloatArray
    one_sided_pvalues: FloatArray
    two_sided_pvalues: FloatArray
    fitted_probabilities: FloatArray
    negative_log_likelihood: float
    null_negative_log_likelihood: float
    likelihood_ratio: float
    degrees_of_freedom: int
    pvalue: float
    intercept: bool
    iterations: int
    column_scale: FloatArray
    column_center: FloatArray
    scaled_coefficients: FloatArray

    def predict(self, x: ArrayLike) -> FloatArray:
        """Predict probabilities using the internally scaled design."""
        if np.iscomplexobj(x):
            raise ValueError("x must be real")
        a = finite(x, "x")
        if a.ndim == 1:
            a = a[:, None]
        if a.ndim != 2 or a.shape[1] != self.column_scale.size:
            raise ValueError("prediction x must have the fitted covariate count")
        with np.errstate(over="ignore", invalid="ignore"):
            z = a / self.column_scale - self.column_center
            eta = z @ self.scaled_coefficients[: a.shape[1]]
            if self.intercept:
                eta += self.scaled_coefficients[-1]
        if not np.isfinite(eta).all():
            raise ArithmeticError("prediction linear predictor exceeds numerical range")
        return _freeze(expit(eta))


def survan_logistic(
    x: ArrayLike,
    event: ArrayLike,
    *,
    intercept: bool = True,
) -> SurvanLogistic:
    """Fit binary logistic regression; coefficients are slopes then intercept.

    Event is explicitly zero/one. Select native event lists or thresholds before
    calling. An intercept is included by default. Information covariance uses
    fixed binomial dispersion, and separation/deficient designs are rejected.
    The no-intercept likelihood-ratio null is beta=0 (probability 1/2), ensuring
    nested models rather than the native nonnested constant-probability null.
    """
    if np.iscomplexobj(x) or np.iscomplexobj(event):
        raise ValueError("x and event must be real")
    a, y = finite(x, "x"), count(event, "event")
    if a.ndim == 1:
        a = a[:, None]
    if (
        a.ndim != 2
        or not 1 <= a.shape[1] <= 100
        or a.shape[0] > 100_000
        or a.size > 2_000_000
        or y.shape != (a.shape[0],)
        or np.any(y > 1)
    ):
        raise ValueError(
            "require a matching binary outcome, 1..100 covariates, <=100000 rows and <=2e6 entries"
        )
    if not isinstance(intercept, bool):
        raise ValueError("intercept must be boolean")
    if a.shape[0] <= a.shape[1] + int(intercept):
        raise ValueError("positive residual degrees of freedom are required")
    scale = np.max(np.abs(a), axis=0)
    if np.any(scale == 0):
        raise ValueError("zero covariate columns are not identifiable")
    normalized = a / scale
    center = normalized.mean(axis=0) if intercept else np.zeros(a.shape[1])
    design = normalized - center
    fit = fit_stukel(
        design,
        y,
        np.ones(y.size),
        family=0,
        intercept=intercept,
        scale="fixed",
        gradient_tolerance=1e-10,
    )
    if fit.covariance is None:
        raise ArithmeticError(f"Logistic covariance unavailable: {fit.inference_message}")
    p = a.shape[1]
    # STUKEL puts the intercept first; SURVAN's table puts it last.
    permutation = np.r_[np.arange(1, p + 1), 0] if intercept else np.arange(p)
    beta = fit.coefficients[permutation]
    covariance = fit.covariance[np.ix_(permutation, permutation)]
    transform = np.eye(beta.size)
    if intercept:
        transform[-1, :p] = -center
    with np.errstate(over="ignore", invalid="ignore"):
        transform[np.arange(p), np.arange(p)] = 1 / scale
        original_beta = transform @ beta
        original_covariance = transform @ covariance @ transform.T
    if not np.isfinite(original_beta).all() or not np.isfinite(original_covariance).all():
        raise ArithmeticError("coefficient units exceed representable numerical range")
    variances = np.diag(original_covariance)
    if np.any(variances <= 0):
        raise ArithmeticError("coefficient standard errors are not representable")
    errors = np.sqrt(variances)
    one_sided = ndtr(-np.abs(original_beta / errors))
    null_nll = fit.null_deviance / 2 if intercept else y.size * np.log(2)
    nll = fit.objective.negative_log_likelihood
    if nll > null_nll + 1e-8 * max(1, null_nll):
        raise ArithmeticError("fitted likelihood is worse than the nested null")
    lr = max(0.0, 2 * (null_nll - nll))
    df = p
    return SurvanLogistic(
        _freeze(original_beta),
        _freeze(original_covariance),
        _freeze(errors),
        _freeze(one_sided),
        _freeze(2 * one_sided),
        _freeze(fit.objective.probabilities),
        float(nll),
        float(null_nll),
        float(lr),
        df,
        float(gammaincc(df / 2, lr / 2)),
        intercept,
        fit.iterations,
        _freeze(scale),
        _freeze(center),
        _freeze(beta),
    )
