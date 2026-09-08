"""Local fixed-design precision for SINGLE dose-response models."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit

from ._validation import FloatArray, finite


@dataclass(frozen=True)
class SingleDesignPrecision:
    information: FloatArray
    probability: FloatArray
    quantile_dose: FloatArray
    slope_variance: FloatArray
    quantile_variance: FloatArray

    @property
    def slope_sd(self) -> FloatArray:
        return np.sqrt(self.slope_variance)

    @property
    def quantile_sd(self) -> FloatArray:
        return np.sqrt(self.quantile_variance)


def single_design_precision(
    doses: ArrayLike,
    subjects: ArrayLike,
    parameters: ArrayLike,
    *,
    model: str = "logistic",
    form: str = "linear",
    quantile: ArrayLike = 0.05,
) -> SingleDesignPrecision:
    """Evaluate a one-sample design at fixed parameters (SINGLE's point prior).

    Doses/subjects are equal-length vectors; allocations may be fractional.
    Parameters end in two entries: (intercept, slope) for linear form or
    (slope, center) for centered form. Leading parameter axes broadcast with
    quantile. Doses are supplied on the model's dose coordinate; no logarithm
    is applied implicitly. Singular information raises ValueError.
    """
    if model not in ("logistic", "loglog") or form not in ("linear", "centered"):
        raise ValueError("model must be logistic/loglog and form linear/centered")
    x, n, b, q = (
        finite(v, name)
        for v, name in [
            (doses, "doses"),
            (subjects, "subjects"),
            (parameters, "parameters"),
            (quantile, "quantile"),
        ]
    )
    if x.ndim != 1 or n.shape != x.shape or x.size < 2 or np.any(n < 0) or not np.any(n > 0):
        raise ValueError(
            "Require at least two doses and matching nonnegative allocations with positive total"
        )
    if b.ndim < 1 or b.shape[-1] != 2 or np.any((q <= 0) | (q >= 1)):
        raise ValueError("parameters must end in two entries and quantile must lie in (0,1)")
    first, second, q = np.broadcast_arrays(b[..., 0], b[..., 1], q)
    slope = second if form == "linear" else first
    if np.any(slope == 0):
        raise ValueError("Slope must be nonzero for quantile estimation")
    if form == "linear":
        u = first[..., None] + second[..., None] * x
        gradient = np.stack(np.broadcast_arrays(np.ones_like(u), x), axis=-1)
    else:
        u = first[..., None] * (x - second[..., None])
        gradient = np.stack(np.broadcast_arrays(x - second[..., None], -first[..., None]), axis=-1)
    if not np.all(np.isfinite(u)):
        raise ValueError("Linear predictor overflowed")
    p, weight = _response_information(u, model)
    link = np.log(q) - np.log1p(-q) if model == "logistic" else -np.log(-np.log(q))
    effective = n * weight
    minimum = np.min(np.where(effective > 0, x, np.inf), axis=-1)
    maximum = np.max(np.where(effective > 0, x, -np.inf), axis=-1)
    if np.any(maximum <= minimum):
        raise ValueError("Design requires at least two distinct informative doses")
    information = np.swapaxes(gradient, -1, -2) @ (effective[..., None] * gradient)
    if form == "linear":
        dose = (link - first) / second
        target = np.stack([-1 / second, -dose / second], axis=-1)
        slope_target = np.broadcast_to([0.0, 1.0], target.shape)
    else:
        dose = second + link / first
        target = np.stack([-link / first**2, np.ones_like(first)], axis=-1)
        slope_target = np.broadcast_to([1.0, 0.0], target.shape)
    try:
        factor = np.linalg.cholesky(information)
        quantile_solution = np.linalg.solve(factor, target[..., None])[..., 0]
        slope_solution = np.linalg.solve(factor, slope_target[..., None])[..., 0]
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Design information is not positive definite; use informative distinct doses"
        ) from error
    variance = np.sum(quantile_solution**2, axis=-1)
    slope_variance = np.sum(slope_solution**2, axis=-1)
    if not np.all(np.isfinite(variance)) or not np.all(np.isfinite(slope_variance)):
        raise ValueError("Design precision overflowed")
    return SingleDesignPrecision(information, p, dose, slope_variance, variance)


def _response_information(u: FloatArray, model: str) -> tuple[FloatArray, FloatArray]:
    if model == "logistic":
        p = expit(u)
        weight = expit(u) * expit(-u)
    else:
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            t = np.exp(-u)
            p = np.exp(-t)
            ratio = np.ones_like(t)
            np.divide(np.expm1(t), t, out=ratio, where=t > 0)
            small = t / ratio
            large = np.exp(2 * np.log(t) - t - np.log(-np.expm1(-t)))
            weight = np.where(t < 1, small, np.where(np.isinf(t), 0, large))
    return p, weight
