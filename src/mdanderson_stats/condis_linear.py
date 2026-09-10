"""The Gaussian identity-link (default glm) refinement in CondiS-X."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite
from .boin import _owned
from .condis import CondiSImputation


@dataclass(frozen=True)
class CondiSLinearRefinement:
    fitted_time: FloatArray
    refined_time: FloatArray
    below_censoring: NDArray[np.bool_]
    above_horizon: NDArray[np.bool_]
    rank: int
    residual_degrees_of_freedom: int
    residual_rmse: float
    enforce_censoring: bool


def condis_linear_refine(
    imputation: CondiSImputation,
    covariates: ArrayLike,
    *,
    enforce_censoring: bool = False,
) -> CondiSLinearRefinement:
    """Fit imputed time on intercept, status and all numeric covariate columns.

    Match the native Gaussian glm final fit, then restore observed event times.
    Raw predictions can violate censoring bounds; flags always report those cases.
    Optional lower-bound clipping is an explicit Python extension, not native glm.
    This refines the supplied sample; it is not an out-of-sample survival predictor.
    """
    if not isinstance(imputation, CondiSImputation):
        raise TypeError("imputation must be CondiSImputation")
    if not isinstance(enforce_censoring, bool):
        raise ValueError("enforce_censoring must be boolean")
    x = finite(covariates, "covariates")
    n = imputation.imputed_time.size
    if x.ndim != 2 or x.shape[0] != n or not 1 <= x.shape[1] <= 500 or x.size > 2000000:
        raise ValueError("covariates must have n rows, 1..500 columns and <=2 million cells")
    # Column scaling and centering preserve the model span with an intercept.
    # Rescale centered columns again so small but representable contrasts survive.
    scale = abs(x).max(axis=0)
    normalized = x / np.where(scale > 0, scale, 1)
    normalized -= normalized.mean(axis=0)
    contrast = abs(normalized).max(axis=0)
    normalized /= np.where(contrast > 0, contrast, 1)
    design = np.column_stack((np.ones(n), imputation.status, normalized))
    time_scale = float(imputation.imputed_time.max()) or 1.0
    target = imputation.imputed_time / time_scale
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ coefficients
    with np.errstate(over="ignore", under="ignore"):
        fitted = predicted * time_scale
        rmse = float(np.sqrt(np.mean((target - predicted) ** 2)) * time_scale)
    if not np.isfinite(fitted).all() or not np.isfinite(rmse):
        raise ArithmeticError("linear fit cannot be represented in these time units; rescale time")
    censored = imputation.status == 0
    below = censored & (fitted < imputation.observed_time)
    above = censored & (fitted > imputation.horizon)
    refined = np.where(censored, fitted, imputation.observed_time)
    if enforce_censoring:
        refined[censored] = np.maximum(refined[censored], imputation.observed_time[censored])
    return CondiSLinearRefinement(
        _owned(fitted),
        _owned(refined),
        _owned(below),
        _owned(above),
        int(rank),
        n - int(rank),
        rmse,
        enforce_censoring,
    )
