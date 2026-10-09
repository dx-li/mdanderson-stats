"""Native WFMM 3.1 inverse-gamma mapping for supplied variance estimates."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray
from .wfmm_model import (
    _MAX_COEFFICIENTS,
    _MAX_RANDOM_EFFECTS,
    _MAX_ROWS,
    WFMMPrior,
    _finite_real,
)


def _variances(value: ArrayLike, name: str) -> FloatArray:
    raw = np.asarray(value)
    if (
        raw.ndim != 2
        or not 1 <= raw.shape[0] <= _MAX_ROWS
        or not 1 <= raw.shape[1] <= _MAX_COEFFICIENTS
    ):
        raise ValueError(f"{name} must be a component-by-coefficient matrix, at most 500 by 512")
    result = _finite_real(raw, name)
    if np.any(result <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return result


def _counts(value: ArrayLike, size: int, name: str, maximum: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.shape != (size,) or raw.dtype.kind == "b":
        raise ValueError(f"{name} must be a vector of {size} positive integer counts")
    result = _finite_real(raw, name)
    if (
        np.any(result < 1)
        or np.any(result > maximum)
        or np.any(result != np.floor(result))
        or float(result.sum()) > maximum
    ):
        raise ValueError(f"{name} must contain positive integers summing to at most {maximum}")
    return result


def _parameters(
    variance: FloatArray, counts: FloatArray, delta: float
) -> tuple[FloatArray, FloatArray]:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        shape = np.broadcast_to(delta * counts[:, None], variance.shape).copy()
        scale = shape * variance
    if (
        not np.isfinite(shape).all()
        or not np.isfinite(scale).all()
        or np.any(shape <= 0)
        or np.any(scale <= 0)
    ):
        raise ValueError("inverse-gamma shape/scale parameters are not representable")
    return shape, scale


def wfmm_variance_prior(
    *,
    inclusion_probability: ArrayLike,
    slab_variance: ArrayLike,
    residual_variance: ArrayLike,
    residual_stratum_sizes: ArrayLike,
    random_variance: ArrayLike | None = None,
    random_effect_counts: ArrayLike | None = None,
    delta_omega: float = 1e-4,
) -> WFMMPrior:
    """Combine fixed-effect priors with the verified native variance mapping.

    Supply positive ``(component, coefficient)`` variance estimates. For each
    random-effect level use its number of columns of Z; for each residual
    stratum use its number of curves. Native parameters are ``a=delta*count``
    and ``b=a*omega_MLE`` for density ``v**(-a-1)*exp(-b/v)``. The count does
    not depend on the number of coefficients in a wavelet partition.

    This mapping does not run the native MOM/profile optimizer. Supplying
    Python REML estimates is an explicit alternative centering choice. The
    prior is centered through ``E[1/v]=1/variance``; its mean for v is infinite
    when a <= 1. The native ``delta_omega`` default is 1e-4.
    """
    if isinstance(delta_omega, (bool, np.bool_)) or not np.isscalar(delta_omega):
        raise ValueError("delta_omega must be a finite positive real scalar")
    delta_array = _finite_real(delta_omega, "delta_omega")
    delta = float(delta_array)
    if delta <= 0:
        raise ValueError("delta_omega must be strictly positive")
    residual = _variances(residual_variance, "residual_variance")
    residual_counts = _counts(
        residual_stratum_sizes, residual.shape[0], "residual_stratum_sizes", _MAX_ROWS
    )
    residual_shape, residual_scale = _parameters(residual, residual_counts, delta)
    if (random_variance is None) != (random_effect_counts is None):
        raise ValueError("random_variance and random_effect_counts must be supplied together")
    random_shape = random_scale = None
    if random_variance is not None and random_effect_counts is not None:
        random = _variances(random_variance, "random_variance")
        if random.shape[1] != residual.shape[1]:
            raise ValueError("random and residual variances must have the same coefficient count")
        random_counts = _counts(
            random_effect_counts, random.shape[0], "random_effect_counts", _MAX_RANDOM_EFFECTS
        )
        random_shape, random_scale = _parameters(random, random_counts, delta)
    return WFMMPrior(
        inclusion_probability=inclusion_probability,
        slab_variance=slab_variance,
        random_shape=random_shape,
        random_scale=random_scale,
        residual_shape=residual_shape,
        residual_scale=residual_scale,
    )
