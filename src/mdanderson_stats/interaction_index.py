"""Lee-Kong log-delta confidence intervals for Loewe interaction indices."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp
from scipy.stats import t

from ._validation import FloatArray, finite, scalar
from .boin import _owned
from .median_effect import MedianEffectFit


@dataclass(frozen=True)
class InteractionIndex:
    log_index: FloatArray
    log_standard_error: FloatArray
    log_interval: FloatArray
    degrees_of_freedom: int
    confidence: float

    @property
    def index(self) -> FloatArray:
        return _exponentiate(self.log_index)

    @property
    def interval(self) -> FloatArray:
        return _exponentiate(self.log_interval)


def _exponentiate(value: FloatArray) -> FloatArray:
    with np.errstate(over="ignore", under="ignore"):
        result = np.exp(value)
    if np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise ArithmeticError("index or confidence limit is not representable; use log outputs")
    return _owned(result)


def _models(models: Sequence[MedianEffectFit]) -> tuple[MedianEffectFit, ...]:
    fits = tuple(models)
    if len(fits) < 2 or any(not isinstance(fit, MedianEffectFit) for fit in fits):
        raise ValueError("require at least two independently fitted median-effect curves")
    if len({np.sign(fit.slope) for fit in fits}) != 1:
        raise ValueError("single-drug curves must have the same effect direction")
    return fits


def _covariance_factor(fit: MedianEffectFit) -> FloatArray:
    scale = np.max(abs(fit.covariance))
    if scale == 0:
        return np.zeros((2, 2))
    values, vectors = np.linalg.eigh(fit.covariance / scale)
    return vectors * np.sqrt(np.maximum(values, 0)) * np.sqrt(scale)


def _variance(fit: MedianEffectFit, gradient: FloatArray) -> FloatArray:
    # A PSD square root avoids negative quadratic forms from cancellation.
    return np.sum((gradient @ _covariance_factor(fit)) ** 2, axis=-1)


def _result(
    log_index: FloatArray, variance: FloatArray, df: int, confidence: float
) -> InteractionIndex:
    level = scalar(confidence, "confidence")
    if not 0 < level < 1:
        raise ValueError("confidence must be in (0,1)")
    se = np.sqrt(variance)
    width = float(t.isf((1 - level) / 2, df)) * se
    interval = np.stack((log_index - width, log_index + width), axis=-1)
    if np.any(~np.isfinite(interval)) or np.any(~np.isfinite(se)):
        raise ArithmeticError("interaction index uncertainty cannot be represented")
    return InteractionIndex(_owned(log_index), _owned(se), _owned(interval), df, level)


def interaction_index(
    models: Sequence[MedianEffectFit],
    doses: ArrayLike,
    effect: ArrayLike,
    *,
    effect_variance: ArrayLike,
    confidence: float = 0.95,
) -> InteractionIndex:
    """Observed-combination log-delta interval, Lee-Kong equations (6)-(9).

    Final dose axis enumerates drugs. effect_variance is the variance of the
    supplied mean effect, not the variance of individual replicates. Zero means
    a known effect. Curves and combination effects must be independently estimated.
    """
    fits = _models(models)
    d = finite(doses, "doses")
    if d.ndim < 1 or d.shape[-1] != len(fits) or np.any(d < 0) or np.any(np.all(d == 0, axis=-1)):
        raise ValueError(
            "require nonnegative doses with one final-axis component per drug and a positive total"
        )
    y, v = np.broadcast_arrays(finite(effect, "effect"), finite(effect_variance, "effect_variance"))
    if np.any(v < 0):
        raise ValueError("effect_variance must be nonnegative")
    log_inverse = np.stack([fit.log_dose(y) for fit in fits], axis=-1)
    with np.errstate(divide="ignore"):
        terms = np.log(d) - log_inverse
    log_index = logsumexp(terms, axis=-1)
    share = np.exp(terms - log_index[..., None])
    variance = np.zeros(log_index.shape)
    response_gradient = np.zeros(log_index.shape)
    for i, fit in enumerate(fits):
        g = share[..., i] / fit.slope
        gradient = np.stack((g, g * log_inverse[..., i]), axis=-1)
        variance += _variance(fit, gradient)
        response_gradient -= g
    with np.errstate(over="ignore", invalid="ignore"):
        # Divide standard error, rather than variance, before squaring.
        effect_se = np.sqrt(v) / y / (1 - y)
        variance += (response_gradient * effect_se) ** 2
    return _result(log_index, variance, sum(f.observations - 2 for f in fits), confidence)


def interaction_index_ray(
    models: Sequence[MedianEffectFit],
    combination: MedianEffectFit,
    proportions: ArrayLike,
    effect: ArrayLike,
    *,
    confidence: float = 0.95,
) -> InteractionIndex:
    """Pointwise log-delta intervals along a fixed composition, equations (12)-(15).

    Combination regression uses total dose, in the same units as the sum of its
    component doses. Proportions are normalized; this is not a simultaneous band.
    """
    fits = _models(models)
    if not isinstance(combination, MedianEffectFit) or np.sign(combination.slope) != np.sign(
        fits[0].slope
    ):
        raise ValueError("combination must be a median-effect fit with matching effect direction")
    p = finite(proportions, "proportions")
    if p.shape != (len(fits),) or np.any(p <= 0):
        raise ValueError("require one positive fixed proportion per drug")
    log_p = np.log(p) - logsumexp(np.log(p))
    inverse = np.stack([fit.log_dose(effect) for fit in fits], axis=-1)
    combo_inverse = combination.log_dose(effect)
    terms = log_p + combo_inverse[..., None] - inverse
    log_index = logsumexp(terms, axis=-1)
    share = np.exp(terms - log_index[..., None])
    variance = np.zeros(log_index.shape)
    for i, fit in enumerate(fits):
        g = share[..., i] / fit.slope
        variance += _variance(fit, np.stack((g, g * inverse[..., i]), axis=-1))
    g = np.full(log_index.shape, -1 / combination.slope)
    variance += _variance(combination, np.stack((g, g * combo_inverse), axis=-1))
    return _result(
        log_index, variance, sum(f.observations - 2 for f in (*fits, combination)), confidence
    )
