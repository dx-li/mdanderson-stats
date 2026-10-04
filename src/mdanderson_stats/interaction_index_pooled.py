"""Pooled transformed-response error for observed-combination interaction CIs."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .interaction_index import InteractionIndex, _interaction_index, _models
from .median_effect import MedianEffectFit


def _pooled_residual_variance(fits: tuple[MedianEffectFit, ...]) -> float:
    """Pool residual mean squares using their residual degrees of freedom."""
    degrees = np.asarray([fit.observations - 2 for fit in fits], dtype=np.float64)
    residual_variances = np.asarray([fit.residual_variance for fit in fits], dtype=np.float64)
    total_df = float(np.sum(degrees))
    scale = float(np.max(residual_variances))
    if scale == 0:
        return 0.0
    # A weighted average avoids overflowing an intermediate sum of squares.
    pooled = scale * float(np.dot(degrees / total_df, residual_variances / scale))
    if not np.isfinite(pooled) or pooled < 0:
        raise ArithmeticError("pooled transformed-response variance is not representable")
    return pooled


def interaction_index_pooled_error(
    models: Sequence[MedianEffectFit],
    doses: ArrayLike,
    effect: ArrayLike,
    *,
    confidence: float = 0.95,
) -> InteractionIndex:
    """Observed-combination CI using the Section 2 pooled-error fallback.

    This is for a supplied mean combination effect with no replicate-based
    variance estimate. It pools the independent single-agent regression residual
    mean squares on the logit-effect scale, with weights ``n_i - 2``, then uses
    that variance directly in the logit-response delta term. Coefficient
    covariance remains the covariance reported by each fitted single-agent
    regression. The resulting interval uses the existing Section 2
    ``sum(n_i - 2)`` Student-t degrees of freedom.

    If replicate effects are available, use :func:`interaction_index` with the
    variance of their mean. This pooled method is not the fixed-ray procedure;
    fixed-ray intervals retain the separate covariances of each single-agent
    and combination regression.
    """
    fits = _models(models)
    pooled = _pooled_residual_variance(fits)
    return _interaction_index(
        fits,
        doses,
        effect,
        effect_variance=None,
        logit_effect_variance=pooled,
        confidence=confidence,
    )
