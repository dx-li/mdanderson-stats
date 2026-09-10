"""Normal-coefficient Monte Carlo comparator from Lee-Kong section 3.1."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp
from scipy.stats import norm, t

from ._validation import FloatArray, scalar
from .boin import _owned
from .interaction_index import _covariance_factor, _models, interaction_index_ray
from .median_effect import MedianEffectFit


@dataclass(frozen=True)
class InteractionMonteCarlo:
    index: FloatArray
    standard_error: FloatArray
    interval: FloatArray
    coefficient_draws: FloatArray
    slope_reversal_fraction: FloatArray
    confidence: float
    critical_distribution: str
    degrees_of_freedom: int


def interaction_index_monte_carlo(
    models: Sequence[MedianEffectFit],
    combination: MedianEffectFit,
    proportions: ArrayLike,
    effect: ArrayLike,
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    critical_distribution: str = "t",
    rng: int | np.random.Generator | None = None,
) -> InteractionMonteCarlo:
    """Untransformed pointwise interval centered on the fitted Loewe index.

    The paper uses RMS deviation of sampled indices from the fitted index, with
    denominator samples, not sample SD about the Monte Carlo mean. Gaussian
    coefficient draws are not truncated; reversed slopes remain in the sample.
    Near-zero slopes can make the method unstable; unrepresentable results raise.
    coefficient_draws has shape (samples, drugs+1, 2), combination last.
    """
    fits = _models(models)
    point = interaction_index_ray(fits, combination, proportions, effect, confidence=confidence)
    repetitions = scalar(samples, "samples")
    effects = np.asarray(effect, dtype=float)
    if repetitions != int(repetitions) or not 2 <= repetitions <= 1000000:
        raise ValueError("samples must be an integer in [2,1000000]")
    n = int(repetitions)
    if n * effects.size > 20000000 or n * (len(fits) + 1) * 2 > 4000000 or effects.size == 0:
        raise ValueError(
            "require nonempty effects, <=20 million sample-effect and <=4 million coefficient cells"
        )
    if critical_distribution not in ("t", "normal"):
        raise ValueError("critical_distribution must be t or normal")
    all_fits = (*fits, combination)
    generator = np.random.default_rng(rng)
    draws = generator.standard_normal((n, len(all_fits), 2))
    for i, fit in enumerate(all_fits):
        draws[:, i] = draws[:, i] @ _covariance_factor(fit).T + [fit.intercept, fit.slope]
    if np.any(~np.isfinite(draws)) or np.any(draws[..., 1] == 0):
        raise ArithmeticError("coefficient sample contains nonfinite values or a zero slope")
    reversal = np.mean(np.sign(draws[..., 1]) != np.sign([f.slope for f in all_fits]), axis=0)
    logit = (np.log(effects) - np.log1p(-effects)).ravel()
    p = np.asarray(proportions, dtype=float)
    log_p = np.log(p) - logsumexp(np.log(p))
    center = point.log_index.ravel()
    log_squares = np.full(center.shape, -np.inf)
    for start in range(0, n, 1024):
        batch = draws[start : start + 1024]
        for column in range(0, logit.size, 32):
            sl = slice(column, column + 32)
            z = logit[sl]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                combo = (z[None, :] - batch[:, -1, 0, None]) / batch[:, -1, 1, None]
                log_index = np.full(combo.shape, -np.inf)
                for i in range(len(fits)):
                    inverse = (z[None, :] - batch[:, i, 0, None]) / batch[:, i, 1, None]
                    log_index = np.logaddexp(log_index, log_p[i] + combo - inverse)
                delta = log_index - center[sl]
                difference = np.maximum(delta, 0) + np.log(-np.expm1(-abs(delta)))
            if np.any(~np.isfinite(log_index)):
                raise ArithmeticError(
                    "sampled interaction indices cannot be represented in log space"
                )
            log_squares[sl] = np.logaddexp(log_squares[sl], logsumexp(2 * difference, axis=0))
    with np.errstate(over="ignore", under="ignore"):
        se = np.exp(center + 0.5 * (log_squares - np.log(n))).reshape(effects.shape)
    estimate = point.index
    critical = (
        t.isf((1 - point.confidence) / 2, point.degrees_of_freedom)
        if critical_distribution == "t"
        else norm.isf((1 - point.confidence) / 2)
    )
    interval = np.stack((estimate - critical * se, estimate + critical * se), axis=-1)
    if np.any(~np.isfinite(se)) or np.any(~np.isfinite(interval)):
        raise ArithmeticError(
            "Monte Carlo uncertainty cannot be represented; normal slope draws may be unstable"
        )
    return InteractionMonteCarlo(
        estimate,
        _owned(se),
        _owned(interval),
        _owned(draws),
        _owned(reversal),
        point.confidence,
        critical_distribution,
        point.degrees_of_freedom,
    )
