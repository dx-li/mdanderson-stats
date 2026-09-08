"""MULTI's Schweder line fit and bootstrap, using explicit random state.

Python adaptation of the archived SCHWED/SWBOOT algorithm; see the MULTI notice.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import stdtrit

from ._validation import FloatArray
from .multiplicity import _alpha, _pvalues


class SchwederFitError(ValueError):
    """The p-value distribution does not support the requested line fit."""


@dataclass(frozen=True)
class SchwederFit:
    null_estimate: float
    one_minus_p: FloatArray
    upper_counts: NDArray[np.int64]
    fitted_points: int
    mean_squared_error: float
    prediction_variance: float


@dataclass(frozen=True)
class SchwederBootstrap:
    fit: SchwederFit
    estimates: FloatArray
    mean: float
    variance: float
    attempts: int


def schweder_fit(pvalues: ArrayLike, *, alpha: float = 0.05) -> SchwederFit:
    """Fit MULTI's origin-constrained line to its upper p-value count plot.

    Distinct p-values are combined and processed from largest to smallest.
    ``upper_counts`` includes observations equal to each p-value. The rightmost
    point is removed until it lies below the source's t prediction bound.
    At least four distinct p-values are required. A failed fit raises
    SchwederFitError, rather than substituting a zero null-count estimate.

    ``null_estimate`` is not clipped to the family size. ``prediction_variance``
    preserves the S library's BETAVR output, on its normalized-ordinate scale;
    it is not the variance of the null-count estimate. Bootstrap for that scale.
    """
    values = _pvalues(pvalues)
    if values.ndim != 1:
        raise ValueError("Schweder fitting accepts one p-value vector")
    alpha = _alpha(alpha)
    if not 0 < alpha < 1:
        raise ValueError("Schweder alpha must be strictly between zero and one")
    distinct, frequencies = np.unique(values, return_counts=True)
    x = 1 - distinct[::-1]
    counts = np.cumsum(frequencies[::-1], dtype=np.int64)
    m = x.size
    if m < 4:
        raise SchwederFitError("At least four distinct p-values are required")
    y = counts / m
    for nn in range(m, 3, -1):
        xx, yy = x[:nn], y[:nn]
        sum_x2 = float(xx @ xx)
        centered = float(np.sum((xx - xx.mean()) ** 2))
        if sum_x2 <= 0 or centered <= 0:
            raise SchwederFitError("The line fit has zero predictor variation")
        slope = float(xx @ yy) / sum_x2
        mse = float(np.sum((yy - slope * xx) ** 2)) / (nn - 1)
        leverage = 1 + 1 / nn + (xx[-1] - xx.mean()) ** 2 / centered
        # Invert the smaller tail directly when alpha is near zero.
        quantile = -float(stdtrit(nn - 2, alpha))
        if yy[-1] <= slope * xx[-1] + quantile * np.sqrt(mse * leverage):
            variance = mse * (1 + 1 / nn + (1 - xx.mean()) ** 2 / centered)
            return SchwederFit(
                m * slope,
                x,
                counts,
                nn,
                mse,
                float(variance),
            )
    raise SchwederFitError("No line passes the stopping rule before only three points remain")


def schweder_bootstrap(
    pvalues: ArrayLike,
    samples: int = 200,
    *,
    alpha: float = 0.05,
    rng: np.random.Generator | int | None = None,
) -> SchwederBootstrap:
    """Resample and refit, requiring samples valid fits within 2*samples attempts.

    Matches SWBOOT's bounded retry and sample-variance convention. The RNG is
    NumPy's Generator, not the legacy global two-seed generator. Returned plot
    coordinates describe the original data, not the last resampled dataset.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    values = _pvalues(pvalues)
    fitted = schweder_fit(values, alpha=alpha)
    generator = np.random.default_rng(rng)
    estimates: list[float] = []
    for attempt in range(1, 2 * samples + 1):
        resampled = generator.choice(values, size=values.size, replace=True)
        try:
            fit = schweder_fit(resampled, alpha=alpha)
        except SchwederFitError:
            continue
        estimates.append(fit.null_estimate)
        if len(estimates) == samples:
            draws = np.array(estimates)
            return SchwederBootstrap(
                fitted,
                draws,
                float(draws.mean()),
                float(draws.var(ddof=1)) if samples > 1 else 0.0,
                attempt,
            )
    raise SchwederFitError(f"Could not obtain {samples} successful fits in {2 * samples} attempts")
