"""Posterior intervals after isotonic transformation for mTPI dose estimates.

The paper proposes drawing each dose probability independently from its beta
posterior, applying an isotonic transformation to each joint draw, and then
computing numerical posterior intervals. The original source does not specify
the isotonic weights, Monte Carlo size, or empirical quantile convention; this
module makes those choices explicit.
"""

from dataclasses import dataclass
from math import ceil, log2

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import isotonic_regression

from ._validation import FloatArray, scalar
from .boin import _owned
from .mtpi import MTPIDesign

_MAX_DRAWS = 100_000
_MAX_DOSES = 100
_MAX_WORK = 50_000_000
_MAX_CELLS = 2_000_000


@dataclass(frozen=True)
class MTPIIsotonicPosteriorIntervals:
    """Marginal summaries of isotonic-transformed dose toxicity probabilities.

    ``lower`` and ``upper`` use equal-tailed ``numpy.quantile(method='linear')``
    intervals. When requested, ``transformed_draws`` has axes draw and dose.
    The full dose grid is included: an untried dose uses its Beta(1, 1) prior.
    """

    lower: FloatArray
    median: FloatArray
    upper: FloatArray
    mean: FloatArray
    confidence: float
    draws: int
    weights: FloatArray
    transformed_draws: FloatArray | None


def mtpi_isotonic_posterior_intervals(
    design: MTPIDesign,
    patients: ArrayLike,
    toxicities: ArrayLike,
    *,
    draws: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
    weights: ArrayLike | None = None,
    retain_draws: bool = False,
    max_work: int = _MAX_WORK,
) -> MTPIIsotonicPosteriorIntervals:
    """Sample marginal intervals after transforming independent beta draws.

    Every sample draws one toxicity probability per dose from its independent
    ``Beta(y + 1, n - y + 1)`` posterior, then applies weighted increasing
    isotonic regression across the complete dose grid. Untried doses therefore
    contribute their design prior ``Beta(1, 1)``. Equal weights are the default;
    explicitly supplied positive weights define an alternate Python policy.

    The Monte Carlo size, Generator, confidence level and empirical linear
    quantile rule are explicit. A retained transformed sample is optional, but
    the internally retained sample matrix is always needed for exact empirical
    quantiles. Bounds are checked before the first random draw.
    """
    if not isinstance(design, MTPIDesign):
        raise TypeError("design must be an MTPIDesign")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy Generator")
    if not isinstance(retain_draws, (bool, np.bool_)):
        raise ValueError("retain_draws must be Boolean")

    draw_value = scalar(draws, "draws")
    confidence_value = scalar(confidence, "confidence")
    work_value = scalar(max_work, "max_work")
    if draw_value != int(draw_value) or not 1 <= draw_value <= _MAX_DRAWS:
        raise ValueError(f"draws must be an integer in 1..{_MAX_DRAWS}")
    if not 0 < confidence_value < 1:
        raise ValueError("confidence must be in (0, 1)")
    lower_probability = 0.5 - confidence_value / 2.0
    upper_probability = 0.5 + confidence_value / 2.0
    if (
        lower_probability <= 0
        or upper_probability >= 1
        or lower_probability == 0.5
        or upper_probability == 0.5
    ):
        raise ValueError("confidence cannot be represented by interior quantile levels")
    if work_value != int(work_value) or not 1 <= work_value <= _MAX_WORK:
        raise ValueError(f"max_work must be an integer in 1..{_MAX_WORK}")
    draw_count = int(draw_value)
    work_limit = int(work_value)
    if np.iscomplexobj(patients) or np.iscomplexobj(toxicities):
        raise ValueError("patients and toxicities must be real counts")
    n, y, _, _ = design._state(patients, toxicities, None)
    doses = n.size
    cells = draw_count * doses
    if doses > _MAX_DOSES or cells > _MAX_CELLS:
        raise ValueError(f"posterior draw matrix exceeds {_MAX_CELLS} cells")
    # Beta generation and isotonic fitting are linear in cells; the extra
    # logarithmic factor conservatively budgets per-dose empirical quantiles.
    work_required = cells * (2 + ceil(log2(draw_count + 1)))
    if work_required > work_limit:
        raise ValueError("max_work is below the required beta-draw/isotonic work")

    if weights is None:
        isotonic_weights = np.ones(doses, dtype=float)
    else:
        raw_weights = np.asarray(weights)
        if np.iscomplexobj(raw_weights):
            raise ValueError("weights must be real")
        isotonic_weights = np.asarray(raw_weights, dtype=float)
        if (
            isotonic_weights.shape != (doses,)
            or not np.isfinite(isotonic_weights).all()
            or np.any(isotonic_weights <= 0)
        ):
            raise ValueError("weights must be a positive finite vector matching the dose grid")
    normalized_weights = isotonic_weights / isotonic_weights.max()
    if np.any(normalized_weights <= 0):
        raise ArithmeticError("relative isotonic weights cannot be represented")

    transformed = np.empty((draw_count, doses), dtype=float)
    alpha = y + 1
    beta = n - y + 1
    for index in range(draw_count):
        sample = rng.beta(alpha, beta)
        transformed[index] = isotonic_regression(sample, weights=normalized_weights).x

    lower, median, upper = np.quantile(
        transformed, [lower_probability, 0.5, upper_probability], axis=0, method="linear"
    )
    retained = _owned(transformed) if retain_draws else None
    return MTPIIsotonicPosteriorIntervals(
        _owned(lower),
        _owned(median),
        _owned(upper),
        _owned(transformed.mean(axis=0)),
        confidence_value,
        draw_count,
        _owned(isotonic_weights),
        retained,
    )
