"""Posterior intervals after isotonic transformation for original TPI posteriors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from ._validation import count
from .mtpi_isotonic_posterior import (
    MTPIIsotonicPosteriorIntervals,
    _isotonic_beta_posterior_intervals,
)
from .tpi import TPIDesign


def _bounded_count_vector(value: ArrayLike, name: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if (
            np.iscomplexobj(value)
            or value.ndim != 1
            or not 1 <= value.size <= 100
            or value.dtype.kind not in "iuf"
        ):
            raise ValueError(f"{name} must be a one-dimensional vector for 1..100 doses")
    elif isinstance(value, Sequence):
        if not 1 <= len(value) <= 100 or any(
            isinstance(item, (list, tuple, np.ndarray))
            or isinstance(item, (complex, np.complexfloating, bool, np.bool_))
            for item in value
        ):
            raise ValueError(f"{name} must be a one-dimensional vector for 1..100 doses")
    else:
        raise ValueError(f"{name} must be a bounded one-dimensional vector")
    return count(value, name)


def tpi_isotonic_posterior_intervals(
    design: TPIDesign,
    patients: ArrayLike,
    toxicities: ArrayLike,
    *,
    draws: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
    weights: ArrayLike | None = None,
    retain_draws: bool = False,
    max_work: int = 50_000_000,
) -> MTPIIsotonicPosteriorIntervals:
    """Sample marginal isotonic intervals from the supplied TPI beta posteriors.

    The mTPI paper proposes independent dose-specific beta posterior draws,
    isotonic transformation of every joint draw, and numerical posterior
    intervals. This applies that inferential recipe to ``TPIDesign``'s original
    TPI beta posterior, including its explicit common or dose-specific priors;
    it does not change TPI's interval-mass decisions or claim native TPI support.

    The complete supplied dose grid is transformed, so an untried dose draws
    from its configured TPI prior. Weights, Monte Carlo count, and the linear
    empirical quantile convention are explicit Python choices.
    """
    if not isinstance(design, TPIDesign):
        raise TypeError("design must be a TPIDesign")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy Generator")
    if not isinstance(retain_draws, (bool, np.bool_)):
        raise ValueError("retain_draws must be Boolean")
    n = _bounded_count_vector(patients, "patients")
    y = _bounded_count_vector(toxicities, "toxicities")
    if n.shape != y.shape or np.any(y > n) or n.sum() > 200 or np.any(n > 200):
        raise ValueError("require matching dose vectors, toxicities <= patients, total <=200")
    if design.has_dose_specific_prior and n.size != design.prior_dose_count:
        raise ValueError("per-dose priors require one count per configured dose")

    prior_alpha, prior_beta = design._prior_for_dose(None)
    alpha = y + prior_alpha
    beta = n - y + prior_beta
    return _isotonic_beta_posterior_intervals(
        alpha,
        beta,
        draws=draws,
        rng=rng,
        confidence=confidence,
        weights=weights,
        retain_draws=retain_draws,
        max_work=max_work,
    )
