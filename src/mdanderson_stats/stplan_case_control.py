"""Power calculations for STPLAN unmatched and matched case-control designs."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri
from scipy.stats import binom

from ._validation import FloatArray
from .stplan_continuous import _alpha, _inputs, _power
from .stplan_discrete import _binomial_exact_power


def _exposure_rates(
    frequency: FloatArray, risk_exposed: FloatArray, risk_unexposed: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return exposure probabilities conditional on disease and nondisease."""
    disease = frequency * risk_exposed + (1 - frequency) * risk_unexposed
    nondisease = frequency * (1 - risk_exposed) + (1 - frequency) * (1 - risk_unexposed)
    if np.any(disease <= 0) or np.any(nondisease <= 0):
        raise ValueError("disease and nondisease groups must both have positive probability")
    among_diseased = frequency * risk_exposed / disease
    among_nondiseased = frequency * (1 - risk_exposed) / nondisease
    return among_diseased, among_nondiseased


def stplan_case_control_power(
    exposure_frequency: ArrayLike,
    disease_risk_exposed: ArrayLike,
    disease_risk_unexposed: ArrayLike,
    n_cases: ArrayLike,
    n_controls: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """Normal-approximation power for the unmatched case-control comparison.

    Disease risks condition on the exposure; exposure_frequency is its population
    prevalence. The default one-sided power retains STPLAN's signed direction
    (greater exposure prevalence among cases). sides=2 is a Python extension that
    halves alpha and retains only the dominant effect direction.
    """
    a = _alpha(alpha, sides)
    frequency, risk_e, risk_u, cases, controls = _inputs(
        (exposure_frequency, "exposure_frequency"),
        (disease_risk_exposed, "disease_risk_exposed"),
        (disease_risk_unexposed, "disease_risk_unexposed"),
        (n_cases, "n_cases"),
        (n_controls, "n_controls"),
    )
    if np.any((frequency < 0) | (frequency > 1)):
        raise ValueError("exposure_frequency must lie in [0,1]")
    if np.any((risk_e < 0) | (risk_e > 1) | (risk_u < 0) | (risk_u > 1)):
        raise ValueError("disease risks must lie in [0,1]")
    if np.any(cases < 2) or np.any(controls < 2):
        raise ValueError("case and control sample sizes must be at least 2")
    p_case, p_control = _exposure_rates(frequency, risk_e, risk_u)
    difference = 2 * np.arcsin(np.sqrt(p_case)) - 2 * np.arcsin(np.sqrt(p_control))
    if sides == 2:
        difference = np.abs(difference)
    standard_error = np.sqrt(1 / cases + 1 / controls)
    return _power(ndtr(difference / standard_error + ndtri(a)))


def stplan_matched_case_control_power(
    exposure_frequency: ArrayLike,
    disease_risk_exposed: ArrayLike,
    disease_risk_unexposed: ArrayLike,
    n_pairs: ArrayLike,
    *,
    alpha: float = 0.05,
    sides: int = 1,
) -> FloatArray:
    """Exact conditional-power mixture for matched case-control pairs.

    Pair members are independently sampled from the case and control exposure
    distributions. The number of discordant pairs is binomial, and their
    case-only exposure probability determines an exact binomial test against
    1/2. The one-sided test is fixed in the direction of excess case exposure,
    matching the STPLAN user guide. sides=2 is an extension that chooses the
    alternative's direction and halves alpha. This corrects a reversed null/alt
    call in the native matched routine.
    """
    a = _alpha(alpha, sides)
    frequency, risk_e, risk_u, pairs = _inputs(
        (exposure_frequency, "exposure_frequency"),
        (disease_risk_exposed, "disease_risk_exposed"),
        (disease_risk_unexposed, "disease_risk_unexposed"),
        (n_pairs, "n_pairs"),
    )
    if np.any((frequency < 0) | (frequency > 1)):
        raise ValueError("exposure_frequency must lie in [0,1]")
    if np.any((risk_e < 0) | (risk_e > 1) | (risk_u < 0) | (risk_u > 1)):
        raise ValueError("disease risks must lie in [0,1]")
    if np.any((pairs < 2) | (pairs != np.floor(pairs)) | (pairs >= 2**53)):
        raise ValueError("n_pairs must be integers >= 2 and below 2**53")
    if pairs.size == 0:
        return np.empty(pairs.shape, dtype=float)
    support_work = sum(int(n) + 1 for n in pairs.flat)
    if support_work > 200_000:
        raise ValueError("matched-case-control mixture exceeds 200000 support terms")
    p_case, p_control = _exposure_rates(frequency, risk_e, risk_u)
    case_only = p_case * (1 - p_control)
    discordance = case_only + (1 - p_case) * p_control
    conditional = np.divide(
        case_only, discordance, out=np.full_like(case_only, 0.5), where=discordance > 0
    )
    flat_pairs = pairs.ravel().astype(np.int64)
    lengths = flat_pairs + 1
    case_index: NDArray[np.int64] = np.repeat(np.arange(flat_pairs.size), lengths)
    starts: NDArray[np.int64] = np.repeat(np.cumsum(lengths, dtype=np.int64) - lengths, lengths)
    counts: NDArray[np.int64] = np.arange(support_work, dtype=np.int64) - starts
    flat_discordance = discordance.ravel()
    flat_conditional = conditional.ravel()
    result = np.zeros(flat_pairs.size, dtype=float)
    max_pairs = int(np.max(flat_pairs))
    if sides == 1:
        candidate: NDArray[np.int64] = np.arange(1, max_pairs + 1, dtype=np.int64)
        lo: NDArray[np.int64] = np.zeros(candidate.shape, dtype=np.int64)
        hi: NDArray[np.int64] = candidate + 1
        while np.any(hi - lo > 1):
            middle = lo + (hi - lo) // 2
            acceptable = binom.sf(middle - 1, candidate, 0.5) <= a
            hi = np.where(acceptable, middle, hi)
            lo = np.where(acceptable, lo, middle)
        critical: NDArray[np.int64] = np.ones(max_pairs + 1, dtype=np.int64)
        critical[1:] = hi
    else:
        critical = np.zeros(max_pairs + 1, dtype=np.int64)
    for start in range(0, support_work, 2048):
        stop = min(start + 2048, support_work)
        idx = case_index[start:stop]
        k = counts[start:stop]
        if sides == 1:
            conditional_power = binom.sf(critical[k] - 1, k, flat_conditional[idx])
        else:
            conditional_power = _binomial_exact_power(
                np.full(k.shape, 0.5), flat_conditional[idx], k.astype(float), a
            )
        weights = binom.pmf(k, flat_pairs[idx], flat_discordance[idx])
        np.add.at(result, idx, weights * conditional_power)
    return _power(result.reshape(pairs.shape))
