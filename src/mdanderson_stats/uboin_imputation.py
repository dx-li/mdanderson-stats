"""Stage-II U-BOIN evaluation over supplied posterior predictive imputations.

This module evaluates efficacy-completion probabilities supplied by the
caller. It does not fit the paper's pending-efficacy model.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import count, finite
from .uboin_conduct import UBOINDesign, _int_scalar, _readonly

_MAX_PENDING = 1_000
_MAX_IMPUTATIONS = 1_000
_MAX_PREDICTIVE_CELLS = 100_000
_MAX_POSTERIOR_CELLS = 2_000_000


def _bounded_vector(value: ArrayLike, name: str, limit: int) -> NDArray[np.float64]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.size > limit:
            raise ValueError(f"{name} must be a one-dimensional vector of at most {limit} values")
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must contain real values")
    elif isinstance(value, (list, tuple)):
        if len(value) > limit or any(isinstance(item, (list, tuple, np.ndarray)) for item in value):
            raise ValueError(f"{name} must be a flat vector of at most {limit} values")
    else:
        raise ValueError(f"{name} must be a one-dimensional array or flat sequence")
    result = finite(value, name)
    if result.ndim != 1 or result.size > limit:
        raise ValueError(f"{name} must be a one-dimensional vector of at most {limit} values")
    return np.array(result, dtype=np.float64, copy=True)


def _predictive_matrix(value: ArrayLike) -> NDArray[np.float64]:
    if isinstance(value, np.ndarray):
        if value.ndim != 2 or value.shape[0] > _MAX_IMPUTATIONS or value.shape[1] > _MAX_PENDING:
            raise ValueError("efficacy_probabilities must be a bounded H by M matrix")
        if np.iscomplexobj(value):
            raise ValueError("efficacy_probabilities must be real")
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        if len(value) > _MAX_IMPUTATIONS:
            raise ValueError("efficacy_probabilities has too many imputation rows")
        widths: list[int] = []
        for row in value:
            if not isinstance(row, (list, tuple, np.ndarray)):
                raise ValueError("efficacy_probabilities must be a rectangular H by M matrix")
            if isinstance(row, np.ndarray):
                if row.ndim != 1 or np.iscomplexobj(row):
                    raise ValueError("efficacy_probabilities rows must be real vectors")
                width = row.size
            else:
                if any(isinstance(item, (list, tuple, np.ndarray)) for item in row):
                    raise ValueError("efficacy_probabilities must be a rectangular H by M matrix")
                width = len(row)
            if width > _MAX_PENDING:
                raise ValueError("efficacy_probabilities has too many pending-patient columns")
            widths.append(width)
        if len(set(widths)) > 1:
            raise ValueError("efficacy_probabilities must be rectangular")
        shape = (len(value), widths[0] if widths else 0)
    else:
        raise ValueError("efficacy_probabilities must be a bounded two-dimensional array")
    if shape[0] < 5:
        raise ValueError("at least five posterior-predictive imputations are required")
    if shape[0] * shape[1] > _MAX_PREDICTIVE_CELLS:
        raise ValueError("efficacy probability matrix exceeds the 100,000-cell limit")
    result = finite(value, "efficacy_probabilities")
    if result.shape != shape or np.any((result < 0) | (result > 1)):
        raise ValueError("efficacy_probabilities must be finite probabilities in [0,1]")
    return np.array(result, dtype=np.float64, copy=True)


@dataclass(frozen=True)
class UBOINMIDecision:
    """Stage-II conduct and compact summaries over completed efficacy data."""

    stage: int
    action: str
    allocation_probabilities: NDArray[np.float64]
    next_dose: int | None
    selected_dose: int | None
    eliminated: NDArray[np.bool_]
    observed_counts: NDArray[np.float64]
    pending_dose: NDArray[np.int64]
    pending_toxicity: NDArray[np.int64]
    efficacy_probabilities: NDArray[np.float64]
    imputed_response: NDArray[np.bool_]
    mean_utility_by_imputation: NDArray[np.float64]
    low_efficacy_probability_by_imputation: NDArray[np.float64]
    overdose_probability: NDArray[np.float64]
    mean_utility: NDArray[np.float64]
    low_efficacy_probability: NDArray[np.float64]
    admissible: NDArray[np.bool_]


def uboin_stage2_multiple_imputation(
    design: UBOINDesign,
    observed_counts: ArrayLike,
    pending_dose: ArrayLike,
    pending_toxicity: ArrayLike,
    efficacy_probabilities: ArrayLike,
    *,
    current_dose: int,
    eliminated: ArrayLike | None = None,
    rng: np.random.Generator,
) -> UBOINMIDecision:
    """Average complete-data posterior functionals across pending-response MI.

    ``efficacy_probabilities[h, j]`` is the externally supplied predictive
    efficacy probability for pending patient ``j`` in posterior draw ``h``.
    A shared row therefore represents one predictive-model draw across that
    row's patients. Each row is completed independently and evaluated with
    the existing Dirichlet posterior; posterior counts are never pooled.
    ``observed_counts`` excludes pending patients, which are added exactly
    once from the two one-based pending ledgers. Efficacy is binary; toxicity
    may have two or three categories.
    """
    if not isinstance(design, UBOINDesign):
        raise TypeError("design must be a UBOINDesign")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit numpy.random.Generator")
    prior_shape = np.asarray(design.prior).shape
    if prior_shape[-2] != 2 or design.response_level != 1:
        raise ValueError("multiple-imputation efficacy evaluation requires binary efficacy")

    observed, d, e, t = design._validate_counts(observed_counts)
    if e != 2:
        raise ValueError("multiple-imputation efficacy evaluation requires two efficacy categories")
    probabilities = _predictive_matrix(efficacy_probabilities)
    h, m = probabilities.shape
    doses = _bounded_vector(pending_dose, "pending_dose", _MAX_PENDING)
    tox = _bounded_vector(pending_toxicity, "pending_toxicity", _MAX_PENDING)
    if doses.size != m or tox.size != m:
        raise ValueError("pending_dose and pending_toxicity must match probability columns")
    dose_codes = count(doses, "pending_dose")
    tox_codes = count(tox, "pending_toxicity")
    if np.any((dose_codes < 1) | (dose_codes > d)):
        raise ValueError("pending_dose values must be one-based configured dose indices")
    if np.any((tox_codes < 1) | (tox_codes > t)):
        raise ValueError("pending_toxicity values must be one-based toxicity categories")
    if int(observed.sum()) + m > design.max_patients or int(observed.sum()) + m > 1_000:
        raise ValueError("observed plus pending patients exceed the design patient limit")
    if h * d * e * t > _MAX_POSTERIOR_CELLS:
        raise ValueError("imputation-by-posterior workload exceeds the 2,000,000-cell limit")

    current = _int_scalar(current_dose, "current_dose", 1, d)
    mask = design._mask(eliminated, d)
    conduct_counts = np.array(observed, copy=True)
    for dose, toxicity in zip(dose_codes, tox_codes, strict=True):
        conduct_counts[int(dose) - 1, 0, int(toxicity) - 1] += 1
    conduct_n = conduct_counts.sum(axis=(1, 2))
    if conduct_n.sum() == 0:
        raise ValueError("Stage II requires at least one observed or pending patient")
    if conduct_n[current - 1] == 0:
        raise ValueError(
            "current_dose must have treated patients in the observed-plus-pending ledger"
        )
    overdose = design._posterior(conduct_counts).overdose_probability

    # A single H x M tape is the only patient-by-imputation state retained.
    response_draws = rng.random((h, m)) < probabilities
    utility_rows = np.empty((h, d), dtype=np.float64)
    low_rows = np.empty((h, d), dtype=np.float64)
    for index in range(h):
        completed = np.array(observed, copy=True)
        for dose, toxicity, response in zip(
            dose_codes, tox_codes, response_draws[index], strict=True
        ):
            completed[int(dose) - 1, int(response), int(toxicity) - 1] += 1
        if int(completed.sum()) > 1_000 or np.any(completed > 1_000_000):
            raise ValueError("completed counts exceed posterior limits")
        posterior = design._posterior(completed)
        utility_rows[index] = posterior.mean_utility
        low_rows[index] = posterior.low_efficacy_probability

    # Centered averaging preserves exact equality for identical completions.
    mean_utility = utility_rows[0] + (utility_rows - utility_rows[0]).mean(axis=0)
    low_probability = low_rows[0] + (low_rows - low_rows[0]).mean(axis=0)
    admissible = (overdose <= design.safety_cutoff) & (low_probability <= design.efficacy_cutoff)
    action, allocation, next_dose, selected = design._stage2_action(
        conduct_counts, mask, mean_utility, admissible
    )
    return UBOINMIDecision(
        2,
        action,
        _readonly(allocation),
        next_dose,
        selected,
        _readonly(mask, dtype=np.bool_),
        _readonly(observed),
        _readonly(dose_codes, dtype=np.int64),
        _readonly(tox_codes, dtype=np.int64),
        _readonly(probabilities),
        _readonly(response_draws, dtype=np.bool_),
        _readonly(utility_rows),
        _readonly(low_rows),
        _readonly(overdose),
        _readonly(mean_utility),
        _readonly(low_probability),
        _readonly(admissible, dtype=np.bool_),
    )
