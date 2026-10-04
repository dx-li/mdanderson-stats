"""Explicit policy for the authors' reference isotonic MTADF implementation.

This module preserves the author R program's fixed toxicity prior, inclusive
admissibility threshold, rightmost efficacy ties and all-dose final fit. The
paper-policy implementation remains available separately in :mod:`mtadf`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import betainc, betaincc

from ._cdflib import _freeze
from ._validation import FloatArray
from .keyboard_combination import _pava
from .mtadf import MTADFPrior, _count_vector, _scalar

_MAX_DOSES = 20
_MAX_SUBJECTS = 10_000
_AUTHOR_PRIOR_CONCENTRATION = 0.5
_AUTHOR_PRIOR_CDF = 0.22
_AUTHOR_PRIOR_TOXICITY_LIMIT = 0.3


@lru_cache(maxsize=1)
def _mtadf_author_prior() -> MTADFPrior:
    """Return the fixed Beta prior calibrated in ``targetAgentDF.r``."""

    def objective(alpha: float) -> float:
        return float(
            betainc(
                alpha,
                _AUTHOR_PRIOR_CONCENTRATION - alpha,
                _AUTHOR_PRIOR_TOXICITY_LIMIT,
            )
            - _AUTHOR_PRIOR_CDF
        )

    alpha = float(brentq(objective, 0.01, 0.49, xtol=1e-14))
    return MTADFPrior(alpha, _AUTHOR_PRIOR_CONCENTRATION - alpha)


@dataclass(frozen=True, slots=True)
class MTADFAuthorDecision:
    """One author-reference MTADF decision with its current safety state.

    Dose indices are zero-based. ``admissible`` is the prefix used to cap this
    decision; ``admissible_dose_count`` is freshly calculated from these data.
    The two can differ only for the author simulation's lagged-cap convention.
    """

    action: str
    dose: int | None
    reason: str
    raw_overdose_probability: FloatArray
    adjusted_overdose_probability: FloatArray
    admissible: NDArray[np.bool_]
    admissible_dose_count: int
    admissibility_count_used: int
    fitted_efficacy: FloatArray
    peak: int | None


def _author_count_inputs(
    subjects: ArrayLike, toxicities: ArrayLike
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    n = _count_vector(subjects, "subjects")
    y = _count_vector(toxicities, "toxicities")
    if n.shape != y.shape or np.any(y > n):
        raise ValueError(
            "subject and toxicity counts must match and toxicities cannot exceed subjects"
        )
    if sum(int(value) for value in n) > _MAX_SUBJECTS:
        raise ValueError(f"total subjects must not exceed {_MAX_SUBJECTS}")
    return n, y


def _author_limits(toxicity_limit: float, safety_cutoff: float) -> tuple[float, float]:
    phi = _scalar(toxicity_limit, "toxicity_limit")
    cutoff = _scalar(safety_cutoff, "safety_cutoff")
    if not 0 < phi < 1 or not 0 < cutoff < 1:
        raise ValueError("toxicity_limit and safety_cutoff must lie in (0,1)")
    return phi, cutoff


def _freeze_bool(values: ArrayLike) -> NDArray[np.bool_]:
    contiguous = np.ascontiguousarray(values, dtype=np.bool_)
    return np.frombuffer(contiguous.tobytes(), dtype=np.bool_)


def _author_unimodal_fit(values: FloatArray) -> FloatArray:
    """Enumerate Iso::ufit's midpoint modes with unit-weight PAVA."""
    if values.size == 1:
        return values.copy()
    weights = np.ones(values.size, dtype=np.float64)
    best_score = np.inf
    best_fit: FloatArray | None = None
    for split in range(values.size - 1):
        candidate = np.empty(values.size, dtype=np.float64)
        candidate[: split + 1] = _pava(values[: split + 1], weights[: split + 1])
        candidate[split + 1 :] = _pava(values[split + 1 :][::-1], weights[split + 1 :][::-1])[::-1]
        residual = candidate - values
        score = float(np.dot(residual, residual))
        # Iso Fortran replaces the best mode only for a strict SSE decrease.
        if score < best_score:
            best_score, best_fit = score, candidate
    assert best_fit is not None
    return best_fit


def _author_safety_state(
    subjects: ArrayLike,
    toxicities: ArrayLike,
    *,
    toxicity_limit: float,
    safety_cutoff: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64], FloatArray, FloatArray, int]:
    n, y = _author_count_inputs(subjects, toxicities)
    phi, cutoff = _author_limits(toxicity_limit, safety_cutoff)
    prior = _mtadf_author_prior()
    raw = np.asarray(betaincc(prior.alpha + y, prior.beta + (n - y), phi), dtype=np.float64)
    if not np.all(np.isfinite(raw)):
        raise ArithmeticError("author Beta posterior tail is not representable")
    adjusted = np.asarray(_pava(raw, np.ones(raw.size)), dtype=np.float64)
    admissible_count = max(1, int(np.count_nonzero(adjusted <= cutoff)))
    return n, y, raw, adjusted, admissible_count


def _mtadf_author_admissible_dose_count(
    subjects: ArrayLike,
    toxicities: ArrayLike,
    *,
    toxicity_limit: float = 0.3,
    safety_cutoff: float = 0.8,
) -> int:
    """Get the fresh inclusive safety cap for the author trial simulator."""
    return _author_safety_state(
        subjects,
        toxicities,
        toxicity_limit=toxicity_limit,
        safety_cutoff=safety_cutoff,
    )[-1]


def _author_efficacy_fit(
    subjects: NDArray[np.int64],
    responses: NDArray[np.int64],
    *,
    final: bool,
) -> tuple[FloatArray, int | None]:
    if final:
        # The epsilon is part of the author reference's all-dose final rule.
        rates = responses.astype(np.float64) / (subjects.astype(np.float64) + 0.0001)
        fitted = _author_unimodal_fit(rates)
        peak = int(np.flatnonzero(fitted == np.max(fitted))[-1])
        return np.asarray(fitted), peak

    tried = np.flatnonzero(subjects)
    if tried.size == 0:
        return np.full(subjects.size, np.nan), None
    expected_prefix = np.arange(tried.size)
    if not np.array_equal(tried, expected_prefix):
        raise ValueError("author interim decisions require a contiguous observed dose prefix")
    rates = responses[: tried.size].astype(np.float64) / subjects[: tried.size]
    fitted_prefix = _author_unimodal_fit(rates)
    fitted = np.full(subjects.size, np.nan)
    fitted[: tried.size] = fitted_prefix
    peak = int(np.flatnonzero(fitted_prefix == np.max(fitted_prefix))[-1])
    return fitted, peak


def _mtadf_author_decision_with_count(
    subjects: ArrayLike,
    toxicities: ArrayLike,
    responses: ArrayLike,
    *,
    current_dose: int | None,
    final: bool,
    toxicity_limit: float,
    safety_cutoff: float,
    admissibility_count: int,
) -> MTADFAuthorDecision:
    n, y, raw, adjusted, fresh_count = _author_safety_state(
        subjects,
        toxicities,
        toxicity_limit=toxicity_limit,
        safety_cutoff=safety_cutoff,
    )
    r = _count_vector(responses, "responses")
    if r.shape != n.shape or np.any(r > n):
        raise ValueError("response counts must match subjects and cannot exceed enrollment")
    if not isinstance(final, (bool, np.bool_)):
        raise ValueError("final must be boolean")
    if isinstance(admissibility_count, (bool, np.bool_)) or not isinstance(
        admissibility_count, (int, np.integer)
    ):
        raise ValueError("admissibility_count must be an integer prefix length")
    used_count = int(admissibility_count)
    if not 1 <= used_count <= n.size:
        raise ValueError("admissibility_count must lie from 1 to the number of doses")
    if current_dose is not None and (
        isinstance(current_dose, (bool, np.bool_))
        or not isinstance(current_dose, (int, np.integer))
        or not 0 <= int(current_dose) < n.size
    ):
        raise ValueError("current_dose must be a zero-based dose index")

    fitted, peak = _author_efficacy_fit(n, r, final=bool(final))
    admissible = np.arange(n.size) < used_count
    action: str
    dose: int | None
    reason: str
    if final:
        if int(n.sum()) == 0:
            raise ValueError("final author selection requires at least one observed subject")
        assert peak is not None
        dose = min(peak, used_count - 1)
        action, reason = "select_obd", "rightmost_unimodal_max_capped_at_author_admissible_dose"
    elif int(n.sum()) == 0:
        if current_dose is not None:
            raise ValueError("current_dose must be omitted before enrollment")
        dose, action, reason = 0, "start", "author_program_starts_at_first_dose"
    else:
        if current_dose is None:
            raise ValueError("current_dose is required after enrollment has begun")
        current = int(current_dose)
        if n[current] == 0:
            raise ValueError("current_dose must have observed subjects")
        assert peak is not None
        nd = peak + 1  # source is one-based
        d = current + 1
        if current == n.size - 1:
            destination = d if nd == n.size else d - 1
        else:
            tried_count = int(np.count_nonzero(n))
            if nd == tried_count or nd > d:
                destination = d + 1
            elif nd < d:
                destination = d - 1
            else:
                destination = d
        dose = min(destination, used_count) - 1
        action = "treat"
        reason = "author_rightmost_mode_one_level_policy_capped_by_admissible_prefix"

    return MTADFAuthorDecision(
        action,
        dose,
        reason,
        _freeze(raw),
        _freeze(adjusted),
        _freeze_bool(admissible),
        fresh_count,
        used_count,
        _freeze(fitted),
        peak,
    )


def mtadf_author_decision(
    subjects: ArrayLike,
    toxicities: ArrayLike,
    responses: ArrayLike,
    *,
    current_dose: int | None = None,
    final: bool = False,
    toxicity_limit: float = 0.3,
    safety_cutoff: float = 0.8,
) -> MTADFAuthorDecision:
    """Apply the author's reference isotonic rule with a fresh safety cap.

    This is an explicitly named author-program policy, separate from the
    paper-policy ``mtadf_decision``. Its fixed Beta prior solves
    ``BetaCDF(0.3; alpha, 0.5-alpha)=0.22`` regardless of the supplied
    toxicity limit or safety cutoff. Safety uses ``adjusted <= cutoff`` and
    retains dose 1 when no dose passes. Interim data must be a contiguous
    observed prefix. Final efficacy fits all doses using ``response /
    (subjects + 0.0001)`` and the rightmost fitted maximum.
    """
    fresh_count = _mtadf_author_admissible_dose_count(
        subjects,
        toxicities,
        toxicity_limit=toxicity_limit,
        safety_cutoff=safety_cutoff,
    )
    return _mtadf_author_decision_with_count(
        subjects,
        toxicities,
        responses,
        current_dose=current_dose,
        final=final,
        toxicity_limit=toxicity_limit,
        safety_cutoff=safety_cutoff,
        admissibility_count=fresh_count,
    )
