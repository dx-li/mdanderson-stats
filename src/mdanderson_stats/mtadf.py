"""Double-sided isotonic optimal biological dose methods.

The safety rule pools marginal Beta-posterior overdose probabilities with an
increasing isotonic fit. Efficacy uses the paper's double-sided isotonic fit:
weighted PAVA within each candidate segment, but unweighted residual sum of
squares to select the split.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import betainc, betaincc

from ._cdflib import _freeze
from ._validation import FloatArray
from .keyboard_combination import _pava

_MAX_DOSES = 20
_MAX_SUBJECTS = 10_000


def _scalar(value: object, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw)
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


@dataclass(frozen=True)
class MTADFPrior:
    """Beta prior for toxicity, parameterized by positive ``alpha`` and ``beta``."""

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        a = _scalar(self.alpha, "alpha")
        b = _scalar(self.beta, "beta")
        if a <= 0 or b <= 0:
            raise ValueError("prior alpha and beta must be positive")
        object.__setattr__(self, "alpha", a)
        object.__setattr__(self, "beta", b)


def mtadf_toxicity_prior(
    toxicity_limit: float = 0.3,
    safety_cutoff: float = 0.8,
    margin: float = 0.05,
    concentration: float = 0.5,
) -> MTADFPrior:
    """Elicit a Beta prior with ``a+b=concentration`` and specified CDF at limit."""
    phi = _scalar(toxicity_limit, "toxicity_limit")
    cutoff = _scalar(safety_cutoff, "safety_cutoff")
    delta = _scalar(margin, "margin")
    total = _scalar(concentration, "concentration")
    cdf = 1.0 - cutoff + delta
    if not 0 < phi < 1 or not 0 < cutoff < 1 or not 0 < delta < cutoff:
        raise ValueError("require limit and cutoff in (0,1), and 0<margin<cutoff")
    if not 0 < cdf < 1 or not 0 < total <= 100:
        raise ValueError("elicited CDF must be in (0,1) and concentration in (0,100]")

    def residual(logit_a: float) -> float:
        fraction = (
            1 / (1 + np.exp(-logit_a)) if logit_a >= 0 else np.exp(logit_a) / (1 + np.exp(logit_a))
        )
        return float(betainc(total * fraction, total * (1 - fraction), phi) - cdf)

    lo, hi = -36.0, 36.0
    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo == 0:
        fraction = 1 / (1 + np.exp(-lo))
    elif f_hi == 0:
        fraction = 1 / (1 + np.exp(-hi))
    elif f_lo * f_hi > 0:
        raise ValueError("requested prior calibration is not representable")
    else:
        root = brentq(residual, lo, hi, xtol=1e-13, rtol=4 * np.finfo(float).eps)
        fraction = 1 / (1 + np.exp(-root)) if root >= 0 else np.exp(root) / (1 + np.exp(root))
    a, b = total * fraction, total * (1 - fraction)
    if not (a > 0 and b > 0 and isfinite(a) and isfinite(b)):
        raise ArithmeticError("calibrated Beta prior is not representable")
    return MTADFPrior(a, b)


@dataclass(frozen=True)
class MTADFIsotonicFit:
    """Candidate unimodal fits and selected split (zero-based end of rising side)."""

    fitted: FloatArray
    candidate_fits: FloatArray
    candidate_scores: FloatArray
    split: int
    peak: int


def double_sided_isotonic(values: ArrayLike, weights: ArrayLike | None = None) -> MTADFIsotonicFit:
    """Fit a unimodal sequence by split-wise increasing/decreasing weighted PAVA.

    Candidate splits are visited from left to right; equal residual scores keep
    the first split. The reported peak is the lowest index attaining the actual
    maximum fitted value and may lie to the right of the selected split.
    """
    xraw = np.asarray(values)
    if xraw.ndim != 1 or not 1 <= xraw.size <= _MAX_DOSES:
        raise ValueError(f"values must be a vector of 1..{_MAX_DOSES} entries")
    if xraw.dtype.kind not in "iuf":
        raise ValueError("values must be real numeric values")
    x = np.asarray(xraw, dtype=float)
    if not np.all(np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("values must be finite probabilities in [0,1]")
    if weights is None:
        w = np.ones_like(x)
    else:
        wraw = np.asarray(weights)
        if wraw.ndim != 1 or wraw.shape != x.shape or wraw.dtype.kind not in "iuf":
            raise ValueError("weights must be a matching real numeric vector")
        w = np.asarray(wraw, dtype=float)
        if not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise ValueError("weights must be finite and positive")
        w = w / np.max(w)
        if np.any(w == 0):
            raise ValueError("weight ratios are too extreme to represent safely")
    candidates = np.empty((x.size, x.size), dtype=float)
    scores = np.empty(x.size, dtype=float)
    for split in range(x.size):
        candidate = np.empty_like(x)
        candidate[: split + 1] = _pava(x[: split + 1], w[: split + 1])
        if split + 1 < x.size:
            candidate[split + 1 :] = _pava(x[split + 1 :][::-1], w[split + 1 :][::-1])[::-1]
        candidates[split] = candidate
        scores[split] = float(np.dot(candidate - x, candidate - x))
    split = int(np.argmin(scores))
    fitted = candidates[split].copy()
    peak = int(np.flatnonzero(fitted == np.max(fitted))[0])
    return MTADFIsotonicFit(_freeze(fitted), _freeze(candidates), _freeze(scores), split, peak)


@dataclass(frozen=True)
class MTADFDecision:
    """MTADF dose recommendation and toxicity/efficacy summaries."""

    action: str
    dose: int | None
    reason: str
    raw_overdose_probability: FloatArray
    adjusted_overdose_probability: FloatArray
    admissible: NDArray[np.bool_]
    fitted_efficacy: FloatArray
    split: int | None
    peak: int | None


def _count_vector(value: ArrayLike, name: str) -> NDArray[np.int64]:
    raw = np.asarray(value)
    if raw.ndim != 1 or not 1 <= raw.size <= _MAX_DOSES or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real vector with 1..{_MAX_DOSES} entries")
    numeric = np.asarray(raw, dtype=float)
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric < 0)
        or np.any(numeric != np.floor(numeric))
        or np.any(numeric >= 2**53)
    ):
        raise ValueError(f"{name} must contain nonnegative integer counts")
    return numeric.astype(np.int64)


def mtadf_decision(
    subjects: ArrayLike,
    toxicities: ArrayLike,
    responses: ArrayLike,
    *,
    current_dose: int | None = None,
    starting_dose: int = 0,
    final: bool = False,
    toxicity_limit: float = 0.3,
    safety_cutoff: float = 0.8,
    prior: MTADFPrior | None = None,
) -> MTADFDecision:
    """Recommend a dose from complete binomial toxicity and response counts.

    Dose indices are zero-based. Safety uses strict ``pooled_probability <
    safety_cutoff``. Efficacy is fit over observed doses only, including
    noncontiguous histories. Interim movement is one level toward the fitted
    peak; an unsafe current dose is capped at the highest admissible level.
    """
    n = _count_vector(subjects, "subjects")
    y = _count_vector(toxicities, "toxicities")
    r = _count_vector(responses, "responses")
    if y.shape != n.shape or r.shape != n.shape or np.any(y > n) or np.any(r > n):
        raise ValueError("count vectors must match and events cannot exceed subjects")
    if int(n.sum(dtype=np.int64)) > _MAX_SUBJECTS:
        raise ValueError(f"total subjects must not exceed {_MAX_SUBJECTS}")
    if not isinstance(final, (bool, np.bool_)):
        raise ValueError("final must be boolean")
    start_raw = np.asarray(starting_dose)
    if (
        start_raw.ndim != 0
        or start_raw.dtype.kind not in "iu"
        or isinstance(starting_dose, (bool, np.bool_))
    ):
        raise ValueError("starting_dose must be an integer index")
    start = int(start_raw)
    if not 0 <= start < n.size:
        raise ValueError("starting_dose must be within the dose range")
    if current_dose is not None:
        cur_raw = np.asarray(current_dose)
        if (
            cur_raw.ndim != 0
            or cur_raw.dtype.kind not in "iu"
            or isinstance(current_dose, (bool, np.bool_))
        ):
            raise ValueError("current_dose must be an integer index")
        current = int(cur_raw)
        if not 0 <= current < n.size:
            raise ValueError("current_dose must be within the dose range")
    else:
        current = None
    phi = _scalar(toxicity_limit, "toxicity_limit")
    cutoff = _scalar(safety_cutoff, "safety_cutoff")
    if not 0 < phi < 1 or not 0 < cutoff < 1:
        raise ValueError("toxicity_limit and safety_cutoff must lie in (0,1)")
    beta_prior = prior if prior is not None else mtadf_toxicity_prior(phi, cutoff)
    if not isinstance(beta_prior, MTADFPrior):
        raise ValueError("prior must be an MTADFPrior")
    tried = n > 0
    if final and not np.any(tried):
        raise ValueError("final selection requires at least one observed subject")
    if np.any(tried):
        if current is None and not final:
            raise ValueError("current_dose is required after enrollment has begun")
        if current is not None and n[current] == 0 and not final:
            raise ValueError("current_dose must have at least one observed subject")

    raw_safe = betaincc(beta_prior.alpha + y, beta_prior.beta + (n - y), phi)
    if not np.all(np.isfinite(raw_safe)):
        raise ArithmeticError("Beta posterior tail probability is not representable")
    adjusted = _pava(np.asarray(raw_safe, dtype=float), np.ones(n.size, dtype=float))
    admissible = adjusted < cutoff
    fitted = np.full(n.size, np.nan, dtype=float)
    split: int | None = None
    peak: int | None = None
    tried_indices = np.flatnonzero(tried)
    if tried_indices.size:
        efficacy_fit = double_sided_isotonic(
            r[tried].astype(float) / n[tried], n[tried].astype(float)
        )
        fitted[tried] = efficacy_fit.fitted
        split = int(tried_indices[efficacy_fit.split])
        peak = int(tried_indices[efficacy_fit.peak])

    def result(action: str, dose: int | None, reason: str) -> MTADFDecision:
        return MTADFDecision(
            action,
            dose,
            reason,
            _freeze(raw_safe),
            _freeze(adjusted),
            np.frombuffer(admissible.tobytes(), dtype=np.bool_),
            _freeze(fitted),
            split,
            peak,
        )

    if not np.any(admissible):
        return result("stop", None, "no_dose_meets_safety_cutoff")
    if not np.any(tried):
        return (
            result("start", start, "start_at_configured_dose")
            if admissible[start]
            else result("stop", None, "starting_dose_is_not_admissible")
        )
    safe_tried = tried & admissible
    if final:
        if not np.any(safe_tried):
            return result("stop", None, "no_tried_dose_meets_safety_cutoff")
        best = float(np.max(fitted[safe_tried]))
        dose = int(np.flatnonzero(safe_tried & (fitted == best))[0])
        return result("select_obd", dose, "lowest_safe_tried_dose_at_maximum_efficacy")

    assert current is not None
    if not admissible[current]:
        candidates = np.flatnonzero(admissible)
        return result("treat", int(candidates[-1]), "deescalate_to_highest_admissible_dose")
    best = float(np.max(fitted[safe_tried]))
    peak_safe = int(np.flatnonzero(safe_tried & (fitted == best))[0])
    if peak_safe != current:
        destination = current + (1 if peak_safe > current else -1)
        return result("treat", destination, "move_one_level_toward_safe_efficacy_peak")
    if current == int(tried_indices[-1]):
        next_dose = current + 1
        if next_dose < n.size and admissible[next_dose]:
            return result("treat", next_dose, "explore_next_admissible_dose_at_highest_tried_peak")
    return result("treat", current, "remain_at_current_efficacy_peak")
