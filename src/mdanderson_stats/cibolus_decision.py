"""Safety-screened concentration/bolus decisions for CiBolus."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray
from .cibolus_fit import CiBolusFit


def _scalar(value: object, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _readonly_bool(value: ArrayLike) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.bool_)
    return np.frombuffer(array.tobytes(), dtype=np.bool_).reshape(array.shape)


def _pair(value: object, shape: tuple[int, int], name: str) -> tuple[int, int]:
    raw = np.asarray(value)
    if raw.shape != (2,) or raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must be an integer (concentration, bolus) pair")
    answer = (int(raw[0]), int(raw[1]))
    if not 0 <= answer[0] < shape[0] or not 0 <= answer[1] < shape[1]:
        raise ValueError(f"{name} is outside the fitted grid")
    return answer


@dataclass(frozen=True)
class CiBolusDecision:
    """Grid summaries and the deterministic highest-mean-utility assignment."""

    action: str
    pair: tuple[int, int] | None
    mean_utility: FloatArray
    toxicity_probability: FloatArray
    inefficacy_probability: FloatArray
    acceptable: np.ndarray
    eligible: np.ndarray
    reason: str


def cibolus_decision(
    fit: CiBolusFit,
    treated: ArrayLike,
    *,
    toxicity_limit: float,
    toxicity_cutoff: float,
    efficacy_limit: float,
    efficacy_cutoff: float,
    starting: tuple[int, int] = (0, 0),
    final: bool = False,
) -> CiBolusDecision:
    """Choose maximum posterior mean utility after strict tail-probability screens.

    A candidate is safe when both posterior tail probabilities are at most
    their cutoffs; the source excludes only when a probability is strictly
    greater than its cutoff. Interim concentration escalation is capped at
    one level above the highest previously treated concentration. Bolus levels
    are unrestricted. The empty-history starting regimen bypasses the screen,
    but returned masks still report the actual posterior classifications.
    """
    if not isinstance(fit, CiBolusFit):
        raise ValueError("fit must be a CiBolusFit")
    if not isinstance(final, (bool, np.bool_)):
        raise ValueError("final must be boolean")
    tox_limit = _scalar(toxicity_limit, "toxicity_limit")
    tox_cutoff = _scalar(toxicity_cutoff, "toxicity_cutoff")
    eff_limit = _scalar(efficacy_limit, "efficacy_limit")
    eff_cutoff = _scalar(efficacy_cutoff, "efficacy_cutoff")
    if any(not 0 <= value <= 1 for value in (tox_limit, tox_cutoff, eff_limit, eff_cutoff)):
        raise ValueError("toxicity/efficacy limits and cutoffs must lie in [0,1]")
    utility_draws = np.asarray(fit.expected_utility)
    tox_draws = np.asarray(fit.toxicity_at_one_response)
    efficacy_draws = np.asarray(fit.response_at_one)
    if (
        utility_draws.ndim != 4
        or utility_draws.shape[0] < 1
        or utility_draws.shape[1] < 1
        or utility_draws.shape[2] > 20
        or utility_draws.shape[3] > 20
        or utility_draws.size > 2_000_000
        or tox_draws.shape != utility_draws.shape
        or efficacy_draws.shape != utility_draws.shape
        or np.any(~np.isfinite(utility_draws))
        or np.any(~np.isfinite(tox_draws))
        or np.any(~np.isfinite(efficacy_draws))
        or np.any((tox_draws < 0) | (tox_draws > 1))
        or np.any((efficacy_draws < 0) | (efficacy_draws > 1))
    ):
        raise ValueError("fit contains invalid posterior grid draws")
    grid = utility_draws.shape[2:]
    raw_treated = np.asarray(treated)
    if (
        raw_treated.shape != grid
        or raw_treated.dtype.kind not in "iu"
        or raw_treated.dtype.kind == "b"
    ):
        raise ValueError("treated must be nonnegative integer counts matching the fitted grid")
    if np.any(raw_treated < 0) or np.sum(raw_treated, dtype=object) > 10_000:
        raise ValueError("treated counts must be nonnegative and total at most 10,000")
    counts = raw_treated.astype(np.int64, copy=False)
    start = _pair(starting, grid, "starting")
    total = int(np.sum(counts, dtype=np.int64))
    if total == 0 and final:
        raise ValueError("final selection requires at least one treated patient")

    mean_utility = utility_draws.mean(axis=(0, 1))
    p_toxic = np.mean(tox_draws > tox_limit, axis=(0, 1))
    p_inefficacy = np.mean(efficacy_draws < eff_limit, axis=(0, 1))
    if not np.all(np.isfinite(mean_utility)):
        raise ArithmeticError("posterior mean utility is not representable")
    acceptable = (p_toxic <= tox_cutoff) & (p_inefficacy <= eff_cutoff)
    eligible = acceptable.copy()
    if total == 0:
        return CiBolusDecision(
            "start",
            start,
            _freeze_float(mean_utility),
            _freeze_float(p_toxic),
            _freeze_float(p_inefficacy),
            _readonly_bool(acceptable),
            _readonly_bool(eligible),
            "first assignment follows the configured starting regimen",
        )
    if not final:
        tried_concentrations = np.flatnonzero(np.any(counts > 0, axis=1))
        maximum = min(int(tried_concentrations.max()) + 1, grid[0] - 1)
        eligible[: maximum + 1, :] &= acceptable[: maximum + 1, :]
        eligible[maximum + 1 :, :] = False
    if not np.any(eligible):
        return CiBolusDecision(
            "stop",
            None,
            _freeze_float(mean_utility),
            _freeze_float(p_toxic),
            _freeze_float(p_inefficacy),
            _readonly_bool(acceptable),
            _readonly_bool(eligible),
            "no regimen satisfies both posterior safety screens and the allocation rule",
        )
    utility = np.where(eligible, mean_utility, -np.inf)
    flat = int(np.argmax(utility))
    chosen_raw = np.unravel_index(flat, grid)
    chosen = (int(chosen_raw[0]), int(chosen_raw[1]))
    return CiBolusDecision(
        "select" if final else "treat",
        chosen,
        _freeze_float(mean_utility),
        _freeze_float(p_toxic),
        _freeze_float(p_inefficacy),
        _readonly_bool(acceptable),
        _readonly_bool(eligible),
        "highest posterior mean utility among acceptable eligible regimens; ties lexicographic",
    )


def _freeze_float(value: ArrayLike) -> FloatArray:
    array = np.ascontiguousarray(value, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)
