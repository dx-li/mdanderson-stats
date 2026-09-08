"""RANLIST unrestricted assignment from weighted cumulative probabilities."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite
from .ranlist_random import _DEFAULT, ranlist_uniform


@dataclass(frozen=True)
class UnrestrictedAllocation:
    patients: NDArray[np.int64]
    treatments: NDArray[np.int64]
    weights: FloatArray
    probabilities: FloatArray
    cumulative_probabilities: FloatArray
    uniforms: FloatArray
    seed: tuple[int, int]
    stream: int
    legacy: bool


def ranlist_unrestricted(
    patients: ArrayLike,
    weights: ArrayLike,
    *,
    seed: tuple[int, int] = _DEFAULT,
    stream: int = 1,
    legacy: bool = False,
) -> UnrestrictedAllocation:
    """Assign one-based patients to one-based treatments in a selected stream.

    Supply 1..20 positive relative weights, not necessarily summing to one.
    Patient arrays retain shape and may be unsorted or repeated. Defaults use
    double-precision probabilities/uniforms; legacy reproduces sequential
    float32 weight accumulation and IGTUT's inclusive cumulative boundary.
    """
    if not isinstance(legacy, (bool, np.bool_)):
        raise ValueError("legacy must be boolean")
    w = finite(weights, "weights").copy()
    if w.ndim != 1 or not 1 <= len(w) <= 20 or np.any(w <= 0):
        raise ValueError("weights must be a vector of 1..20 strictly positive values")
    patient = count(patients, "patients").astype(np.int64)
    if np.any(patient < 1):
        raise ValueError("patients must be positive one-based numbers")
    if legacy:
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            source_weights = w.astype(np.float32)
            total = np.cumsum(source_weights, dtype=np.float32)[-1]
            source_probabilities = source_weights / total
            cumulative = np.cumsum(source_probabilities, dtype=np.float32).astype(np.float64)
            probabilities = source_probabilities.astype(np.float64)
        if not np.isfinite(total) or np.any(source_weights <= 0):
            raise ValueError("weights and their sum must be representable in float32 for legacy")
    else:
        scaled = w / w.max()
        probabilities = scaled / scaled.sum()
        cumulative = np.cumsum(probabilities)
        cumulative[-1] = 1.0
    intervals = np.diff(np.r_[0.0, cumulative])
    if not np.all(np.isfinite(cumulative)) or np.any(intervals < 0 if legacy else intervals <= 0):
        raise ValueError("weights cannot be resolved as positive cumulative intervals")
    uniforms = ranlist_uniform(patient, seed=seed, stream=stream, legacy=legacy)
    assigned = np.asarray(np.searchsorted(cumulative, uniforms, side="left") + 1, dtype=np.int64)
    if np.any(assigned > len(w)):
        raise ArithmeticError(
            "legacy cumulative probabilities do not cover the requested uniform draw"
        )
    for value in (patient, assigned, w, probabilities, cumulative):
        value.flags.writeable = False
    return UnrestrictedAllocation(
        patient,
        assigned,
        w,
        probabilities,
        cumulative,
        uniforms,
        (int(seed[0]), int(seed[1])),
        int(stream),
        bool(legacy),
    )
