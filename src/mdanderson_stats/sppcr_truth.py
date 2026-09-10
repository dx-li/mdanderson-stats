"""Explicit truth parameters for SPPCR simulations in model DNA units."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import count, finite
from .sppcr_fit import FloatArray, _freeze
from .sppcr_generate import _wells, sppcr_detection_probabilities


@dataclass(frozen=True)
class SPPCRTruth:
    """One immutable design; dna is used directly, without genome conversion."""

    dna: FloatArray
    wells: FloatArray
    frequency: FloatArray
    calibration: float
    progenitor: tuple[int, int]
    mu: FloatArray
    probability: FloatArray


def sppcr_truth(
    dna: ArrayLike,
    wells: ArrayLike,
    weights: ArrayLike,
    calibration: float,
    *,
    progenitor: tuple[int, int],
) -> SPPCRTruth:
    """Normalize allele weights and prepare a single truth-generation design.

    DNA is in the numerical model's units, as in native truth entry. Calibration
    is positive and mu = calibration * normalized weights. Wells may vary by DNA
    level (an extension of native interactive entry). Progenitors are zero-based.
    Positive weights or means that would round to zero raise ArithmeticError.
    Detection probabilities may underflow or saturate as documented by
    sppcr_detection_probabilities. No random state is consumed here.
    """
    d, w, c = finite(dna, "dna"), finite(weights, "weights"), finite(calibration, "calibration")
    if d.ndim != 1 or d.size == 0 or np.any(d <= 0):
        raise ValueError("dna must be a nonempty positive vector")
    if w.ndim != 1 or w.size == 0 or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("weights must be a nonempty nonnegative vector with positive total")
    if c.ndim != 0 or c <= 0:
        raise ValueError("calibration must be a positive scalar")
    parents = count(progenitor, "progenitor")
    if parents.shape != (2,) or np.any(parents >= w.size):
        raise ValueError("progenitor must contain two valid zero-based allele indices")
    n = _wells(wells, d.shape)
    with np.errstate(under="ignore"):
        scaled = w / np.max(w)
        frequency = scaled / np.sum(scaled)
        mu = float(c) * frequency
    if np.any((w > 0) & ((frequency == 0) | (mu == 0))):
        raise ArithmeticError("positive truth frequency or mean underflows float64")
    return SPPCRTruth(
        _freeze(d),
        _freeze(n),
        _freeze(frequency),
        float(c),
        (int(parents[0]), int(parents[1])),
        _freeze(mu),
        sppcr_detection_probabilities(d, mu),
    )
