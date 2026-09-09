"""SPPCR detection probabilities and reproducible independent binomial samples."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import count, finite
from .sppcr_fit import FloatArray, _freeze


def _wells(wells: ArrayLike, shape: tuple[int, ...]) -> FloatArray:
    n = np.broadcast_to(count(wells, "wells"), shape)
    if np.any(n == 0):
        raise ValueError("wells must be positive")
    return n


def _cells(values: ArrayLike, name: str) -> FloatArray:
    a = finite(values, name)
    if a.ndim < 2 or 0 in a.shape[-2:]:
        raise ValueError(f"{name} must have nonempty DNA-level and allele axes")
    return a


def sppcr_detection_probabilities(dna: ArrayLike, mu: ArrayLike) -> FloatArray:
    """Return 1-exp(-dna*mu) with shape (..., levels, alleles).

    dna is a positive vector and mu has shape (..., alleles), with finite
    nonnegative means. Large products saturate at one; tiny probabilities may
    underflow to zero. expm1 preserves representable rare detection probabilities.
    """
    d, m = finite(dna, "dna"), finite(mu, "mu")
    if d.ndim != 1 or d.size == 0 or np.any(d <= 0):
        raise ValueError("dna must be a nonempty positive vector")
    if m.ndim < 1 or m.shape[-1] == 0 or np.any(m < 0):
        raise ValueError("mu must have a nonempty allele axis and nonnegative values")
    with np.errstate(over="ignore", under="ignore"):
        p = -np.expm1(-d[:, None] * m[..., None, :])
    return _freeze(p)


def sppcr_observed_probabilities(seen: ArrayLike, wells: ArrayLike) -> FloatArray:
    """Observed cell fractions used by the original SPPCR bootstrap.

    seen has shape (..., levels, alleles); wells broadcasts to (..., levels).
    Counts must be integers below 2**53 and wells must be positive.
    """
    s = count(_cells(seen, "seen"), "seen")
    n = _wells(wells, s.shape[:-1])
    if np.any(s > n[..., None]):
        raise ValueError("seen must not exceed wells")
    return _freeze(s / n[..., None])


@dataclass(frozen=True)
class SPPCRSamples:
    """Owned immutable float64 arrays, directly accepted by sppcr_fit_means.

    seen and unseen have shape (replicates, ..., levels, alleles). probability
    and wells describe the unreplicated design. Integer counts are exact float64.
    """

    probability: FloatArray
    wells: FloatArray
    seen: FloatArray
    unseen: FloatArray


def sppcr_generate(
    probability: ArrayLike,
    wells: ArrayLike,
    *,
    rng: np.random.Generator,
    replicates: int = 1,
) -> SPPCRSamples:
    """Generate independent binomial counts in one NumPy call.

    probability has shape (..., levels, alleles), with values in [0, 1]. wells
    broadcasts to (..., levels). An explicit Generator is consumed only after
    input validation; no global RNG or clock reseeding is used. This samples the
    source model, not the historical Fortran random sequence. Zero replicates and
    empty leading batch axes are supported and consume no randomness.
    """
    p = _cells(probability, "probability")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probability must be in [0, 1]")
    n = _wells(wells, p.shape[:-1])
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 0:
        raise ValueError("replicates must be a nonnegative integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    shape = (replicates,) + p.shape
    samples = rng.binomial(n.astype(np.int64)[..., None], p, size=shape)
    return SPPCRSamples(_freeze(p), _freeze(n), _freeze(samples), _freeze(n[..., None] - samples))
