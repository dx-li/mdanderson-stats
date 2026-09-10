"""Explicit historical SPPCR binomial sampling through reconciled RANDLIB."""

import numpy as np
from numpy.typing import ArrayLike

from .randlib import RandlibGenerator
from .sppcr_fit import _freeze
from .sppcr_generate import SPPCRSamples, _cells, _wells


def sppcr_generate_legacy(
    probability: ArrayLike,
    wells: ArrayLike,
    *,
    rng: RandlibGenerator,
    replicates: int = 1,
) -> SPPCRSamples:
    """Sample in replicate/batch/DNA/allele order with source float32 binomials.

    Shapes match sppcr_generate. All inputs are validated before drawing. Positive
    probabilities that round to zero, or subunit probabilities rounding to one,
    are rejected by the existing RANDLIB safety contract. Exact endpoints work.
    Legacy well counts cannot exceed 2147483646. The explicit RNG is consumed
    cell by cell; a later numerical sampling failure does not roll back earlier
    cells. Use sppcr_generate with NumPy for fast modern batch simulation.
    """
    p = _cells(probability, "probability")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probability must be in [0, 1]")
    n = _wells(wells, p.shape[:-1])
    if np.any(n > 2147483646):
        raise ValueError("legacy wells must not exceed 2147483646")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 0:
        raise ValueError("replicates must be a nonnegative integer")
    if not isinstance(rng, RandlibGenerator):
        raise TypeError("rng must be a RandlibGenerator")
    narrowed = p.astype(np.float32)
    if np.any(((p > 0) & (narrowed == 0)) | ((p < 1) & (narrowed == 1))):
        raise ValueError("legacy probability must not round to an endpoint")
    counts = np.empty((replicates,) + p.shape, dtype=np.int64)
    trials = np.broadcast_to(n[..., None], p.shape).reshape(-1)
    probabilities = narrowed.reshape(-1)
    for i in range(replicates):
        row = counts[i].reshape(-1)
        for j in range(row.size):
            row[j] = rng.binomial(
                1, n=int(trials[j]), p=float(probabilities[j]), legacy=True, source="fortran"
            )[0]
    return SPPCRSamples(
        _freeze(narrowed), _freeze(n), _freeze(counts), _freeze(n[..., None] - counts)
    )
