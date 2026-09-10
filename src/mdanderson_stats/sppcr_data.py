"""Validated SPPCR input data with explicit genome-to-allele DNA conversion."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import count, finite
from .sppcr_fit import FloatArray, _freeze


@dataclass(frozen=True)
class SPPCRData:
    """One experiment; dna is twice genome_dna and is ready for numerical APIs."""

    genome_dna: FloatArray
    dna: FloatArray
    wells: FloatArray
    seen: FloatArray
    allele_sizes: tuple[int, ...]
    progenitor_sizes: tuple[int, int]
    progenitor: tuple[int, int]
    omitted_allele_sizes: tuple[int, ...]


def sppcr_data(
    genome_dna: ArrayLike,
    seen: ArrayLike,
    wells: ArrayLike,
    allele_sizes: ArrayLike,
    progenitor_sizes: tuple[int, int],
    *,
    unseen_alleles: str = "drop",
) -> SPPCRData:
    """Validate one input experiment and map allele sizes to zero-based indices.

    Counts and sizes are nonnegative integers below 2**53; sizes and wells must
    be positive. The default drops never-seen alleles as in native input conversion,
    but refuses to drop a progenitor. Use 'retain' to keep every supplied allele.
    """
    d = finite(genome_dna, "genome_dna")
    s, n = count(seen, "seen"), count(wells, "wells")
    sizes, parents = (
        count(allele_sizes, "allele_sizes"),
        count(progenitor_sizes, "progenitor_sizes"),
    )
    if d.ndim != 1 or d.size == 0 or np.any(d <= 0):
        raise ValueError("genome_dna must be a nonempty positive vector")
    if s.ndim != 2 or s.shape[0] != d.size or s.shape[1] == 0:
        raise ValueError("seen must have shape (DNA levels, nonempty alleles)")
    n = np.broadcast_to(n, d.shape)
    if np.any(n <= 0) or np.any(s > n[:, None]):
        raise ValueError("wells must be positive and seen must not exceed wells")
    if sizes.shape != (s.shape[1],) or np.any(sizes <= 0) or len(np.unique(sizes)) != len(sizes):
        raise ValueError("allele_sizes must contain one distinct positive size per allele")
    if parents.shape != (2,) or not np.all(np.isin(parents, sizes)):
        raise ValueError("progenitor_sizes must contain two sizes present in allele_sizes")
    if unseen_alleles not in ("drop", "retain"):
        raise ValueError("unseen_alleles must be 'drop' or 'retain'")
    keep = np.any(s > 0, axis=0) if unseen_alleles == "drop" else np.ones(len(sizes), dtype=bool)
    if not np.all(np.isin(parents, sizes[keep])):
        raise ValueError("cannot drop an unseen progenitor; use unseen_alleles='retain'")
    with np.errstate(over="ignore"):
        dna = 2 * d
    if np.any(~np.isfinite(dna)):
        raise ArithmeticError("genome-to-allele DNA conversion exceeds float64 range")
    labels = tuple(int(x) for x in sizes[keep])
    p = (int(parents[0]), int(parents[1]))
    return SPPCRData(
        _freeze(d),
        _freeze(dna),
        _freeze(n),
        _freeze(s[:, keep]),
        labels,
        p,
        (labels.index(p[0]), labels.index(p[1])),
        tuple(int(x) for x in sizes[~keep]),
    )
