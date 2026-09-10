"""Chromosome recombination, marker screening and selection kernels for SOGS."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray, finite

SOGS_MOUSE_LENGTHS = _freeze(
    np.array(
        [
            55.04,
            115.53,
            99.48,
            68.52,
            75.45,
            84.56,
            64.05,
            69.63,
            75.22,
            71.34,
            74.17,
            85.42,
            62.46,
            65.86,
            67.93,
            63.09,
            51.83,
            40.35,
            62.43,
        ]
    )
)


def _chromosome(segments: ArrayLike, length: float) -> FloatArray:
    if np.ndim(length) != 0 or not np.isfinite(length) or not 0 < length <= 10000:
        raise ValueError("length must be a finite value in (0,10000] cM")
    x = finite(segments, "donor_segments")
    if x.shape == (0,):
        x = x.reshape(0, 2)
    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] > 10000:
        raise ValueError("donor_segments must contain at most 10000 [start,end] intervals")
    if (
        np.any(x[:, 0] < 0)
        or np.any(x[:, 1] > length)
        or np.any(x[:, 0] >= x[:, 1])
        or np.any(x[1:, 0] <= x[:-1, 1])
    ):
        raise ValueError("donor intervals must be increasing, separated, and within the chromosome")
    return x


@dataclass(frozen=True)
class SOGSCrossover:
    retained: FloatArray
    transferred: FloatArray


def sogs_recombine(
    donor_segments: ArrayLike, length: float, breakpoints: ArrayLike
) -> SOGSCrossover:
    """Cross a donor/mixed chromosome with a pure inbred-partner chromosome.

    Start on the donor-containing homologue and switch at each breakpoint.
    Return both resulting donor-segment sets; no random draw is performed.
    Segment endpoints are closed for marker screening, matching the source.
    """
    x = _chromosome(donor_segments, length)
    breaks = finite(breakpoints, "breakpoints")
    if (
        breaks.ndim != 1
        or breaks.size > 10000
        or np.any(breaks < 0)
        or np.any(breaks > length)
        or np.any(np.diff(breaks) <= 0)
    ):
        raise ValueError(
            "breakpoints must be at most 10000 strictly increasing positions in [0,length]"
        )
    retained: list[tuple[float, float]] = []
    transferred: list[tuple[float, float]] = []
    # Each breakpoint lies in at most one donor segment, bounding total output.
    for start, end in x:
        lo = np.searchsorted(breaks, start, side="right")
        hi = np.searchsorted(breaks, end, side="left")
        cuts = np.r_[start, breaks[lo:hi], end]
        for j in range(cuts.size - 1):
            target = transferred if (lo + j) % 2 else retained
            target.append((cuts[j], cuts[j + 1]))
    return SOGSCrossover(
        _freeze(np.asarray(retained).reshape(-1, 2)),
        _freeze(np.asarray(transferred).reshape(-1, 2)),
    )


@dataclass(frozen=True)
class SOGSScreen:
    probes: FloatArray
    detected: bool
    pure_ipt: bool
    missed: bool
    donor_length: float


def sogs_screen(donor_segments: ArrayLike, length: float) -> SOGSScreen:
    """Screen donor segments using SOGS's approximately 5-cM marker grid.

    floor(length/5)+1 markers include both chromosome ends. A shorter chromosome
    uses two endpoint markers, avoiding the source's division by zero.
    """
    x = _chromosome(donor_segments, length)
    intervals = max(1, int(length / 5))
    probes = np.arange(intervals + 1) * (length / intervals)
    probes[-1] = length
    detected = False
    if x.size:
        first = np.searchsorted(probes, x[:, 0], side="left")
        detected = bool(np.any(probes[first] <= x[:, 1]))
    return SOGSScreen(
        _freeze(probes),
        detected,
        not bool(x.size),
        bool(x.size) and not detected,
        float(np.sum(x[:, 1] - x[:, 0])),
    )


def sogs_eligible(
    apparent_ipt: ArrayLike, chromosome_lengths: ArrayLike, *, rule: int = 2
) -> NDArray[np.int64]:
    """Return zero-based eligible offspring; random tie-breaking is separate.

    Rules: 0 all; 1 most apparent IPT chromosomes; 2 most IPT chromosomes then
    greatest total length of those chromosomes; 3 greatest IPT chromosome
    length alone. Length refers to whole apparent-IPT chromosomes, not the
    amount of inbred material within a mixed chromosome.
    """
    if isinstance(rule, bool) or not isinstance(rule, int) or rule not in (0, 1, 2, 3):
        raise ValueError("rule must be 0, 1, 2 or 3")
    a = np.asarray(apparent_ipt)
    lengths = finite(chromosome_lengths, "chromosome_lengths")
    if (
        a.ndim != 2
        or a.dtype.kind != "b"
        or not 1 <= a.shape[0] <= 100000
        or a.size > 2000000
        or lengths.shape != (a.shape[1],)
        or not lengths.size
        or np.any(lengths <= 0)
        or np.any(lengths > 10000)
    ):
        raise ValueError("supply an offspring-by-chromosome boolean matrix and positive cM lengths")
    eligible = np.ones(a.shape[0], dtype=bool)
    if rule in (1, 2):
        counts = np.sum(a, axis=1)
        eligible &= counts == counts.max()
    if rule in (2, 3):
        # Fixed summation order makes ties independent of the BLAS configuration.
        totals = np.sum(a * lengths, axis=1)
        eligible &= totals == totals[eligible].max()
    result = np.flatnonzero(eligible)
    result.flags.writeable = False
    return result
