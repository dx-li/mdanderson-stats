"""SEQBIN null boundary tables; compaction never mutates the trial design."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ._validation import FloatArray
from .seqbin import SeqBinDesign


@dataclass(frozen=True)
class SeqBinBoundaryTable:
    rows: FloatArray
    columns: ClassVar[tuple[str, ...]] = (
        "subjects",
        "continue_low",
        "quit_low",
        "cumulative_low",
        "continue_high",
        "quit_high",
        "cumulative_high",
        "quit_total",
        "cumulative_total",
    )


def _boundary_table(
    design: SeqBinDesign, low: FloatArray, high: FloatArray, compact: bool
) -> SeqBinBoundaryTable:
    if not isinstance(compact, (bool, np.bool_)):
        raise ValueError("compact must be boolean")
    subjects = np.arange(1, design.max_subjects + 1)
    lower, upper = np.zeros_like(subjects), subjects.copy()
    lower[design.looks - 1], upper[design.looks - 1] = design.continue_low, design.continue_high
    quit_low, quit_high = np.zeros(subjects.size), np.zeros(subjects.size)
    quit_low[design.looks - 1], quit_high[design.looks - 1] = low, high
    low, high = quit_low, quit_high
    total = low + high
    rows = np.column_stack(
        [
            subjects,
            lower,
            low,
            np.cumsum(low),
            upper,
            high,
            np.cumsum(high),
            total,
            np.cumsum(total),
        ]
    )
    if compact:
        rows = rows[total > 0]
    rows.flags.writeable = False
    return SeqBinBoundaryTable(rows)


def seqbin_boundary_table(design: SeqBinDesign, *, compact: bool = False) -> SeqBinBoundaryTable:
    """Null continuation bounds, look-specific and cumulative stopping probabilities.

    Verbose output includes every subject count, including unscheduled nonlooks.
    Compact output drops only rows with exactly zero computed null stopping
    probability. All scheduled looks and the underlying design remain intact.
    """
    properties = design.operating_characteristics(design.null_probability)
    return _boundary_table(design, properties.quit_low, properties.quit_high, compact)
