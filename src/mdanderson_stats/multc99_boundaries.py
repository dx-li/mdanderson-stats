"""Multc99 manual boundary editing and portable delta/probability curve data.

Adapted workflow retains the upstream noncommercial terms. See the bundled
Multc99 notice. The legacy plotting callback's C signature defect is corrected.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .multc99 import Multc99Design, _immutable
from .multc_study import _atomic_write


def multc99_with_boundaries(
    design: Multc99Design,
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
) -> Multc99Design:
    """Return a design using caller-supplied monotone integer stopping tables.

    Rows follow event order; columns are conditioning counts 0--max_subjects.
    Bounds from min_subjects onward must be valid nondecreasing count limits.
    Earlier limits are regenerated using the recovered semi-free-ride/run-back
    policy. This corrects the native editor leaving conditional prefix tables
    stale. Cutoffs describe the posterior design; explicit tables drive conduct.
    """
    if not isinstance(design, Multc99Design):
        raise TypeError("design must be a Multc99Design")
    shape = (len(design.events), design.max_subjects + 1)
    raw_lower, raw_upper = np.asarray(lower_bounds), np.asarray(upper_bounds)
    if any(raw.shape != shape or raw.dtype.kind not in "iu" for raw in (raw_lower, raw_upper)):
        raise ValueError("manual boundary tables must be integer matrices matching the design")
    minimum = design.min_subjects
    sizes = np.arange(minimum, design.max_subjects + 1)
    lower_suffix, upper_suffix = raw_lower[:, minimum:], raw_upper[:, minimum:]
    if (
        np.any(lower_suffix < -1)
        or np.any(lower_suffix > sizes)
        or np.any(upper_suffix < 0)
        or np.any(upper_suffix > sizes + 1)
        or np.any(np.diff(lower_suffix, axis=1) < 0)
        or np.any(np.diff(upper_suffix, axis=1) < 0)
    ):
        raise ValueError("manual suffix bounds must be valid nondecreasing count limits")
    lower = np.full(shape, -1, dtype=np.int64)
    upper = np.broadcast_to(np.arange(design.max_subjects + 1) + 1, shape).copy()
    lower[:, minimum:], upper[:, minimum:] = lower_suffix, upper_suffix
    for j, event in enumerate(design.events):
        for n in range(1, minimum):
            lower[j, n] = lower[j, minimum] + n - minimum
            if event.conditioning_definition is None:
                upper[j, n] = upper[j, minimum]
            else:
                upper[j, n] = raw_upper[j, n]
                if not 0 <= upper[j, n] <= n + 1:
                    raise ValueError("conditional upper prefix limits must lie in [0,n+1]")
    return replace(
        design,
        lower_bounds=_immutable(lower),
        upper_bounds=_immutable(upper),
        boundary_origin="manual",
    )


@dataclass(frozen=True)
class Multc99ProbabilityCurve:
    event_name: str
    margins: FloatArray
    probabilities: FloatArray
    absolute_errors: FloatArray

    def write_csv(self, path: str | Path) -> Path:
        text = "margin,probability,absolute_error\n"
        for margin, probability, error in zip(
            self.margins, self.probabilities, self.absolute_errors, strict=True
        ):
            text += f"{margin:.17g},{probability:.17g},{error:.17g}\n"
        return _atomic_write(path, text)


def multc99_probability_curve(
    design: Multc99Design,
    event_name: str,
    elementary_counts: ArrayLike,
    margins: ArrayLike,
) -> Multc99ProbabilityCurve:
    """Compute bounded curve data for explicit margins, preserving their order.

    Native plotting wrote a delta/probability text file for external S plotting.
    Python exports CSV including numerical errors; it does not repeat the
    original incompatible function-pointer call or floating step accumulation.
    """
    if not isinstance(design, Multc99Design):
        raise TypeError("design must be a Multc99Design")
    if isinstance(margins, (list, tuple)) and len(margins) > 1000:
        raise ValueError("margins exceeds 1,000 points")
    raw = np.asarray(margins)
    if raw.ndim != 1 or not 1 <= raw.size <= 1000 or raw.dtype.kind not in "iuf":
        raise ValueError("margins must be a nonempty real vector of at most 1,000 points")
    values = np.asarray(raw, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(np.abs(values) > 1):
        raise ValueError("margins must be finite and lie in [-1,1]")
    probabilities, errors = np.empty(values.size), np.empty(values.size)
    for j, delta in enumerate(values):
        result = design.event_probability(event_name, elementary_counts, margin=float(delta))
        probabilities[j], errors[j] = result.probability, result.absolute_error
    return Multc99ProbabilityCurve(
        event_name, _freeze(values), _freeze(probabilities), _freeze(errors)
    )
