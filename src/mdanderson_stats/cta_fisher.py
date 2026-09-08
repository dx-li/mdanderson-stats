"""Fixed-margin Fisher probabilities and CTA's original selected tail."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import hypergeom

from ._validation import FloatArray, count


@dataclass(frozen=True)
class FisherExact:
    pvalue: FloatArray
    observed_probability: FloatArray
    terms: NDArray[np.int64]
    support_size: NDArray[np.int64]
    source_lower_tail: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    alternative: str
    legacy: bool


def fisher_exact(
    observed: ArrayLike,
    *,
    alternative: Literal["two-sided", "less", "greater", "source"] | None = None,
    legacy: bool = False,
) -> FisherExact:
    """Enumerate fixed-margin probabilities for final 2x2 integer-count axes.

    The default is a probability-ordered two-sided test. Legacy defaults to
    CTA's data-selected one-sided tail and relative-term cutoff; explicit
    alternative='source' without legacy sums that entire tail. Less/greater
    refer to the upper-left count. Each table must total at most 50,000 (CTA's
    original workspace limit). Leading dimensions form a batch.
    """
    ob = count(observed, "observed")
    if ob.ndim < 2 or ob.shape[-2:] != (2, 2):
        raise ValueError("observed must contain 2x2 tables")
    if np.any(ob.sum(axis=(-2, -1)) > 50_000):
        raise ValueError("each table total must be at most 50,000, CTA's supported limit")
    if not isinstance(legacy, (bool, np.bool_)):
        raise ValueError("legacy must be boolean")
    if alternative is None:
        alternative = "source" if legacy else "two-sided"
    if alternative not in ("two-sided", "less", "greater", "source"):
        raise ValueError("alternative must be two-sided, less, greater or source")
    if legacy and alternative != "source":
        raise ValueError("legacy requires the source alternative")
    shape = ob.shape[:-2]
    pvalue, observed_probability = np.empty(shape), np.empty(shape)
    terms, support_size = np.empty(shape, dtype=np.int64), np.empty(shape, dtype=np.int64)
    source_lower = np.empty(shape, dtype=np.bool_)
    truncated = np.zeros(shape, dtype=np.bool_)
    # Supports differ across tables; vectorize the probability calculation
    # within each support without allocating a batch-by-maximum-support array.
    for i, table in enumerate(ob.reshape(-1, 4)):
        support, probabilities, selected, stopped = _fisher_support(
            table, alternative, bool(legacy)
        )
        a, b, c, d = map(int, table)
        observed_probability.flat[i] = probabilities[a - support[0]]
        source_lower.flat[i] = a * d < b * c
        support_size.flat[i] = len(support)
        truncated.flat[i] = stopped
        terms.flat[i] = len(selected)
        pvalue.flat[i] = min(1.0, float(np.sum(probabilities[selected])))
    for value in (pvalue, observed_probability, terms, support_size, source_lower, truncated):
        value.flags.writeable = False
    return FisherExact(
        pvalue,
        observed_probability,
        terms,
        support_size,
        source_lower,
        truncated,
        alternative,
        bool(legacy),
    )


def _fisher_support(
    table: FloatArray, alternative: str, legacy: bool
) -> tuple[NDArray[np.int64], FloatArray, NDArray[np.int64], bool]:
    """Enumerate a validated table once for both tests and detailed reports."""
    truncated = False
    a, b, c, d = map(int, table)
    total, row, col = a + b + c + d, a + b, a + c
    lo, hi = max(0, row + col - total), min(row, col)
    support = np.arange(lo, hi + 1)
    probabilities = np.ones(1) if total == 0 else hypergeom.pmf(support, total, row, col)
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("nonfinite hypergeometric probabilities")
    at = a - lo
    if alternative == "two-sided":
        # Relative tolerance includes theoretically equal masses despite
        # floating-point evaluation of symmetric tables.
        if probabilities[at] > 0:
            selected = np.flatnonzero(probabilities <= probabilities[at] * (1 + 1e-12))
        else:
            logs = hypergeom.logpmf(support, total, row, col)
            selected = np.flatnonzero(logs <= logs[at] + 1e-10)
    else:
        lower = alternative == "less" or (alternative == "source" and a * d < b * c)
        selected = np.arange(at, -1, -1) if lower else np.arange(at, len(support))
        if legacy and len(selected) > 1:
            logs = hypergeom.logpmf(support[selected], total, row, col)
            stop = np.flatnonzero(logs - logs[0] < np.log(1e-5))
            if len(stop):
                end = int(stop[0]) + 1  # Source includes the triggering term.
                truncated = end < len(selected)
                selected = selected[:end]
    return support, probabilities, selected, truncated
