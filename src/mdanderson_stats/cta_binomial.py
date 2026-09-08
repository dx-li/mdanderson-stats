"""CTA BINCOMP conditional Poisson-model comparison of two event counts."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count
from .onesample import binomial_test


@dataclass(frozen=True)
class BinomialComparison:
    group_sizes: FloatArray
    events: FloatArray
    event_index: NDArray[np.int64]
    null_probability: FloatArray
    p_less: FloatArray
    p_greater: FloatArray
    pvalue: FloatArray
    reported_tails: FloatArray
    groups: str
    legacy: bool


def binomial_comparison(
    observed: ArrayLike,
    *,
    groups: Literal["rows", "columns"] = "rows",
    event_index: int | None = None,
    legacy: bool = False,
) -> BinomialComparison:
    """Condition on total events, using group sizes as Poisson exposures.

    Final axes are 2x2 nonnegative integer counts; leading axes form batches.
    Event index chooses the event category on the axis opposite groups. None
    uses CTA's common-minority rule, rejecting inconsistent group directions.
    Default pvalue is min(1, 2*min(inclusive tails)). Legacy doubles the tail
    selected by the smaller raw event count, without capping it at one.
    This is exact under the conditional Poisson model, not a Fisher test.
    """
    ob = count(observed, "observed")
    if ob.ndim < 2 or ob.shape[-2:] != (2, 2):
        raise ValueError("observed must contain 2x2 tables")
    if groups not in ("rows", "columns"):
        raise ValueError("groups must be rows or columns")
    if not isinstance(legacy, (bool, np.bool_)):
        raise ValueError("legacy must be boolean")
    if event_index is not None and (
        isinstance(event_index, (bool, np.bool_))
        or not isinstance(event_index, (int, np.integer))
        or event_index not in (0, 1)
    ):
        raise ValueError("event_index must be zero, one or None")
    table = ob if groups == "rows" else np.swapaxes(ob, -1, -2)
    sizes = table.sum(axis=-1)
    total = sizes.sum(axis=-1)
    if np.any(sizes <= 0) or np.any(total >= 2**53):
        raise ValueError("group sizes must be positive and table totals smaller than 2**53")
    if event_index is None:
        first = np.all(table[..., :, 0] < table[..., :, 1], axis=-1)
        second = np.all(table[..., :, 0] >= table[..., :, 1], axis=-1)
        if np.any(~(first | second)):
            raise ValueError("no common minority category; specify event_index explicitly")
        index = np.asarray(np.where(first, 0, 1), dtype=np.int64)
    else:
        index = np.full(ob.shape[:-2], event_index, dtype=np.int64)
    events = np.where(index[..., None] == 0, table[..., :, 0], table[..., :, 1])
    n = events.sum(axis=-1)
    p = np.asarray(sizes[..., 0] / total)
    # The shared binomial engine requires positive trials. Handle the genuine
    # degenerate zero-event distribution explicitly, without inventing trials.
    less, greater = np.ones(n.shape), np.ones(n.shape)
    active = n > 0
    fit = binomial_test(events[..., 0][active], n[active], p[active])
    less[active], greater[active] = fit.p_less, fit.p_greater
    if legacy:
        chosen = np.where(events[..., 0] < events[..., 1], less, greater)
        reported = np.stack((chosen, chosen), axis=-1)
        pvalue = np.asarray(2 * chosen)
    else:
        reported = np.stack((less, greater), axis=-1)
        pvalue = np.asarray(np.minimum(1, 2 * np.minimum(less, greater)))
    for value in (sizes, events, index, p, less, greater, pvalue, reported):
        value.flags.writeable = False
    return BinomialComparison(
        sizes, events, index, p, less, greater, pvalue, reported, groups, bool(legacy)
    )
