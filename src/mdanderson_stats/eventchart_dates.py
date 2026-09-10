"""Calendar conversion for EVENTCHART, using days since an explicit origin."""

from datetime import date, timedelta

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, finite


def event_dates(values: tuple[str | date | None, ...], *, origin: str = "1960-01-01") -> FloatArray:
    """Convert ISO calendar dates to integer days; None becomes NaN."""
    start = date.fromisoformat(origin)
    if len(values) > 1_000_000:
        raise ValueError("at most 1000000 dates are supported")
    result = []
    for value in values:
        if value is None:
            result.append(np.nan)
        else:
            parsed = date.fromisoformat(value) if isinstance(value, str) else value
            if type(parsed) is not date:
                raise ValueError("dates must be ISO date strings, date objects or None")
            result.append(float((parsed - start).days))
    return _freeze(np.asarray(result))


def event_date_labels(values: ArrayLike, *, origin: str = "1960-01-01") -> tuple[str, ...]:
    """Format finite day offsets as ISO dates, rounding to the nearest day."""
    start = date.fromisoformat(origin)
    x = finite(values, "values")
    if x.ndim != 1 or x.size > 1_000_000:
        raise ValueError("values must be a vector of at most 1000000 day offsets")
    result = []
    for v in np.rint(x):
        if not (date.min - start).days <= v <= (date.max - start).days:
            raise ValueError("date offset is outside calendar years 1..9999")
        result.append((start + timedelta(days=int(v))).isoformat())
    return tuple(result)
