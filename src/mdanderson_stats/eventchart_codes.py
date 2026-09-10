"""Numeric/string category validation for EVENTCHART event conversion."""

import math
from numbers import Integral, Real

import numpy as np

EventCode = str | int | float


def _code(value: object) -> EventCode | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        v = float(value)
        if math.isnan(v):
            return None
        if math.isfinite(v):
            return v
    raise ValueError("event codes must be finite numbers, strings, None or NaN")


def _code_levels(
    values: np.ndarray,
    declared: tuple[EventCode, ...] | None,
) -> tuple[np.ndarray, tuple[EventCode, ...]]:
    normalized = [_code(v) for v in values]
    seen = {v for v in normalized if v is not None}
    if declared is None:
        if any(isinstance(v, str) for v in seen) and any(not isinstance(v, str) for v in seen):
            raise ValueError("each code column must contain only strings or only numbers")
        levels = tuple(sorted(seen))
    else:
        checked = tuple(_code(v) for v in declared)
        if any(v is None for v in checked):
            raise ValueError("declared levels must not contain missing values")
        levels = tuple(v for v in checked if v is not None)
        if len(set(levels)) != len(levels):
            raise ValueError("declared levels must be unique")
        all_values = seen | set(levels)
        if any(isinstance(v, str) for v in all_values) and any(
            not isinstance(v, str) for v in all_values
        ):
            raise ValueError("each code column and its levels must have the same category type")
        if not seen.issubset(set(levels)):
            raise ValueError("an observed event code is absent from declared levels")
    return np.asarray(normalized, dtype=object), levels
