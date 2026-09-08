"""Pure bounded draws with explicit state and a finite rejection budget."""

import numpy as np
from numpy.typing import NDArray

from .ranlist_random import _M1, _M2, _mod_power


def raw_batch(
    state: tuple[int, int], size: int, antithetic: bool
) -> tuple[NDArray[np.int64], tuple[int, int]]:
    positions = np.arange(1, size + 1, dtype=np.int64)
    first = state[0] * _mod_power(40014, positions, _M1) % _M1
    second = state[1] * _mod_power(40692, positions, _M2) % _M2
    difference = first - second
    result = np.where(difference < 1, difference + _M1 - 1, difference)
    if antithetic:
        result = _M1 - result
    final = (int(first[-1]), int(second[-1])) if size else state
    return result, final


def bounded(
    state: tuple[int, int],
    size: int,
    low: int,
    high: int,
    antithetic: bool,
    legacy: bool,
    budget: int,
) -> tuple[NDArray[np.int64], tuple[int, int], int]:
    result = np.full(size, low, dtype=np.int64)
    if low == high:
        return result, state, 0
    width = high - low + 1
    limit = ((_M1 - 2 if legacy else _M1 - 1) // width) * width
    filled = used = 0
    while filled < size:
        if used == budget:
            raise ArithmeticError(
                "integer rejection sampling exceeded max_attempts; state unchanged"
            )
        remaining = size - filled
        if remaining < 16:
            a, b = state[0] * 40014 % _M1, state[1] * 40692 % _M2
            state = (a, b)
            z = a - b
            if z < 1:
                z += _M1 - 1
            value = (_M1 - z if antithetic else z) - 1
            used += 1
            if value <= limit if legacy else value < limit:
                result[filled] = low + value % width
                filled += 1
        else:
            amount = min(remaining, budget - used)
            raw, state = raw_batch(state, amount, antithetic)
            values = raw - 1
            accepted = values[values <= limit if legacy else values < limit]
            result[filled : filled + accepted.size] = low + accepted % width
            filled += accepted.size
            used += amount
    return result, state, used
