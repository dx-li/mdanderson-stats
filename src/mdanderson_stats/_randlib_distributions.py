"""Local source-compatible sampling state for RANDLIB distribution algorithms."""

import numpy as np
from numpy.typing import NDArray

from .ranlist_random import _M1, _M2

_Q = np.array(
    [0.6931472, 0.9333737, 0.9888778, 0.9984959, 0.9998293, 0.9999833, 0.9999986, 0.9999999],
    dtype=np.float32,
)
_Q.flags.writeable = False


class DistributionStream:
    """Local state committed by the public bank only after successful sampling."""

    def __init__(self, state: tuple[int, int], antithetic: bool, source: str, budget: int) -> None:
        self.state = state
        self.antithetic = antithetic
        self.source = source
        self.budget = budget
        self.used = 0

    def uniform(self) -> np.float32:
        if self.used >= self.budget:
            raise ArithmeticError("distribution sampling exceeded max_attempts; state unchanged")
        a, b = self.state[0] * 40014 % _M1, self.state[1] * 40692 % _M2
        self.state = (a, b)
        self.used += 1
        z = a - b
        if z < 1:
            z += _M1 - 1
        if self.antithetic:
            z = _M1 - z
        return (
            np.float32(z * 4.656613057e-10)
            if self.source == "c"
            else np.float32(z) * np.float32(4.656613057e-10)
        )

    def standard_exponential(self) -> np.float32:
        offset = np.float32(0)
        u = self.uniform()
        while True:
            u = u + u
            if u >= 1:
                break
            offset = offset + _Q[0]
        u = u - np.float32(1)
        if u <= _Q[0]:
            return offset + u
        if u > _Q[-1]:
            raise ArithmeticError("legacy exponential exceeds its source table; state unchanged")
        minimum = self.uniform()
        for boundary in _Q[1:]:
            minimum = min(minimum, self.uniform())
            if u <= boundary:
                return offset + minimum * _Q[0]
        raise ArithmeticError("unreachable exponential table state")


def legacy_exponential(
    stream: DistributionStream, size: int, mean: np.float32
) -> NDArray[np.float64]:
    result = np.empty(size, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for i in range(size):
            result[i] = stream.standard_exponential() * mean
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("exponential samples overflow float32; state unchanged")
    return result


def source_log(value: np.float32 | np.float64, source: str) -> np.float32 | np.float64:
    """Evaluate a logarithm in double precision, then apply source rounding."""
    result = np.log(np.float64(value))
    return np.float32(result) if source == "fortran" else np.float64(result)


def source_exp(value: np.float32 | np.float64, source: str) -> np.float32 | np.float64:
    """Evaluate an exponential in double precision, then apply source rounding."""
    result = np.exp(np.float64(value))
    return np.float32(result) if source == "fortran" else np.float64(result)
