"""RANDLIB's GS/GD gamma algorithms; attribution in THIRD_PARTY_NOTICES.md."""

import numpy as np
from numpy.typing import NDArray

from ._randlib_distributions import DistributionStream
from ._randlib_normal import standard_normal

_Q = np.array(
    [0.04166669, 0.02083148, 0.00801191, 0.00144121, -0.00007388, 0.00024511, 0.00024240],
    dtype=np.float32,
)
_A = np.array(
    [0.3333333, -0.2500030, 0.2000062, -0.1662921, 0.1423657, -0.1367177, 0.1233795],
    dtype=np.float32,
)
_E = np.array([1, 0.4999897, 0.1668290, 0.0407753, 0.0102930], dtype=np.float32)
for table in (_Q, _A, _E):
    table.flags.writeable = False


def _polynomial(coefficients: NDArray[np.float32], x: np.float32) -> np.float32:
    result = coefficients[-1]
    for coefficient in coefficients[-2::-1]:
        result = result * x + coefficient
    return result * x


class GammaSampler:
    """Shape-specific constants, without the source's process-global cache."""

    def __init__(self, shape: np.float32, source: str) -> None:
        self.shape = shape
        self.k = np.float64 if source == "c" else np.float32
        k = self.k
        self.b0 = np.float32(k(1) + k(0.3678794) * shape)
        if shape < 1:
            return
        self.s2 = np.float32(shape - k(0.5))
        self.s = np.float32(np.sqrt(k(self.s2)))
        self.d = np.float32(np.float32(5.656854) - k(12) * self.s)
        self.q0 = _polynomial(_Q, np.float32(k(1) / shape))
        if shape <= k(3.686):
            self.b = np.float32(k(0.463) + self.s + k(0.178) * self.s2)
            self.si = np.float32(1.235)
            self.c = np.float32(k(0.195) / self.s - k(0.079) + k(0.16) * self.s)
        elif shape <= k(13.022):
            self.b = np.float32(k(1.654) + k(0.0076) * self.s2)
            self.si = np.float32(k(1.68) / self.s + k(0.275))
            self.c = np.float32(k(0.062) / self.s + k(0.024))
        else:
            self.b, self.si = np.float32(1.77), np.float32(0.75)
            self.c = np.float32(k(0.1515) / self.s)

    def quotient(self, t: np.float32) -> np.float32:
        k = self.k
        v = t / (self.s + self.s)
        if abs(v) > 0.25:
            q = np.float32(
                self.q0 - self.s * t + k(0.25) * t * t + (self.s2 + self.s2) * np.log(k(1) + v)
            )
        else:
            # The final multiplication by v belongs after the t factors.
            polynomial = _A[-1]
            for coefficient in _A[-2::-1]:
                polynomial = polynomial * v + coefficient
            q = np.float32(self.q0 + k(0.5) * t * t * polynomial * v)
        if not np.isfinite(q):
            raise ArithmeticError("nonfinite legacy gamma quotient; state unchanged")
        return q

    def sample(self, stream: DistributionStream) -> np.float32:
        k, a = self.k, self.shape
        if a < 1:
            while True:
                p = self.b0 * stream.uniform()
                if p < 1:
                    value = np.float32(np.exp(np.log(k(p)) / a))
                    if stream.standard_exponential() >= value:
                        return value
                else:
                    value = np.float32(-np.log(k((self.b0 - p) / a)))
                    if stream.standard_exponential() >= (k(1) - a) * np.log(k(value)):
                        return value
        t = standard_normal(stream)
        x = np.float32(self.s + k(0.5) * t)
        value = x * x
        if t >= 0:
            return value
        u = stream.uniform()
        if self.d * u <= t * t * t:
            return value
        if x > 0 and np.log(k(1) - u) <= self.quotient(t):
            return value
        while True:
            e = stream.standard_exponential()
            u = stream.uniform()
            u = np.float32(u + (u - k(1)) if stream.source == "c" else u + u - k(1))
            magnitude = self.si * e
            t = self.b + (-magnitude if u < 0 else magnitude)
            if t < k(-0.7187449):
                continue
            q = self.quotient(t)
            if q <= 0:
                continue
            # C fabs/exp return doubles; Fortran ABS/EXP retain REAL precision.
            left = self.c * np.abs(k(u))
            if q >= 15:
                exponent = q + e - k(0.5) * t * t
                accepted = exponent > k(87.49823) or left <= np.exp(exponent)
            else:
                w = np.float32(np.exp(k(q)) - k(1)) if q > 0.5 else _polynomial(_E, q)
                accepted = left <= w * np.exp(e - k(0.5) * t * t)
            if accepted:
                x = np.float32(self.s + k(0.5) * t)
                return x * x


def legacy_gamma(
    stream: DistributionStream, size: int, shape: np.float32, rate: np.float32
) -> NDArray[np.float64]:
    result = np.empty(size, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        sampler = GammaSampler(shape, stream.source)
        for i in range(size):
            result[i] = sampler.sample(stream) / rate
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("gamma samples overflow float32; state unchanged")
    return result
