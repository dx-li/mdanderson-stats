"""Cheng BB/BC beta sampling from RANDLIB; retained attribution applies."""

import numpy as np
from numpy.typing import NDArray
from scipy.special import betaincinv

from ._randlib_distributions import DistributionStream, source_log
from ._randlib_sampling import raw_batch
from ._validation import scalar
from .ranlist_random import _M1


class BetaSampler:
    """Local shape-dependent BB/BC coefficients, with source arithmetic."""

    def __init__(self, aa: np.float32, bb: np.float32, source: str) -> None:
        self.source = source
        self.k = np.float64 if source == "c" else np.float32
        k = self.k
        self.bb_algorithm = min(aa, bb) > 1
        self.a, self.b = (
            (min(aa, bb), max(aa, bb)) if self.bb_algorithm else (max(aa, bb), min(aa, bb))
        )
        a, b = self.a, self.b
        self.reverse = aa != a
        self.alpha = a + b
        if self.bb_algorithm:
            self.beta = np.float32(np.sqrt((self.alpha - k(2)) / (k(2) * a * b - self.alpha)))
            self.gamma = np.float32(a + k(1) / self.beta)
            coefficients = [self.alpha, self.beta, self.gamma]
        else:
            self.beta = np.float32(k(1) / b)
            delta = np.float32(k(1) + a - b)
            self.k1 = np.float32(
                delta * (k(0.0138889) + k(0.0416667) * b) / (a * self.beta - k(0.777778))
            )
            self.k2 = np.float32(k(0.25) + (k(0.5) + k(0.25) / delta) * b)
            coefficients = [self.alpha, self.beta, self.k1, self.k2]
        if not np.all(np.isfinite(coefficients)) or self.beta <= 0:
            raise ArithmeticError("invalid legacy beta coefficients; state unchanged")

    def weight(self, v: np.float32) -> np.float32:
        a, k = self.a, self.k
        if a <= 1:
            if v <= k(87.49823):
                return np.float32(a * np.exp(k(v)))
            w = np.float32(v + source_log(a, self.source))
            return np.float32(1e38) if w > k(87.49823) else np.float32(np.exp(k(w)))
        if v > k(87.49823):
            return np.float32(1e38)
        w = np.float32(np.exp(k(v)))
        return np.float32(1e38) if w > k(1e38) / a else a * w

    def sample(self, stream: DistributionStream) -> np.float32:
        a, b, k = self.a, self.b, self.k
        while True:
            u1, u2 = stream.uniform(), stream.uniform()
            direct = False
            if self.bb_algorithm or u1 >= 0.5:
                z = np.float32(k(u1) * k(u1) * u2)
                if not self.bb_algorithm:
                    direct = bool(z <= 0.25)
                    if not direct and z >= self.k2:
                        continue
            else:
                y = u1 * u2
                z = u1 * y
                if k(0.25) * u2 + z - y >= self.k1:
                    continue
            v = np.float32(self.beta * source_log(u1 / (k(1) - u1), self.source))
            w = self.weight(v)
            if self.bb_algorithm:
                r = np.float32(self.gamma * v - k(1.3862944))
                s = a + r - w
                if s + k(2.609438) >= k(5) * z:
                    direct = True
                else:
                    t = np.float32(source_log(z, self.source))
                    if s > t:
                        direct = True
                if not direct:
                    ratio = self.alpha / (b + w)
                    if ratio < k(1e-37) or r + self.alpha * source_log(ratio, self.source) < t:
                        continue
            elif not direct:
                ratio = self.alpha / (b + w)
                if ratio < k(1e-37):
                    continue
                if self.alpha * (source_log(ratio, self.source) + v) - k(1.3862944) < source_log(
                    z, self.source
                ):
                    continue
            return (b if self.reverse else w) / (b + w)


def sample_beta(
    state: tuple[int, int],
    antithetic: bool,
    size: int,
    a: float,
    b: float,
    legacy: bool,
    source: str,
    budget: int,
) -> tuple[NDArray[np.float64], tuple[int, int]]:
    a, b = scalar(a, "a"), scalar(b, "b")
    if a <= 0 or b <= 0:
        raise ValueError("beta shape parameters must be positive")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        if legacy:
            aa, bb = np.float32(a), np.float32(b)
            k = np.float64 if source == "c" else np.float32
            if not np.isfinite(aa) or not np.isfinite(bb) or aa < k(1e-37) or bb < k(1e-37):
                raise ValueError(
                    "legacy beta parameters must fit float32 and meet source minimum 1e-37"
                )
            stream = DistributionStream(state, antithetic, source, budget)
            sampler = BetaSampler(aa, bb, source)
            result = np.empty(size)
            for i in range(size):
                result[i] = sampler.sample(stream)
            state = stream.state
        else:
            if size > budget:
                raise ArithmeticError(
                    "distribution sampling exceeded max_attempts; state unchanged"
                )
            raw, state = raw_batch(state, size, antithetic)
            result = betaincinv(a, b, raw / _M1)
    if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise ArithmeticError("invalid beta samples; state unchanged")
    result.flags.writeable = False
    return result, state
