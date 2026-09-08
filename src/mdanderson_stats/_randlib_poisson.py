"""RANDLIB Poisson inversion and modified-normal rejection sampling."""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import poisson

from ._randlib_binomial import _integer_power
from ._randlib_distributions import DistributionStream, source_exp, source_log
from ._randlib_normal import standard_normal
from ._randlib_sampling import raw_batch
from ._validation import scalar
from .ranlist_random import _M1

_A = np.array(
    [-0.5, 0.3333333, -0.2500068, 0.2000118, -0.1661269, 0.1421878, -0.1384794, 0.1250060],
    dtype=np.float32,
)
_FACT = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880], dtype=np.float32)
_A.flags.writeable = _FACT.flags.writeable = False


class PoissonSampler:
    """Per-request source coefficients, with no shared mutable table cache."""

    def __init__(self, mu: np.float32, source: str) -> None:
        self.mu, self.source = mu, source
        self.k = np.float64 if source == "c" else np.float32
        k = self.k
        if mu < 10:
            self.table = np.empty(36, dtype=np.float32)
            p = q = np.float32(source_exp(-mu, source))
            self.table[0] = q
            for i in range(1, 36):
                p = p * mu / np.float32(i)
                q = q + p
                self.table[i] = q
            return
        self.s = np.float32(np.sqrt(k(mu)))
        self.d = np.float32(k(6) * mu * mu)
        self.ll = int(mu - k(1.1484))
        self.omega = np.float32(k(0.3989423) / self.s)
        b1 = np.float32(k(0.04166667) / mu)
        b2 = np.float32(k(0.3) * b1 * b1)
        self.c3 = np.float32(k(0.1428571) * b1 * b2)
        self.c2 = np.float32(b2 - k(15) * self.c3)
        self.c1 = np.float32(b1 - k(6) * b2 + k(45) * self.c3)
        self.c0 = np.float32(k(1) - b1 + k(3) * b2 - k(15) * self.c3)
        self.c = np.float32(k(0.1069) / mu)

    def densities(self, count: int) -> tuple[np.float32, np.float32, np.float32, np.float32]:
        k, mu = self.k, self.mu
        fk = np.float32(count)
        difference = mu - fk
        if count < 10:
            px = -mu
            py = np.float32(
                np.float64(mu) ** count / _FACT[count]
                if self.source == "c"
                else _integer_power(mu, count) / _FACT[count]
            )
        else:
            delta = np.float32(k(0.08333333) / fk)
            delta = np.float32(delta - k(4.8) * delta * delta * delta)
            v = difference / fk
            if abs(v) > 0.25:
                px = np.float32(fk * source_log(k(1) + v, self.source) - difference - delta)
            else:
                polynomial = _A[-1]
                for coefficient in _A[-2::-1]:
                    polynomial = polynomial * v + coefficient
                px = fk * v * v * polynomial - delta
            py = np.float32(k(0.3989423) / np.sqrt(k(fk)))
        x = np.float32((k(0.5) - difference) / self.s)
        xx = x * x
        fx = np.float32(-k(0.5) * xx)
        fy = self.omega * (((self.c3 * xx + self.c2) * xx + self.c1) * xx + self.c0)
        return px, py, fx, fy

    @staticmethod
    def count(value: np.float32) -> int:
        if not np.isfinite(value) or value < 0 or value >= 2147483648:
            raise ArithmeticError("legacy Poisson integer overflow; state unchanged")
        return int(value)

    def sample(self, stream: DistributionStream) -> int:
        if self.mu < 10:
            while True:
                u = stream.uniform()
                index = int(np.searchsorted(self.table, u, side="left"))
                if index < len(self.table):
                    return index
        k = self.k
        g = self.mu + self.s * standard_normal(stream)
        if g >= 0:
            count = self.count(g)
            if count >= self.ll:
                return count
            difference = self.mu - np.float32(count)
            u = stream.uniform()
            if self.d * u >= difference * difference * difference:
                return count
            px, py, fx, fy = self.densities(count)
            if fy - u * fy <= py * source_exp(px - fx, self.source):
                return count
        while True:
            e = stream.standard_exponential()
            u = stream.uniform()
            u = np.float32(u + (u - k(1)) if self.source == "c" else u + u - k(1))
            t = np.float32(k(1.8) + (e if u >= 0 else -e))
            if t <= -k(0.6744):
                continue
            count = self.count(self.mu + self.s * t)
            px, py, fx, fy = self.densities(count)
            if self.c * np.abs(k(u)) <= py * source_exp(px + e, self.source) - fy * source_exp(
                fx + e, self.source
            ):
                return count


def sample_poisson(
    state: tuple[int, int],
    antithetic: bool,
    size: int,
    mu: float,
    legacy: bool,
    source: str,
    budget: int,
) -> tuple[NDArray[np.int64], tuple[int, int]]:
    mu = scalar(mu, "mu")
    if not 0 <= mu <= 2**53 - 1:
        raise ValueError("mu must be in [0, 2**53 - 1]")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        if legacy:
            mean = np.float32(mu)
            if mean >= 2147483648 or (mu > 0 and mean == 0):
                raise ValueError("legacy mu must round to a value in [0, 2**31) without underflow")
            stream = DistributionStream(state, antithetic, source, budget)
            sampler = PoissonSampler(mean, source)
            result = np.empty(size, dtype=np.int64)
            for i in range(size):
                result[i] = sampler.sample(stream)
            state = stream.state
        else:
            if size > budget:
                raise ArithmeticError(
                    "distribution sampling exceeded max_attempts; state unchanged"
                )
            raw, state = raw_batch(state, size, antithetic)
            u = raw / _M1
            values = poisson.ppf(u, mu)
            if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 2**53 - 1)):
                raise ArithmeticError("invalid Poisson quantiles; state unchanged")
            lower = np.where(u <= 0.5, poisson.cdf(values - 1, mu), poisson.sf(values, mu))
            upper = np.where(u <= 0.5, poisson.cdf(values, mu), poisson.sf(values - 1, mu))
            target = np.minimum(u, 1 - u)
            tolerance = 64 * np.finfo(float).eps + 1e-10 * target
            if (
                np.any(values != np.floor(values))
                or not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
                or np.any(lower > target + tolerance)
                or np.any(upper < target - tolerance)
            ):
                raise ArithmeticError("invalid Poisson probability bracket; state unchanged")
            result = values.astype(np.int64)
    result.flags.writeable = False
    return result, state
