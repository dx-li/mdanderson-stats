"""RANDLIB binomial inverse/BTPE sampling; see retained third-party notices."""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom

from ._randlib_distributions import DistributionStream, source_log
from ._randlib_sampling import raw_batch
from ._validation import scalar
from .ranlist_random import _M1


def _integer_power(base: np.float32, exponent: int) -> np.float32:
    result = np.float32(1)
    while exponent:
        if exponent & 1:
            result *= base
        exponent >>= 1
        if exponent:
            base *= base
    return result


class BinomialSampler:
    """Local inverse-CDF/BTPE coefficients and bounded acceptance tests."""

    def __init__(self, n: int, probability: np.float32, source: str) -> None:
        self.n, self.source = n, source
        self.k = np.float64 if source == "c" else np.float32
        k = self.k
        self.reflect = probability > 0.5
        self.p = np.float32(min(probability, k(1) - probability))
        self.q = np.float32(k(1) - self.p)
        self.mean = np.float32(n) * self.p
        self.r = self.p / self.q
        self.g = np.float32(n + 1) * self.r
        if self.mean < 30:
            self.qn = (
                np.float32(np.float64(self.q) ** n) if source == "c" else _integer_power(self.q, n)
            )
            return
        ffm = self.mean + self.p
        self.m = int(ffm)
        self.fm = np.float32(self.m)
        self.variance = self.mean * self.q
        self.p1 = np.float32(int(k(2.195) * np.sqrt(k(self.variance)) - k(4.6) * self.q) + 0.5)
        self.xm = np.float32(self.fm + k(0.5))
        self.xl, self.xr = self.xm - self.p1, self.xm + self.p1
        self.c = np.float32(k(0.134) + k(20.5) / (k(15.3) + self.fm))
        al = (ffm - self.xl) / (ffm - self.xl * self.p)
        self.xll = np.float32(al * (k(1) + k(0.5) * al))
        al = (self.xr - ffm) / (self.xr * self.q)
        self.xlr = np.float32(al * (k(1) + k(0.5) * al))
        self.p2 = np.float32(self.p1 * (k(1) + self.c + self.c))
        self.p3 = self.p2 + self.c / self.xll
        self.p4 = self.p3 + self.c / self.xlr

    def correction(self, x: np.float32) -> np.float32 | np.float64:
        k = self.k
        square = x * x
        return (
            (k(13860) - (k(462) - (k(132) - (k(99) - k(140) / square) / square) / square) / square)
            / x
            / k(166320)
        )

    def accept(self, ix: int, v: np.float32) -> bool:
        distance = abs(ix - self.m)
        k = self.k
        if distance <= 20 or distance >= self.variance / np.float32(2) - np.float32(1):
            if distance > 100000:
                raise ArithmeticError("legacy binomial PMF work limit exceeded; state unchanged")
            density = np.float32(1)
            for i in range(min(ix, self.m) + 1, max(ix, self.m) + 1):
                ratio = self.g / np.float32(i) - self.r
                density = density * ratio if ix > self.m else density / ratio
            return bool(v <= density)
        if self.source == "fortran" and distance > 46340:
            raise ArithmeticError("legacy binomial integer-square overflow; state unchanged")
        amaxp = np.float32(
            (np.float32(distance) / self.variance)
            * (
                (k(distance) * (k(distance) / k(3) + k(0.625)) + k(0.1666666666666)) / self.variance
                + k(0.5)
            )
        )
        ynorm = np.float32(-k(distance * distance) / (k(2) * self.variance))
        alv = np.float32(source_log(v, self.source))
        if alv < ynorm - amaxp:
            return True
        if alv > ynorm + amaxp:
            return False
        x1 = np.float32(ix + 1)
        f1 = np.float32(self.fm + k(1))
        z = np.float32(k(self.n + 1) - self.fm)
        w = np.float32(k(self.n - ix) + k(1))
        log_density = (
            self.xm * source_log(f1 / x1, self.source)
            + (k(self.n - self.m) + k(0.5)) * source_log(z / w, self.source)
            + k(ix - self.m) * source_log(w * self.p / (x1 * self.q), self.source)
            + self.correction(f1)
            + self.correction(z)
            + self.correction(x1)
            + self.correction(w)
        )
        return bool(alv <= log_density)

    def sample(self, stream: DistributionStream) -> int:
        k = self.k
        if self.mean < 30:
            while True:
                ix = 0
                density = self.qn
                u = stream.uniform()
                while True:
                    if u < density:
                        return self.n - ix if self.reflect else ix
                    if ix > 110:
                        break
                    u = u - density
                    ix += 1
                    density = density * (self.g / np.float32(ix) - self.r)
        while True:
            u = stream.uniform() * self.p4
            v = stream.uniform()
            if u <= self.p1:
                ix = int(self.xm - self.p1 * v + u)
                return self.n - ix if self.reflect else ix
            if u <= self.p2:
                x = self.xl + (u - self.p1) / self.c
                v = np.float32(v * self.c + k(1) - abs(self.xm - x) / self.p1)
                if v > 1 or v <= 0:
                    continue
                ix = int(x)
            elif u <= self.p3:
                ix = int(self.xl + source_log(v, self.source) / self.xll)
                if ix < 0:
                    continue
                v = (
                    v * ((u - self.p2) * self.xll)
                    if self.source == "c"
                    else v * (u - self.p2) * self.xll
                )
            else:
                ix = int(self.xr - source_log(v, self.source) / self.xlr)
                if ix > self.n:
                    continue
                v = (
                    v * ((u - self.p3) * self.xlr)
                    if self.source == "c"
                    else v * (u - self.p3) * self.xlr
                )
            if self.accept(ix, v):
                return self.n - ix if self.reflect else ix


def sample_binomial(
    state: tuple[int, int],
    antithetic: bool,
    size: int,
    n: int,
    p: float,
    legacy: bool,
    source: str,
    budget: int,
) -> tuple[NDArray[np.int64], tuple[int, int]]:
    p = scalar(p, "p")
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        if legacy:
            if n > 2147483646:
                raise ValueError("legacy n must be <= 2147483646")
            probability = np.float32(p)
            if p > 0 and probability == 0 or p < 1 and probability == 1:
                raise ValueError("legacy probability must not round to an endpoint")
            stream = DistributionStream(state, antithetic, source, budget)
            sampler = BinomialSampler(n, probability, source)
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
            values = binom.ppf(raw / _M1, n, p)
            if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > n)):
                raise ArithmeticError("invalid binomial quantiles; state unchanged")
            u = raw / _M1
            lower, upper = binom.cdf(values - 1, n, p), binom.cdf(values, n, p)
            tolerance = 64 * np.finfo(float).eps + 1e-10 * np.minimum(u, 1 - u)
            if (
                np.any(values != np.floor(values))
                or not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
                or np.any(lower > u + tolerance)
                or np.any(upper < u - tolerance)
            ):
                raise ArithmeticError("invalid binomial probability bracket; state unchanged")
            result = values.astype(np.int64)
    if np.any((result < 0) | (result > n)):
        raise ArithmeticError("invalid legacy binomial values; state unchanged")
    result.flags.writeable = False
    return result, state
