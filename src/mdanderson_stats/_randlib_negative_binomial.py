"""RANDLIB negative-binomial counts via quantiles or gamma–Poisson mixing."""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import nbinom

from ._randlib_distributions import DistributionStream
from ._randlib_gamma import GammaSampler
from ._randlib_poisson import PoissonSampler
from ._randlib_sampling import raw_batch
from ._validation import scalar
from .ranlist_random import _M1


def sample_negative_binomial(
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
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        if legacy:
            if n > 2147483647:
                raise ValueError("legacy n must be <= 2147483647")
            probability = np.float32(p)
            if not 0 < probability < 1:
                raise ValueError("legacy p must round to a value strictly between zero and one")
            rate = probability / (np.float32(1) - probability)
            gamma = GammaSampler(np.float32(n), source)
            stream = DistributionStream(state, antithetic, source, budget)
            result = np.empty(size, dtype=np.int64)
            for i in range(size):
                mean = gamma.sample(stream) / rate
                if not np.isfinite(mean) or not 0 <= mean < 2147483648:
                    raise ArithmeticError(
                        "legacy negative-binomial Poisson mean overflow; state unchanged"
                    )
                result[i] = PoissonSampler(mean, source).sample(stream)
            state = stream.state
        else:
            if size > budget:
                raise ArithmeticError(
                    "distribution sampling exceeded max_attempts; state unchanged"
                )
            raw, state = raw_batch(state, size, antithetic)
            u = raw / _M1
            values = nbinom.ppf(u, n, p)
            if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 2**53 - 1)):
                raise ArithmeticError("invalid negative-binomial quantiles; state unchanged")
            lower = np.where(u <= 0.5, nbinom.cdf(values - 1, n, p), nbinom.sf(values, n, p))
            upper = np.where(u <= 0.5, nbinom.cdf(values, n, p), nbinom.sf(values - 1, n, p))
            target = np.minimum(u, 1 - u)
            tolerance = 64 * np.finfo(float).eps + 1e-10 * target
            if (
                np.any(values != np.floor(values))
                or not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
                or np.any(lower > target + tolerance)
                or np.any(upper < target - tolerance)
            ):
                raise ArithmeticError(
                    "invalid negative-binomial probability bracket; state unchanged"
                )
            result = values.astype(np.int64)
    result.flags.writeable = False
    return result, state
