"""RANDLIB multinomial counts through conditional binomial sampling."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import binom

from ._randlib_binomial import BinomialSampler
from ._randlib_distributions import DistributionStream
from ._randlib_sampling import raw_batch
from .ranlist_random import _M1


def probabilities(p: ArrayLike, max_categories: int) -> NDArray[np.float64]:
    values = np.asarray(p, dtype=np.float64)
    if values.ndim != 1 or not 1 <= values.size <= max_categories:
        raise ValueError("p must be a nonempty vector with at most max_draws categories")
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p must contain finite probabilities in [0, 1]")
    if abs(float(values.sum()) - 1) > 1e-12:
        raise ValueError("p must sum to one within 1e-12")
    return values


def sample_multinomial(
    state: tuple[int, int],
    antithetic: bool,
    size: int,
    n: int,
    p: NDArray[np.float64],
    legacy: bool,
    source: str,
    budget: int,
) -> tuple[NDArray[np.int64], tuple[int, int]]:
    categories = len(p)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        result = np.zeros((size, categories), dtype=np.int64)
        if legacy:
            if n > 2147483646 or categories < 2:
                raise ValueError(
                    "legacy multinomial requires n <= 2147483646 and at least 2 categories"
                )
            source_p = p[:-1].astype(np.float32)
            if np.any((p[:-1] > 0) & (source_p == 0)):
                raise ValueError("legacy probabilities must not underflow float32")
            total = np.float32(0)
            for probability in source_p:
                total += probability
            if total > np.float32(0.99999):
                raise ValueError("legacy first K-1 probabilities must sum to <= float32(0.99999)")
            tail = np.float32(1)
            conditional = np.empty(categories - 1, dtype=np.float32)
            for j, probability in enumerate(source_p):
                conditional[j] = probability / tail
                tail -= probability
            if np.any((conditional < 0) | (conditional > 1)) or not np.all(
                np.isfinite(conditional)
            ):
                raise ArithmeticError("invalid legacy conditional probabilities; state unchanged")
            stream = DistributionStream(state, antithetic, source, budget)
            for i in range(size):
                remaining = n
                for j, probability in enumerate(conditional):
                    draw = BinomialSampler(remaining, probability, source).sample(stream)
                    if not 0 <= draw <= remaining:
                        raise ArithmeticError("invalid legacy multinomial count; state unchanged")
                    result[i, j] = draw
                    remaining -= draw
                    if remaining == 0:
                        break
                result[i, -1] = remaining
            state = stream.state
        else:
            draws = size * (categories - 1)
            if draws > budget:
                raise ArithmeticError(
                    "distribution sampling exceeded max_attempts; state unchanged"
                )
            raw, state = raw_batch(state, draws, antithetic)
            uniform = (raw / _M1).reshape(size, categories - 1)
            remaining_counts = np.full(size, n, dtype=np.int64)
            # Reverse sums retain small tails instead of subtracting them from one.
            tails = np.cumsum(p[::-1])[::-1]
            for j in range(categories - 1):
                probability = p[j] / tails[j] if tails[j] > 0 else 0.0
                u = uniform[:, j]
                values = binom.ppf(u, remaining_counts, probability)
                if (
                    not np.all(np.isfinite(values))
                    or np.any(values != np.floor(values))
                    or np.any((values < 0) | (values > remaining_counts))
                ):
                    raise ArithmeticError(
                        "invalid multinomial conditional quantiles; state unchanged"
                    )
                lower = binom.cdf(values - 1, remaining_counts, probability)
                upper = binom.cdf(values, remaining_counts, probability)
                tolerance = 64 * np.finfo(float).eps + 1e-10 * np.minimum(u, 1 - u)
                if (
                    not np.all(np.isfinite(lower))
                    or not np.all(np.isfinite(upper))
                    or np.any(lower > u + tolerance)
                    or np.any(upper < u - tolerance)
                ):
                    raise ArithmeticError(
                        "invalid multinomial probability bracket; state unchanged"
                    )
                result[:, j] = values.astype(np.int64)
                remaining_counts -= result[:, j]
            result[:, -1] = remaining_counts
    result.flags.writeable = False
    return result, state
