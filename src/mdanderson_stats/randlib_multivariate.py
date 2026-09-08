"""Reusable RANDLIB multivariate-normal parameters (SETGMN / GENMN)."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtri

from ._randlib_distributions import DistributionStream
from ._randlib_normal import standard_normal
from ._randlib_sampling import raw_batch
from .ranlist_random import _M1


def _source_dot(x: NDArray[np.float32], y: NDArray[np.float32], source: str) -> np.float32:
    products = x * y
    total = np.float32(0)
    if source == "fortran":
        for value in products:
            total += value
        return total
    prefix = len(x) % 5
    for value in products[:prefix]:
        total += value
    block_size = 5 if source == "c" else max(1, len(x) - prefix)
    for start in range(prefix, len(x), block_size):
        block = np.float32(0)
        for value in products[start : start + block_size]:
            block += value
        total += block
    return total


def _source_cholesky(covariance: NDArray[np.float32], source: str) -> NDArray[np.float32]:
    upper = np.triu(covariance).copy()
    for j in range(len(upper)):
        square_sum = np.float32(0)
        for k in range(j):
            t = (upper[k, j] - _source_dot(upper[:k, k], upper[:k, j], source)) / upper[k, k]
            upper[k, j] = t
            square_sum += t * t
        diagonal = upper[j, j] - square_sum
        if not np.isfinite(diagonal) or diagonal <= 0:
            raise ValueError(f"legacy covariance is not positive definite at leading minor {j + 1}")
        upper[j, j] = np.float32(np.sqrt(np.float64(diagonal)))
    if not np.all(np.isfinite(upper)):
        raise ValueError("legacy Cholesky factor overflow")
    return upper.T.copy()


def _immutable(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class RandlibMultivariateNormal:
    """Immutable mean and lower Cholesky factor, prepared once for repeated draws."""

    mean: NDArray[np.float64]
    factor: NDArray[np.float64]
    legacy: bool
    source: str

    def __init__(
        self,
        mean: ArrayLike,
        covariance: ArrayLike,
        *,
        legacy: bool = False,
        source: Literal["fortran", "fortran95", "c"] = "fortran",
        max_dimension: int = 256,
    ) -> None:
        if not isinstance(legacy, (bool, np.bool_)):
            raise ValueError("legacy must be boolean")
        if not isinstance(source, str) or source not in ("fortran", "fortran95", "c"):
            raise ValueError("source must be fortran, fortran95 or c")
        if not legacy and source != "fortran":
            raise ValueError("source selection requires legacy=True")
        if (
            isinstance(max_dimension, (bool, np.bool_))
            or not isinstance(max_dimension, (int, np.integer))
            or max_dimension < 1
        ):
            raise ValueError("max_dimension must be a positive integer")
        mean_array = np.asarray(mean, dtype=np.float64)
        covariance_array = np.asarray(covariance, dtype=np.float64)
        if mean_array.ndim != 1 or not 1 <= mean_array.size <= max_dimension:
            raise ValueError("mean must be a nonempty vector within max_dimension")
        if covariance_array.shape != (mean_array.size, mean_array.size):
            raise ValueError("covariance must be a square matrix matching mean")
        if not np.all(np.isfinite(mean_array)):
            raise ValueError("mean must be finite")
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            if legacy:
                upper = np.triu(covariance_array)
                upper32, mean32 = upper.astype(np.float32), mean_array.astype(np.float32)
                if (
                    not np.all(np.isfinite(upper32))
                    or not np.all(np.isfinite(mean32))
                    or np.any((upper != 0) & (upper32 == 0))
                    or np.any((mean_array != 0) & (mean32 == 0))
                ):
                    raise ValueError(
                        "legacy upper covariance and mean must fit float32 without underflow"
                    )
                factor = _source_cholesky(upper32, source).astype(np.float64)
                mean_array = mean32.astype(np.float64)
            else:
                if not np.all(np.isfinite(covariance_array)) or not np.array_equal(
                    covariance_array, covariance_array.T
                ):
                    raise ValueError("covariance must be finite and symmetric")
                try:
                    factor = np.linalg.cholesky(covariance_array)
                except np.linalg.LinAlgError as exc:
                    raise ValueError("covariance must be positive definite") from exc
                if not np.all(np.isfinite(factor)):
                    raise ValueError("nonfinite Cholesky factor")
        object.__setattr__(self, "mean", _immutable(mean_array))
        object.__setattr__(self, "factor", _immutable(factor))
        object.__setattr__(self, "legacy", bool(legacy))
        object.__setattr__(self, "source", source)


def sample_multivariate_normal(
    state: tuple[int, int],
    antithetic: bool,
    size: int,
    parameters: RandlibMultivariateNormal,
    budget: int,
) -> tuple[NDArray[np.float64], tuple[int, int]]:
    dimension = len(parameters.mean)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if parameters.legacy:
            stream = DistributionStream(
                state, antithetic, "c" if parameters.source == "c" else "fortran", budget
            )
            result = np.empty((size, dimension), dtype=np.float64)
            factor = parameters.factor.astype(np.float32)
            mean = parameters.mean.astype(np.float32)
            for row in range(size):
                normal = np.array(
                    [standard_normal(stream) for _ in range(dimension)], dtype=np.float32
                )
                for j in range(dimension):
                    total = np.float32(0)
                    for k in range(j + 1):
                        total += factor[j, k] * normal[k]
                    result[row, j] = total + mean[j]
            state = stream.state
        else:
            draws = size * dimension
            if draws > budget:
                raise ArithmeticError(
                    "distribution sampling exceeded max_attempts; state unchanged"
                )
            raw, state = raw_batch(state, draws, antithetic)
            normal = ndtri(raw / _M1).reshape(size, dimension)
            result = normal @ parameters.factor.T + parameters.mean
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("multivariate normal samples overflow; state unchanged")
    result.flags.writeable = False
    return result, state
