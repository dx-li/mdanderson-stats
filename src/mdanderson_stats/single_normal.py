"""SINGLE normal/log-normal prior criteria with explicit covariance scaling."""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite
from .single_uniform import _local_criterion


@dataclass(frozen=True)
class SingleNormalCriterion:
    value: float
    parameters: FloatArray
    weights: FloatArray
    local_criterion: FloatArray
    order: int
    legacy_scale: bool


def single_normal_criterion(
    doses: ArrayLike | tuple[ArrayLike, ArrayLike],
    subjects: ArrayLike | tuple[ArrayLike, ArrayLike],
    mean: ArrayLike,
    covariance: ArrayLike,
    *,
    lognormal: ArrayLike | None = None,
    criterion: str = "quantile_sd",
    model: str = "logistic",
    form: str = "linear",
    comparison: str | None = None,
    quantile: float = 0.05,
    order: int = 6,
    legacy_scale: bool = False,
) -> SingleNormalCriterion:
    """Approximate 1/E[1/local criterion] using tensor Hermite quadrature.

    Mean and covariance describe the latent normal vector. Coordinates marked
    by a boolean lognormal vector are exponentiated afterward. Other coordinates
    remain normal. This explicitly avoids SINGLE's approximate conversion from
    raw log-normal moments. Default scaling respects the given covariance;
    legacy_scale=True reproduces RECHRM's quarter-covariance node scaling.

    Criteria and parameter order follow single_uniform_criterion. Covariance
    may be positive semidefinite, including zero for fixed parameters. No
    singular node is discarded. Compare orders to assess quadrature convergence.
    """
    dimension = 2 if comparison is None else 3
    mu, cov = finite(mean, "mean"), finite(covariance, "covariance")
    if mu.shape != (dimension,) or cov.shape != (dimension, dimension):
        raise ValueError(f"Require a {dimension}-entry mean and matching covariance matrix")
    if not np.array_equal(cov, cov.T):
        raise ValueError("covariance must be symmetric")
    if not isinstance(legacy_scale, (bool, np.bool_)):
        raise ValueError("legacy_scale must be boolean")
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)):
        raise ValueError("order must be an integer from 2 to 32")
    if not 2 <= order <= 32:
        raise ValueError("order must be an integer from 2 to 32")
    mask = np.zeros(dimension, dtype=bool) if lognormal is None else np.asarray(lognormal)
    if mask.shape != (dimension,) or mask.dtype != np.bool_:
        raise ValueError("lognormal must be a boolean vector matching the mean")
    allowed = (
        ("slope_sd", "slope_variance", "quantile_sd", "quantile_variance")
        if comparison is None
        else ("sd", "variance")
    )
    if criterion not in allowed:
        raise ValueError(f"criterion must be one of {allowed}")
    q = finite(quantile, "quantile")
    if q.ndim != 0 or not 0 < q < 1:
        raise ValueError("quantile must be a scalar in (0,1)")
    eigenvalues, vectors = np.linalg.eigh(cov)
    tolerance = 32 * np.finfo(float).eps * np.max(np.abs(cov))
    if np.any(eigenvalues < -tolerance):
        raise ValueError("covariance must be positive semidefinite")
    positive = eigenvalues > tolerance
    factor = vectors[:, positive] * np.sqrt(eigenvalues[positive])
    rank = factor.shape[1]
    if rank:
        x, w = hermgauss(order)
        scale = 1 / np.sqrt(2) if legacy_scale else np.sqrt(2)
        grid = np.stack(np.meshgrid(*([x * scale] * rank), indexing="ij"), axis=-1)
        parameters = mu + grid.reshape(-1, rank) @ factor.T
        weights = np.prod(
            np.stack(np.meshgrid(*([w / np.sqrt(np.pi)] * rank), indexing="ij"), axis=-1),
            axis=-1,
        ).ravel()
    else:
        parameters, weights = mu[None, :].copy(), np.ones(1)
    with np.errstate(over="ignore", under="ignore"):
        parameters[:, mask] = np.exp(parameters[:, mask])
    if not np.all(np.isfinite(parameters)) or np.any(parameters[:, mask] == 0):
        raise ValueError("Log-normal parameter transformation overflowed or underflowed")
    local = _local_criterion(
        doses, subjects, parameters, criterion, model, form, comparison, quantile
    )
    if np.any(local <= 0) or not np.all(np.isfinite(local)):
        raise ValueError("Local criteria must be finite and positive")
    minimum = np.min(local)
    value = float(minimum / (weights @ (minimum / local)))
    if not np.isfinite(value):
        raise ValueError("Prior-averaged criterion overflowed")
    return SingleNormalCriterion(value, parameters, weights, local, int(order), bool(legacy_scale))
