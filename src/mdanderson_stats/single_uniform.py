"""SINGLE's arithmetic averaging of criteria under independent uniform priors."""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite
from .single import single_design_precision
from .single_two_sample import single_two_sample_precision


@dataclass(frozen=True)
class SingleUniformCriterion:
    value: float
    parameters: FloatArray
    weights: FloatArray
    local_criterion: FloatArray
    order: int


def single_uniform_criterion(
    doses: ArrayLike | tuple[ArrayLike, ArrayLike],
    subjects: ArrayLike | tuple[ArrayLike, ArrayLike],
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    criterion: str = "quantile_sd",
    model: str = "logistic",
    form: str = "linear",
    comparison: str | None = None,
    quantile: float = 0.05,
    order: int = 6,
) -> SingleUniformCriterion:
    """Tensor Gauss-Legendre average over independent parameter intervals.

    One sample: criterion is slope_sd, slope_variance, quantile_sd or
    quantile_variance. Two samples: specify comparison=location/slope and
    criterion=sd/variance. Parameter order follows the corresponding fixed-design
    API. Equal interval endpoints denote fixed parameters. Order is per varying
    parameter, from 1 to 32; SINGLE uses 6. This is a quadrature approximation,
    not an error-certified integral. Increase order to assess convergence.
    """
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)):
        raise ValueError("order must be an integer from 1 to 32")
    if not 1 <= order <= 32:
        raise ValueError("order must be an integer from 1 to 32")
    count = 2 if comparison is None else 3
    low, high = finite(lower, "lower"), finite(upper, "upper")
    if low.shape != (count,) or high.shape != low.shape or np.any(low > high):
        raise ValueError(f"Require {count} ordered finite parameter intervals")
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
    if comparison is None and criterion.startswith("quantile"):
        slope = 1 if form == "linear" else 0
        if low[slope] <= 0 <= high[slope]:
            raise ValueError("Quantile precision requires a slope interval excluding zero")
    nodes, weights = leggauss(order)
    axes, masses = [], []
    for a, b in zip(low, high, strict=True):
        if a == b:
            axes.append(np.array([a]))
            masses.append(np.array([1.0]))
        else:
            axes.append((1 - nodes) * (a / 2) + (1 + nodes) * (b / 2))
            masses.append(weights / 2)
    parameters = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, count)
    probability = np.prod(np.stack(np.meshgrid(*masses, indexing="ij"), axis=-1), axis=-1).ravel()
    local = _local_criterion(
        doses, subjects, parameters, criterion, model, form, comparison, quantile
    )
    value = float(probability @ local)
    if not np.isfinite(value):
        raise ValueError("Prior-averaged criterion overflowed")
    return SingleUniformCriterion(value, parameters, probability, local, int(order))


def _local_criterion(
    doses: ArrayLike | tuple[ArrayLike, ArrayLike],
    subjects: ArrayLike | tuple[ArrayLike, ArrayLike],
    parameters: FloatArray,
    criterion: str,
    model: str,
    form: str,
    comparison: str | None,
    quantile: float,
) -> FloatArray:
    if comparison is None:
        result = single_design_precision(
            np.asarray(doses, dtype=float),
            np.asarray(subjects, dtype=float),
            parameters,
            model=model,
            form=form,
            quantile=quantile,
        )
        local = getattr(result, criterion)
    else:
        if not isinstance(doses, (tuple, list, np.ndarray)) or not isinstance(
            subjects, (tuple, list, np.ndarray)
        ):
            raise ValueError("Require two groups of doses and allocations")
        if len(doses) != 2 or len(subjects) != 2:
            raise ValueError("Require two groups of doses and allocations")
        two_sample = single_two_sample_precision(
            (doses[0], doses[1]),
            (subjects[0], subjects[1]),
            parameters,
            model=model,
            form=form,
            comparison=comparison,
        )
        local = getattr(two_sample, criterion)
    return local
