"""SINGLE's reference-design information for eliciting prior correlations."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite
from .single import single_design_precision
from .single_two_sample import single_two_sample_precision


@dataclass(frozen=True)
class SingleDesignCorrelation:
    doses: tuple[FloatArray, ...]
    subjects: tuple[FloatArray, ...]
    covariance: FloatArray
    correlation: FloatArray


def single_design_correlation(
    dose_bounds: ArrayLike,
    parameters: ArrayLike,
    *,
    model: str = "logistic",
    form: str = "linear",
    comparison: str | None = None,
) -> SingleDesignCorrelation:
    """Estimate parameter correlations from SINGLE's DSTCOV reference design.

    Group 1 receives 50 subjects at each interior third of the dose interval.
    If comparison=location/slope, group 2 receives 100 subjects at the midpoint
    (represented as two 50-subject doses, following the source). Parameters are
    actual response-model values, not latent log-normal coordinates. Leading
    parameter axes are supported. A singular reference design raises ValueError.
    """
    bounds = finite(dose_bounds, "dose_bounds")
    if bounds.shape != (2,) or bounds[0] >= bounds[1]:
        raise ValueError("dose_bounds must contain two finite increasing values")
    low, high = bounds
    x = np.array([low * (2 / 3) + high / 3, low / 3 + high * (2 / 3)])
    n = np.array([50.0, 50.0])
    doses: tuple[FloatArray, ...]
    subjects: tuple[FloatArray, ...]
    if comparison is None:
        information = single_design_precision(x, n, parameters, model=model, form=form).information
        doses, subjects = (x,), (n,)
    else:
        midpoint = np.full(2, low / 2 + high / 2)
        information = single_two_sample_precision(
            (x, midpoint), (n, n), parameters, model=model, form=form, comparison=comparison
        ).information
        doses, subjects = (x, midpoint), (n, n.copy())
    identity = np.broadcast_to(np.eye(information.shape[-1]), information.shape)
    factor = np.linalg.cholesky(information)
    solution = np.linalg.solve(factor, identity)
    covariance = np.swapaxes(solution, -1, -2) @ solution
    sd = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    correlation = covariance / (sd[..., :, None] * sd[..., None, :])
    # Normalize the diagonal exactly for use by single_prior_parameters.
    indices = np.arange(information.shape[-1])
    correlation[..., indices, indices] = 1
    if not np.all(np.isfinite(correlation)) or not np.all(np.isfinite(covariance)):
        raise ValueError("Reference-design covariance overflowed")
    return SingleDesignCorrelation(doses, subjects, covariance, correlation)
