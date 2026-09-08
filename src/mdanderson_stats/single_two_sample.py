"""SINGLE's shared-parameter two-sample fixed-design criteria."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite
from .single import _response_information


@dataclass(frozen=True)
class SingleTwoSamplePrecision:
    information: FloatArray
    probability: tuple[FloatArray, FloatArray]
    difference: FloatArray
    variance: FloatArray

    @property
    def sd(self) -> FloatArray:
        return np.sqrt(self.variance)


def single_two_sample_precision(
    doses: tuple[ArrayLike, ArrayLike],
    subjects: tuple[ArrayLike, ArrayLike],
    parameters: ArrayLike,
    *,
    comparison: str = "location",
    model: str = "logistic",
    form: str = "linear",
) -> SingleTwoSamplePrecision:
    """Precision of group 1 minus group 2, with the other parameter shared.

    Parameters use SINGLE's order: the two group-1 parameters followed by
    the group-2 parameter being compared. Linear form starts (intercept, slope),
    centered form (slope, center). Leading parameter axes are evaluated together.
    Each group has its own nonempty dose/allocation vectors, possibly differing
    in length. Allocations may be fractional. Information must have full rank.
    """
    if model not in ("logistic", "loglog") or form not in ("linear", "centered"):
        raise ValueError("model must be logistic/loglog and form linear/centered")
    if comparison not in ("location", "slope"):
        raise ValueError("comparison must be location or slope")
    if len(doses) != 2 or len(subjects) != 2:
        raise ValueError("Require two groups of doses and allocations")
    b = finite(parameters, "parameters")
    if b.ndim < 1 or b.shape[-1] != 3:
        raise ValueError("parameters must end in three entries")
    compared = 0 if (form == "linear") == (comparison == "location") else 1
    information = np.zeros(b.shape[:-1] + (3, 3))
    probabilities = []
    informative_gradients = []
    for group in range(2):
        x = finite(doses[group], "doses")
        n = finite(subjects[group], "subjects")
        if x.ndim != 1 or not x.size or n.shape != x.shape or np.any(n < 0):
            raise ValueError(
                "Each group requires matching nonempty doses and nonnegative allocations"
            )
        indices = [0, 1]
        if group == 1:
            indices[compared] = 2
        first, second = b[..., indices[0], None], b[..., indices[1], None]
        if form == "linear":
            u = first + second * x
            derivatives = np.broadcast_arrays(np.ones_like(u), x)
        else:
            u = first * (x - second)
            derivatives = np.broadcast_arrays(x - second, -first)
        if not np.all(np.isfinite(u)):
            raise ValueError("Linear predictor overflowed")
        p, weight = _response_information(u, model)
        gradient = np.zeros(u.shape + (3,))
        for index, derivative in zip(indices, derivatives, strict=True):
            gradient[..., index] = derivative
        effective = n * weight
        informative_gradients.append(np.where(effective[..., None] > 0, gradient, 0))
        information += np.swapaxes(gradient, -1, -2) @ (effective[..., None] * gradient)
        probabilities.append(p)
    # Check identifiability explicitly: floating-point Cholesky can occasionally
    # succeed on a structurally singular matrix after accumulation roundoff.
    if np.any(np.linalg.matrix_rank(np.concatenate(informative_gradients, axis=-2)) < 3):
        raise ValueError("Design requires three identifiable parameters across informative doses")
    target = np.zeros(b.shape)
    target[..., compared] = 1
    target[..., 2] = -1
    try:
        factor = np.linalg.cholesky(information)
        solution = np.linalg.solve(factor, target[..., None])[..., 0]
    except np.linalg.LinAlgError as error:
        raise ValueError("Design information is not positive definite") from error
    variance = np.sum(solution**2, axis=-1)
    if not np.all(np.isfinite(variance)):
        raise ValueError("Design precision overflowed")
    return SingleTwoSamplePrecision(
        information, (probabilities[0], probabilities[1]), b[..., compared] - b[..., 2], variance
    )
