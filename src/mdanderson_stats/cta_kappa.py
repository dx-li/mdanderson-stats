"""Cohen agreement with multinomial and independence variances for CTA."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite


@dataclass(frozen=True)
class CohenKappa:
    proportions: FloatArray
    observed_agreement: FloatArray
    chance_agreement: FloatArray
    kappa: FloatArray
    variance: FloatArray
    null_variance: FloatArray
    legacy: bool


def cohen_kappa(observed: ArrayLike, *, legacy: bool = False) -> CohenKappa:
    """Unweighted two-rater kappa on final square table axes; leading axes batch.

    Default variances use the multinomial delta method and independent raters
    with the observed margins. Legacy retains CTA's asymmetric theta4/theta5
    indexing, including potentially negative variances. Kappa is NaN when
    chance disagreement is zero. Finite nonnegative fractional counts are valid.
    """
    ob = finite(observed, "observed")
    if ob.ndim < 2 or ob.shape[-1] != ob.shape[-2] or ob.shape[-1] < 2 or np.any(ob < 0):
        raise ValueError("observed must contain nonnegative square tables of size at least two")
    if not isinstance(legacy, (bool, np.bool_)):
        raise ValueError("legacy must be boolean")
    with np.errstate(over="ignore"):
        n = ob.sum(axis=(-2, -1))
    if np.any(n <= 0) or not np.all(np.isfinite(n)):
        raise ValueError("each table must have a positive finite total")
    p = ob / n[..., None, None]
    rows, cols = p.sum(axis=-1), p.sum(axis=-2)
    independent = rows[..., :, None] * cols[..., None, :]
    eye = np.eye(ob.shape[-1])
    off = 1 - eye
    po = np.asarray(np.trace(p, axis1=-2, axis2=-1))
    pe = np.asarray(np.sum(rows * cols, axis=-1))
    denominator = np.sum(independent * off, axis=(-2, -1))
    disagreement = np.sum(p * off, axis=(-2, -1))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        kappa = np.asarray(1 - disagreement / denominator)
        if legacy:
            diagonal = np.diagonal(p, axis1=-2, axis2=-1)
            t3 = np.sum(diagonal * (rows + cols), axis=-1)
            t4 = np.sum(p * (rows[..., :, None] + cols[..., None, :]) ** 2, axis=(-2, -1))
            t5 = np.sum(cols**2 * (cols + rows), axis=-1)
            variance = np.asarray(
                (
                    po * (1 - po) / denominator**2
                    + 2 * (1 - po) * (2 * po * pe - t3) / denominator**3
                    + (1 - po) ** 2 * (t4 - 4 * pe**2) / denominator**4
                )
                / n
            )
            null_variance = np.asarray((pe + pe**2 - t5) / (n * denominator**2))
        else:
            derivative_pe = cols[..., :, None] + rows[..., None, :]
            gradient = (eye - (1 - kappa[..., None, None]) * derivative_pe) / denominator[
                ..., None, None
            ]
            center = np.sum(p * gradient, axis=(-2, -1), keepdims=True)
            variance = np.asarray(
                np.sum((np.sqrt(p) * (gradient - center)) ** 2, axis=(-2, -1)) / n
            )
            null_influence = (eye - derivative_pe + pe[..., None, None]) / denominator[
                ..., None, None
            ]
            null_variance = np.asarray(
                np.sum(
                    ((np.sqrt(rows)[..., :, None] * np.sqrt(cols)[..., None, :]) * null_influence)
                    ** 2,
                    axis=(-2, -1),
                )
                / n
            )
    undefined = denominator == 0
    kappa = np.where(undefined, np.nan, kappa)
    variance = np.where(undefined, np.nan, variance)
    null_variance = np.where(undefined, np.nan, null_variance)
    for value in (p, po, pe, kappa, variance, null_variance):
        value.flags.writeable = False
    return CohenKappa(p, po, pe, kappa, variance, null_variance, bool(legacy))
