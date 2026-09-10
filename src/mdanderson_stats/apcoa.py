"""Covariate-adjusted principal coordinates from distance matrices (aPCoA)."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite, scalar
from .bayesian_monitoring import _integer
from .boin import _owned


def _squared_units(values: FloatArray, scale: float) -> FloatArray:
    with np.errstate(over="ignore", under="ignore"):
        result = (values * scale) * scale
    if not np.isfinite(result).all() or np.any((values != 0) & (result == 0)):
        raise ArithmeticError("squared distance units are unrepresentable; use scaled values")
    return _owned(result)


@dataclass(frozen=True)
class PCoAOrdination:
    coordinates: FloatArray
    scaled_eigenvalues: FloatArray
    eigenvectors: FloatArray
    scaled_gram_matrix: FloatArray
    distance_scale: float
    positive_fraction: FloatArray
    negative_inertia_fraction: float

    @property
    def eigenvalues(self) -> FloatArray:
        """Full signed spectrum in squared distance units; scaled values always available."""
        return _squared_units(self.scaled_eigenvalues, self.distance_scale)

    @property
    def gram_matrix(self) -> FloatArray:
        return _squared_units(self.scaled_gram_matrix, self.distance_scale)


@dataclass(frozen=True)
class AdjustedPCoA:
    original: PCoAOrdination
    adjusted: PCoAOrdination
    covariate_rank: int
    intercept: bool
    eigenvalue_tolerance: float


def _ordination(
    gram: FloatArray, scale: float, components: int | None, tolerance: float
) -> PCoAOrdination:
    values, vectors = np.linalg.eigh((gram + gram.T) * 0.5)
    values, vectors = values[::-1].copy(), vectors[:, ::-1].copy()
    values[abs(values) <= tolerance] = 0
    # Fix arbitrary signs for repeatable display; repeated eigenspaces may rotate.
    pivots = np.argmax(abs(vectors), axis=0)
    vectors *= np.where(vectors[pivots, np.arange(values.size)] < 0, -1, 1)
    positive = np.flatnonzero(values > 0)
    selected = positive if components is None else positive[:components]
    coordinates = (vectors[:, selected] * np.sqrt(values[selected])) * scale
    if not np.isfinite(coordinates).all():
        raise ArithmeticError("coordinates overflow; rescale distances")
    total_positive, total_absolute = values[positive].sum(), abs(values).sum()
    fraction = values[selected] / total_positive if total_positive else np.zeros(selected.size)
    negative = float(-values[values < 0].sum() / total_absolute) if total_absolute else 0.0
    return PCoAOrdination(
        _owned(coordinates),
        _owned(values),
        _owned(vectors),
        _owned(gram),
        scale,
        _owned(fraction),
        negative,
    )


def adjusted_pcoa(
    distances: ArrayLike,
    covariates: ArrayLike | None = None,
    *,
    components: int | None = 2,
    intercept: bool = False,
    eigenvalue_tolerance: float = 1e-12,
) -> AdjustedPCoA:
    """Compute G=-J D² J/2 and adjusted G=(I-H)G(I-H), H projecting onto X.

    Covariates are an explicitly encoded numeric n-by-p matrix, not centered
    automatically. intercept=True adds a constant column. Dependent columns are
    handled by rank-revealing SVD after column scaling, without normal equations.
    Positive eigenvectors define coordinates; negative eigenvalues remain reported.
    Tolerance is relative to the original centered Gram Frobenius norm.
    """
    d = finite(distances, "distances")
    if d.ndim != 2 or d.shape[0] != d.shape[1] or not 2 <= d.shape[0] <= 2000:
        raise ValueError("distances must be a square matrix with 2..2000 samples")
    if np.any(d < 0):
        raise ValueError("distances must be nonnegative")
    n = len(d)
    if components is not None:
        components = _integer(components, "components")
        if not 1 <= components <= n:
            raise ValueError("components must be in [1,n] or None for all positive axes")
    if not isinstance(intercept, bool):
        raise ValueError("intercept must be boolean")
    tolerance = scalar(eigenvalue_tolerance, "eigenvalue_tolerance")
    if not 0 < tolerance < 1:
        raise ValueError("eigenvalue_tolerance must be in (0,1)")
    scale = float(d.max())
    scaled = d / scale if scale else d.copy()
    if np.max(abs(scaled - scaled.T)) > 1e-12 or np.max(abs(np.diag(scaled))) > 1e-12:
        raise ValueError(
            "distances must be symmetric with zero diagonal (relative tolerance 1e-12)"
        )
    scaled = (scaled + scaled.T) * 0.5
    np.fill_diagonal(scaled, 0)
    a = -0.5 * scaled**2
    row_mean = a.mean(axis=1)
    gram = a - row_mean[:, None] - row_mean[None, :] + a.mean()
    cutoff = tolerance * float(np.linalg.norm(gram))
    x = np.empty((n, 0)) if covariates is None else finite(covariates, "covariates")
    if x.ndim != 2 or x.shape[0] != n or x.shape[1] > n:
        raise ValueError("covariates must be a matching n-by-p matrix with p<=n")
    if intercept:
        x = np.column_stack((np.ones(n), x))
    rank = 0
    adjusted = gram.copy()
    if x.shape[1]:
        column_scale = abs(x).max(axis=0)
        nonzero = column_scale > 0
        normalized = x[:, nonzero] / column_scale[nonzero]
        if normalized.shape[1]:
            u, singular, _ = np.linalg.svd(normalized, full_matrices=False)
            rank = int(np.sum(singular > np.finfo(float).eps * max(normalized.shape) * singular[0]))
            q = u[:, :rank]
            # Two low-rank matrix products, without constructing an n-by-n hat matrix.
            residual = gram - q @ (q.T @ gram)
            adjusted = residual - (residual @ q) @ q.T
            adjusted = (adjusted + adjusted.T) * 0.5
    original = _ordination(gram, scale, components, cutoff)
    return AdjustedPCoA(
        original,
        _ordination(adjusted, scale, components, cutoff) if rank else original,
        rank,
        intercept,
        tolerance,
    )
