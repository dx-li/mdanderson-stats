"""Optional side-by-side original and adjusted PCoA plots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist
from scipy.stats import f

from .apcoa import AdjustedPCoA

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dataclass(frozen=True)
class PCoAGroupPlotGeometry:
    """Optional native-style plot geometry for one group.

    Medoid indices use zero-based sample positions. Ellipse arrays contain the
    52 closed vertices used by the recovered ``car::dataEllipse`` convention;
    ``None`` means that feature was not requested.
    """

    label: str
    sample_indices: tuple[int, ...]
    original_medoid_index: int | None
    adjusted_medoid_index: int | None
    original_ellipse: NDArray[np.float64] | None
    adjusted_ellipse: NDArray[np.float64] | None
    original_ellipse_rank: int | None
    adjusted_ellipse_rank: int | None


@dataclass(frozen=True)
class AdjustedPCoAPlotGeometry:
    """Reusable optional overlay geometry and distance-recovery diagnostics."""

    groups: tuple[PCoAGroupPlotGeometry, ...]
    clipped_original_distance_cells: int


_ELLIPSE_SEGMENTS = 51
_MAX_MEDOID_PROFILE_CELLS = 4_000_000
_MAX_MEDOID_WORK = 50_000_000


def _groups(labels: ArrayLike | None, n: int) -> tuple[NDArray[np.str_], tuple[str, ...]]:
    values = np.full(n, "All samples") if labels is None else np.asarray(labels)
    if values.shape != (n,) or not all(isinstance(value, str) and value for value in values):
        raise ValueError("groups must be one nonempty text label per sample")
    text_values = values.astype(str, copy=False)
    order = tuple(dict.fromkeys(text_values.tolist()))
    return text_values, order


def _profile_medoid(profile: NDArray[np.float64]) -> int:
    """Return a one-medoid row by Euclidean row-profile distances.

    The source calls ``cluster::pam(profile, 1)`` with a matrix, so rows are
    objects and the columns are their features. The k=1 BUILD result minimizes
    total pairwise Euclidean distance; its ``<=`` gain update selects the last
    exactly tied candidate. Only one group-by-group difference matrix is live.
    """
    pairwise = cdist(profile, profile, metric="euclidean")
    baseline = 1.1 * float(np.max(pairwise)) + 1.0
    gains = np.empty(profile.shape[0])
    for candidate in range(profile.shape[0]):
        gains[candidate] = np.cumsum(baseline - pairwise[candidate])[-1]
    best, best_gain = 0, -np.inf
    for candidate, gain in enumerate(gains):
        if gain >= best_gain:
            best, best_gain = candidate, float(gain)
    return best


def _original_distance_profile(
    scaled_gram: NDArray[np.float64], indices: NDArray[np.intp]
) -> tuple[NDArray[np.float64], int]:
    gram = scaled_gram[np.ix_(indices, indices)]
    diagonal = np.diag(gram)
    squared = gram.copy()
    squared *= -2.0
    squared += diagonal[:, None]
    squared += diagonal[None, :]
    gram_scale = float(np.max(np.abs(diagonal)))
    tolerance = 64 * np.finfo(float).eps * gram_scale
    if np.any(squared < -tolerance):
        raise ArithmeticError(
            "original distances could not be recovered from the centered Gram matrix"
        )
    clipped = int(np.count_nonzero(squared < 0))
    np.maximum(squared, 0.0, out=squared)
    np.sqrt(squared, out=squared)
    return squared, clipped


def _ellipse(
    coordinates: NDArray[np.float64], label: str, *, confidence: float = 0.95
) -> tuple[NDArray[np.float64], int]:
    count = len(coordinates)
    if count < 2:
        raise ValueError(f"ellipse for group {label!r} requires at least two samples")
    coordinate_scale = float(np.max(np.abs(coordinates)))
    if coordinate_scale == 0:
        raise ValueError(f"ellipse for group {label!r} is undefined for zero covariance")
    normalized = coordinates / coordinate_scale
    center_normalized = normalized.mean(axis=0)
    centered = normalized - center_normalized
    spread = float(np.max(np.abs(centered)))
    if spread == 0:
        raise ValueError(f"ellipse for group {label!r} is undefined for zero covariance")
    centered /= spread
    covariance = centered.T @ centered / (count - 1)
    pivot = np.array([0, 1]) if covariance[0, 0] >= covariance[1, 1] else np.array([1, 0])
    permuted = covariance[np.ix_(pivot, pivot)]
    pivot_tolerance = 2.0 * np.finfo(float).eps * float(np.max(np.diag(permuted)))
    if permuted[0, 0] <= 0:
        raise ArithmeticError(
            f"ellipse covariance for group {label!r} is not positive semidefinite"
        )
    upper = np.zeros((2, 2))
    upper[0, 0] = np.sqrt(permuted[0, 0])
    upper[0, 1] = permuted[0, 1] / upper[0, 0]
    last_pivot = permuted[1, 1] - upper[0, 1] ** 2
    if last_pivot < -pivot_tolerance:
        raise ArithmeticError(
            f"ellipse covariance for group {label!r} is not positive semidefinite"
        )
    rank = 1 + int(last_pivot > pivot_tolerance)
    upper[1, 1] = np.sqrt(last_pivot) if rank == 2 else 0.0
    upper = upper[:, np.argsort(pivot)]
    radius = float(np.sqrt(2.0 * f.ppf(confidence, 2, count - 1)))
    angles = np.arange(_ELLIPSE_SEGMENTS + 1) * (2.0 * np.pi / _ELLIPSE_SEGMENTS)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    with np.errstate(over="ignore", invalid="ignore"):
        vertices = (center_normalized + (radius * spread) * (unit @ upper)) * coordinate_scale
    if not np.isfinite(vertices).all():
        raise ArithmeticError(f"ellipse for group {label!r} exceeds floating-point range")
    vertices.setflags(write=False)
    return vertices, rank


def adjusted_pcoa_plot_geometry(
    result: AdjustedPCoA,
    groups: ArrayLike | None = None,
    *,
    show_ellipses: bool = False,
    show_medoid_connectors: bool = False,
) -> AdjustedPCoAPlotGeometry:
    """Return optional aPCoA ellipse and medoid geometry without plotting.

    The ellipses use the contemporaneous ``car::dataEllipse`` 95% convention:
    group mean, sample covariance, and ``sqrt(2*F(0.95; 2, n_group-1))`` radius.
    Rank-one covariance is represented as a collapsed line; zero covariance is
    undefined and raises. Medoids use the source's row-profile PAM objective.
    """
    if not isinstance(result, AdjustedPCoA):
        raise TypeError("result must be AdjustedPCoA")
    if not isinstance(show_ellipses, bool) or not isinstance(show_medoid_connectors, bool):
        raise ValueError("overlay switches must be boolean")
    n = result.original.coordinates.shape[0]
    label_values, label_order = _groups(groups, n)
    if min(result.original.coordinates.shape[1], result.adjusted.coordinates.shape[1]) < 2:
        raise ValueError("both ordinations require at least two positive coordinate axes")
    group_indices = tuple(np.flatnonzero(label_values == label) for label in label_order)
    if show_medoid_connectors:
        if n * n > _MAX_MEDOID_PROFILE_CELLS:
            raise ValueError("medoid row-profile matrices exceed the 4,000,000-cell limit")
        work = 2 * sum(int(indices.size) ** 3 for indices in group_indices)
        if work > _MAX_MEDOID_WORK:
            raise ValueError("medoid computation exceeds the 50,000,000-operation work limit")

    geometry: list[PCoAGroupPlotGeometry] = []
    clipped_count = 0
    for label, indices in zip(label_order, group_indices, strict=True):
        original_medoid = adjusted_medoid = None
        if show_medoid_connectors:
            original_profile, clipped = _original_distance_profile(
                result.original.scaled_gram_matrix, indices
            )
            clipped_count += clipped
            original_medoid = int(indices[_profile_medoid(original_profile)])
            del original_profile
            adjusted_profile = result.adjusted.scaled_gram_matrix[np.ix_(indices, indices)]
            adjusted_medoid = int(indices[_profile_medoid(adjusted_profile)])
            del adjusted_profile
        original_ellipse = adjusted_ellipse = None
        original_rank = adjusted_rank = None
        if show_ellipses:
            original_ellipse, original_rank = _ellipse(
                result.original.coordinates[indices, :2], label
            )
            adjusted_ellipse, adjusted_rank = _ellipse(
                result.adjusted.coordinates[indices, :2], label
            )
        geometry.append(
            PCoAGroupPlotGeometry(
                label,
                tuple(int(i) for i in indices),
                original_medoid,
                adjusted_medoid,
                original_ellipse,
                adjusted_ellipse,
                original_rank,
                adjusted_rank,
            )
        )
    return AdjustedPCoAPlotGeometry(tuple(geometry), clipped_count)


def plot_adjusted_pcoa(
    result: AdjustedPCoA,
    groups: ArrayLike | None = None,
    *,
    show_ellipses: bool = False,
    show_medoid_connectors: bool = False,
) -> tuple[Axes, Axes]:
    """Plot the first two positive axes, colored by supplied text group labels.

    Requires the plot extra; returns axes without showing/saving the figure.
    Axis percentages use positive inertia, not the possibly negative signed trace.
    Optional native-style ellipses and medoid connectors are computed by
    :func:`adjusted_pcoa_plot_geometry`; both overlays are off by default.
    """
    if not isinstance(result, AdjustedPCoA):
        raise TypeError("result must be AdjustedPCoA")
    if not isinstance(show_ellipses, bool) or not isinstance(show_medoid_connectors, bool):
        raise ValueError("overlay switches must be boolean")
    if min(result.original.coordinates.shape[1], result.adjusted.coordinates.shape[1]) < 2:
        raise ValueError("both ordinations require at least two positive coordinate axes")
    n = result.original.coordinates.shape[0]
    labels, group_order = _groups(groups, n)
    geometry = (
        adjusted_pcoa_plot_geometry(
            result,
            labels,
            show_ellipses=show_ellipses,
            show_medoid_connectors=show_medoid_connectors,
        )
        if show_ellipses or show_medoid_connectors
        else None
    )
    if geometry is None:
        # Retain the existing default plotting order; opt-in geometry follows
        # the native source's first-occurrence group traversal.
        group_order = tuple(np.unique(labels))
    else:
        group_order = tuple(group.label for group in geometry.groups)
    from matplotlib import pyplot as plt

    _, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for panel, (ax, fit, title) in enumerate(
        zip(
            axes,
            [result.original, result.adjusted],
            ["Original PCoA", "Adjusted PCoA"],
            strict=True,
        )
    ):
        for group_index, label in enumerate(group_order):
            selected = labels == label
            points = fit.coordinates[selected, :2]
            collection = ax.scatter(points[:, 0], points[:, 1], label=label)
            color = collection.get_facecolor()[0]
            if geometry is not None:
                group_geometry = geometry.groups[group_index]
                ellipse = (
                    group_geometry.original_ellipse
                    if panel == 0
                    else group_geometry.adjusted_ellipse
                )
                medoid = (
                    group_geometry.original_medoid_index
                    if panel == 0
                    else group_geometry.adjusted_medoid_index
                )
                if ellipse is not None:
                    ax.plot(ellipse[:, 0], ellipse[:, 1], color=color, linewidth=1.0)
                if medoid is not None:
                    medoid_point = fit.coordinates[medoid, :2]
                    segments_x = np.vstack((np.full(len(points), medoid_point[0]), points[:, 0]))
                    segments_y = np.vstack((np.full(len(points), medoid_point[1]), points[:, 1]))
                    ax.plot(segments_x, segments_y, color=color, linewidth=0.8, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(f"Axis 1 ({100 * fit.positive_fraction[0]:.1f}% positive inertia)")
        ax.set_ylabel(f"Axis 2 ({100 * fit.positive_fraction[1]:.1f}% positive inertia)")
        ax.set_aspect("equal", adjustable="datalim")
    axes[1].legend()
    return axes[0], axes[1]
