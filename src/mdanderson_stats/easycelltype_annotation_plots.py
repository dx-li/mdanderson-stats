"""Optional plots for source-selected EasyCellType annotations.

The functions display already selected Fisher or GSEA labels; they do not
rerank candidates or perform additional inference.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import TYPE_CHECKING

import numpy as np

from .easycelltype import EasyCellTypeLabel
from .easycelltype_gsea_labels import EasyCellTypeGSEALabel

if TYPE_CHECKING:
    from matplotlib.axes import Axes

type AnnotationLabel = EasyCellTypeLabel | EasyCellTypeGSEALabel
_MAX_LABELS = 100
_HARD_COLOR = "#1f77b4"
_SOFT_COLOR = "#ff7f0e"
_PLOT_FLOOR = float(np.nextafter(0.0, 1.0))


def _labels(value: Sequence[AnnotationLabel]) -> tuple[AnnotationLabel, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("labels must be a sequence of Fisher or GSEA label records")
    try:
        size = len(value)
    except TypeError as exc:
        raise TypeError("labels must be a sized sequence") from exc
    if size > _MAX_LABELS:
        raise ValueError(f"labels exceed the {_MAX_LABELS}-row plotting limit")
    rows = tuple(value)
    if len(rows) != size or any(
        not isinstance(x, (EasyCellTypeLabel, EasyCellTypeGSEALabel)) for x in rows
    ):
        raise TypeError("labels must contain only EasyCellType Fisher or GSEA label records")
    if not rows:
        raise ValueError("at least one selected label is required")
    families = {type(row) for row in rows}
    if len(families) != 1:
        raise ValueError("a plot must contain Fisher labels or GSEA labels, not a mixture")
    clusters: set[str] = set()
    for row in rows:
        method = row.method
        expected = "hard_fisher" if isinstance(row, EasyCellTypeLabel) else "hard_enrich"
        soft = "soft_fisher" if isinstance(row, EasyCellTypeLabel) else "soft_enrich"
        if (
            not isinstance(row.cluster, str)
            or not row.cluster
            or not isinstance(row.cell_type, str)
            or not row.cell_type
        ):
            raise ValueError("cluster and cell_type labels must be nonempty strings")
        if method not in (expected, soft):
            raise ValueError(f"unsupported annotation method {method!r}")
        clusters.add(row.cluster)
        pvalue = row.pvalue if isinstance(row, EasyCellTypeLabel) else row.p_value
        adjusted = (
            row.adjusted_pvalue if isinstance(row, EasyCellTypeLabel) else row.adjusted_p_value
        )
        score = (
            row.mean_score
            if isinstance(row, EasyCellTypeLabel)
            else row.normalized_enrichment_score
        )
        if not all(isfinite(float(x)) for x in (pvalue, adjusted, score)) or not (
            0 <= pvalue <= 1 and 0 <= adjusted <= 1
        ):
            raise ValueError("selected labels must contain finite scores and p-values in [0, 1]")
    if len(clusters) > 20:
        raise ValueError("plots support at most 20 clusters")
    return rows


def _pvalue(row: AnnotationLabel, kind: str) -> float:
    if kind not in ("raw", "adjusted"):
        raise ValueError("pvalue must be 'raw' or 'adjusted'")
    if isinstance(row, EasyCellTypeLabel):
        return row.pvalue if kind == "raw" else row.adjusted_pvalue
    return row.p_value if kind == "raw" else row.adjusted_p_value


def _score(row: AnnotationLabel) -> float:
    return row.mean_score if isinstance(row, EasyCellTypeLabel) else row.normalized_enrichment_score


def _core_size(row: AnnotationLabel) -> int:
    genes = row.overlap_genes if isinstance(row, EasyCellTypeLabel) else row.core_enrichment
    return len(genes) if genes is not None else 0


def _axis(ax: Axes | None, *, figsize: tuple[float, float]) -> Axes:
    if ax is not None:
        return ax
    try:
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise ImportError("EasyCellType plots require the optional 'plot' extra") from exc
    _, axis = plt.subplots(figsize=figsize, layout="constrained")
    return axis


def plot_easycelltype_candidates(
    labels: Sequence[AnnotationLabel], *, ax: Axes | None = None
) -> Axes:
    """Plot selected candidates' Fisher mean scores or GSEA NES by cluster.

    Input order is retained. Hard and soft annotations use distinct colors;
    bar lengths are the actual score values and may be negative.
    """
    rows = _labels(labels)
    axis = _axis(ax, figsize=(8, min(18.0, max(3.5, 0.34 * len(rows)))))
    y = np.arange(len(rows))
    scores = np.asarray([_score(row) for row in rows], dtype=float)
    colors = [
        _HARD_COLOR if row.method in ("hard_fisher", "hard_enrich") else _SOFT_COLOR for row in rows
    ]
    axis.barh(y, scores, color=colors)
    axis.set_yticks(y, [f"{row.cluster} / {row.cell_type}" for row in rows])
    axis.invert_yaxis()
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel(
        "Mean marker score"
        if isinstance(rows[0], EasyCellTypeLabel)
        else "Normalized enrichment score (NES)"
    )
    axis.set_ylabel("Cluster / candidate cell type")
    axis.set_title("Selected EasyCellType candidates")
    from matplotlib.patches import Patch

    axis.legend(
        handles=[
            Patch(color=_HARD_COLOR, label="Hard label"),
            Patch(color=_SOFT_COLOR, label="Soft label"),
        ]
    )
    return axis


def plot_easycelltype_annotation_dots(
    labels: Sequence[AnnotationLabel],
    *,
    pvalue: str = "raw",
    ax: Axes | None = None,
) -> Axes:
    """Plot selected cluster/type labels by significance, method and core size.

    X is cluster, Y is candidate cell type. Color is ``-log10`` of the
    explicitly selected raw or adjusted p-value; exact zeros display at the
    finite float64 ceiling and the underlying result is not changed. Marker
    area scales with overlapping Fisher genes or GSEA core-enrichment genes.
    Hard/soft status uses circle/triangle markers.
    """
    rows = _labels(labels)
    if pvalue not in ("raw", "adjusted"):
        raise ValueError("pvalue must be 'raw' or 'adjusted'")
    clusters = tuple(dict.fromkeys(row.cluster for row in rows))
    cell_types = tuple(dict.fromkeys(row.cell_type for row in rows))
    cluster_index = {name: i for i, name in enumerate(clusters)}
    type_index = {name: i for i, name in enumerate(cell_types)}
    probabilities = np.asarray([_pvalue(row, pvalue) for row in rows], dtype=float)
    significance = -np.log10(np.maximum(probabilities, _PLOT_FLOOR))
    sizes = np.asarray([_core_size(row) for row in rows], dtype=float)
    areas = 24.0 + 18.0 * np.sqrt(sizes)
    axis = _axis(
        ax,
        figsize=(
            min(11.0, max(6.0, 0.55 * len(clusters))),
            min(18.0, max(4.0, 0.34 * len(cell_types))),
        ),
    )
    for method, marker in (("hard", "o"), ("soft", "^")):
        selected = np.asarray([row.method.startswith(f"{method}_") for row in rows], dtype=bool)
        if not np.any(selected):
            continue
        axis.scatter(
            [cluster_index[row.cluster] for row, keep in zip(rows, selected, strict=True) if keep],
            [type_index[row.cell_type] for row, keep in zip(rows, selected, strict=True) if keep],
            c=significance[selected],
            s=areas[selected],
            marker=marker,
            cmap="viridis",
            vmin=0.0,
            vmax=max(1.0, float(np.max(significance))),
            edgecolors="black",
            linewidths=0.35,
            label=method.capitalize(),
        )
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    color_max = max(1.0, float(np.max(significance)))
    axis.figure.colorbar(
        ScalarMappable(norm=Normalize(vmin=0.0, vmax=color_max), cmap="viridis"),
        ax=axis,
        label=f"−log10({pvalue} p-value)",
    )
    axis.set_xticks(np.arange(len(clusters)), clusters)
    axis.set_yticks(np.arange(len(cell_types)), cell_types)
    axis.invert_yaxis()
    axis.set_xlabel("Cluster")
    axis.set_ylabel("Candidate cell type")
    axis.set_title(f"Selected annotations colored by −log10({pvalue} p-value)")
    axis.legend(title="Label method")
    return axis
