"""BLiP standard distribution-plot geometry and optional rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, finite

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dataclass(frozen=True)
class BLiPGroup:
    label: str
    low: float
    high: float
    coordinates: FloatArray
    counts: FloatArray
    heights: FloatArray
    outliers: FloatArray
    missing: int


@dataclass(frozen=True)
class BLiPPlot:
    mode: str
    groups: tuple[BLiPGroup, ...]


def _prepare_groups(
    groups: tuple[ArrayLike, ...],
    labels: tuple[str, ...] | None,
    nclass: int,
    uniform: bool,
    region: tuple[float, float],
) -> tuple[list[FloatArray], list[int], tuple[str, ...], FloatArray]:
    if not 1 <= len(groups) <= 100:
        raise ValueError("provide 1..100 group vectors")
    if not isinstance(uniform, bool):
        raise ValueError("uniform must be boolean")
    if isinstance(nclass, bool) or not isinstance(nclass, int) or not 1 <= nclass <= 1000:
        raise ValueError("nclass must be an integer in 1..1000")
    rr = finite(region, "region")
    if rr.shape != (2,) or not 0 <= rr[0] < rr[1] <= 1:
        raise ValueError("region must be two increasing values in [0,1]")
    if labels is None:
        labels = tuple(str(i + 1) for i in range(len(groups)))
    if len(labels) != len(groups) or any(not isinstance(v, str) for v in labels):
        raise ValueError("labels must contain one string per group")
    data, missing = [], []
    for raw in groups:
        if np.iscomplexobj(raw):
            raise ValueError("BLiP groups must be real")
        x = np.asarray(raw, dtype=float)
        if x.ndim != 1 or not 1 <= x.size <= 1_000_000 or np.isinf(x).any():
            raise ValueError("groups must be vectors with <=1000000 entries and no infinity")
        missing.append(int(np.isnan(x).sum()))
        x = np.sort(x[~np.isnan(x)])
        if not x.size:
            raise ValueError("each group must have at least one observed value")
        data.append(x)
    if sum(x.size for x in data) > 1_000_000:
        raise ValueError("combined nonmissing observations must not exceed 1000000")
    return data, missing, labels, rr


def blip_data(
    *groups: ArrayLike,
    mode: str = "boxplot",
    labels: tuple[str, ...] | None = None,
    nclass: int = 5,
    breaks: ArrayLike | None = None,
    uniform: bool = True,
    region: tuple[float, float] = (0.25, 0.75),
) -> BLiPPlot:
    """Standard BLiP geometry; groups are positional vectors, NaNs are omitted.

    Histogram/polygon bins are right-closed, including both exterior endpoints.
    Boxplot whiskers are interpolated 1.5-IQR fences clipped to sample extrema,
    not snapped to observed values. See blip_custom_data for custom layouts.
    """
    if mode not in ("boxplot", "histogram", "polygon"):
        raise ValueError("mode must be boxplot, histogram or polygon")
    data, missing, labels, rr = _prepare_groups(groups, labels, nclass, uniform, region)
    explicit = None if breaks is None else finite(breaks, "breaks")
    if explicit is not None and (
        explicit.ndim != 1 or not 2 <= explicit.size <= 1001 or np.any(np.diff(explicit) <= 0)
    ):
        raise ValueError("breaks must contain 2..1001 strictly increasing values")
    if mode == "boxplot" and breaks is not None:
        raise ValueError("breaks do not apply to boxplots")
    overall_low, overall_high = min(x[0] for x in data), max(x[-1] for x in data)
    geometry, frequencies, outliers = [], [], []
    for x in data:
        if mode == "boxplot":
            q = np.quantile(x, [0.25, 0.5, 0.75])
            with np.errstate(over="ignore"):
                spread = q[2] - q[0]
                whiskers = [max(x[0], q[0] - 1.5 * spread), min(x[-1], q[2] + 1.5 * spread)]
            coordinates = np.r_[whiskers[0], q, whiskers[1]]
            if not np.isfinite(coordinates).all():
                raise ValueError("boxplot coordinates exceed numerical range")
            geometry.append(coordinates)
            frequencies.append(np.empty(0))
            outliers.append(x[(x < whiskers[0]) | (x > whiskers[1])])
        else:
            lo, hi = (overall_low, overall_high) if uniform else (x[0], x[-1])
            if explicit is None:
                if lo == hi:
                    raise ValueError("constant-range histograms require explicit breaks")
                edge = np.linspace(lo, hi, nclass + 1)
            else:
                edge = explicit
            if x[0] < edge[0] or x[-1] > edge[-1]:
                raise ValueError("breaks must cover every observation")
            if not np.isfinite(edge).all() or np.any(np.diff(edge) <= 0):
                raise ValueError("bin spacing is not numerically resolved")
            if mode == "polygon":
                with np.errstate(over="ignore"):
                    width = edge[1] - edge[0]
                    tips = np.array([edge[0] - width / 2, edge[-1] + width / 2])
                if not np.isfinite(tips).all():
                    raise ValueError("polygon endpoints exceed numerical range")
            index = np.maximum(0, np.searchsorted(edge, x, side="left") - 1)
            geometry.append(edge)
            frequencies.append(np.bincount(index, minlength=edge.size - 1).astype(float))
            outliers.append(np.empty(0))
    peak = max(float(f.max()) for f in frequencies) if mode != "boxplot" else 1
    result = []
    for i, (x, coord, freq, out) in enumerate(
        zip(data, geometry, frequencies, outliers, strict=True)
    ):
        low, high = (i + rr) / len(data)
        if mode == "boxplot":
            height = np.empty(0)
        else:
            denominator = peak if uniform else freq.max()
            height = low + (high - low) * freq / denominator
        result.append(
            BLiPGroup(
                labels[i],
                float(low),
                float(high),
                _freeze(coord),
                _freeze(freq),
                _freeze(height),
                _freeze(out),
                missing[i],
            )
        )
    return BLiPPlot(mode, tuple(result))


def plot_blip(
    data: BLiPPlot,
    *,
    axes: Axes | None = None,
    color: str = "#126e75",
    fill: str = "#d4e9e6",
    count_labels: bool = False,
    xlabel: str = "",
) -> Axes:
    """Render prepared BLiP geometry; install the plot extra. No show/save occurs."""
    from matplotlib import pyplot as plt

    if not isinstance(count_labels, bool):
        raise ValueError("count_labels must be boolean")
    if axes is None:
        _, axes = plt.subplots(layout="constrained")
    if axes.get_xscale() != "linear" or axes.get_yscale() != "linear":
        raise ValueError("BLiP requires linear plotting axes")
    for g in data.groups:
        c = g.coordinates
        middle = (g.low + g.high) / 2
        if data.mode == "boxplot":
            a, q1, median, q3, b = c
            axes.fill([q1, q3, q3, q1], [g.low, g.low, g.high, g.high], color=fill)
            axes.plot([q1, q3, q3, q1, q1], [g.low, g.low, g.high, g.high, g.low], color=color)
            axes.plot([median, median], [g.low, g.high], color=color)
            axes.plot([a, q1, np.nan, q3, b], [middle] * 5, color=color)
            axes.plot(
                g.outliers,
                np.full(g.outliers.size, middle),
                linestyle="None",
                marker="o",
                color=color,
            )
        else:
            midpoint = c[:-1] / 2 + c[1:] / 2
            if data.mode == "histogram":
                for a, b, h in zip(c[:-1], c[1:], g.heights, strict=True):
                    axes.fill([a, a, b, b], [g.low, h, h, g.low], color=fill)
                    axes.plot([a, a, b, b], [g.low, h, h, g.low], color=color)
                axes.plot([c[0], c[-1]], [g.low, g.low], color=color)
            else:
                width = c[1] - c[0]
                xx = np.r_[c[0] - width / 2, midpoint, c[-1] + width / 2]
                yy = np.r_[g.low, g.heights, g.low]
                axes.fill(xx, yy, color=fill)
                axes.plot(xx, yy, color=color)
                axes.plot([xx[0], xx[-1]], [g.low, g.low], color=color)
            if count_labels:
                for x, h, n in zip(midpoint, g.heights, g.counts, strict=True):
                    if n:
                        axes.text(x, h + 0.02, str(int(n)), ha="center", va="bottom", fontsize=8)
    axes.set_yticks(
        [(g.low + g.high) / 2 for g in data.groups], labels=[g.label for g in data.groups]
    )
    axes.set(ylim=(0, 1.08 if count_labels else 1), xlabel=xlabel)
    return axes
