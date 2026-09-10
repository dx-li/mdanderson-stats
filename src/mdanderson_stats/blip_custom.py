"""BLiP custom percentile boxes, line overlays and point layout geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .blip import _prepare_groups

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dataclass(frozen=True)
class BLiPBox:
    probabilities: FloatArray
    coordinates: FloatArray
    lower: FloatArray
    upper: FloatArray


@dataclass(frozen=True)
class BLiPCustomGroup:
    label: str
    low: float
    high: float
    baseline: float
    boxes: tuple[BLiPBox, ...]
    lines: tuple[FloatArray, ...]
    mean: float | None
    points: FloatArray
    point_lower: FloatArray
    point_upper: FloatArray
    missing: int


@dataclass(frozen=True)
class BLiPCustomPlot:
    groups: tuple[BLiPCustomGroup, ...]
    point_pattern: str


def _percentiles(parts: tuple[ArrayLike, ...], name: str) -> list[FloatArray]:
    if len(parts) > 100:
        raise ValueError(f"{name} allows at most 100 pieces")
    result = []
    for part in parts:
        p = finite(part, name)
        if (
            p.ndim != 1
            or not (1 if name == "boxes" else 2) <= p.size <= 1000
            or np.any(np.diff(p) <= 0)
            or p[0] < 0
            or p[-1] > 1
        ):
            raise ValueError(f"{name} pieces need increasing probabilities in [0,1]")
        result.append(p)
    return result


def _mass(x: FloatArray, q: FloatArray, delta: float) -> FloatArray:
    """Source's interpolated right-rank CDF, with exterior values zero/one."""
    unique, counts = np.unique(x, return_counts=True)
    cdf = np.cumsum(counts) / x.size
    return np.interp(q + delta, unique, cdf, left=0, right=1) - np.interp(
        q - delta, unique, cdf, left=0, right=1
    )


def blip_custom_data(
    *groups: ArrayLike,
    boxes: tuple[ArrayLike, ...] = ((0.25, 0.5, 0.75),),
    lines: tuple[ArrayLike, ...] = (),
    width: str = "fixed",
    placement: str = "centered",
    uniform: bool = True,
    nclass: int = 5,
    region: tuple[float, float] = (0.25, 0.75),
    labels: tuple[str, ...] | None = None,
    mean: bool = False,
    sd: float | None = None,
    se: float | None = None,
    point_pattern: str = "on-line",
    seed: int = 500,
) -> BLiPCustomPlot:
    """Prepare source BLiP geometry, omitting observations covered by boxes/lines.

    Each boxes/lines item is a separate increasing percentile sequence. Variable
    width uses a smoothed empirical CDF, not a kernel density estimate. Jitter
    uses a local NumPy generator restarted for each group, not the S RNG.
    """
    data, missing, labels, rr = _prepare_groups(groups, labels, nclass, uniform, region)
    if width not in ("fixed", "variable") or placement not in ("centered", "based"):
        raise ValueError("width must be fixed/variable and placement centered/based")
    patterns = ("on-line", "stacking", "evenly-spaced", "jittered", "max-range", "vertical-bar")
    if point_pattern not in patterns:
        raise ValueError(f"point_pattern must be one of {patterns}")
    if not isinstance(mean, bool):
        raise ValueError("mean must be boolean")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for value in (sd, se):
        if value is not None and (np.ndim(value) != 0 or not np.isfinite(value) or value < 0):
            raise ValueError("sd and se must be nonnegative finite scalar multipliers")
    bp, lp = _percentiles(boxes, "boxes"), _percentiles(lines, "lines")
    # Normalize horizontal units before interpolation, means and squared deviations.
    scale = max(float(np.max(np.abs(x))) for x in data) or 1.0
    normalized = [x / scale for x in data]
    for original, x in zip(data, normalized, strict=True):
        if np.any((original != 0) & (x == 0)) or np.any(
            (original[1:] != original[:-1]) & (x[1:] == x[:-1])
        ):
            raise ValueError("combined data range is not numerically resolved")
    global_range = max(x[-1] for x in normalized) - min(x[0] for x in normalized)
    deltas = [(global_range if uniform else x[-1] - x[0]) / (2 * nclass) for x in normalized]
    peaks = []
    if width == "variable":
        for x, delta in zip(normalized, deltas, strict=True):
            if delta <= 0 or np.unique(x).size < 2:
                raise ValueError("variable widths require at least two distinct values per group")
            peak = float(np.max(_mass(x, np.unique(x), delta)))
            if peak <= 0:
                raise ValueError("variable-width smoothing window is not numerically resolved")
            peaks.append(peak)
        if uniform:
            peaks = [max(peaks)] * len(data)
    else:
        peaks = [1.0] * len(data)
    prepared = []
    max_n = max(x.size for x in data)
    for k, (original, x) in enumerate(zip(data, normalized, strict=True)):
        low, high = (k + rr) / len(data)
        center = (low + high) / 2
        baseline = center if placement == "centered" else low

        def edges(q: FloatArray) -> tuple[FloatArray, FloatArray]:
            h = np.full(q.shape, high - low)
            if width == "variable":
                h *= np.minimum(_mass(x, q, deltas[k]) / peaks[k], 1.0)
                if uniform:
                    h *= x.size / max_n
            lower = center - h / 2 if placement == "centered" else np.full(q.shape, low)
            return lower, lower + h

        keep = np.ones(x.size, dtype=bool)
        box_geometry = []
        line_geometry = []
        for p in bp:
            q = np.quantile(x, p)
            lo, hi = edges(q)
            box_geometry.append(BLiPBox(_freeze(p), _freeze(q * scale), _freeze(lo), _freeze(hi)))
            if q.size > 1:
                keep &= (x < q[0]) | (x > q[-1])
        for p in lp:
            q = np.quantile(x, p)
            line_geometry.append(_freeze(q * scale))
            keep &= (x < q[0]) | (x > q[-1])
        average = None
        if mean or sd is not None or se is not None:
            # Subtract before scaling to retain small spreads on large offsets.
            anchor = original[0]
            with np.errstate(over="ignore"):
                differences = original - anchor
            if not np.isfinite(differences).all():
                anchor = 0.0
                differences = original
            spread_scale = float(np.max(np.abs(differences))) or 1.0
            centered = differences / spread_scale
            mu_original = float(anchor + np.mean(centered) * spread_scale)
            if mean:
                average = mu_original
            if sd is not None or se is not None:
                if x.size < 2:
                    raise ValueError("SD/SE overlays require at least two observations")
                spread = float(np.std(centered, ddof=1))
                for multiplier, divisor in ((sd, 1.0), (se, np.sqrt(x.size))):
                    if multiplier is not None:
                        with np.errstate(over="ignore", invalid="ignore"):
                            radius = (spread * (multiplier / divisor)) * spread_scale
                            endpoints = mu_original + np.array([-1.0, 1.0]) * radius
                        if not np.isfinite(endpoints).all():
                            raise ValueError("SD/SE endpoints exceed numerical range")
                        line_geometry.append(_freeze(endpoints))
                        keep &= (original < endpoints[0]) | (original > endpoints[-1])
        points = original[keep]
        lo, hi = edges(x[keep])
        prepared.append(
            (
                labels[k],
                float(low),
                float(high),
                float(baseline),
                tuple(box_geometry),
                tuple(line_geometry),
                average,
                points,
                lo,
                hi,
                missing[k],
            )
        )
    ties = [np.unique(g[7], return_counts=True)[1] for g in prepared]
    max_ties = [int(t.max()) if t.size else 1 for t in ties]
    result = []
    for k, g in enumerate(prepared):
        label, low, high, baseline, bg, lg, average, points, lo, hi, nmissing = g
        if point_pattern == "vertical-bar":
            yy = hi
        elif point_pattern == "max-range":
            yy = hi
        elif point_pattern == "jittered":
            yy = lo + np.random.default_rng(seed).random(points.size) * (hi - lo)
        elif point_pattern == "on-line":
            yy = np.full(points.size, baseline)
        else:
            yy = np.empty(points.size)
            start = 0
            increment = (high - low) / (max(max_ties) if uniform else max_ties[k])
            for count in ties[k]:
                stop = start + int(count)
                if point_pattern == "stacking":
                    offsets = np.arange(count) * increment
                    yy[start:stop] = baseline + offsets
                    if placement == "centered":
                        yy[start:stop] -= increment * (count - 1) / 2
                elif count > 1:
                    yy[start:stop] = np.linspace(lo[start], hi[start], count)
                else:
                    yy[start] = (lo[start] + hi[start]) / 2 if width == "variable" else baseline
                start = stop
        result.append(
            BLiPCustomGroup(
                label,
                low,
                high,
                baseline,
                bg,
                lg,
                average,
                _freeze(points),
                _freeze(lo if point_pattern == "vertical-bar" else yy),
                _freeze(yy),
                nmissing,
            )
        )
    return BLiPCustomPlot(tuple(result), point_pattern)


def plot_blip_custom(
    data: BLiPCustomPlot,
    *,
    axes: Axes | None = None,
    color: str = "#126e75",
    fill: str = "#d4e9e6",
    bars: bool | ArrayLike = True,
    percentile_labels: bool = False,
    point_type: str = "p",
    marker: str = "o",
    xlabel: str = "",
) -> Axes:
    """Render custom BLiP geometry. bars selects flattened box percentile bars."""
    from matplotlib import pyplot as plt

    if point_type not in ("p", "l", "b", "n"):
        raise ValueError("point_type must be p, l, b or n")
    if not isinstance(percentile_labels, bool):
        raise ValueError("percentile_labels must be boolean")
    size = sum(b.probabilities.size for b in data.groups[0].boxes)
    selection = np.asarray(bars)
    if selection.dtype != bool or (selection.ndim != 0 and selection.shape != (size,)):
        raise ValueError("bars must be boolean or one boolean per box percentile")
    selection = np.broadcast_to(selection, (size,))
    if axes is None:
        _, axes = plt.subplots(layout="constrained")
    if axes.get_xscale() != "linear" or axes.get_yscale() != "linear":
        raise ValueError("BLiP requires linear plotting axes")
    for g in data.groups:
        start = 0
        for box in g.boxes:
            q, lo, hi = box.coordinates, box.lower, box.upper
            axes.fill(np.r_[q, q[::-1]], np.r_[hi, lo[::-1]], color=fill)
            axes.plot(q, lo, color=color)
            axes.plot(q, hi, color=color)
            for i, p in enumerate(box.probabilities):
                if selection[start + i]:
                    axes.plot([q[i], q[i]], [lo[i], hi[i]], color=color)
                    if percentile_labels:
                        axes.text(q[i], hi[i] + 0.02, f"{100 * p:g}%", ha="center", fontsize=8)
            start += q.size
        for line in g.lines:
            axes.plot(line, np.full(line.size, g.baseline), color=color)
        if g.mean is not None:
            axes.plot([g.mean], [g.baseline], marker="D", color=color, linestyle="None")
        if data.point_pattern == "vertical-bar":
            xx = np.column_stack((g.points, g.points, np.full(g.points.size, np.nan))).ravel()
            yy = np.column_stack(
                (g.point_lower, g.point_upper, np.full(g.points.size, np.nan))
            ).ravel()
            axes.plot(xx, yy, color=color)
        elif point_type != "n":
            axes.plot(
                g.points,
                g.point_upper,
                color=color,
                marker=marker if point_type in ("p", "b") else "None",
                linestyle="-" if point_type in ("l", "b") else "None",
                markersize=4,
            )
    axes.set_yticks(
        [(g.low + g.high) / 2 for g in data.groups], labels=[g.label for g in data.groups]
    )
    axes.set(ylim=(0, 1.08 if percentile_labels else 1), xlabel=xlabel)
    return axes
