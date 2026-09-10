"""EVENTCHART coded-event conversion and calendar/interval chart geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .eventchart_codes import EventCode, _code_levels
from .eventchart_style import EventLineStyle, LegendLocation, _line_groups

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _matrix(data: ArrayLike) -> FloatArray:
    if np.iscomplexobj(data):
        raise ValueError("data must be real")
    x = np.asarray(data, dtype=float)
    if (
        x.ndim != 2
        or not 1 <= x.shape[0] <= 1_000_000
        or not 1 <= x.shape[1] <= 1000
        or x.size > 2_000_000
        or np.isinf(x).any()
    ):
        raise ValueError("data must be a nonempty matrix with <=2000000 entries and no infinity")
    return x


def _indices(value: ArrayLike, n: int, name: str, *, allow_repeats: bool = False) -> np.ndarray:
    v = np.asarray(value)
    if v.ndim != 1 or v.size == 0 or v.dtype.kind not in "iu" or np.any((v < 0) | (v >= n)):
        raise ValueError(f"{name} must contain nonempty zero-based integer indices in range")
    if not allow_repeats and np.unique(v).size != v.size:
        raise ValueError(f"{name} must not repeat indices")
    return v.astype(np.int64)


@dataclass(frozen=True)
class ConvertedEvents:
    times: FloatArray
    names: tuple[str, ...]
    source_columns: tuple[int, ...]
    codes: tuple[EventCode, ...]


def event_convert(
    data: ArrayLike,
    *,
    time_columns: ArrayLike = (0,),
    code_columns: ArrayLike = (1,),
    names: tuple[str, ...] | None = None,
    code_levels: tuple[tuple[EventCode, ...] | None, ...] | None = None,
) -> ConvertedEvents:
    """Expand time/code pairs, retaining numeric or string category identity.

    None/NaN codes create no event. Codes are sorted within each pair unless
    code_levels explicitly supplies their order, including unobserved levels.
    String sorting uses Unicode order, independent of system locale.
    """
    x = np.asarray(data, dtype=object)
    if (
        x.ndim != 2
        or not 1 <= x.shape[0] <= 1_000_000
        or not 1 <= x.shape[1] <= 1000
        or x.size > 2_000_000
    ):
        raise ValueError("data must be a nonempty matrix with <=2000000 entries")
    tc = _indices(time_columns, x.shape[1], "time_columns", allow_repeats=True)
    cc = _indices(code_columns, x.shape[1], "code_columns", allow_repeats=True)
    if x.shape[0] * tc.size > 2_000_000:
        raise ValueError("conversion exceeds 2000000 row/pair combinations")
    if tc.size > 1000:
        raise ValueError("at most 1000 time/code pairs are supported")
    if tc.size != cc.size:
        raise ValueError("time_columns and code_columns must have equal lengths")
    if names is None:
        names = tuple(f"V{i + 1}" for i in range(x.shape[1]))
    if len(names) != x.shape[1] or any(not isinstance(n, str) for n in names):
        raise ValueError("names must contain one string per data column")
    if code_levels is None:
        code_levels = (None,) * tc.size
    if len(code_levels) != tc.size:
        raise ValueError("code_levels must have one item per time/code pair")
    prepared = [
        _code_levels(x[:, c], declared) for c, declared in zip(cc, code_levels, strict=True)
    ]
    size = sum(len(levels) for _, levels in prepared)
    if x.shape[0] * size > 2_000_000:
        raise ValueError("expanded event matrix exceeds 2000000 entries")
    times = np.full((x.shape[0], size), np.nan)
    labels, sources, codes = [], [], []
    column = 0
    for t, (values, levels) in zip(tc, prepared, strict=True):
        # Convert only selected time columns; other columns may contain metadata.
        if any(isinstance(v, (complex, np.complexfloating)) for v in x[:, t]):
            raise ValueError("event times must be real")
        time = np.asarray(x[:, t], dtype=float)
        if np.isinf(time).any():
            raise ValueError("event times must not contain infinity")
        for v in levels:
            mask = values == v
            times[mask, column] = time[mask]
            label = v if isinstance(v, str) else str(v) if isinstance(v, int) else f"{v:.17g}"
            labels.append(f"{names[t]}.{label}")
            sources.append(int(t))
            codes.append(v)
            column += 1
    return ConvertedEvents(_freeze(times), tuple(labels), tuple(sources), tuple(codes))


@dataclass(frozen=True)
class EventChart:
    times: FloatArray
    positions: FloatArray
    spans: FloatArray
    overlays: tuple[FloatArray, ...]
    rows: tuple[int, ...]
    columns: tuple[int, ...]
    labels: tuple[str, ...]
    x_range: tuple[float, float]
    relative: bool
    time_scale: float = 1.0


def event_chart_data(
    data: ArrayLike,
    *,
    columns: ArrayLike | None = None,
    rows: ArrayLike | None = None,
    reference: int | None = None,
    scale: float = 1.0,
    sort_by: ArrayLike | None = None,
    ascending: bool | tuple[bool, ...] = True,
    na_last: bool = True,
    sort_after_subset: bool = True,
    drop_missing: bool = False,
    renumber: bool = False,
    y_column: int | None = None,
    jitter: float = 0.0,
    seed: int = 500,
    line_pairs: tuple[tuple[int, int], ...] = (),
    all_rows_range: bool = True,
    labels: tuple[str, ...] | None = None,
) -> EventChart:
    """Prepare subject spans/events, with zero-based column and row indices.

    When sorting precedes subsetting, rows selects positions in the sorted data.
    The result always reports original row indices. Missing-only rows can be
    kept as empty chart rows. Numeric y covariates omit missing y observations.
    """
    x = _matrix(data)
    n, p = x.shape
    cols = _indices(np.arange(p) if columns is None else columns, p, "columns")
    selected = _indices(np.arange(n) if rows is None else rows, n, "rows")
    for flag in (na_last, sort_after_subset, drop_missing, renumber, all_rows_range):
        if not isinstance(flag, bool):
            raise ValueError("layout flags must be boolean")
    if np.ndim(scale) != 0 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if np.ndim(jitter) != 0 or not np.isfinite(jitter) or jitter < 0:
        raise ValueError("jitter must be finite and nonnegative")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if labels is None:
        labels = tuple(str(i + 1) for i in range(n))
    if len(labels) != n or any(not isinstance(v, str) for v in labels):
        raise ValueError("labels must contain one string per original row")
    for value, name in ((reference, "reference"), (y_column, "y_column")):
        if value is not None:
            _indices([value], p, name)
    if y_column is not None and sort_by is not None:
        raise ValueError("y_column and sort_by cannot be combined")
    if jitter and y_column is None:
        raise ValueError("jitter requires y_column")
    order = selected.copy() if sort_after_subset else np.arange(n)
    if sort_by is not None:
        keys = _indices(sort_by, p, "sort_by")
        directions = (ascending,) * keys.size if isinstance(ascending, bool) else ascending
        if len(directions) != keys.size or any(not isinstance(a, bool) for a in directions):
            raise ValueError("ascending must be boolean or one boolean per sort key")
        for col, asc in reversed(list(zip(keys, directions, strict=True))):
            v = x[order, col]
            good, bad = order[~np.isnan(v)], order[np.isnan(v)]
            values = x[good, col]
            good = good[np.argsort(values if asc else -values, kind="stable")]
            order = np.r_[good, bad] if na_last else np.r_[bad, good]
    if not sort_after_subset:
        order = order[selected]
    positions = selected.astype(float) + 1
    if renumber:
        positions = np.arange(1.0, order.size + 1)
    with np.errstate(over="ignore", invalid="ignore"):
        times = x[:, cols].copy()
        if reference is not None:
            times -= x[:, reference, None]
        times /= scale
    if np.isinf(times).any():
        raise ValueError("event coordinates exceed numerical range")
    keep = ~np.all(np.isnan(times[order]), axis=1) if drop_missing else np.ones(order.size, bool)
    if y_column is not None:
        keep &= ~np.isnan(x[order, y_column])
    order, positions = order[keep], positions[keep]
    if order.size == 0:
        raise ValueError("no rows remain for plotting")
    if renumber:
        positions = np.arange(1.0, order.size + 1)
    if y_column is not None:
        positions = x[order, y_column].copy()
        if jitter and order.size > 1:
            with np.errstate(over="ignore", invalid="ignore"):
                radius = jitter * ((positions.max() - positions.min()) / (2 * (order.size - 1)))
                positions += np.random.default_rng(seed).uniform(-1.0, 1.0, order.size) * radius
            if not np.isfinite(positions).all():
                raise ValueError("jitter coordinates exceed numerical range")
    plotted = times[order]
    spans = np.column_stack(
        (
            np.min(np.where(np.isnan(plotted), np.inf, plotted), axis=1),
            np.max(np.where(np.isnan(plotted), -np.inf, plotted), axis=1),
        )
    )
    spans[~np.isfinite(spans)] = np.nan
    bounds = times if all_rows_range else plotted
    observed = bounds[~np.isnan(bounds)]
    if not observed.size:
        raise ValueError("chart has no observed event times")
    overlays = []
    if len(line_pairs) > 100:
        raise ValueError("at most 100 overlay pairs are supported")
    for pair in line_pairs:
        pair_array = _indices(pair, p, "line_pairs")
        if pair_array.size != 2 or not np.isin(pair_array, cols).all():
            raise ValueError("each line pair must contain two selected columns")
        indices = [int(np.flatnonzero(cols == c)[0]) for c in pair_array]
        overlays.append(_freeze(plotted[:, indices]))
    return EventChart(
        _freeze(plotted),
        _freeze(positions),
        _freeze(spans),
        tuple(overlays),
        tuple(map(int, order)),
        tuple(map(int, cols)),
        tuple(labels[i] for i in order),
        (float(observed.min()), float(observed.max())),
        reference is not None,
        float(scale),
    )


def plot_event_chart(
    data: EventChart,
    *,
    axes: Axes | None = None,
    event_labels: tuple[str, ...] | None = None,
    markers: tuple[str, ...] | None = None,
    color: str = "#52636a",
    xlabel: str | None = None,
    calendar: bool = False,
    date_origin: str = "1960-01-01",
    calendar_y: bool = False,
    ylabel: str = "",
    line_groups: tuple[EventCode | None, ...] | None = None,
    group_styles: dict[EventCode, EventLineStyle] | None = None,
    overlay_styles: tuple[EventLineStyle, ...] | None = None,
    point_colors: tuple[str, ...] | None = None,
    point_sizes: ArrayLike | None = None,
    legend: bool = True,
    legend_location: LegendLocation = "best",
) -> Axes:
    """Render calendar/interval event geometry on linear numeric axes."""
    from matplotlib import pyplot as plt

    if any(not isinstance(v, bool) for v in (calendar, calendar_y, legend)):
        raise ValueError("calendar, calendar_y and legend must be boolean")
    y_ticks = None
    if calendar_y:
        from .eventchart_dates import event_date_labels

        y_ticks = np.unique(np.rint(np.linspace(data.positions.min(), data.positions.max(), 5)))
        y_labels = event_date_labels(y_ticks, origin=date_origin)
    calendar_ticks = None
    if calendar:
        from .eventchart_dates import event_date_labels

        if data.relative:
            raise ValueError("calendar labels require absolute event times")
        calendar_ticks = np.unique(np.linspace(*data.x_range, 5))
        calendar_labels = event_date_labels(calendar_ticks * data.time_scale, origin=date_origin)
    count = len(data.columns)
    if event_labels is None:
        event_labels = tuple(f"Event {c + 1}" for c in data.columns)
    if markers is None:
        shapes = ("o", "^", "s", "D", "x", "+", "v", "*")
        markers = tuple(shapes[i % len(shapes)] for i in range(count))
    if len(event_labels) != count or len(markers) != count:
        raise ValueError("event_labels and markers need one entry per event column")
    sizes = np.full(count, 5.0) if point_sizes is None else np.asarray(point_sizes, dtype=float)
    if sizes.shape != (count,) or not np.isfinite(sizes).all() or np.any(sizes < 0):
        raise ValueError("point_sizes must contain one nonnegative finite size per event")
    if point_colors is not None and len(point_colors) != count:
        raise ValueError("point_colors must contain one color per event")
    if overlay_styles is None:
        overlay_styles = (EventLineStyle(color, "--"),) * len(data.overlays)
    if len(overlay_styles) != len(data.overlays):
        raise ValueError("overlay_styles must contain one style per interval pair")
    layers: list[tuple[FloatArray, FloatArray, EventLineStyle, str | None]] = []
    if line_groups is None:
        if group_styles is not None:
            raise ValueError("group_styles requires line_groups")
        layers.append((data.spans, data.positions, EventLineStyle(color), None))
    else:
        selected, levels = _line_groups(data.rows, line_groups)
        for i, level in enumerate(levels):
            if group_styles is not None:
                if level not in group_styles:
                    raise ValueError("group_styles must cover every retained line group")
                style = group_styles[level]
            else:
                style = EventLineStyle(f"C{i % 10}", ("-", "--", ":", "-.")[i % 4])
            mask = selected == level
            layers.append((data.spans[mask], data.positions[mask], style, str(level)))
    layers.extend(
        (a, data.positions, style, None)
        for a, style in zip(data.overlays, overlay_styles, strict=True)
    )
    if any(not isinstance(layer[2], EventLineStyle) for layer in layers):
        raise ValueError("line styles must be EventLineStyle instances")
    if axes is None:
        _, axes = plt.subplots(layout="constrained")
    if axes.get_xscale() != "linear" or axes.get_yscale() != "linear":
        raise ValueError("event charts require linear axes")
    for spans, positions, style, label in layers:
        xx = np.column_stack((spans, np.full(spans.shape[0], np.nan))).ravel()
        yy = np.repeat(positions, 3)
        axes.plot(
            xx,
            yy,
            color=style.color,
            linestyle=style.linestyle,
            linewidth=style.linewidth,
            label=label,
        )
    for j in range(count):
        axes.plot(
            data.times[:, j],
            data.positions,
            marker=markers[j],
            linestyle="None",
            markersize=sizes[j],
            color=None if point_colors is None else point_colors[j],
            label=event_labels[j],
        )
    if y_ticks is None:
        axes.set_yticks(data.positions, labels=data.labels)
    else:
        axes.set_yticks(y_ticks, labels=y_labels)
    axes.set_ylabel(ylabel)
    lo, hi = data.x_range
    pad = 0.03 * (hi - lo) if hi > lo else max(1.0, abs(lo) * 0.03)
    if np.isfinite([lo - pad, hi + pad]).all():
        axes.set_xlim(lo - pad, hi + pad)
    axes.set_xlabel(
        xlabel if xlabel is not None else ("Elapsed time" if data.relative else "Event time")
    )
    if calendar_ticks is not None:
        axes.set_xticks(calendar_ticks, labels=calendar_labels)
    if legend:
        axes.legend(loc=legend_location)
    return axes
