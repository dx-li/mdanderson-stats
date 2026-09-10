"""Goldman event charts: elapsed event times against calendar entry dates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .eventchart import EventChart, _matrix, event_chart_data, plot_event_chart
from .eventchart_codes import EventCode
from .eventchart_dates import event_date_labels
from .eventchart_style import EventLineStyle, LegendLocation

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dataclass(frozen=True)
class GoldmanChart:
    chart: EventChart
    boundary: FloatArray
    now: float
    native_boundary: bool
    entry_axis: bool = True


def goldman_chart_data(
    data: ArrayLike,
    *,
    reference: int,
    columns: ArrayLike | None = None,
    rows: ArrayLike | None = None,
    scale: float = 1.0,
    now: float | None = None,
    drop_missing: bool = False,
    line_pairs: tuple[tuple[int, int], ...] = (),
    native_boundary: bool = False,
    y_column: int | None = None,
) -> GoldmanChart:
    """Plot elapsed times against the reference column's calendar dates.

    The boundary is calendar date = now: elapsed time = (now - entry)/scale.
    native_boundary reproduces the source's range-dependent slope instead.
    """
    if not isinstance(native_boundary, bool):
        raise ValueError("native_boundary must be boolean")
    if y_column is None:
        y_column = reference
    if y_column != reference and not native_boundary:
        raise ValueError("a different calendar y_column requires native_boundary=True")
    x = _matrix(data)
    chart = event_chart_data(
        x,
        columns=columns,
        rows=rows,
        reference=reference,
        scale=scale,
        y_column=y_column,
        drop_missing=drop_missing,
        line_pairs=line_pairs,
    )
    observed = x[:, chart.columns]
    observed = observed[~np.isnan(observed)]
    if now is None:
        now = float(observed.max())
    if np.ndim(now) != 0 or not np.isfinite(now):
        raise ValueError("now must be a finite calendar day")
    low, high = float(chart.positions.min()), float(chart.positions.max())
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if native_boundary:
            left = chart.x_range[0]
            right = (now - float(observed.min())) / scale
            denominator = right - left
            if denominator == 0:
                raise ValueError("native current-date boundary has a zero denominator")
            slope = (low - now) / denominator
            boundary = np.array([[left, now + slope * left], [right, now + slope * right]])
        else:
            boundary = np.array([[(now - low) / scale, low], [(now - high) / scale, high]])
    if not np.isfinite(boundary).all():
        raise ValueError("current-date boundary exceeds numerical range")
    return GoldmanChart(
        chart, _freeze(boundary), float(now), native_boundary, y_column == reference
    )


def plot_goldman_chart(
    data: GoldmanChart,
    *,
    axes: Axes | None = None,
    origin: str = "1960-01-01",
    calendar_labels: bool = True,
    event_labels: tuple[str, ...] | None = None,
    xlabel: str = "Elapsed time",
    line_groups: tuple[EventCode | None, ...] | None = None,
    group_styles: dict[EventCode, EventLineStyle] | None = None,
    boundary_style: EventLineStyle = EventLineStyle("#963e35", "--"),
    legend: bool = True,
    legend_location: LegendLocation = "best",
    square: bool = False,
) -> Axes:
    """Render the chart and current-date line, with ISO calendar y-axis labels."""
    if any(not isinstance(v, bool) for v in (calendar_labels, legend, square)):
        raise ValueError("calendar_labels, legend and square must be boolean")
    if not isinstance(boundary_style, EventLineStyle):
        raise ValueError("boundary_style must be an EventLineStyle")
    # Prepare ticks before drawing, so invalid calendar inputs fail without plotting.
    low, high = float(data.chart.positions.min()), float(data.chart.positions.max())
    ticks = (
        np.unique(np.rint(np.linspace(low, high, 5)))
        if calendar_labels
        else np.unique(np.linspace(low, high, 5))
    )
    if square:
        ticks = np.unique(np.r_[ticks, data.now])
    labels = (
        event_date_labels(ticks, origin=origin)
        if calendar_labels
        else tuple(f"{v:g}" for v in ticks)
    )
    axes = plot_event_chart(
        data.chart,
        axes=axes,
        event_labels=event_labels,
        xlabel=xlabel,
        line_groups=line_groups,
        group_styles=group_styles,
        legend=False,
    )
    boundary = np.vstack((data.boundary, [0.0, data.now])) if square else data.boundary
    axes.plot(
        boundary[:, 0],
        boundary[:, 1],
        linestyle=boundary_style.linestyle,
        color=boundary_style.color,
        linewidth=boundary_style.linewidth,
        label="Current-date boundary",
    )
    left = min(data.chart.x_range[0], float(data.boundary[:, 0].min()))
    right = max(data.chart.x_range[1], float(data.boundary[:, 0].max()))
    if left < right:
        pad = 0.03 * (right - left)
        if np.isfinite([left - pad, right + pad]).all():
            axes.set_xlim(left - pad, right + pad)
    if square:
        axes.set_box_aspect(1)
        pad_y = 0.03 * abs(data.now - low)
        if not np.isfinite([low - pad_y, data.now + pad_y]).all():
            raise ValueError("square chart limits exceed numerical range")
        axes.set_ylim(low - pad_y, data.now + pad_y)
    axes.set_yticks(ticks, labels=labels)
    axes.set_ylabel(
        ("Entry date" if data.entry_axis else "Calendar date")
        if calendar_labels
        else ("Entry time" if data.entry_axis else "Time")
    )
    if legend:
        axes.legend(loc=legend_location)
    return axes
