"""Covariate grouping and line appearance for EVENTCHART rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from .eventchart_codes import EventCode, _code

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

LegendLocation = (
    Literal[
        "best",
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
    ]
    | tuple[float, float]
)


@dataclass(frozen=True)
class EventLineStyle:
    color: str = "#52636a"
    linestyle: str = "-"
    linewidth: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.color, str) or not isinstance(self.linestyle, str):
            raise ValueError("line color and linestyle must be strings")
        if np.ndim(self.linewidth) != 0 or not np.isfinite(self.linewidth) or self.linewidth < 0:
            raise ValueError("line width must be finite and nonnegative")


def _line_groups(
    rows: tuple[int, ...],
    values: tuple[EventCode | None, ...],
) -> tuple[np.ndarray, tuple[EventCode, ...]]:
    if len(values) <= max(rows):
        raise ValueError("line_groups must cover the original data rows")
    selected = [_code(values[i]) for i in rows]
    # Source table(as.character(...)) excludes empty strings and literal NA.
    selected = [None if v in ("", "NA") else v for v in selected]
    present = {v for v in selected if v is not None}
    if any(isinstance(v, str) for v in present) and any(not isinstance(v, str) for v in present):
        raise ValueError("line groups must be homogeneous strings or numbers")
    levels = tuple(sorted({v for v in selected if v is not None}, key=str))
    return np.asarray(selected, dtype=object), levels


def event_chart_legend(
    axes: Axes, *, title: str = "Event chart legend", columns: int = 1
) -> Figure:
    """Create a separate legend page from a rendered chart; no show/save occurs."""
    from matplotlib import pyplot as plt

    if isinstance(columns, bool) or not isinstance(columns, int) or not 1 <= columns <= 10:
        raise ValueError("columns must be an integer in 1..10")
    handles, labels = axes.get_legend_handles_labels()
    if not handles:
        raise ValueError("chart has no labelled artists for a legend")
    rows = (len(handles) + columns - 1) // columns
    figure = plt.figure(figsize=(max(5.0, 2.5 * columns), max(2.0, 0.4 * rows + 1)))
    figure.legend(handles, labels, loc="center", title=title, ncols=columns)
    return figure
