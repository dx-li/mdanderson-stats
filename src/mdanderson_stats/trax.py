"""TRAX coordinate transforms with original-unit tick labels and overlays."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray, finite

if TYPE_CHECKING:
    from matplotlib.axes import Axes

Transform = Callable[[FloatArray], ArrayLike]


def _apply(values: FloatArray, function: Transform | None) -> FloatArray:
    with np.errstate(all="ignore"):
        result = np.asarray(values if function is None else function(_freeze(values)))
    if np.iscomplexobj(result):
        raise ValueError("axis transforms must return real coordinates")
    result = np.asarray(result, dtype=float)
    if result.shape != values.shape:
        raise ValueError("axis transforms must preserve the input vector shape")
    return result


@dataclass(frozen=True)
class TRAXData:
    original_x: FloatArray
    original_y: FloatArray
    x: FloatArray
    y: FloatArray
    retained_indices: NDArray[np.intp]
    dropped_indices: NDArray[np.intp]


def trax_data(
    x: ArrayLike,
    y: ArrayLike,
    *,
    xfunc: Transform | None = None,
    yfunc: Transform | None = None,
    invalid: str = "drop",
) -> TRAXData:
    """Transform paired data with pointwise callbacks; inspect omitted row indices.

    Requires no plotting dependency. Callbacks receive immutable vectors and
    must return same-shaped real vectors. Their exceptions propagate.
    """
    if np.iscomplexobj(x) or np.iscomplexobj(y):
        raise ValueError("TRAX requires real coordinates")
    xx, yy = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if xx.ndim != 1 or xx.shape != yy.shape or not 2 <= xx.size <= 1_000_000:
        raise ValueError("x and y must be matching vectors with 2..1000000 entries")
    if invalid not in ("drop", "raise"):
        raise ValueError("invalid must be drop or raise")
    tx, ty = _apply(xx, xfunc), _apply(yy, yfunc)
    keep = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(tx) & np.isfinite(ty)
    if invalid == "raise" and not keep.all():
        raise ValueError("input or transformed coordinates contain nonfinite values")
    if keep.sum() < 2:
        raise ValueError("at least two jointly finite transformed observations are required")
    retained = np.flatnonzero(keep).astype(np.intp)
    dropped = np.flatnonzero(~keep).astype(np.intp)
    return TRAXData(
        _freeze(xx[keep]),
        _freeze(yy[keep]),
        _freeze(tx[keep]),
        _freeze(ty[keep]),
        np.frombuffer(retained.tobytes(), dtype=np.intp),
        np.frombuffer(dropped.tobytes(), dtype=np.intp),
    )


def _ticks(
    original: FloatArray,
    function: Transform | None,
    ticks: ArrayLike | None,
    intervals: int,
    limits: ArrayLike | None,
) -> tuple[FloatArray, list[str], tuple[float, float] | None]:
    from matplotlib.ticker import MaxNLocator

    if (
        isinstance(intervals, (bool, np.bool_))
        or not isinstance(intervals, (int, np.integer))
        or not 2 <= intervals <= 100
    ):
        raise ValueError("tick interval requests must be integers in 2..100")
    bounds = None if limits is None else finite(limits, "axis limits")
    transformed_limits = None
    if bounds is not None:
        if bounds.shape != (2,) or bounds[0] == bounds[1]:
            raise ValueError("axis limits must have two distinct original-unit endpoints")
        transformed = _apply(bounds, function)
        if not np.isfinite(transformed).all() or transformed[0] == transformed[1]:
            raise ValueError("transformed axis limits must be finite and distinct")
        transformed_limits = (float(transformed[0]), float(transformed[1]))
    raw_range = original if bounds is None else bounds
    lo, hi = float(raw_range.min()), float(raw_range.max())
    if ticks is None:
        raw = np.asarray(
            MaxNLocator(nbins=intervals, steps=[1, 2, 5, 10]).tick_values(lo, hi), dtype=float
        )
        raw = raw[(raw >= lo) & (raw <= hi)]
        if raw.size == 0:
            raw = np.unique([lo, hi])
    else:
        raw = finite(ticks, "ticks")
        if raw.ndim != 1 or not 1 <= raw.size <= 1000:
            raise ValueError("explicit ticks must have 1..1000 original-unit values")
    positions = _apply(raw, function)
    valid = np.isfinite(positions)
    if ticks is not None and not valid.all():
        raise ValueError("explicit tick values must transform to finite coordinates")
    raw, positions = raw[valid], positions[valid]
    if not raw.size:
        raise ValueError("no finite tick positions; supply ticks inside the transform domain")
    if np.unique(positions).size != positions.size:
        raise ValueError("tick transforms overlap; supply distinct explicit tick positions")
    return positions, [f"{v:.8g}" for v in raw], transformed_limits


def _draw(axes: Axes, data: TRAXData, kind: str, options: dict[str, Any]) -> None:
    if axes.get_xscale() != "linear" or axes.get_yscale() != "linear":
        raise ValueError("TRAX requires linear plotting axes; put transformations in callbacks")
    if kind not in ("points", "line", "both"):
        raise ValueError("kind must be points, line or both")
    style = dict(options)
    style.setdefault("linestyle", "None" if kind == "points" else "-")
    style.setdefault("marker", "o" if kind in ("points", "both") else "None")
    axes.plot(data.x, data.y, **style)


class TRAXPlot:
    """A plot and its transform contract; overlays reuse transforms and view limits."""

    def __init__(
        self,
        axes: Axes,
        data: TRAXData,
        xfunc: Transform | None,
        yfunc: Transform | None,
        invalid: str,
    ):
        self.axes = axes
        self._series = [data]
        self._xfunc, self._yfunc, self._invalid = xfunc, yfunc, invalid

    @property
    def series(self) -> tuple[TRAXData, ...]:
        return tuple(self._series)

    def add(
        self, x: ArrayLike, y: ArrayLike, *, kind: str = "line", **plot_options: Any
    ) -> TRAXData:
        data = trax_data(x, y, xfunc=self._xfunc, yfunc=self._yfunc, invalid=self._invalid)
        limits = self.axes.get_xlim(), self.axes.get_ylim()
        _draw(self.axes, data, kind, plot_options)
        self.axes.set_xlim(limits[0])
        self.axes.set_ylim(limits[1])
        self._series.append(data)
        return data


def trax(
    x: ArrayLike,
    y: ArrayLike,
    *,
    xfunc: Transform | None = None,
    yfunc: Transform | None = None,
    xnint: int = 5,
    ynint: int = 5,
    xticks: ArrayLike | None = None,
    yticks: ArrayLike | None = None,
    xlim: ArrayLike | None = None,
    ylim: ArrayLike | None = None,
    xgrid: bool = False,
    ygrid: bool = False,
    invalid: str = "drop",
    axes: Axes | None = None,
    kind: str = "points",
    xlabel: str = "x",
    ylabel: str = "y",
    **plot_options: Any,
) -> TRAXPlot:
    """Plot transformed coordinates with original-unit tick labels.

    Install the plot extra. Limits and explicit ticks use original units.
    Add overlays with returned_plot.add(), preserving its axes and transforms.
    Remaining keywords are Matplotlib line/marker properties. No show/save occurs.
    """
    data = trax_data(x, y, xfunc=xfunc, yfunc=yfunc, invalid=invalid)
    if not isinstance(xgrid, (bool, np.bool_)) or not isinstance(ygrid, (bool, np.bool_)):
        raise ValueError("grid flags must be boolean")
    xp, xlabels, xlimits = _ticks(data.original_x, xfunc, xticks, xnint, xlim)
    yp, ylabels, ylimits = _ticks(data.original_y, yfunc, yticks, ynint, ylim)
    from matplotlib import pyplot as plt

    if axes is None:
        _, axes = plt.subplots(layout="constrained")
    _draw(axes, data, kind, plot_options)
    xview, yview = axes.get_xlim(), axes.get_ylim()
    axes.set_xticks(xp, labels=xlabels)
    axes.set_yticks(yp, labels=ylabels)
    axes.set_xlim(xview if xlimits is None else xlimits)
    axes.set_ylim(yview if ylimits is None else ylimits)
    axes.set(xlabel=xlabel, ylabel=ylabel)
    axes.grid(bool(xgrid), axis="x")
    axes.grid(bool(ygrid), axis="y")
    return TRAXPlot(axes, data, xfunc, yfunc, invalid)
