"""Actual mouse/key callbacks across all three linked EXPSURV displays."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.backend_bases import Event, KeyEvent, MouseEvent  # noqa: E402
from numpy.testing import assert_array_equal  # noqa: E402

from mdanderson_stats import (  # noqa: E402
    plot_censored_box,
    plot_event_scatter,
    plot_survival_scatter,
)


@pytest.fixture(params=["survival", "event", "box"])
def view(request):
    args = ([1, 2, 3], [1, 0, 1], [[0, 0], [1, 1], [2, 2]])
    if request.param == "survival":
        v = plot_survival_scatter(*args)
    elif request.param == "box":
        v = plot_censored_box(*args)
    else:
        v = plot_event_scatter([2, 4, 6], *args)
    v.figure.canvas.draw()
    yield v
    v.close()
    assert not plt.get_fignums()


def mouse(v, name, xy, button=None, key=None):
    x, y = v.axes[1, 0].transData.transform(xy)
    event = MouseEvent(name, v.figure.canvas, x, y, button=button, key=key)
    v.figure.canvas.callbacks.process(name, event)


def key(v, text):
    v.figure.canvas.callbacks.process(
        "key_press_event", KeyEvent("key_press_event", v.figure.canvas, key=text)
    )


def test_hover_replaces_selection_resizes_and_clears(view):
    v = view
    key(v, "b")
    assert v.selection_mode == "brush"
    assert all(not s.active for s in v.selectors)
    v.set_brush_size(15, 15)
    mouse(v, "motion_notify_event", (1, 1))
    assert_array_equal(v.selected_indices, [1])
    assert sum(p.get_visible() for p in v._brushes) == 1
    before = v.brush_size
    key(v, "+")
    assert v.brush_size[0] > before[0]
    mouse(v, "motion_notify_event", (2, 2))
    assert_array_equal(v.selected_indices, [2])
    mouse(v, "motion_notify_event", (0.5, 1.5))
    assert not v.selected_indices.size
    v.figure.canvas.callbacks.process(
        "figure_leave_event", Event("figure_leave_event", v.figure.canvas)
    )
    assert not any(p.get_visible() for p in v._brushes)
    key(v, "b")
    assert all(s.active for s in v.selectors)


def test_click_shift_add_and_blank_click(view):
    v = view
    for xy, modifier, expected in [
        ((0, 0), None, [0]),
        ((2, 2), "shift", [0, 2]),
        ((0.5, 1.5), None, []),
    ]:
        mouse(v, "button_press_event", xy, 1, modifier)
        mouse(v, "button_release_event", xy, 1, modifier)
        assert_array_equal(v.selected_indices, expected)
    v.select([1])
    key(v, "escape")
    assert not v.selected_indices.size


def test_bad_controls_preserve_state_and_same_brush_preserves_summary(view):
    v = view
    for size in [(0, 1), (1, np.nan), (-1, 2)]:
        with pytest.raises(ValueError):
            v.set_brush_size(*size)
    assert v.brush_size == (40, 40)
    with pytest.raises(ValueError):
        v.set_selection_mode("bad")
    assert v.selection_mode == "select"
    v.set_selection_mode("brush")
    mouse(v, "motion_notify_event", (1, 1))
    selected = v.selected_indices
    mouse(v, "motion_notify_event", (1, 1))
    assert v.selected_indices is selected
