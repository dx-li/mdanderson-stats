"""Event coordinates, censoring symbols and linked mouse selection."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.backend_bases import KeyEvent, MouseEvent  # noqa: E402
from numpy.testing import assert_array_equal  # noqa: E402

from mdanderson_stats import plot_event_scatter  # noqa: E402


@pytest.fixture(autouse=True)
def cleanup():
    yield
    plt.close("all")


def make_view():
    return plot_event_scatter([7, 1, 4], [3, 8, 0], [1, 0, 1], [[0, 0], [1, 1], [2, 2]])


def test_source_segment_geometry_and_status_partition_keep_original_rows():
    v = make_view()
    assert_array_equal(v.selected_indices, [0, 1, 2])
    assert_array_equal(
        v.segments.get_segments(), [[[0, 7], [3, 7]], [[0, 1], [8, 1]], [[0, 4], [0, 4]]]
    )
    assert_array_equal(v.failures.get_offsets(), [[3, 7], [0, 4]])
    assert_array_equal(v.censored.get_offsets(), [[8, 1]])
    limits = v.event_axes.get_xlim(), v.event_axes.get_ylim()
    v.select([1])
    assert_array_equal(v.segments.get_segments(), [[[0, 1], [8, 1]]])
    assert v.failures.get_offsets().shape == (0, 2)
    v.select([0], add=True)
    assert_array_equal(v.selected_indices, [0, 1])
    assert_array_equal(v.failures.get_offsets(), [[3, 7]])
    assert limits == (v.event_axes.get_xlim(), v.event_axes.get_ylim())
    v.clear()
    assert not v.segments.get_segments()
    assert v.failures.get_offsets().shape == v.censored.get_offsets().shape == (0, 2)


def test_real_mouse_selection_and_escape_update_event_chart():
    v = make_view()
    canvas = v.figure.canvas
    canvas.draw()
    ax = v.axes[1, 0]
    for name, xy in [
        ("button_press_event", (-0.05, -0.05)),
        ("motion_notify_event", (0.5, 0.5)),
        ("button_release_event", (0.5, 0.5)),
    ]:
        x, y = ax.transData.transform(xy)
        canvas.callbacks.process(name, MouseEvent(name, canvas, x, y, button=1))
    assert_array_equal(v.selected_indices, [0])
    assert_array_equal(v.segments.get_segments(), [[[0, 7], [3, 7]]])
    canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, key="escape"))
    assert not v.segments.get_segments()


def test_input_snapshots_zero_ranges_and_cleanup():
    arrival = np.zeros(2)
    v = plot_event_scatter(arrival, [0, 0], [0, 0], [1, 1])
    arrival[:] = 9
    assert_array_equal(v.arrival, [0, 0])
    assert v.event_axes.get_xlim() == v.event_axes.get_ylim() == (0, 1)
    assert v.failures.get_offsets().shape == (0, 2)
    assert_array_equal(v.censored.get_offsets(), [[0, 0], [0, 0]])
    for array in (v.arrival, v.duration, v.status, v.covariates, v.selected_indices):
        assert not array.flags.writeable
    v.close()
    assert not plt.get_fignums()


@pytest.mark.parametrize(
    "arrival,duration,status",
    [
        ([], [], []),
        ([-1], [1], [1]),
        ([np.nan], [1], [1]),
        ([[1]], [1], [1]),
        ([1], [-1], [1]),
        ([1], [np.inf], [1]),
        ([1], [1, 2], [1]),
        ([1], [1], [2]),
        ([1], [1], [0.5]),
    ],
)
def test_invalid_event_data_creates_no_figures(arrival, duration, status):
    with pytest.raises(ValueError):
        plot_event_scatter(arrival, duration, status, [1])
    assert not plt.get_fignums()


def test_invalid_selection_preserves_event_chart():
    v = make_view()
    before = np.array(v.segments.get_segments())
    with pytest.raises(ValueError):
        v.select([3])
    assert_array_equal(v.segments.get_segments(), before)
    assert_array_equal(v.selected_indices, [0, 1, 2])
