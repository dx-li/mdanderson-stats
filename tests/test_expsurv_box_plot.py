"""Linked box display callbacks, undefined samples, and lifecycle."""

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.backend_bases import MouseEvent  # noqa: E402
from numpy.testing import assert_array_equal  # noqa: E402

from mdanderson_stats import plot_censored_box  # noqa: E402


@pytest.fixture(autouse=True)
def cleanup():
    yield
    plt.close("all")


def test_selected_box_all_censored_and_empty_are_distinct():
    v = plot_censored_box([1, 2, 3, 4], [1, 1, 0, 0], [0, 1, 2, 3])
    assert v.summary.curve.n_observations == 4
    assert_array_equal(v.segments.get_segments(), v.summary.segments)
    v.select([2, 3])
    assert v.summary is not None and v.summary.last_failure_time is None
    assert not v.segments.get_segments()
    assert v.annotation.get_text() == "No observed failures"
    v.select([0], add=True)
    assert v.summary.last_failure_time == 1
    assert "S(last failure)" in v.annotation.get_text()
    v.clear()
    assert v.summary is None
    assert v.annotation.get_text() == "No selection"
    v.close()
    assert not plt.get_fignums()


def test_real_rectangle_updates_box_and_invalid_selection_preserves_it():
    v = plot_censored_box([1, 2, 3], [1, 0, 1], [[0, 0], [1, 1], [2, 2]])
    canvas = v.figure.canvas
    canvas.draw()
    for name, xy in [
        ("button_press_event", (-0.05, -0.05)),
        ("motion_notify_event", (0.5, 0.5)),
        ("button_release_event", (0.5, 0.5)),
    ]:
        x, y = v.axes[1, 0].transData.transform(xy)
        canvas.callbacks.process(name, MouseEvent(name, canvas, x, y, button=1))
    assert_array_equal(v.selected_indices, [0])
    assert v.summary.curve.n_observations == 1
    assert v.summary.last_failure_time == 1
    before = v.summary
    with pytest.raises(ValueError):
        v.select([3])
    assert v.summary is before


def test_invalid_inputs_create_no_figures():
    with pytest.raises(ValueError):
        plot_censored_box([1], [2], [1])
    with pytest.raises(ValueError):
        plot_censored_box([1], [1], [1], quantile_method="bad")
    assert not plt.get_fignums()
