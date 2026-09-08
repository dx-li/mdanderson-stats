"""Linked selection through row IDs and actual canvas mouse events."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.backend_bases import KeyEvent, MouseEvent  # noqa: E402
from numpy.testing import assert_allclose, assert_array_equal  # noqa: E402

from mdanderson_stats import exploratory_survival, plot_survival_scatter  # noqa: E402


@pytest.fixture(autouse=True)
def cleanup():
    yield
    plt.close("all")


def make_view():
    return plot_survival_scatter(
        [4, 1, 3, 2], [1, 0, 1, 1], [[0, 0], [1, 1], [2, 2], [3, 3]], labels=["A", "B"]
    )


def test_original_row_selection_addition_and_empty_clear():
    v = make_view()
    v.select([2, 0, 2])
    assert_array_equal(v.selected_indices, [0, 2])
    expected = exploratory_survival([4, 3], [1, 1])
    assert_allclose(v.curve.survival, expected.survival)
    assert_array_equal(v.line.get_xdata(), expected.step_time)
    v.select([1], add=True)
    assert_array_equal(v.selected_indices, [0, 1, 2])
    for points in v.points:
        colors = points.get_facecolors()
        assert_array_equal(colors[0], colors[1])
        assert not np.array_equal(colors[0], colors[3])
    v.clear()
    assert v.curve is None and not v.selected_indices.size
    assert len(v.line.get_xdata()) == 0


def drag(v, lo, hi, key=None):
    canvas = v.figure.canvas
    canvas.draw()
    ax = v.axes[1, 0]
    for name, xy in [
        ("button_press_event", lo),
        ("motion_notify_event", hi),
        ("button_release_event", hi),
    ]:
        x, y = ax.transData.transform(xy)
        event = MouseEvent(name, canvas, x, y, button=1, key=key)
        canvas.callbacks.process(name, event)


def test_real_rectangle_selection_and_shift_add_update_linked_curve():
    v = make_view()
    drag(v, (-0.1, -0.1), (1.2, 1.2))
    assert_array_equal(v.selected_indices, [0, 1])
    drag(v, (1.8, 1.8), (2.2, 2.2), key="shift")
    assert_array_equal(v.selected_indices, [0, 1, 2])
    assert v.curve.n_observations == 3
    v.figure.canvas.callbacks.process(
        "key_press_event", KeyEvent("key_press_event", v.figure.canvas, key="escape")
    )
    assert v.curve is None
    v.survival_figure.canvas.draw()


@pytest.mark.parametrize("indices", [[-1], [4], [1.5], [[1]], [np.nan]])
def test_invalid_selection_does_not_change_view(indices):
    v = make_view()
    with pytest.raises(ValueError):
        v.select(indices)
    assert_array_equal(v.selected_indices, [0, 1, 2, 3])


def test_one_covariate_and_close_both_figures():
    v = plot_survival_scatter([1, 2], [0, 1], [3, 4])
    assert v.axes.shape == (1, 1)
    numbers = [v.figure.number, v.survival_figure.number]
    v.close()
    assert all(n not in plt.get_fignums() for n in numbers)


@pytest.mark.parametrize(
    "covariates,labels",
    [([[1], [2], [3]], None), ([[], []], None), ([1, np.nan], None), ([[1, 2], [3, 4]], ["one"])],
)
def test_invalid_matrix_fails_before_figure_creation(covariates, labels):
    with pytest.raises(ValueError):
        plot_survival_scatter([1, 2], [1, 0], covariates, labels=labels)
    assert not plt.get_fignums()
