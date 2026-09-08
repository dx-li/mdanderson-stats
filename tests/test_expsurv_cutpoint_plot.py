"""Slider callbacks update both linked views and handle empty groups."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from numpy.testing import assert_allclose, assert_array_equal  # noqa: E402

from mdanderson_stats import plot_cutpoint, survival_cutpoint  # noqa: E402
from mdanderson_stats.expsurv_cutpoint_plot import _density  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_initial_cut_and_slider_callback_update_both_panels():
    data = survival_cutpoint([1, 2, 3, 4], [1, 0, 1, 1], [0, 1, 2, 3])
    view = plot_cutpoint(data, cut=0.5)
    assert view.slider.val == view.comparison.cut == 0.5
    view.slider.set_val(2)
    assert view.comparison.cut == 2
    assert_array_equal(view.cut_line.get_xdata(), [2, 2])
    assert_array_equal(view.lower_line.get_ydata(), view.comparison.lower.step_survival)
    assert_array_equal(view.upper_line.get_ydata(), view.comparison.upper.step_survival)
    assert "n=3" in view.lower_line.get_label()
    view.figure.canvas.draw()


def test_endpoint_empty_curve_and_invalid_update_is_atomic():
    view = plot_cutpoint(survival_cutpoint([1, 2, 3], [1, 0, 1], [1, 2, 3]))
    assert view.comparison.cut == np.linspace(1, 3, 50)[25]
    view.set_cut(3)
    assert view.comparison.upper is None
    assert len(view.upper_line.get_xdata()) == 0
    with pytest.raises(ValueError):
        view.set_cut(4)
    assert view.comparison.cut == view.slider.val == 3
    view.figure.canvas.draw()
    number = view.figure.number
    view.close()
    assert number not in plt.get_fignums()


def test_constant_covariate_has_explicit_noninteractive_path():
    data = survival_cutpoint([1, 2], [1, 1], [3, 3])
    assert data.compare(3).upper is None
    with pytest.raises(ValueError, match="varying covariate"):
        plot_cutpoint(data)
    assert not plt.get_fignums()


def test_density_guide_is_normalized_and_symmetric_with_explicit_bandwidth():
    x, y, h = _density(np.array([-1.0, 1.0]), 0.5)
    assert h == 0.5
    assert_allclose(y, y[::-1], rtol=1e-13, atol=1e-15)
    assert np.trapezoid(y, x) == pytest.approx(1, abs=0.002)


@pytest.mark.parametrize(
    "kwargs", [{"cut": np.nan}, {"cut": 4}, {"bandwidth": 0}, {"bandwidth": np.inf}]
)
def test_invalid_plot_settings(kwargs):
    with pytest.raises(ValueError):
        plot_cutpoint(survival_cutpoint([1, 2], [1, 0], [1, 2]), **kwargs)
    assert not plt.get_fignums()
