"""Scientific transformation identities and real slider callbacks."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from numpy.testing import assert_allclose, assert_array_equal  # noqa: E402

from mdanderson_stats import exploratory_survival, plot_survival_alignment  # noqa: E402


@pytest.fixture(autouse=True)
def cleanup():
    yield
    plt.close("all")


def test_accelerated_failure_scaling_aligns_rescaled_samples():
    a = exploratory_survival([1, 2, 3], [1, 0, 1])
    b = exploratory_survival([2, 4, 6], [1, 0, 1])
    v = plot_survival_alignment(a, b, method="accelerated-failure")
    assert [s.valmax for s in v.sliders] == [2, 1]
    assert_array_equal(v.lines[0].get_xdata(), v.lines[1].get_xdata())
    v.sliders[0].set_val(1)
    assert_array_equal(v.lines[0].get_xdata(), a.step_time)
    assert_array_equal(v.lines[0].get_ydata(), a.step_survival)
    assert_array_equal(a.time, [1, 2, 3])
    v.set_parameters(0, 1)
    assert_array_equal(v.lines[0].get_xdata(), 0)
    v.figure.canvas.draw()


def test_proportional_hazards_power_scales_cumulative_hazard():
    a = exploratory_survival([1, 2, 3, 4], [1, 1, 0, 1])
    v = plot_survival_alignment(a, a)
    v.set_parameters(0.5, 0.25)
    assert_allclose(v.lines[0].get_ydata(), np.sqrt(a.step_survival))
    positive = a.step_survival > 0
    assert_allclose(
        -np.log(v.lines[1].get_ydata()[positive]), -0.25 * np.log(a.step_survival[positive])
    )
    v.set_parameters(0, 1)
    assert_array_equal(v.lines[0].get_ydata(), 1)
    assert_array_equal(v.lines[1].get_ydata(), a.step_survival)
    v.figure.canvas.draw()
    number = v.figure.number
    v.close()
    assert number not in plt.get_fignums()


@pytest.mark.parametrize("values", [(-1, 1), (1, 2), (np.nan, 1), (1, np.inf)])
def test_invalid_update_preserves_both_sliders(values):
    a = exploratory_survival([1, 2])
    v = plot_survival_alignment(a, a)
    with pytest.raises(ValueError):
        v.set_parameters(*values)
    assert [s.val for s in v.sliders] == [1, 1]
    assert_array_equal(v.lines[0].get_ydata(), a.step_survival)


def test_zero_followup_has_no_defined_accelerated_time_ratio():
    a = exploratory_survival([0])
    with pytest.raises(ValueError, match="positive follow-up"):
        plot_survival_alignment(a, a, method="accelerated-failure")
    assert not plt.get_fignums()
    v = plot_survival_alignment(a, a)
    assert v.axes.get_xlim() == (0, 1)


def test_invalid_method_creates_no_figure():
    a = exploratory_survival([1, 2])
    with pytest.raises(ValueError):
        plot_survival_alignment(a, a, method="unknown")
    assert not plt.get_fignums()
