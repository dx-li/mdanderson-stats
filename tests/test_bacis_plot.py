import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from mdanderson_stats.bacis import bacis_classify  # noqa: E402
from mdanderson_stats.bacis_plot import plot_bacis_classification_posterior  # noqa: E402


def test_classification_plot_keeps_the_zero_density_jump_disconnected():
    classification = bacis_classify(
        [1, 8], [10, 10], classification_precision=4.0, classification_cutoff=0.5
    )
    figure, axes = plt.subplots()
    try:
        result = plot_bacis_classification_posterior(
            classification,
            latent_precision=4.0,
            grid=[-1.0, 0.0, 1.0],
            xlim="native",
            ax=axes,
        )
        assert result is axes
        assert len(axes.lines) == 4
        for group in range(2):
            left, right = axes.lines[2 * group : 2 * group + 2]
            assert left.get_xdata()[-1] == 0.0
            assert right.get_xdata()[0] == 0.0
            assert np.all(left.get_xdata() <= 0.0)
            assert np.all(right.get_xdata() >= 0.0)
            scale = np.sqrt(4.0) * np.sqrt(2.0 / np.pi)
            assert left.get_ydata()[-1] == pytest.approx(
                scale * classification.low_probability[group]
            )
            assert right.get_ydata()[0] == pytest.approx(
                scale * classification.high_probability[group]
            )
    finally:
        plt.close(figure)


def test_supplied_axes_keep_limits_and_labels_on_a_one_sided_view():
    classification = bacis_classify([2], [10])
    figure, axes = plt.subplots()
    try:
        axes.set_xlim(0.0, 1.0)
        axes.set_ylim(0.2, 0.8)
        axes.set_xlabel("existing x")
        axes.set_ylabel("existing y")
        axes.set_title("existing title")
        plot_bacis_classification_posterior(classification, ax=axes)
        assert axes.get_xlim() == (0.0, 1.0)
        assert axes.get_ylim() == (0.2, 0.8)
        assert axes.get_xlabel() == "existing x"
        assert axes.get_ylabel() == "existing y"
        assert axes.get_title() == "existing title"
        assert axes.lines[0].get_xdata()[0] == 0.0
    finally:
        plt.close(figure)


def test_numpy_window_pair_is_validated_without_string_comparison():
    classification = bacis_classify([2], [10])
    figure, axes = plt.subplots()
    try:
        plot_bacis_classification_posterior(classification, xlim=np.array([-2.0, 3.0]), ax=axes)
        assert axes.get_xlim() == (-2.0, 3.0)
    finally:
        plt.close(figure)
