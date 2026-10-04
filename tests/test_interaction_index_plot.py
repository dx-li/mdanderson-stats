import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from mdanderson_stats.interaction_index import InteractionIndex  # noqa: E402
from mdanderson_stats.interaction_index_plot import (  # noqa: E402
    plot_interaction_index,
    plot_median_effect,
)
from mdanderson_stats.median_effect import MedianEffectFit  # noqa: E402


def test_median_effect_plot_uses_transformed_observations_and_existing_fit():
    fit = MedianEffectFit(0.2, 1.5, np.zeros((2, 2)), 3, 0)
    figure, axes = plt.subplots()
    returned = plot_median_effect(fit, [1, 2, 4], [0.5, 0.75, 0.9], ax=axes)
    assert returned is axes
    offsets = axes.collections[0].get_offsets()
    np.testing.assert_allclose(offsets[:, 0], np.log([1, 2, 4]))
    np.testing.assert_allclose(offsets[:, 1], np.log(np.array([0.5, 0.75, 0.9]) / [0.5, 0.25, 0.1]))
    line = axes.lines[0]
    np.testing.assert_allclose(line.get_ydata(), fit.intercept + fit.slope * line.get_xdata())
    assert axes.get_xlabel() == "log(dose)"
    assert axes.get_ylabel() == "logit(effect)"
    with pytest.raises(ValueError, match="finite real"):
        plot_median_effect(fit, [1 + 2j, 2, 4], [0.5, 0.75, 0.9], ax=axes)
    plt.close(figure)


def test_interaction_plot_sorts_line_and_keeps_extreme_values_in_log_space():
    result = InteractionIndex(
        np.array([1000.0, 999.0]),
        np.zeros(2),
        np.array([[999.5, 1000.5], [998.5, 999.5]]),
        20,
        0.95,
    )
    figure, axes = plt.subplots()
    plot_interaction_index([0.8, 0.2], result, ax=axes, log_index=True)
    np.testing.assert_allclose(axes.lines[0].get_xdata(), [0.2, 0.8])
    np.testing.assert_allclose(axes.lines[0].get_ydata(), [999, 1000])
    assert axes.get_ylabel() == "Log interaction index"
    with pytest.raises(ArithmeticError, match="log_index=True"):
        plot_interaction_index([0.8, 0.2], result, ax=axes)
    plt.close(figure)

    ordinary = InteractionIndex(
        np.log([2.0, 1.0]),
        np.zeros(2),
        np.log([[1.5, 2.5], [0.7, 1.3]]),
        20,
        0.95,
    )
    figure, axes = plt.subplots()
    plot_interaction_index([0.8, 0.2], ordinary, ax=axes, kind="points", label="A")
    np.testing.assert_allclose(axes.containers[0].lines[0].get_xdata(), [0.8, 0.2])
    plot_interaction_index([0.8, 0.2], ordinary, ax=axes, kind="points", label="B")
    legend = axes.get_legend_handles_labels()[1]
    assert legend.count("Additivity") == 1
    assert "95% pointwise interval" in " ".join(legend)
    plt.close(figure)
