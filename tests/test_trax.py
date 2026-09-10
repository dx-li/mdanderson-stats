import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import trax, trax_data


def logit(p):
    return np.log(p / (1 - p))


def test_native_logit_transformed_coordinates():
    fixture = json.loads((Path(__file__).parent / "fixtures/trax-native.json").read_text())
    result = trax_data(np.arange(1, 10), np.arange(1, 10) / 10, yfunc=logit)
    assert_allclose(np.column_stack((result.x, result.y)), fixture["data"], rtol=1e-13, atol=1e-14)
    overlay = trax_data(np.arange(1, 10), np.linspace(0.5, 0.6, 9), yfunc=logit)
    assert_allclose(
        np.column_stack((overlay.x, overlay.y)), fixture["overlay"], rtol=1e-13, atol=1e-14
    )


def test_joint_finite_filter_preserves_pairs_and_reports_omissions():
    result = trax_data([-1, 1, 2, 4, 8], [4, 2, 0, 8, 16], xfunc=np.log, yfunc=np.log)
    assert_array_equal(result.retained_indices, [1, 3, 4])
    assert_array_equal(result.dropped_indices, [0, 2])
    assert_allclose(result.x, np.log([1, 4, 8]))
    assert_allclose(result.y, np.log([2, 8, 16]))
    with pytest.raises(ValueError, match="nonfinite"):
        trax_data([-1, 1, 2], [1, 2, 3], xfunc=np.log, invalid="raise")
    with pytest.raises(ValueError, match="shape"):
        trax_data([1, 2, 3], [1, 2, 3], xfunc=lambda x: x[:1])


def test_tick_labels_original_units_and_overlay_keeps_axes():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    p = np.arange(1, 10) / 10
    plot = trax(np.arange(1, 10), p, yfunc=logit, yticks=p, ygrid=True)
    try:
        assert_allclose(plot.axes.get_yticks(), logit(p), atol=1e-14)
        assert [t.get_text() for t in plot.axes.get_yticklabels()] == [f"{v:.1f}" for v in p]
        limits = plot.axes.get_xlim(), plot.axes.get_ylim()
        added = plot.add(np.arange(1, 10), np.linspace(0.5, 0.6, 9))
        assert_allclose(added.y, logit(np.linspace(0.5, 0.6, 9)), atol=1e-14)
        assert (plot.axes.get_xlim(), plot.axes.get_ylim()) == limits
        assert len(plot.series) == 2
        plot.axes.figure.canvas.draw()
    finally:
        plt.close(plot.axes.figure)
