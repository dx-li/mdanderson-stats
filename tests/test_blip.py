import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats.blip import blip_data, plot_blip

GROUPS = ([0, 1, 2, 3, 4, 5, 20], [0, 1, 1, 2, 3, 3, 4, 4])


def test_original_standard_plot_coordinates():
    ref = json.loads((Path(__file__).parent / "fixtures/blip-native.json").read_text())
    box = blip_data(*GROUPS).groups[0]
    assert_allclose(box.coordinates[1:4], ref["box_quantiles"])
    assert_allclose(box.coordinates[[0, 4]], ref["whiskers"])
    assert_array_equal(box.outliers, [20])
    hist = blip_data(*GROUPS, mode="histogram", breaks=[0, 2, 4, 10, 20])
    polygons = []
    for g in hist.groups:
        for a, b, h in zip(g.coordinates[:-1], g.coordinates[1:], g.heights, strict=True):
            polygons.append([[a, g.low], [a, h], [b, h], [b, g.low]])
    assert_allclose(polygons, ref["histogram_polygons"])


def test_right_closed_bins_missing_values_and_separate_scaling():
    data = blip_data(
        [0, 1, 2, np.nan], [0, 2, 2, 2], mode="polygon", breaks=[0, 1, 2], uniform=False
    )
    assert_array_equal(data.groups[0].counts, [2, 1])
    assert data.groups[0].missing == 1
    assert data.groups[0].heights.max() == data.groups[0].high
    assert data.groups[1].heights.max() == data.groups[1].high
    with pytest.raises(ValueError, match="cover"):
        blip_data([0, 2], mode="histogram", breaks=[0, 1])


def test_standard_plot_rendering():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    from matplotlib import pyplot as plt

    for mode in ["boxplot", "histogram", "polygon"]:
        axes = plot_blip(blip_data(*GROUPS, mode=mode), count_labels=True)
        try:
            axes.figure.canvas.draw()
            assert len(axes.get_yticks()) == 2
        finally:
            plt.close(axes.figure)
