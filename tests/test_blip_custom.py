import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import blip_custom_data, plot_blip_custom

GROUPS = ([0, 1, 1, 2, 3, 4, 5, 8], [-1, 0, 0, 0, 2, 3, 5, 7, 9, 10])


def test_original_custom_boxes_and_point_layouts():
    fixture = json.loads((Path(__file__).parent / "fixtures/blip-custom-native.json").read_text())
    for case, ref in fixture.items():
        options = dict(
            width="fixed" if case == "fixed" else "variable",
            uniform=case != "separate",
            placement="based" if case == "separate" else "centered",
            nclass=4,
        )
        data = blip_custom_data(*GROUPS, boxes=((0.2, 0.4, 0.6, 0.8),), **options)
        vertices = []
        for g in data.groups:
            b = g.boxes[0]
            vertices.extend(
                np.column_stack(
                    (np.r_[b.coordinates, b.coordinates[::-1]], np.r_[b.upper, b.lower[::-1]])
                )
            )
        assert_allclose(vertices, ref["polygons"], atol=1e-14)
        for pattern, expected in ref["points"].items():
            data = blip_custom_data(*GROUPS, boxes=(), point_pattern=pattern, **options)
            values = np.concatenate(
                [
                    np.column_stack((g.points, g.point_upper, g.points, g.point_lower))
                    if pattern == "vertical-bar"
                    else np.column_stack((g.points, g.point_upper))
                    for g in data.groups
                ]
            )
            assert_allclose(values, expected, atol=1e-14)


def test_overlays_exclusions_extreme_units_and_jitter():
    g = blip_custom_data(
        np.arange(11.0),
        boxes=((0.0,), (0.2, 0.4), (0.6, 0.8), (1.0,)),
        lines=((0, 0.1),),
        mean=True,
    ).groups[0]
    assert_array_equal(g.points, [5, 9, 10])
    assert_allclose(g.lines[0], [0, 1])
    assert g.mean == 5
    for scale in (1e-200, 1.0, 1e200):
        g = blip_custom_data(np.array([0, 1, 2]) * scale, boxes=(), mean=True, sd=1, se=1).groups[0]
        assert_allclose(g.lines[0] / scale, [0, 2], atol=1e-14)
        assert_allclose(g.lines[1] / scale, 1 + np.array([-1.0, 1.0]) / np.sqrt(3))
        assert g.mean / scale == pytest.approx(1.0)
    g = blip_custom_data(1e12 + np.arange(3), boxes=(), sd=1).groups[0]
    assert_array_equal(g.lines[0] - 1e12, [0, 2])
    a = blip_custom_data(*GROUPS, boxes=(), width="variable", point_pattern="jittered", seed=42)
    b = blip_custom_data(*GROUPS, boxes=(), width="variable", point_pattern="jittered", seed=42)
    for ga, gb in zip(a.groups, b.groups, strict=True):
        assert_array_equal(ga.point_upper, gb.point_upper)
        assert np.all((ga.point_upper >= ga.low) & (ga.point_upper <= ga.high))
    assert blip_custom_data([1, 1], boxes=((0, 1),)).groups[0].points.size == 0
    with pytest.raises(ValueError, match="not numerically resolved"):
        blip_custom_data([1.0, np.nextafter(1.0, 2.0)], width="variable", nclass=1000)
    with pytest.raises(ValueError, match="distinct"):
        blip_custom_data([1, 1], width="variable")


def test_custom_plot_rendering():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    from matplotlib import pyplot as plt

    for pattern in (
        "on-line",
        "stacking",
        "evenly-spaced",
        "jittered",
        "max-range",
        "vertical-bar",
    ):
        data = blip_custom_data(
            *GROUPS,
            width="variable",
            point_pattern=pattern,
            mean=True,
            boxes=((0.2, 0.4), (0.6, 0.8)),
            lines=((0.4, 0.6),),
        )
        axes = plot_blip_custom(data, percentile_labels=True, bars=[True, False, True, True])
        try:
            axes.figure.canvas.draw()
            assert len(axes.get_yticks()) == 2
        finally:
            plt.close(axes.figure)
