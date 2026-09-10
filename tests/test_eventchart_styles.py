import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import EventLineStyle, event_chart_data, goldman_chart_data, plot_event_chart

DATA = np.array(
    [
        [7305, 7400, 8000, 2],
        [7340, 7700, 8100, 1],
        [7400, 7800, 8200, 2],
        [7500, 7900, 8300, 1],
        [7600, 8000, 8500, 1],
        [7700, 8100, 8600, 2],
    ]
)


def test_native_grouped_line_coordinates_and_styles():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    from matplotlib import pyplot as plt

    ref = json.loads((Path(__file__).parent / "fixtures/eventchart-groups-native.json").read_text())
    g = event_chart_data(
        DATA,
        columns=[0, 1, 2],
        rows=[1, 2, 4, 5],
        sort_by=[0],
        ascending=False,
        renumber=True,
        reference=0,
    )
    axes = plot_event_chart(
        g,
        line_groups=tuple(DATA[:, 3]),
        group_styles={1: EventLineStyle("black", "-", 1), 2: EventLineStyle("red", "--", 2)},
    )
    try:
        got = []
        for line in axes.lines[:2]:
            xy = line.get_xydata().reshape(-1, 3, 2)
            group = 1 if line.get_color() == "black" else 2
            assert line.get_linestyle() == ("-" if group == 1 else "--")
            got.extend(
                [a[0, 0], a[0, 1], a[1, 0], a[1, 1], group, line.get_linewidth(), group] for a in xy
            )
        assert_allclose(sorted(got, key=lambda a: a[1]), ref["grouped_lines"], rtol=0, atol=1e-12)
        axes.figure.canvas.draw()
    finally:
        plt.close(axes.figure)
    x = DATA.copy()
    x[:, 3] = x[:, 0] + [10, 20, 40, 60, 80, 100]
    general = goldman_chart_data(
        x, columns=[0, 1, 2], reference=0, y_column=3, native_boundary=True
    )
    a, b = ref["general_boundary"]
    assert_allclose(general.boundary[:, 1], a + b * general.boundary[:, 0], atol=1e-10, rtol=0)
    assert not general.entry_axis


def test_missing_groups_calendar_covariates_and_overlays():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    from matplotlib import pyplot as plt

    g = event_chart_data(DATA, columns=[0, 1, 2], y_column=0, reference=0, line_pairs=((0, 2),))
    axes = plot_event_chart(
        g,
        line_groups=("A", None, "", "NA", "B", "A"),
        calendar_y=True,
        overlay_styles=(EventLineStyle("gray", ":", 2),),
        point_colors=("black", "orange", "blue"),
        point_sizes=[3, 4, 5],
        legend=False,
    )
    try:
        assert len(axes.lines[0].get_xdata()) + len(axes.lines[1].get_xdata()) == 9
        assert len(axes.lines[2].get_xdata()) == 18  # overlays remain for all six subjects
        assert axes.get_legend() is None
        assert axes.get_yticklabels()[0].get_text() == "1980-01-01"
        axes.figure.canvas.draw()
    finally:
        plt.close(axes.figure)
