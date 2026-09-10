import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import event_chart_data, event_convert, plot_event_chart

DATA = np.array(
    [
        [10, 13, 17, 3, 40],
        [12, np.nan, 20, 1, 50],
        [9, 14, np.nan, 2, 60],
        [11, 15, 21, 1, 45],
        [15, 19, 25, 2, 55],
        [16, 18, 24, 3, 65],
    ]
)


def test_original_eventchart_and_conversion_coordinates():
    fixture = json.loads((Path(__file__).parent / "fixtures/eventchart-native.json").read_text())
    layouts = [
        event_chart_data(DATA, columns=[0, 1, 2]),
        event_chart_data(DATA, columns=[0, 1, 2], reference=0, scale=2),
        event_chart_data(
            DATA,
            columns=[0, 1, 2],
            reference=0,
            rows=[0, 2, 3, 5],
            sort_by=[3, 0],
            ascending=(True, False),
            renumber=True,
        ),
    ]
    for case, g in zip(("calendar", "interval", "sorted"), layouts, strict=True):
        lines = np.stack([g.spans, g.positions[:, None].repeat(2, axis=1)], axis=2).reshape(-1, 2)
        points = np.concatenate([np.column_stack((g.times[:, j], g.positions)) for j in range(3)])
        assert_allclose(lines, np.array(fixture[case]["lines"], dtype=float), rtol=0, atol=1e-14)
        assert_allclose(points, np.array(fixture[case]["points"], dtype=float), rtol=0, atol=1e-14)
    z = np.column_stack(([5, 6, 3, 1, 2], [1, 0, 1, 1, 0], [8, 9, 4, 2, 7], [3, 2, 2, 3, 1]))
    converted = event_convert(z, time_columns=[0, 2], code_columns=[1, 3])
    assert_array_equal(converted.times, np.array(fixture["converted"], dtype=float))
    assert converted.names == ("V1.0", "V1.1", "V3.1", "V3.2", "V3.3")


def test_missing_rows_keep_overlays_and_provenance_aligned():
    x = [[0, 2, 5], [np.nan, 3, 4], [1, 4, 8], [np.nan, np.nan, np.nan]]
    g = event_chart_data(x, reference=0, drop_missing=True, line_pairs=((0, 2),))
    assert g.rows == (0, 2)
    assert_array_equal(g.positions, [1, 3])
    assert_array_equal(g.overlays[0], [[0, 5], [0, 7]])
    assert_array_equal(g.times, [[0, 2, 5], [0, 3, 7]])
    assert not g.times.flags.writeable
    g = event_chart_data(DATA, columns=[0, 1, 2], sort_by=[3], rows=[0, 2], sort_after_subset=False)
    assert g.rows == (1, 2)
    assert_array_equal(g.positions, [1, 3])
    c = event_convert([[5, 1], [6, np.nan], [np.nan, 0]])
    assert_array_equal(c.times, [[np.nan, 5], [np.nan, np.nan], [np.nan, np.nan]])
    assert event_convert([[1, np.nan]]).times.shape == (1, 0)
    a = event_chart_data(DATA, y_column=4, jitter=1, seed=20)
    b = event_chart_data(DATA, y_column=4, jitter=1, seed=20)
    assert_array_equal(a.positions, b.positions)
    with pytest.raises(ValueError, match="numerical range"):
        event_chart_data([[-1e308, 1e308]], reference=0)


def test_eventchart_rendering():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    from matplotlib import pyplot as plt

    axes = plot_event_chart(
        event_chart_data(DATA, columns=[0, 1, 2], reference=0, line_pairs=((0, 2),)),
        event_labels=("Entry", "Recurrence", "Last contact"),
    )
    try:
        axes.figure.canvas.draw()
        assert len(axes.get_legend().get_texts()) == 3
    finally:
        plt.close(axes.figure)
