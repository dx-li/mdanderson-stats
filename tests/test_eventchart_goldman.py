import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import (
    event_chart_data,
    event_date_labels,
    event_dates,
    goldman_chart_data,
    plot_event_chart,
    plot_goldman_chart,
)

DATA = [
    [7305, 7400, 8000],
    [7340, np.nan, 8100],
    [7400, 7800, 8200],
    [7500, 7900, np.nan],
    [7600, 8000, 8500],
    [7700, 8100, 8600],
]


def test_native_goldman_points_and_boundary():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/eventchart-goldman-native.json").read_text()
    )
    for case, rows in [("all", None), ("subset", [1, 2, 3, 4])]:
        result = goldman_chart_data(DATA, reference=0, rows=rows, scale=365, native_boundary=True)
        g = result.chart
        got = np.concatenate([np.column_stack((g.times[:, j], g.positions)) for j in range(3)])
        assert_allclose(got, np.asarray(fixture[case]["points"], dtype=float), rtol=0, atol=1e-11)
        intercept, slope = fixture[case]["boundary"]
        assert_allclose(
            result.boundary[:, 1], intercept + slope * result.boundary[:, 0], rtol=0, atol=1e-10
        )
        corrected = goldman_chart_data(DATA, reference=0, rows=rows, scale=365)
        assert_allclose(corrected.boundary[:, 0] * 365 + corrected.boundary[:, 1], corrected.now)
    assert not result.boundary.flags.writeable
    assert not np.allclose(corrected.boundary, result.boundary)


def test_calendar_origin_leap_years_missing_dates_and_scaling():
    offsets = event_dates(("1960-01-01", "1960-02-29", "1960-03-01", None))
    assert_array_equal(offsets, [0, 59, 60, np.nan])
    assert event_date_labels(offsets[:3]) == ("1960-01-01", "1960-02-29", "1960-03-01")
    assert event_dates(("2000-02-29", "2000-03-01"), origin="2000-02-28").tolist() == [1, 2]
    with pytest.raises(ValueError):
        event_dates(("1900-02-29",))
    assert event_date_labels([0, 1], origin="1900-02-28") == ("1900-02-28", "1900-03-01")
    assert event_chart_data(DATA, scale=365).time_scale == 365
    with pytest.raises(ValueError, match="outside calendar"):
        event_date_labels([1e308])


def test_goldman_and_calendar_rendering():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    from matplotlib import pyplot as plt

    axes = plot_goldman_chart(goldman_chart_data(DATA, reference=0, scale=365))
    calendar = plot_event_chart(event_chart_data(DATA, scale=365), calendar=True)
    try:
        axes.figure.canvas.draw()
        calendar.figure.canvas.draw()
        assert axes.get_ylabel() == "Entry date"
        assert calendar.get_xticklabels()[0].get_text() == "1980-01-01"
    finally:
        plt.close(axes.figure)
        plt.close(calendar.figure)
