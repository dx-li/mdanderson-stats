"""End-to-end original demos.S comparison, including both archived datasets."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import StukelFitError, compare_stukel, stukel_demo

REFERENCE = json.loads((Path(__file__).parent / "fixtures/stukel_fit.json").read_text())["cases"]


@pytest.mark.parametrize("dataset", ["beetles", "warsaw"])
def test_demo_fits_all_families_against_native_examples(dataset):
    result = stukel_demo(dataset)
    cases = [c for c in REFERENCE if c["dataset"] == dataset]
    assert len(result.fits) == 6
    assert_array_equal(result.x, np.array(cases[0]["x"])[:, 0])
    assert_array_equal(result.successes, cases[0]["successes"])
    assert_array_equal(result.trials, cases[0]["trials"])
    for family, (fit, case) in enumerate(zip(result.fits, cases, strict=True)):
        assert fit.family == family
        if family != 4:
            assert_allclose(
                fit.objective.negative_log_likelihood, case["objective"], atol=1e-6, rtol=0
            )
        else:
            # The original opposite-shape derivative bug causes false convergence.
            assert case["status"] == 8
            assert fit.objective.negative_log_likelihood < case["objective"]
        assert_allclose(fit.predict(result.x), fit.objective.probabilities)
    assert result.report().count("STUKEL generalized logistic regression") == 6
    assert result.report().count("raw data\t") == 6


def test_combined_plot_uses_baseline_log_odds_not_family_predictors():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    result = stukel_demo()
    dose, link = result.plot()
    try:
        assert len(dose.lines) == 7 and len(link.lines) == 6
        order = np.argsort(result.x)
        assert_allclose(dose.lines[0].get_ydata(), (result.successes / result.trials)[order])
        baseline = result.fits[0].objective.log_odds
        link_order = np.argsort(baseline)
        for index, fit in enumerate(result.fits):
            assert_allclose(dose.lines[index + 1].get_ydata(), fit.objective.probabilities[order])
            assert_allclose(link.lines[index].get_xdata(), baseline[link_order])
            assert_allclose(link.lines[index].get_ydata(), fit.objective.log_odds[link_order])
        dose.figure.canvas.draw()
    finally:
        plt.close(dose.figure)


def test_custom_data_are_snapshotted_and_input_order_preserved():
    case = REFERENCE[0]
    x = np.array(case["x"])[::-1].copy()
    y, n = np.array(case["successes"])[::-1].copy(), np.array(case["trials"])[::-1].copy()
    result = compare_stukel(x, y, n, scale="fixed")
    expected = x[:, 0].copy()
    x[:] = 0
    y[:] = 0
    assert_array_equal(result.x, expected)
    assert np.any(result.successes > 0)
    assert all(f.dispersion == 1 for f in result.fits)
    with pytest.raises(ValueError):
        result.x[0] = 0


def test_failed_family_is_explicit():
    case = REFERENCE[0]
    with pytest.raises(StukelFitError, match="Comparison family 0"):
        compare_stukel(case["x"], case["successes"], case["trials"], max_iterations=1)


@pytest.mark.parametrize("dataset", ["", "Warsaw", 1, None])
def test_unknown_demo_dataset(dataset):
    with pytest.raises(ValueError):
        stukel_demo(dataset)


@pytest.mark.parametrize(
    "x,y,n",
    [
        ([], [], []),
        ([[1, 2]], [2], [10]),
        ([1], [11], [10]),
        ([1], [2], [0]),
        ([1, 2], [2], [10]),
        ([np.nan], [2], [10]),
    ],
)
def test_invalid_comparison_data(x, y, n):
    with pytest.raises(ValueError):
        compare_stukel(x, y, n)
