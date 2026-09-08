"""Check report statistics and plotted observation/prediction pairings."""

from dataclasses import replace
from math import erfc, sqrt

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import fit_stukel, format_stukel, plot_stukel, stukel_log_odds


@pytest.fixture
def fitted():
    return fit_stukel([2, -1, 0, 1], [15, 2, 4, 7], [20, 10, 12, 15], fixed_alpha=(-0.2, 0.3))


@pytest.fixture
def close_plots():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    yield plt
    plt.close("all")


def test_dose_plot_keeps_unequal_trials_paired_after_sorting(fitted, close_plots):
    x = np.array([2, -1, 0, 1])
    y, n = np.array([15, 2, 4, 7]), np.array([20, 10, 12, 15])
    dose, link = plot_stukel(fitted, x, y, n, plot_dose=True)
    order = np.argsort(x)
    assert_allclose(dose.lines[0].get_xdata(), x[order])
    assert_allclose(dose.lines[0].get_ydata(), (y / n)[order])
    assert_allclose(dose.lines[1].get_ydata(), fitted.predict(x)[order])
    eta = fitted.coefficients[0] + x * fitted.coefficients[1]
    order = np.argsort(eta)
    assert_allclose(link.lines[0].get_xdata(), eta[order])
    assert_allclose(link.lines[0].get_ydata(), stukel_log_odds(eta[order], *fitted.alpha))
    assert_allclose(link.lines[1].get_ydata(), eta[order])


def test_link_with_multiple_covariates_and_no_intercept(close_plots):
    x = np.array([[1, 2], [-1, 3], [2, 0], [1, -1], [0, 2]])
    y, n = [7, 3, 8, 6, 5], [10] * 5
    fit = fit_stukel(x, y, n, intercept=False, fixed_alpha=(0.1, -0.2))
    (ax,) = plot_stukel(fit, x, y, n)
    eta = np.sort(x @ fit.coefficients)
    assert_allclose(ax.lines[0].get_xdata(), eta)
    assert_allclose(ax.lines[0].get_ydata(), stukel_log_odds(eta, *fit.alpha))
    with pytest.raises(ValueError, match="exactly one"):
        plot_stukel(fit, x, y, n, plot_dose=True)


def test_dose_only_returns_one_axes(fitted, close_plots):
    assert len(plot_stukel(fitted, [0, 1], [2, 5], [10, 10], plot_dose=True, plot_link=False)) == 1


def test_report_wald_values_and_comparison_statistics(fitted):
    text = format_stukel(fitted, digits=15)
    lines = text.splitlines()
    for line, coef, se in zip(lines[5:7], fitted.coefficients, fitted.standard_errors, strict=True):
        fields = line.split("\t")
        assert_allclose(float(fields[1]), coef, rtol=1e-14)
        assert_allclose(float(fields[2]), se, rtol=1e-14)
        assert_allclose(float(fields[3]), coef / se, rtol=1e-14)
        assert_allclose(float(fields[4]), erfc(abs(coef / se) / sqrt(2)), rtol=1e-14)
    rows = {line.split("\t")[0]: line.split("\t")[1:] for line in lines if "\t" in line}
    assert_allclose(
        list(map(float, rows["raw data"])), [fitted.null_deviance, 3, fitted.null_dispersion]
    )
    assert_allclose(list(map(float, rows["regression"])), [fitted.deviance, 2, fitted.dispersion])
    assert "alpha1 = -0.2; alpha2 = 0.3" in text


def test_report_retains_extreme_tail_and_handles_missing_inference(fitted):
    artificial = replace(fitted, coefficients=np.array([10.0, 0.0]), covariance=np.eye(2))
    rows = format_stukel(artificial, digits=15).splitlines()
    assert_allclose(float(rows[5].split("\t")[-1]), 1.5239706048321e-23, rtol=1e-13)
    assert float(rows[6].split("\t")[-1]) == 1
    missing = replace(fitted, covariance=None, inference_message="Covariance unavailable: bound")
    text = format_stukel(missing)
    assert text.count("NA\tNA\tNA") == 2
    assert "Covariance unavailable: bound" in text
    zero = replace(fitted, covariance=np.zeros((2, 2)))
    assert format_stukel(zero).count("0\tNA\tNA") == 2


@pytest.mark.parametrize(
    "family,extra,names",
    [
        (0, [], []),
        (1, [0.2], ["alpha1"]),
        (2, [0.2], ["alpha2"]),
        (3, [0.2], ["alpha1"]),
        (4, [0.2], ["alpha1"]),
        (5, [0.2, 0.3], ["alpha1", "alpha2"]),
    ],
)
def test_report_names_all_families(fitted, family, extra, names):
    artificial = replace(
        fitted, family=family, coefficients=np.r_[fitted.coefficients, extra], covariance=None
    )
    rows = format_stukel(artificial).splitlines()[5 : 7 + len(extra)]
    assert [row.split("\t")[0] for row in rows] == ["Intercept", "beta1", *names]


@pytest.mark.parametrize("digits", [0, 18, True, 1.5])
def test_invalid_report_precision(fitted, digits):
    with pytest.raises(ValueError):
        format_stukel(fitted, digits=digits)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"plot_dose": False, "plot_link": False},
        {"plot_link": 1},
        {"successes": [11, 2]},
        {"trials": [0, 10]},
        {"successes": [1]},
        {"x": [np.nan, 1]},
        {"x": [[1, 2], [3, 4]]},
    ],
)
def test_invalid_plot_inputs(fitted, kwargs, close_plots):
    arguments = dict(x=[0, 1], successes=[2, 5], trials=[10, 10])
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        plot_stukel(fitted, **arguments)
    assert not close_plots.get_fignums()
