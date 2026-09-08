"""Exact rational KM products, plotting corners and inverse-survival semantics."""

from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats.expsurv import exploratory_survival


@pytest.mark.parametrize("status", [[1] * 6, [0] * 6, [1, 0, 1, 0, 1, 0], [0, 1, 1, 1, 0, 1]])
@pytest.mark.parametrize("times", [[0, 1, 2, 3, 4, 5], [1, 1, 2, 2, 2, 4]])
def test_legacy_matches_exact_per_record_source_product(times, status):
    fit = exploratory_survival(times, status, legacy=True)
    prob = Fraction(1)
    expected = []
    for i, event in enumerate(status):
        prob *= 1 - Fraction(event, len(times) - i)
        expected.append(float(prob))
    assert_allclose(fit.survival, expected, rtol=2e-15, atol=1e-16)
    assert_array_equal(fit.step_time, np.r_[0, np.repeat(times, 2)])
    assert_allclose(
        fit.step_survival, np.r_[1, np.column_stack((np.r_[1, expected[:-1]], expected)).ravel()]
    )


def test_grouped_ties_use_common_risk_and_are_permutation_invariant():
    t = np.array([1, 1, 2, 2])
    d = np.array([1, 0, 1, 0])
    a = exploratory_survival(t, d)
    b = exploratory_survival(t[::-1], d[::-1])
    assert_array_equal(a.time, [1, 2])
    assert_allclose(a.survival, [0.75, 0.375])
    assert_allclose(b.survival, a.survival)
    assert_array_equal(a.at_risk, [4, 2])
    assert_array_equal(a.events, [1, 1])
    assert_array_equal(a.censored, [1, 1])
    old = exploratory_survival(t[::-1], d[::-1], legacy=True)
    assert old.survival[-1] == 0


def test_right_continuous_queries_and_unreached_quantiles():
    fit = exploratory_survival([1, 2, 3], [1, 0, 0])
    assert_allclose(fit.at([[-1, 0, 1], [2, 3, 4]]), [[1, 1, 2 / 3], [2 / 3, 2 / 3, 2 / 3]])
    assert np.isnan(fit.survival_quantile(0.5))
    assert np.isnan(fit.survival_quantile(0.5, method="source"))
    assert fit.survival_quantile(1) == 0


def test_step_and_source_interpolation_and_exact_plateau_conventions():
    fit = exploratory_survival([1, 2, 3, 4], [1, 1, 0, 1])
    assert_allclose(fit.survival, [0.75, 0.5, 0.5, 0])
    assert_allclose(fit.survival_quantile([0.9, 0.625, 0.5, 0]), [1, 2, 2, 4])
    assert_allclose(fit.survival_quantile([0.9, 0.625, 0.5, 0], method="source"), [0.4, 1.5, 3, 4])


def test_time_zero_jump_and_time_scaling():
    fit = exploratory_survival([0, 1, 3], [1, 0, 1])
    assert fit.at(0) == pytest.approx(2 / 3)
    scaled = exploratory_survival([0, 2, 6], [1, 0, 1])
    assert_allclose(
        scaled.survival_quantile([0.9, 0.5, 0]), fit.survival_quantile([0.9, 0.5, 0]) * 2
    )
    for array in [
        fit.time,
        fit.survival,
        fit.step_time,
        fit.step_survival,
        fit.at([0, 1]),
        fit.survival_quantile([0.5]),
    ]:
        assert not array.flags.writeable


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(time=[]),
        dict(time=[-1, 2]),
        dict(time=[1, np.nan]),
        dict(time=[1, 2], status=[1]),
        dict(time=[1, 2], status=[1, 2]),
        dict(time=[1, 2], legacy=1),
    ],
)
def test_invalid_fit_inputs(kwargs):
    with pytest.raises(ValueError):
        exploratory_survival(**kwargs)


@pytest.mark.parametrize("level", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_quantile_levels(level):
    with pytest.raises(ValueError):
        exploratory_survival([1, 2]).survival_quantile(level)
