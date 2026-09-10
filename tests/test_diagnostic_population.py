"""Published examples, exact tables, Bayes identities and binormal ROC checks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from mdanderson_stats import BinormalROC, diagnostic_population, diagnostic_population_from_counts


def test_diag_manual_table_and_population_projection():
    table = np.array([[5, 4], [3, 2]])
    result = diagnostic_population_from_counts(table)
    assert_allclose(
        [result.sensitivity, result.specificity, result.prevalence], [5 / 8, 1 / 3, 4 / 7]
    )
    assert_allclose(
        [
            result.positive_predictive_value,
            result.negative_predictive_value,
            result.false_discovery_rate,
        ],
        [5 / 9, 2 / 5, 4 / 9],
    )
    assert_allclose(result.expected_counts, table * (10000 / 14))
    assert not result.expected_counts.flags.writeable


def test_prevalence_changes_target_not_sensitivity_or_specificity():
    p = np.linspace(0, 1, 101)
    result = diagnostic_population_from_counts([[5, 4], [3, 2]], prevalence=p)
    expected_ppv = (p * 5 / 8) / (p * 5 / 8 + (1 - p) * 2 / 3)
    assert_allclose(result.positive_predictive_value, expected_ppv, atol=2e-15)
    assert np.all(np.diff(result.positive_predictive_value) > 0)
    assert np.all(np.diff(result.negative_predictive_value) < 0)
    assert_allclose(result.expected_counts.sum(axis=(-2, -1)), 10000)
    assert_allclose(result.expected_counts[..., :, 0].sum(axis=-1), p * 10000, atol=1e-11)


def test_boundary_and_underflow_conditionals():
    perfect = diagnostic_population(1, 1, [0.0, 0.5, 1])
    assert np.isnan(perfect.positive_predictive_value[0])
    assert np.isnan(perfect.negative_predictive_value[-1])
    assert_allclose(perfect.positive_predictive_value[1:], 1)
    # Both joint event probabilities can underflow; their ratio still exists.
    extreme = diagnostic_population(1e-300, 0, 1e-300)
    assert extreme.positive_predictive_value == 0
    point = BinormalROC(0, 1, 0, 1).at_threshold(40, prevalence=0.3)
    assert point.diagnostic.joint_probabilities[0].sum() == 0
    assert_allclose(point.diagnostic.positive_predictive_value, 0.3, rtol=1e-12)
    assert_allclose(point.diagnostic.false_discovery_rate, 0.7, rtol=1e-12)
    assert np.all(diagnostic_population(0.8, 0.9, 0.2, population=0).expected_counts == 0)


def test_dtroc_manual_example():
    result = BinormalROC().at_percentile(0.8)
    d = result.diagnostic
    assert_allclose(result.threshold, -0.1583787664, atol=1e-10)
    assert_allclose(
        [
            d.sensitivity,
            d.specificity,
            result.auc,
            d.positive_predictive_value,
            d.negative_predictive_value,
        ],
        [0.8766457, 0.8, 0.9213504, 0.8142379, 0.8664062],
        atol=1e-6,
    )


def test_auc_against_integrated_curve_and_pairwise_simulation():
    model = BinormalROC(-1, 0.7, 1, 1.5)
    numerical, error = quad(lambda fpr: float(model.curve(fpr)), 0, 1, epsabs=1e-10)
    assert error < 1e-8
    assert_allclose(numerical, model.auc, atol=2e-10)
    rng = np.random.default_rng(104)
    control = rng.normal(-1, 0.7, 100000)
    disease = rng.normal(1, 1.5, 100000)
    assert_allclose(np.mean(disease > control), model.auc, atol=0.003)


def test_diagonal_reflection_and_endpoints():
    fpr = np.array([0, 1e-50, 0.1, 0.5, 0.9, 1])
    equal = BinormalROC(2, 3, 2, 3)
    assert_allclose(equal.curve(fpr), fpr, rtol=1e-12)
    assert equal.auc == 0.5
    forward, reverse = BinormalROC(), BinormalROC(1, 1, -1, 1)
    assert_allclose(forward.auc + reverse.auc, 1)
    ends = forward.at_percentile([0, 1]).diagnostic
    assert_allclose(ends.sensitivity, [1, 0])
    assert_allclose(ends.specificity, [0, 1])
    assert np.isnan(ends.negative_predictive_value[0])
    assert np.isnan(ends.positive_predictive_value[1])


def test_large_scale_and_quantile_representability():
    baseline = BinormalROC(-1, 1, 1, 1).auc
    scaled = BinormalROC(-1e308, 1e308, 1e308, 1e308)
    assert_allclose(scaled.auc, baseline)
    offset = BinormalROC(1e20, 1, 1e20, 1)
    assert_allclose(offset.curve([0.1, 0.5, 0.9]), [0.1, 0.5, 0.9])
    with pytest.raises(ArithmeticError):
        offset.at_percentile(0.8)


def test_batched_models_and_density_normalization():
    model = BinormalROC(control_mean=np.array([0, 1])[:, None])
    curve = model.curve([0.1, 0.5, 0.9])
    assert curve.shape == (2, 3)
    density = model.densities(np.linspace(-8, 10, 10001))
    assert density.shape == (2, 10001, 2)
    assert_allclose(np.trapezoid(density, np.linspace(-8, 10, 10001), axis=1), 1, atol=1e-12)


def test_invalid_inputs():
    for function in [
        lambda: diagnostic_population(1.1, 0.5, 0.5),
        lambda: diagnostic_population_from_counts([[1, 0], [0, 0]]),
        lambda: diagnostic_population_from_counts([[1, 0.5], [2, 3]]),
        lambda: BinormalROC(control_sd=0),
        lambda: BinormalROC().at_threshold(np.nan),
    ]:
        with pytest.raises(ValueError):
            function()
