"""Rank normalization and independent PDNN energy/expression identities."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import pdnn_binding_energy, pdnn_expression, pdnn_signal, quantile_normalize


def test_quantile_normalization_reference_average_and_ties():
    x = [[3, 20], [1, 30], [2, 10]]
    assert_array_equal(quantile_normalize(x), [[16.5, 11], [5.5, 16.5], [11, 5.5]])
    assert_array_equal(
        quantile_normalize(x, reference=[10, 30, 20]), [[30, 20], [10, 30], [20, 10]]
    )
    assert_array_equal(
        quantile_normalize([[1, 4], [1, 6], [3, 8]], reference=[10, 20, 30]),
        [[15, 10], [15, 20], [30, 30]],
    )
    large = quantile_normalize(np.full((3, 2), 1e308))
    assert_allclose(large, 1e308)


def test_sequence_energies_and_known_expression_recovery():
    sequences = ["ACGT" * 6 + "A", "T" * 25, "AC" * 12 + "G", "G" * 25]
    stacking = np.arange(16).reshape(4, 4) / 100
    weights = np.linspace(0.5, 1.5, 24)
    expected = [
        sum(stacking["ACGT".index(s[i]), "ACGT".index(s[i + 1])] * weights[i] for i in range(24))
        for s in sequences
    ]
    e = pdnn_binding_energy(sequences, stacking, weights)
    assert_allclose(e, expected, atol=1e-14)
    ns = np.array([1.0, 2.0, 3.0, 4.0])
    ids = [10, 20, 10, 20]
    expression = np.array([100.0, 200.0, 100.0, 200.0])
    observed = expression / (1 + np.exp(e)) + 50 / (1 + np.exp(ns)) + 10
    result = pdnn_expression(observed, ids, e, ns, nonspecific_amount=50, background=10)
    assert_array_equal(result.probeset_ids, [10, 20])
    assert_allclose(np.exp(result.log_expression), [100, 200], rtol=1e-13)
    assert_allclose(result.fitted_signal, observed, rtol=1e-13)
    assert_array_equal(result.probes_used, [2, 2])


def test_weighted_equation_exclusions_and_log_stability():
    x = np.array([4.0, 9.0, 16.0, 0.1, 10000])
    e, ns = np.zeros(5), np.zeros(5)
    fit = pdnn_expression(
        x,
        [0] * 5,
        e,
        ns,
        nonspecific_amount=0,
        background=1,
        reference_fitted=[4, 9, 16, 0.1, 1],
        fitness=1,
    )
    # Eq. (5), with lambda=sqrt(I/affinity); negative residual and outlier excluded.
    invlambda = np.sqrt(0.5 / x[:3])
    expected = np.sum((x[:3] - 1) * invlambda) / np.sum(0.5 * invlambda)
    assert_allclose(fit.log_expression, np.log(expected), rtol=1e-14)
    assert_array_equal(fit.included, [True, True, True, False, False])
    assert_array_equal(fit.probes_used, [3])
    assert_allclose(
        pdnn_signal([1000], [-1000], log_expression=[1000], nonspecific_amount=2, background=3),
        [6],
        rtol=1e-13,
    )
    huge = pdnn_expression([1], [0], [1000], [0], nonspecific_amount=0, background=0)
    assert_allclose(huge.log_expression, 1000, atol=1e-12)
    assert_allclose(huge.fitted_signal, 1)
    with pytest.raises(ValueError, match="no usable"):
        pdnn_expression([1], [0], [0], [0], nonspecific_amount=0, background=2)
