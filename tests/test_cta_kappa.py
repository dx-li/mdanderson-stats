"""Native CTA legacy variances and exact multinomial delta-method oracle."""

import json
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import cohen_kappa

CASES = json.loads((Path(__file__).parent / "fixtures/cta_kappa.json").read_text())["cases"]


def exact_kappa_variance(table):
    n = sum(map(sum, table))
    m = len(table)
    p = [[F(value) / n for value in row] for row in table]
    rows = list(map(sum, p))
    cols = [sum(p[i][j] for i in range(m)) for j in range(m)]
    po = sum(p[i][i] for i in range(m))
    pe = sum(r * c for r, c in zip(rows, cols, strict=True))
    gradient = [
        [(F(i == j) * (1 - pe) - (1 - po) * (cols[i] + rows[j])) / (1 - pe) ** 2 for j in range(m)]
        for i in range(m)
    ]
    mean = sum(p[i][j] * gradient[i][j] for i in range(m) for j in range(m))
    second = sum(p[i][j] * gradient[i][j] ** 2 for i in range(m) for j in range(m))
    return float((po - pe) / (1 - pe)), float((second - mean**2) / n)


@pytest.mark.parametrize("case", CASES)
def test_native_asymmetric_formula_compatibility(case):
    fit = cohen_kappa(case["observed"], legacy=True)
    assert_allclose(
        [fit.kappa, fit.variance, fit.null_variance], case["result"], rtol=1e-5, atol=1e-8
    )


@pytest.mark.parametrize(
    "table",
    [[[10, 5], [7, 12]], [[3, 1, 7], [2, 6, 3], [4, 8, 9]], [[7, 0, 1], [0, 3, 2], [3, 1, 5]]],
)
def test_exact_multinomial_variance_and_null_model(table):
    fit = cohen_kappa(table)
    k, var = exact_kappa_variance(table)
    assert_allclose([fit.kappa, fit.variance], [k, var], rtol=2e-14, atol=1e-16)
    n = sum(map(sum, table))
    rows = list(map(sum, table))
    cols = list(map(sum, zip(*table, strict=True)))
    independent = [[F(r * c, n) for c in cols] for r in rows]
    null_k, null_var = exact_kappa_variance(independent)
    assert null_k == 0
    assert_allclose(fit.null_variance, null_var, rtol=2e-14)
    transposed = cohen_kappa(np.array(table).T)
    assert_allclose(
        [transposed.kappa, transposed.variance, transposed.null_variance],
        [fit.kappa, fit.variance, fit.null_variance],
        rtol=2e-14,
    )


def test_batch_scaling_perfect_agreement_and_undefined_single_category():
    a = np.array([[10, 5], [7, 12]])
    fit = cohen_kappa(np.stack([a, a * 10]))
    assert_allclose(fit.kappa[0], fit.kappa[1])
    assert_allclose(fit.variance[0], 10 * fit.variance[1])
    perfect = cohen_kappa([[10, 0], [0, 20]])
    assert perfect.kappa == 1 and perfect.variance == 0
    undefined = cohen_kappa([[10, 0], [0, 0]])
    assert np.isnan(undefined.kappa) and np.isnan(undefined.variance)
    assert np.isnan(undefined.null_variance)
    assert not fit.proportions.flags.writeable


def test_legacy_null_variance_is_orientation_dependent():
    table = np.array([[20, 2], [9, 1]])
    a, b = cohen_kappa(table, legacy=True), cohen_kappa(table.T, legacy=True)
    assert a.null_variance != b.null_variance
    assert_allclose(cohen_kappa(table).null_variance, cohen_kappa(table.T).null_variance)


@pytest.mark.parametrize(
    "table",
    [
        [[0, 0], [0, 0]],
        [[1, 2, 3], [4, 5, 6]],
        [[1, np.nan], [2, 3]],
        [[-1, 2], [3, 4]],
        [[1]],
        [[1e308, 1e308], [1, 1]],
    ],
)
def test_invalid_input(table):
    with pytest.raises(ValueError):
        cohen_kappa(table)


def test_no_input_mutation_and_boolean_option():
    a = np.array([[1.0, 2], [3, 4]])
    before = a.copy()
    fit = cohen_kappa(a)
    assert_array_equal(a, before)
    a[:] = 0
    assert_allclose(fit.proportions, before / 10)
    with pytest.raises(ValueError):
        cohen_kappa(before, legacy=1)


@pytest.mark.parametrize("rare", [1e-150, 1e-250])
def test_extreme_rare_agreement_avoids_zero_times_squared_overflow(rare):
    fit = cohen_kappa([[1, 0], [0, rare]])
    assert fit.kappa == 1 and fit.variance == 0
    assert_allclose(fit.null_variance, 1, rtol=2e-14)
