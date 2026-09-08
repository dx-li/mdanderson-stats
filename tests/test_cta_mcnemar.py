"""CTA native decomposition and independent weighted-contrast calculations."""

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import mcnemar_analysis

CASES = json.loads((Path(__file__).parent / "fixtures/cta_mcnemar.json").read_text())["cases"]


def integer_chi_tail(statistic, df):
    """Finite gamma recurrence from exponential or erfc, independent of SciPy."""
    x = statistic / 2
    if df % 2 == 0:
        return math.exp(-x) * sum(x**j / math.factorial(j) for j in range(df // 2))
    result = math.erfc(math.sqrt(x))
    for j in range((df - 1) // 2):
        shape = j + 0.5
        result += x**shape * math.exp(-x) / math.gamma(shape + 1)
    return result


@pytest.mark.parametrize("case", CASES)
def test_native_defined_mcnemar_cases(case):
    result = mcnemar_analysis(case["observed"])
    expected = case["result"]
    assert_allclose(
        [result.summed_statistic, result.pooled_statistic, result.heterogeneity_statistic],
        expected[:3],
        rtol=3e-6,
        atol=2e-7,
    )
    native_sum = result.summed_pvalue
    df = int(result.summed_df)
    if df > 2 and result.summed_statistic >= df:
        # The archived OVERFL stub always returns 1, forcing this approximation.
        z = ((float(result.summed_statistic) / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(
            2 / (9 * df)
        )
        native_sum = 0.5 * math.erfc(z / math.sqrt(2))
    assert_allclose([native_sum, result.pooled_pvalue], expected[3:5], rtol=0, atol=5e-7)
    if df > 0:
        assert_allclose(
            result.summed_pvalue, integer_chi_tail(float(result.summed_statistic), df), rtol=2e-14
        )
    if len(case["observed"]) > 2:
        assert_allclose(result.heterogeneity_pvalue, expected[5], rtol=0, atol=5e-7)
    else:
        assert result.heterogeneity_df == 0 and np.isnan(result.heterogeneity_pvalue)


def test_rational_decomposition_and_orientation():
    table = np.array([[5, 1, 7], [2, 8, 3], [4, 8, 2]])
    result = mcnemar_analysis(table)
    pair = [Fraction(1, 3), Fraction(9, 11), Fraction(25, 11)]
    summed = sum(pair)
    pooled = Fraction((11 - 14) ** 2, 25)
    assert_allclose(result.pair_statistic, list(map(float, pair)), rtol=1e-15)
    assert_allclose(result.summed_statistic, float(summed), rtol=1e-15)
    assert_allclose(result.pooled_statistic, float(pooled), rtol=1e-15)
    assert_allclose(result.heterogeneity_statistic, float(summed - pooled), rtol=1e-15)
    assert_array_equal(result.pairs, [[0, 1], [0, 2], [1, 2]])
    transposed = mcnemar_analysis(table.T)
    assert_allclose(transposed.heterogeneity_statistic, result.heterogeneity_statistic)
    assert_array_equal(transposed.above, result.below)
    assert result.summed_df == 3 and result.heterogeneity_df == 2


def test_zero_pairs_no_discordance_and_single_active_pair():
    none = mcnemar_analysis(np.eye(3) * 10)
    assert none.summed_statistic == none.pooled_statistic == none.heterogeneity_statistic == 0
    assert none.summed_df == none.pooled_df == none.heterogeneity_df == 0
    assert np.isnan(none.summed_pvalue) and np.isnan(none.pooled_pvalue)
    assert np.isnan(none.pair_pvalue).all()
    one = mcnemar_analysis([[1, 2, 0], [4, 1, 0], [0, 0, 1]])
    assert one.summed_df == 1 and one.heterogeneity_df == 0
    assert one.heterogeneity_statistic == 0
    assert_allclose(one.summed_statistic, 2 / 3)


def test_batch_large_counts_and_nearly_homogeneous_contrasts():
    table = np.array([[1.0, 2, 4], [1, 1, 6], [2, 3, 1]])
    result = mcnemar_analysis(np.stack([table, table * 1e200]))
    assert_allclose(result.summed_statistic[1], result.summed_statistic[0] * 1e200)
    assert np.all(result.heterogeneity_statistic / result.summed_statistic < 1e-30)
    assert not result.pair_statistic.flags.writeable
    table[0, 1] = 2.00000001
    nearby = mcnemar_analysis(table)
    assert nearby.heterogeneity_statistic > 0
    assert np.isfinite(nearby.heterogeneity_statistic)


@pytest.mark.parametrize(
    "table",
    [
        [[1, 2, 3], [4, 5, 6]],
        [[-1, 2], [3, 4]],
        [[1, np.nan], [2, 3]],
        [[1]],
        [[1, 1e308], [1e308, 1]],
    ],
)
def test_invalid_tables(table):
    with pytest.raises(ValueError):
        mcnemar_analysis(table)
