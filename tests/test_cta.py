"""Native Fortran CHISQT and independent contingency-table calculations."""

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import contingency_chi_square

CASES = json.loads((Path(__file__).parent / "fixtures/cta_chisqt.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_native_single_precision_chisqt(case):
    fit = contingency_chi_square(case["observed"], legacy=True)
    c, y, cc, p, yp, cp, minimum = case["statistics"]
    assert_allclose(fit.expected, case["expected"], rtol=2e-6)
    assert_allclose(
        [fit.statistic, fit.yates_statistic, fit.minimum_expected], [c, y, minimum], rtol=2e-6
    )
    assert_allclose([fit.pvalue, fit.yates_pvalue], [p, yp], atol=5e-7, rtol=0)
    if cc >= 0:
        assert_allclose(fit.cochran_statistic, cc, rtol=2e-6)
        assert_allclose(fit.cochran_pvalue, cp, atol=5e-7, rtol=0)
    else:
        assert fit.cochran_statistic is fit.cochran_pvalue is None


def test_fraction_oracle_and_df_two_exact_survival():
    data = [[2, 3, 5], [7, 11, 13]]
    rows = list(map(sum, data))
    cols = [sum(column) for column in zip(*data, strict=True)]
    total = sum(rows)
    expected = [[Fraction(r * c, total) for c in cols] for r in rows]
    statistic = sum(
        (Fraction(data[i][j]) - expected[i][j]) ** 2 / expected[i][j]
        for i in range(2)
        for j in range(3)
    )
    fit = contingency_chi_square(data)
    assert_allclose(fit.statistic, float(statistic), rtol=2e-15)
    assert_allclose(fit.pvalue, math.exp(-float(statistic) / 2), rtol=2e-15)
    assert fit.degrees_of_freedom == 2
    assert fit.yates_statistic is fit.yates_pvalue is None
    assert_allclose(fit.row_percent.sum(axis=-1), 100)
    assert_allclose(fit.column_percent.sum(axis=-2), 100)


def test_yates_does_not_create_disagreement_at_exact_independence():
    fit = contingency_chi_square([[10, 10], [10, 10]])
    assert fit.statistic == fit.yates_statistic == fit.cochran_statistic == 0
    assert fit.pvalue == fit.yates_pvalue == 1
    source = contingency_chi_square([[10, 10], [10, 10]], legacy=True)
    assert source.yates_statistic == pytest.approx(0.1)


def test_batches_transpose_permutations_and_threshold_equality():
    tables = np.array([[[12, 5], [7, 16]], [[7, 2], [2, 7]]])
    batch = contingency_chi_square(tables, expected_threshold=4.5)
    for i, table in enumerate(tables):
        fit = contingency_chi_square(table)
        assert_allclose(batch.statistic[i], fit.statistic)
        assert_allclose(contingency_chi_square(table.T).statistic, fit.statistic)
        assert_allclose(contingency_chi_square(table[::-1, ::-1]).statistic, fit.statistic)
    assert batch.percent_small_expected[1] == 100
    assert not batch.expected.flags.writeable and not batch.pvalue.flags.writeable


def test_large_counts_avoid_squared_count_and_margin_product_overflow():
    base = contingency_chi_square([[12, 5], [7, 16]])
    huge = contingency_chi_square(np.array([[12, 5], [7, 16]]) * 1e200)
    assert_allclose(huge.statistic, base.statistic * 1e200)
    assert np.isfinite(huge.cochran_statistic)
    assert huge.pvalue == 0


@pytest.mark.parametrize(
    "observed",
    [
        [[0, 0], [1, 2]],
        [[1, 0], [2, 0]],
        [[-1, 2], [3, 4]],
        [[np.nan, 2], [3, 4]],
        [1, 2],
        [[1, 2]],
        [[1e308, 1e308], [1, 2]],
    ],
)
def test_invalid_tables(observed):
    with pytest.raises(ValueError):
        contingency_chi_square(observed)


def test_input_snapshot_and_invalid_options():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    fit = contingency_chi_square(data)
    data[:] = 0
    assert_array_equal(fit.observed, [[1, 2], [3, 4]])
    for kwargs in [{"expected_threshold": -1}, {"expected_threshold": np.inf}, {"legacy": 1}]:
        with pytest.raises(ValueError):
            contingency_chi_square([[1, 2], [3, 4]], **kwargs)
