import json
import math
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.goodness_of_fit import chi_square_gof


def test_original_gofchi_examples():
    cases = json.loads((Path(__file__).parent / "fixtures/numerics.json").read_text())["gof"]
    for case in cases:
        result = chi_square_gof(
            case["observed"],
            case["weights"],
            degrees_of_freedom=case["degrees_of_freedom"],
            legacy_zero_expected=True,
        )
        # The legacy program prints the statistic to 2 and p-value to 4 decimals.
        assert float(result.statistic) == pytest.approx(case["statistic"], abs=0.00501)
        assert float(result.pvalue) == pytest.approx(case["pvalue"], abs=0.000051)


def test_hand_calculation_and_df_two_exact_tail():
    result = chi_square_gof([6, 9, 5], [2, 3, 5])
    np.testing.assert_allclose(result.expected, [4, 6, 10])
    np.testing.assert_allclose(result.contributions, [1, 1.5, 2.5])
    assert result.statistic == pytest.approx(5)
    assert result.pvalue == pytest.approx(math.exp(-2.5), rel=1e-14)


def test_small_pvalue_without_subtractive_cancellation():
    result = chi_square_gof([300, 0, 0], [1, 1, 1])
    assert result.pvalue == pytest.approx(math.exp(-300), rel=1e-13, abs=0)
    assert result.pvalue > 0


def test_weights_are_relative_and_batch_dimensions_broadcast():
    observed = np.array([[6, 9, 5], [4, 6, 10]])
    result = chi_square_gof(observed, [2e300, 3e300, 5e300])
    np.testing.assert_allclose(result.statistic, [5, 0])
    np.testing.assert_allclose(result.pvalue, [math.exp(-2.5), 1])
    alternative = chi_square_gof(observed, [2, 3, 5], degrees_of_freedom=[1, 2])
    assert alternative.pvalue[0] != result.pvalue[0]
    assert alternative.pvalue[1] == 1
    np.testing.assert_array_equal(observed, [[6, 9, 5], [4, 6, 10]])


def test_zero_null_probability_is_not_ignored():
    result = chi_square_gof([5, 10, 15], [0, 1, 1])
    assert result.statistic == np.inf and result.pvalue == 0
    legacy = chi_square_gof([5, 10, 15], [0, 1, 1], legacy_zero_expected=True)
    assert legacy.statistic == pytest.approx(25 / 15)
    assert legacy.pvalue > 0
    empty_category = chi_square_gof([0, 12, 8], [0, 3, 2])
    assert empty_category.degrees_of_freedom == 1
    assert empty_category.pvalue == pytest.approx(1, abs=1e-14)


@pytest.mark.parametrize(
    "observed,weights,kwargs",
    [
        ([1], [1], {}),
        ([0, 0], [1, 1], {}),
        ([1, 2], [0, 0], {}),
        ([-1, 2], [1, 1], {}),
        ([1, 2], [-1, 1], {}),
        ([1, 2], [1, math.nan], {}),
        ([1, math.inf], [1, 1], {}),
        ([1, 2], [1, 1], {"degrees_of_freedom": 0}),
        ([1, 2], [1, 1], {"degrees_of_freedom": 0.5}),
        ([1, 2], [1, 0], {}),
    ],
)
def test_invalid_gof_inputs(observed, weights, kwargs):
    with pytest.raises(ValueError):
        chi_square_gof(observed, weights, **kwargs)
