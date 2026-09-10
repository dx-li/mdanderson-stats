import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats.survan_descriptive import survan_describe, survan_frequencies


def test_original_summaries_and_frequency_table():
    root = Path(__file__).parent / "fixtures"
    x = np.asarray(json.loads((root / "survan-native.json").read_text())[0]["records"])
    ref = json.loads((root / "survan-descriptive-native.json").read_text())
    r = survan_describe(x)
    assert_allclose(
        np.column_stack(
            (r.mean, r.standard_deviation, r.legacy_standard_error, r.minimum, r.maximum, r.count)
        ),
        ref["summary"],
        rtol=2e-6,
    )
    assert_allclose(r.standard_error, r.standard_deviation / np.sqrt(40))
    f = survan_frequencies(x[:, 2])
    assert_array_equal(np.column_stack((f.value, f.count)), ref["frequency"])
    assert_allclose(f.percent, [47.5, 52.5])


def test_shift_stability_missing_values_and_unrepresentable_variance():
    r = survan_describe(1e12 + np.array([0.0, 1.0, 2.0]))
    assert_allclose(r.variance, [1], atol=1e-14)
    assert_allclose(r.mean, [1e12 + 1], rtol=0, atol=0)
    missing = survan_describe([[1, np.nan], [np.nan, np.nan]])
    assert_array_equal(missing.count, [1, 0])
    assert np.isnan(missing.variance).all()
    assert np.isnan(missing.mean[1])
    with pytest.raises(ArithmeticError, match="variance"):
        survan_describe([-1e200, 1e200])
    with pytest.raises(ValueError, match="infinite"):
        survan_describe([1, np.inf])


def test_native_grouping_is_order_dependent_and_missing_denominator():
    values = [1, 1.000015, 1.00003, np.nan]
    exact = survan_frequencies(values)
    assert_array_equal(exact.count, [1, 1, 1])
    legacy = survan_frequencies(values, legacy_grouping=True)
    assert_array_equal(legacy.count, [2, 1])
    reordered = survan_frequencies([values[1], values[0], values[2]], legacy_grouping=True)
    assert_array_equal(reordered.count, [3])
    assert legacy.missing == 1 and legacy.observations == 3
    assert legacy.cumulative_percent[-1] == 100
