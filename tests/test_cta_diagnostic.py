"""Native SENSPEC orientations and independent conditional binomial errors."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import diagnostic_accuracy

CASES = json.loads((Path(__file__).parent / "fixtures/cta_diagnostic.json").read_text())["cases"]


def probabilities(fit):
    return np.stack(
        (
            fit.sensitivity,
            fit.specificity,
            fit.positive_predictive_value,
            fit.negative_predictive_value,
        ),
        axis=-1,
    )


@pytest.mark.parametrize("case", CASES)
def test_native_both_standard_axes_and_positive_classes(case):
    result = diagnostic_accuracy(
        case["observed"],
        standard=case["standard"],
        positive_index=case["positive_index"],
        legacy=True,
    )
    # PPV/NPV are available only in the original five-decimal text report.
    assert_allclose(probabilities(result), case["probabilities"], atol=5.1e-6, rtol=0)
    assert_allclose(result.standard_errors, case["source_errors"], atol=5.1e-6, rtol=0)


def test_conditional_binomial_errors_and_reference_orientation():
    fit = diagnostic_accuracy([[80, 10], [20, 90]])
    expected = [80 / 100, 90 / 100, 80 / 90, 90 / 110]
    sizes = [100, 100, 90, 110]
    assert_allclose(probabilities(fit), expected)
    assert_allclose(
        fit.standard_errors,
        [math.sqrt(p * (1 - p) / n) for p, n in zip(expected, sizes, strict=True)],
    )
    assert_array_equal(fit.denominators, sizes)
    transpose = diagnostic_accuracy([[80, 20], [10, 90]], standard="rows")
    assert_allclose(probabilities(transpose), probabilities(fit))
    assert_allclose(transpose.standard_errors, fit.standard_errors)
    source = diagnostic_accuracy([[80, 10], [20, 90]], legacy=True)
    assert_allclose(source.standard_errors, fit.standard_errors * fit.denominators)


def test_batch_scaling_and_positive_class_reversal():
    a = np.array([[80.0, 10], [20, 90]])
    fit = diagnostic_accuracy(np.stack([a, a * 100]))
    assert_allclose(probabilities(fit)[0], probabilities(fit)[1])
    assert_allclose(fit.standard_errors[0], 10 * fit.standard_errors[1])
    reversed_fit = diagnostic_accuracy(a, positive_index=1)
    assert_allclose(probabilities(reversed_fit), probabilities(fit)[0][[1, 0, 3, 2]])
    assert not fit.standard_errors.flags.writeable and not fit.sensitivity.flags.writeable


def test_zero_margins_are_local_undefined_metrics():
    fit = diagnostic_accuracy([[0, 2], [0, 5]])
    assert np.isnan(fit.sensitivity) and np.isnan(fit.standard_errors[0])
    assert fit.specificity == 5 / 7
    assert fit.positive_predictive_value == 0 and fit.negative_predictive_value == 1
    assert_array_equal(fit.standard_errors[2:], [0, 0])
    empty = diagnostic_accuracy(np.zeros((2, 2)))
    assert np.isnan(probabilities(empty)).all() and np.isnan(empty.standard_errors).all()


def test_rare_errors_are_retained_when_probability_rounds_to_one():
    fit = diagnostic_accuracy([[1e300, 1], [1, 1e300]])
    assert fit.sensitivity == 1
    assert_allclose(fit.standard_errors, [1e-300] * 4, rtol=1e-14, atol=0)
    source = diagnostic_accuracy([[1e300, 1], [1, 1e300]], legacy=True)
    assert_allclose(source.standard_errors, [1] * 4, rtol=1e-14)


@pytest.mark.parametrize(
    "table",
    [
        [[1, 2, 3], [4, 5, 6]],
        [[1, np.nan], [2, 3]],
        [[-1, 2], [3, 4]],
        [[1]],
        [[1e308, 1e308], [1, 2]],
    ],
)
def test_invalid_tables(table):
    with pytest.raises(ValueError):
        diagnostic_accuracy(table)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"standard": "bad"},
        {"positive_index": 2},
        {"positive_index": True},
        {"positive_index": 0.5},
        {"legacy": 1},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        diagnostic_accuracy([[1, 2], [3, 4]], **kwargs)
