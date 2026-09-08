"""Native shared-parameter information and independent contrast identities."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import single_two_sample_precision

CASES = json.loads((Path(__file__).parent / "fixtures/single_two_sample.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_native_two_group_information(case):
    result = single_two_sample_precision(
        ([-1, 0, 1, 2], [-1, 0, 1, 2]),
        ([10, 20, 30, 40], [10, 20, 30, 40]),
        case["parameters"],
        model=case["model"],
        form=case["form"],
        comparison=case["comparison"],
    )
    assert_allclose(result.probability, case["probabilities"], rtol=1e-13)
    assert_allclose(result.information, case["information"], rtol=1e-12, atol=1e-13)
    # Source CRIT uses Var(theta1)+Var(theta2)-2 Cov(theta1,theta2).
    compared = 0 if (case["form"] == "linear") == (case["comparison"] == "location") else 1
    covariance = np.linalg.inv(case["information"])
    expected = covariance[compared, compared] + covariance[2, 2] - 2 * covariance[compared, 2]
    assert_allclose(result.variance, expected, rtol=1e-12)
    assert_allclose(result.sd, np.sqrt(expected))


@pytest.mark.parametrize("form", ["linear", "centered"])
@pytest.mark.parametrize("comparison", ["location", "slope"])
def test_symmetric_logistic_closed_form(form, comparison):
    first = [0, 1] if form == "linear" else [1, 0]
    b = first + [0 if comparison == "location" else 1]
    result = single_two_sample_precision(
        ([-1, 1], [-1, 1]), ([50, 50], [50, 50]), b, form=form, comparison=comparison
    )
    weight = np.exp(-1) / (1 + np.exp(-1)) ** 2
    assert_allclose(result.variance, 2 / (100 * weight))
    assert_allclose(result.difference, 0)


@pytest.mark.parametrize("model", ["logistic", "loglog"])
@pytest.mark.parametrize("form", ["linear", "centered"])
@pytest.mark.parametrize("comparison", ["location", "slope"])
def test_group_swap_broadcast_and_allocation_scaling(model, form, comparison):
    x = ([-1, 1], [-0.5, 0, 1.5])
    n = (np.array([20, 30]), np.array([30, 20, 50]))
    b = np.array([[0.3, 1.2, 0.7], [-0.5, 0.75, -0.2]])
    options = dict(model=model, form=form, comparison=comparison)
    result = single_two_sample_precision(x, n, b, **options)
    index = 0 if (form == "linear") == (comparison == "location") else 1
    swapped_b = b.copy()
    swapped_b[:, [index, 2]] = b[:, [2, index]]
    swapped = single_two_sample_precision(x[::-1], n[::-1], swapped_b, **options)
    assert_allclose(swapped.variance, result.variance)
    assert_allclose(swapped.difference, -result.difference)
    scaled = single_two_sample_precision(x, (2 * n[0], 2 * n[1]), b, **options)
    assert_allclose(scaled.variance, result.variance / 2)
    for i in range(2):
        scalar = single_two_sample_precision(x, n, b[i], **options)
        assert_allclose(scalar.variance, result.variance[i])
        assert_allclose(scalar.information, result.information[i])


def test_one_dose_in_a_group_can_be_identifiable():
    result = single_two_sample_precision(([-1, 1], [0]), ([50, 50], [100]), [0, 1, 0])
    w = np.exp(-1) / (1 + np.exp(-1)) ** 2
    assert_allclose(result.variance, 1 / (100 * w) + 1 / 25)


def test_zero_slope_is_valid_in_linear_form():
    result = single_two_sample_precision(([-1, 1], [-1, 1]), ([50, 50], [50, 50]), [0, 0, 0])
    assert_allclose(result.variance, 0.08)


@pytest.mark.parametrize(
    "changes",
    [
        {"doses": ([0], [0]), "subjects": ([100], [100])},
        {"doses": ([1, 1], [1, 1])},
        {"subjects": ([0, 0], [50, 50])},
        {"subjects": ([-1, 50], [50, 50])},
        {"subjects": ([50], [50, 50])},
        {"parameters": [1, 2]},
        {"parameters": [0, 0, 0], "form": "centered"},
        {"parameters": [0, np.nan, 0]},
        {"comparison": "ratio"},
        {"model": "probit"},
        {"form": "quadratic"},
        {"doses": ([-1, 1],)},
    ],
)
def test_invalid_or_unidentified_design(changes):
    args = dict(doses=([-1, 1], [-1, 1]), subjects=([50, 50], [50, 50]), parameters=[0, 1, 0])
    args.update(changes)
    with pytest.raises(ValueError):
        single_two_sample_precision(**args)
