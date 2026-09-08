"""Native assignments and independent weighted-category checks."""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import ranlist_unrestricted

CASES = json.loads((Path(__file__).parent / "fixtures/ranlist_unrestricted.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", CASES)
def test_native_assignments_and_float32_cumulative_weights(case):
    fit = ranlist_unrestricted(
        case["patients"],
        case["weights"],
        seed=tuple(case["seed"]),
        stream=case["stream"],
        legacy=True,
    )
    assert_array_equal(fit.treatments, case["treatments"])
    assert_array_equal(
        fit.cumulative_probabilities.astype(np.float32),
        np.array(case["cumulative"], dtype=np.float32),
    )


def test_independent_integer_recurrence_and_rational_weighted_choice():
    weights = [1, 2, 3]
    cumulative = [Fraction(1, 6), Fraction(3, 6), Fraction(6, 6)]
    first, second = 1234567890, 123456789
    expected = []
    for _ in range(1000):
        first = 40014 * first % 2147483563
        second = 40692 * second % 2147483399
        raw = first - second
        if raw < 1:
            raw += 2147483562
        u = Fraction(raw, 2147483563)
        expected.append(next(i + 1 for i, p in enumerate(cumulative) if u <= p))
    patients = np.array([[1000, 1, 30], [10, 30, 2]])
    fit = ranlist_unrestricted(patients, weights)
    assert_array_equal(fit.treatments, np.array(expected)[patients - 1])
    assert_allclose(fit.probabilities, [1 / 6, 2 / 6, 3 / 6], rtol=1e-15)
    assert_array_equal(fit.patients, patients)


def test_inclusive_source_boundary_and_default_precision_difference():
    seed = ((1073741825 * pow(40014, -1, 2147483563)) % 2147483563, pow(40692, -1, 2147483399))
    source = ranlist_unrestricted(1, [1, 1], seed=seed, legacy=True)
    modern = ranlist_unrestricted(1, [1, 1], seed=seed)
    assert source.uniforms == 0.5 and source.treatments == 1
    assert modern.uniforms > 0.5 and modern.treatments == 2


def test_weight_scaling_snapshots_and_single_treatment():
    weights = np.array([1.0, 2, 3])
    patients = np.arange(1, 1001)
    fit = ranlist_unrestricted(patients, weights)
    scaled = ranlist_unrestricted(patients, weights * 1e300)
    assert_array_equal(fit.treatments, scaled.treatments)
    weights[:] = 1
    patients[:] = 1
    assert_array_equal(fit.weights, [1, 2, 3])
    assert_array_equal(fit.patients, np.arange(1, 1001))
    for a in (
        fit.patients,
        fit.treatments,
        fit.weights,
        fit.probabilities,
        fit.cumulative_probabilities,
        fit.uniforms,
    ):
        assert not a.flags.writeable
    assert_array_equal(ranlist_unrestricted([1, 10000], [1]).treatments, [1, 1])
    assert ranlist_unrestricted(np.empty((0, 2)), [1, 1]).treatments.shape == (0, 2)


def test_repeated_and_interleaved_stream_requests_are_reproducible():
    first = ranlist_unrestricted([1, 2, 3], [1, 2, 3], stream=1)
    other = ranlist_unrestricted([10, 1, 2], [1, 2, 3], stream=20)
    assert_array_equal(
        first.treatments, ranlist_unrestricted([1, 2, 3], [1, 2, 3], stream=1).treatments
    )
    assert_array_equal(
        other.treatments[1:], ranlist_unrestricted([1, 2], [1, 2, 3], stream=20).treatments
    )


def test_large_fixed_sample_allocation_frequencies():
    fit = ranlist_unrestricted(np.arange(1, 100001), [1, 2, 3])
    frequencies = np.bincount(fit.treatments, minlength=4)[1:] / 100000
    assert_allclose(frequencies, [1 / 6, 2 / 6, 3 / 6], atol=0.005, rtol=0)


def test_legacy_uncovered_tail_is_an_error_not_an_out_of_bounds_assignment():
    seed = (65421664, 1481316021)
    with pytest.raises(ArithmeticError, match="do not cover"):
        ranlist_unrestricted(1, np.ones(12), seed=seed, legacy=True)
    assert 1 <= ranlist_unrestricted(1, np.ones(12), seed=seed).treatments <= 12


@pytest.mark.parametrize(
    "weights", [[], [[1, 2]], [0, 1], [-1, 2], [np.nan, 1], [np.inf, 1], np.ones(21), [1, 1e-300]]
)
def test_invalid_or_unresolvable_weights(weights):
    with pytest.raises(ValueError):
        ranlist_unrestricted(1, weights)


@pytest.mark.parametrize("weights", [[1e300, 1e300], [1e-300, 1e-300]])
def test_legacy_rejects_unrepresentable_weights_but_default_scales_safely(weights):
    with pytest.raises(ValueError):
        ranlist_unrestricted(1, weights, legacy=True)
    assert_allclose(ranlist_unrestricted(1, weights).probabilities, [0.5, 0.5])


@pytest.mark.parametrize("patients", [0, -1, 1.5, np.nan, 2**53])
def test_invalid_patients(patients):
    with pytest.raises(ValueError):
        ranlist_unrestricted(patients, [1, 1])


def test_invalid_settings():
    for kwargs in ({"legacy": 1}, {"stream": 0}, {"seed": (0, 1)}):
        with pytest.raises(ValueError):
            ranlist_unrestricted(1, [1, 1], **kwargs)


def test_legacy_retains_collapsed_interval_from_source_rounding():
    seed = (65421664, 1481316021)
    source = ranlist_unrestricted(1, [1, 1e-8], seed=seed, legacy=True)
    assert_array_equal(source.cumulative_probabilities, [1, 1])
    assert source.treatments == 1
    assert ranlist_unrestricted(1, [1, 1e-8], seed=seed).treatments == 2
