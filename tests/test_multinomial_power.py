"""Independent exact probabilities and invariants for MULTINOMPOW."""

from fractions import Fraction
from itertools import product
from math import factorial, log

import numpy as np
import pytest

from mdanderson_stats import MultinomialPower, multinomial_power


def reference(n, null, alternative, alpha, statistic):
    rows = []
    for counts in product(range(n + 1), repeat=len(null)):
        if sum(counts) != n:
            continue
        coefficient = Fraction(factorial(n))
        for count in counts:
            coefficient /= factorial(count)
        p0 = coefficient
        pa = coefficient
        for count, p, q in zip(counts, null, alternative, strict=True):
            p0 *= p**count
            pa *= q**count
        if statistic == "pearson":
            value = sum((count - n * p) ** 2 / (n * p) for count, p in zip(counts, null))
        else:
            value = sum(
                2 * count * log(float(count / (n * p))) for count, p in zip(counts, null) if count
            )
        rows.append((float(value), p0, pa))
    rows.sort(reverse=True)
    groups = []
    for value, p0, pa in rows:
        if groups and abs(value - groups[-1][0]) < 1e-11:
            groups[-1][1] += p0
            groups[-1][2] += pa
        else:
            groups.append([value, p0, pa])
    size = power = Fraction(0)
    for _, p0, pa in groups:
        if size + p0 > alpha:
            break
        size += p0
        power += pa
    return float(size), float(power)


@pytest.mark.parametrize("n", [1, 3, 7])
@pytest.mark.parametrize(
    "null", [[Fraction(1, 3)] * 3, [Fraction(1, 5), Fraction(3, 10), Fraction(1, 2)]]
)
@pytest.mark.parametrize(
    "level",
    [Fraction(0), Fraction(1, 100), Fraction(1, 20), Fraction(1, 4), Fraction(4, 5), Fraction(1)],
)
@pytest.mark.parametrize("statistic", ["pearson", "likelihood_ratio"])
def test_rational_enumeration(n, null, level, statistic):
    alternative = [Fraction(1, 10), Fraction(1, 5), Fraction(7, 10)]
    expected_size, expected_power = reference(n, null, alternative, level, statistic)
    result = multinomial_power(n, null, alternative, float(level), statistic=statistic)
    assert isinstance(result, MultinomialPower)
    assert result.actual_size[0, 0] == pytest.approx(expected_size, abs=2e-14)
    assert result.power[0, 0, 0] == pytest.approx(expected_power, abs=2e-14)


def test_binary_ties_and_empty_region():
    result = multinomial_power(3, [0.5, 0.5], [[0.8, 0.2], [0.5, 0.5], [1, 0]], [0, 0.24, 0.25, 1])
    np.testing.assert_allclose(result.actual_size, [[0, 0, 0.25, 1]] * 2)
    np.testing.assert_allclose(result.power[:, 0], [[0, 0, 0.52, 1]] * 2)
    np.testing.assert_allclose(result.power[:, 1], result.actual_size)
    np.testing.assert_allclose(result.power[:, 2], [[0, 0, 1, 1]] * 2)
    assert np.isinf(result.critical_values[:, :2]).all()


def test_category_permutation_and_unsorted_duplicate_levels():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    q = np.array([[0, 0.2, 0.3, 0.5], [0.4, 0.3, 0.2, 0.1]])
    levels = [0.5, 0, 0.05, 1, 0.05]
    first = multinomial_power(12, p, q, levels)
    second = multinomial_power(12, p[::-1], q[:, ::-1], levels)
    np.testing.assert_allclose(first.actual_size, second.actual_size, atol=2e-14)
    np.testing.assert_allclose(first.power, second.power, atol=2e-14)
    np.testing.assert_allclose(first.critical_values, second.critical_values)
    np.testing.assert_array_equal(first.power[:, :, 2], first.power[:, :, 4])
    assert first.sample_space_size == 455


def test_owned_immutable_results():
    p = np.array([0.4, 0.6])
    result = multinomial_power(10, p, p)
    p[0] = 0
    assert result.null[0] == 0.4
    for field in (
        result.null,
        result.alternatives,
        result.alpha,
        result.power,
        result.actual_size,
        result.critical_values,
    ):
        with pytest.raises(ValueError):
            field.setflags(write=True)


@pytest.mark.parametrize(
    "options",
    [
        {"n": 0},
        {"n": True},
        {"n": 2.5},
        {"n": 10001},
        {"null": [0, 1]},
        {"null": [0.4, 0.4]},
        {"null": [0.5, float("nan")]},
        {"null": [[0.5, 0.5]]},
        {"null": [1]},
        {"alternatives": []},
        {"alternatives": [[0.5, 0.4]]},
        {"alternatives": [0.2, 0.3, 0.5]},
        {"alternatives": [-0.1, 1.1]},
        {"alpha": []},
        {"alpha": [[0.05]]},
        {"alpha": -1},
        {"alpha": 1.1},
        {"statistic": "unknown"},
        {"max_points": True},
        {"max_points": 0},
        {"max_points": 2.5},
        {"max_points": 1},
    ],
)
def test_invalid_inputs(options):
    arguments = {"n": 3, "null": [0.5, 0.5], "alternatives": [0.8, 0.2]}
    arguments.update(options)
    with pytest.raises(ValueError):
        multinomial_power(**arguments)


def test_preallocation_budget():
    with pytest.raises(ValueError, match="exceeding max_points"):
        multinomial_power(10000, [0.1] * 10, [0.1] * 10)


def test_normalization_and_numerical_overflow():
    result = multinomial_power(3, [0.5, 0.5 + 1e-14], [0.2, 0.8 + 1e-14], 1)
    np.testing.assert_array_equal(result.power, 1)
    with pytest.raises(ArithmeticError, match="overflow"):
        multinomial_power(3, [1e-320, 1], [0.5, 0.5])
