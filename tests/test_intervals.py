"""Numerical checks independent of the implementation's inverse functions."""

import json
from math import comb, exp, factorial
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import binomial_interval, bp1ci_poisson_interval, poisson_interval


def test_published_binomial_example():
    low, high = binomial_interval(12, 30)
    assert float(low) == pytest.approx(0.2266, abs=0.00005)
    assert float(high) == pytest.approx(0.5940, abs=0.00005)


@pytest.mark.parametrize("n", [1, 5, 20, 50])
def test_binomial_tails_and_frequentist_coverage(n):
    low, high = binomial_interval(np.arange(n + 1), n)
    for k in range(n + 1):
        if k > 0:
            tail = sum(comb(n, j) * low[k] ** j * (1 - low[k]) ** (n - j) for j in range(k, n + 1))
            assert tail == pytest.approx(0.025, abs=2e-12)
        if k < n:
            tail = sum(comb(n, j) * high[k] ** j * (1 - high[k]) ** (n - j) for j in range(k + 1))
            assert tail == pytest.approx(0.025, abs=2e-12)
    for p in np.linspace(0, 1, 101):
        coverage = sum(
            comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(n + 1) if low[j] <= p <= high[j]
        )
        assert coverage >= 0.95 - 1e-12


@pytest.mark.parametrize("k", [0, 1, 2, 10, 30])
def test_poisson_tail_equations(k):
    low, high = map(float, poisson_interval(k))
    assert sum(exp(-high) * high**j / factorial(j) for j in range(k + 1)) == pytest.approx(
        0.025, abs=2e-12
    )
    if k == 0:
        assert low == 0
    else:
        assert 1 - sum(exp(-low) * low**j / factorial(j) for j in range(k)) == pytest.approx(
            0.025, abs=2e-12
        )


def test_no_data_and_all_successes():
    assert binomial_interval(0, 0) == (0, 1)
    assert binomial_interval(10, 10)[1] == 1


def test_broadcasting_and_exposure_scaling():
    count = np.array([0, 1, 20])[:, None]
    confidence = np.array([0.9, 0.95, 0.99])
    low, high = poisson_interval(count, confidence, exposure=100)
    assert low.shape == high.shape == (3, 3)
    for i, k in enumerate(count[:, 0]):
        for j, level in enumerate(confidence):
            expected = poisson_interval(k, level)
            np.testing.assert_allclose([low[i, j], high[i, j]], np.array(expected) / 100)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, np.nan, np.inf])
def test_invalid_confidence(confidence):
    with pytest.raises(ValueError, match="confidence"):
        binomial_interval(1, 10, confidence)


@pytest.mark.parametrize("count", [-1, 0.5, np.nan, np.inf, 2**53])
def test_invalid_counts(count):
    with pytest.raises(ValueError, match="events"):
        poisson_interval(count)


def test_impossible_binomial_data():
    with pytest.raises(ValueError, match="exceed"):
        binomial_interval(11, 10)


@pytest.mark.parametrize("exposure", [0, -1, np.nan, np.inf])
def test_invalid_exposure(exposure):
    with pytest.raises(ValueError, match="exposure"):
        poisson_interval(10, exposure=exposure)


def test_extreme_confidence_remains_finite():
    confidence = np.nextafter(1.0, 0.0)
    for fn, args in [(binomial_interval, (1, 100)), (poisson_interval, (1,))]:
        low, high = fn(*args, confidence=confidence)
        assert np.isfinite(low) and np.isfinite(high) and low < high


def test_bp1ci_reference_executable():
    fixture = json.loads((Path(__file__).parent / "fixtures/bp1ci.json").read_text())
    for case in fixture["cases"]:
        if case["distribution"] == "binomial":
            actual = binomial_interval(case["events"], case["trials"], case["confidence"])
        else:
            actual = bp1ci_poisson_interval(case["events"], case["confidence"])
        np.testing.assert_allclose(actual, case["bounds"], rtol=0, atol=case["atol"])


def test_legacy_poisson_differs_from_exact_at_zero():
    assert poisson_interval(0)[0] == 0
    with pytest.raises(ValueError, match="events"):
        bp1ci_poisson_interval(0)
    assert bp1ci_poisson_interval(1)[0] > poisson_interval(1)[0]
