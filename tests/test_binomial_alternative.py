"""Alternative probability solves: original mode 2 and independent identities."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import binomial_alternative, binomial_test

CASES = json.loads((Path(__file__).parent / "fixtures/binomial_alternative.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", CASES)
def test_native_mode_two(case):
    c = case
    r = binomial_alternative(
        c["trials"], c["null"], c["alpha"], c["target_power"], alternative=c["direction"]
    )
    assert c["status"] == 0
    assert_allclose(r.alternative_probability, c["alternative"], rtol=2e-6, atol=1e-9)
    assert_allclose(r.significance, c["significance"], rtol=2e-9, atol=1e-14)
    assert r.power >= r.target_power
    # Native mode 2 nudges pa outward by its relative solver tolerance.
    # Compare its reported power at that actual pa, not at the tighter root.
    native = binomial_test(r.critical, r.trials, c["alternative"])
    native_power = native.p_less if c["direction"] == "less" else native.p_greater
    assert_allclose(native_power, c["power"], rtol=2e-9, atol=1e-14)
    assert_allclose(r.power, r.target_power, rtol=2e-12, atol=1e-14)


@pytest.mark.parametrize("direction", ["less", "greater"])
def test_closed_form_and_adjacent_bracket(direction):
    lower = direction == "less"
    r = binomial_alternative(
        30, 0.2 if lower else 0.8, 0.01, [0.5, 0.8, 0.99], alternative=direction
    )
    expected = (
        -np.expm1(np.log(r.target_power) / 30) if lower else np.exp(np.log(r.target_power) / 30)
    )
    assert_allclose(r.alternative_probability, expected, rtol=2e-13)
    assert np.all(np.nextafter(r.probability_lower, np.inf) >= r.probability_upper)
    inside = r.probability_upper if lower else r.probability_lower
    probe = binomial_test(r.critical, r.trials, inside)
    power = probe.p_less if lower else probe.p_greater
    assert np.all(power < r.target_power)


def test_broadcasting_and_equal_power_at_null():
    r = binomial_alternative([[30], [50]], 0.2, 0.05, [0.5, 0.8, 0.95])
    assert r.power.shape == (2, 3)
    assert np.all(r.power >= r.target_power)
    for direction in ["less", "greater"]:
        same = binomial_alternative(1, 0.5, 0.5, 0.5, alternative=direction)
        assert same.alternative_probability == 0.5
        assert same.probability_lower == same.probability_upper == 0.5


def test_tiny_probability_avoids_absolute_tolerance_floor():
    r = binomial_alternative(1, 1e-320, 1e-315, 1e-310)
    assert_allclose(r.alternative_probability, 1e-310, atol=np.nextafter(0.0, 1.0), rtol=0)
    assert r.power >= 1e-310


@pytest.mark.parametrize(
    "n,p0,alpha,power,direction",
    [
        (1, 0.5, 0.05, 0.8, "less"),
        (1, 0.5, 0.05, 0.8, "greater"),
        (0, 0.2, 0.05, 0.8, "less"),
        (2.5, 0.2, 0.05, 0.8, "less"),
        (10, 0, 0.05, 0.8, "less"),
        (10, 1, 0.05, 0.8, "less"),
        (10, 0.2, 0, 0.8, "less"),
        (10, 0.2, 0.5, 0.4, "less"),
        (10, 0.2, 0.05, 1, "less"),
        (10, 0.2, 0.05, np.nan, "less"),
        (10, 0.2, 0.05, 0.8, "other"),
    ],
)
def test_invalid_or_impossible(n, p0, alpha, power, direction):
    with pytest.raises(ValueError):
        binomial_alternative(n, p0, alpha, power, alternative=direction)
