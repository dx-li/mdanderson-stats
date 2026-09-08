"""Null probability inversion compared with mode 1 and independent identities."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import binomial_null, binomial_test

CASES = json.loads((Path(__file__).parent / "fixtures/binomial_null.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_native_mode_one(case):
    c = case
    r = binomial_null(
        c["trials"], c["alternative"], c["alpha"], c["target_power"], alternative=c["direction"]
    )
    assert c["status"] == 0
    assert_allclose(r.null_probability, c["null"], rtol=2e-6, atol=1e-9)
    assert_allclose(r.power, c["power"], rtol=2e-9, atol=1e-14)
    native = binomial_test(r.critical, r.trials, c["null"])
    size = native.p_less if c["direction"] == "less" else native.p_greater
    assert_allclose(size, c["significance"], rtol=2e-9, atol=1e-14)
    assert r.significance <= r.alpha and r.power >= r.target_power
    assert r.previous_power < r.target_power


@pytest.mark.parametrize("direction", ["less", "greater"])
def test_closed_form_and_closer_null_fails(direction):
    lower = direction == "less"
    r = binomial_null(30, 0.01 if lower else 0.99, [0.01, 0.05], 0.7, alternative=direction)
    expected = -np.expm1(np.log(r.alpha) / 30) if lower else np.exp(np.log(r.alpha) / 30)
    assert_allclose(r.null_probability, expected, rtol=2e-13)
    assert np.all(np.nextafter(r.probability_lower, np.inf) >= r.probability_upper)
    closer = r.probability_lower if lower else r.probability_upper
    probe = binomial_test(r.critical, r.trials, closer)
    assert np.all((probe.p_less if lower else probe.p_greater) > r.alpha)


def test_broadcasting_and_null_boundary_equality():
    r = binomial_null([[30], [50]], 0.6, 0.05, [0.5, 0.8, 0.95])
    assert r.power.shape == (2, 3)
    assert np.all(r.significance <= 0.05) and np.all(r.power >= r.target_power)
    for direction in ["less", "greater"]:
        same = binomial_null(1, 0.5, 0.5, 0.5, alternative=direction)
        assert same.null_probability == 0.5
        assert same.probability_lower == same.probability_upper == 0.5


def test_subnormal_null_probability():
    r = binomial_null(1, 1e-300, 1e-310, 1e-301)
    assert_allclose(r.null_probability, 1e-310, atol=np.nextafter(0.0, 1.0), rtol=0)
    assert r.significance <= 1e-310 and r.power >= 1e-301


@pytest.mark.parametrize(
    "n,pa,alpha,power,direction",
    [
        (1, 0.4, 0.05, 0.8, "greater"),
        (1, 0.6, 0.05, 0.8, "less"),
        (0, 0.6, 0.05, 0.8, "greater"),
        (2.5, 0.6, 0.05, 0.8, "greater"),
        (10, 0, 0.05, 0.8, "greater"),
        (10, 1, 0.05, 0.8, "greater"),
        (10, 0.6, 0, 0.8, "greater"),
        (10, 0.6, 0.5, 0.4, "greater"),
        (10, 0.6, 0.05, 1, "greater"),
        (10, 0.6, 0.05, np.nan, "greater"),
        (10, 0.6, 0.05, 0.8, "other"),
    ],
)
def test_invalid_or_impossible(n, pa, alpha, power, direction):
    with pytest.raises(ValueError):
        binomial_null(n, pa, alpha, power, alternative=direction)
