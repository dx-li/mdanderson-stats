"""Significance inversion checked against Fortran and enumerated critical regions."""

import json
from math import comb, fsum
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import binomial_power, binomial_significance

CASES = json.loads((Path(__file__).parent / "fixtures/binomial_significance.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", CASES)
def test_native_mode_four(case):
    c = case
    r = binomial_significance(c["trials"], c["null"], c["alternative"], c["target_power"])
    assert c["status"] == 0
    expected = r
    if r.significance < 1e-8:
        # The original searches alpha only from SRANGE=1e-8 upward.
        expected = binomial_power(c["trials"], c["null"], c["alternative"], 1e-8)
        assert r.significance < expected.significance
    assert_allclose(
        [expected.significance, expected.power],
        [c["significance"], c["power"]],
        rtol=2e-9,
        atol=1e-14,
    )
    assert r.power >= c["target_power"] and r.previous_power < c["target_power"]


@pytest.mark.parametrize("p0,pa", [(0.2, 0.06), (0.3, 0.6), (0.8, 0.94)])
@pytest.mark.parametrize("n", [1, 5, 10])
def test_all_resolvable_power_steps(n, p0, pa):
    masses = [[comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)] for p in [p0, pa]]
    if pa > p0:
        masses = [row[::-1] for row in masses]
    sizes, powers = [np.array([0] + [fsum(row[:j]) for j in range(1, n + 2)]) for row in masses]
    targets = (powers[:-1] + powers[1:]) / 2
    valid = (targets > 0) & (targets < 1) & (np.diff(powers) > 1e-13)
    r = binomial_significance(n, p0, pa, targets[valid])
    selected = np.arange(1, n + 2)[valid]
    assert_allclose(r.critical, selected - 1 if pa < p0 else n - selected + 1)
    assert_allclose(r.power, powers[1:][valid], atol=1e-14)
    assert_allclose(r.significance, sizes[1:][valid], atol=1e-14)
    assert_allclose(r.previous_power, powers[:-1][valid], atol=1e-14)
    assert_allclose(r.previous_significance, sizes[:-1][valid], atol=1e-14)


def test_exact_tie_full_region_and_broadcast_directions():
    r = binomial_significance([[1], [2]], [0.8, 0.2], [0.5, 0.5], 0.5)
    assert_allclose(r.critical, [[0, 1], [1, 1]])
    assert_allclose(r.power, [[0.5, 0.5], [0.75, 0.75]])
    full = binomial_significance(1, [0.8, 0.2], [0.6, 0.4], 0.9)
    assert_allclose(full.critical, [1, 0])
    assert_allclose(full.significance, 1)
    assert_allclose(full.power, 1)
    assert np.all(full.previous_power < 0.9)


def test_large_n_and_tiny_target():
    r = binomial_significance([1e6, 1e10], 1e-5, 2e-5, 0.8)
    assert np.all(r.power >= 0.8) and np.all(r.previous_power < 0.8)
    tiny = binomial_significance(200, 0.01, 0.02, 3e-307)
    assert tiny.critical == 189
    assert tiny.power >= 3e-307 and tiny.previous_power < 3e-307


@pytest.mark.parametrize(
    "n,p0,pa,power",
    [
        (0, 0.2, 0.4, 0.8),
        (1.5, 0.2, 0.4, 0.8),
        (10, 0, 0.4, 0.8),
        (10, 0.2, 1, 0.8),
        (10, 0.2, 0.2, 0.8),
        (10, 0.2, 0.4, 0),
        (10, 0.2, 0.4, 1),
        (10, 0.2, 0.4, np.nan),
    ],
)
def test_invalid_inputs(n, p0, pa, power):
    with pytest.raises(ValueError):
        binomial_significance(n, p0, pa, power)


def test_significance_below_original_solver_floor_is_verified_independently():
    r = binomial_significance(100, 0.3, 0.6, 0.5)
    assert r.critical == 60

    def tail(p, k):
        return fsum(comb(100, j) * p**j * (1 - p) ** (100 - j) for j in range(k, 101))

    assert_allclose(r.significance, tail(0.3, 60), rtol=1e-13)
    assert_allclose(r.power, tail(0.6, 60), rtol=1e-13)
    assert tail(0.6, 61) < 0.5 <= tail(0.6, 60)
