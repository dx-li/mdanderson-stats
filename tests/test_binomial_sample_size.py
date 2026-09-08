"""Native design checks and exhaustive integer-sample-size minimality checks."""

import json
from math import comb, fsum
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import binomial_power, binomial_sample_size

CASES = json.loads((Path(__file__).parent / "fixtures/binomial_sample_size.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", CASES)
def test_native_and_exhaustive_minimum(case):
    c = case
    r = binomial_sample_size(c["null"], c["alternative"], c["alpha"], c["target_power"])
    assert r.significance <= c["alpha"] and r.power >= c["target_power"]
    assert c["status"] == 0
    if c["significance"] <= c["alpha"]:
        native = binomial_power(c["trials"], c["null"], c["alternative"], c["alpha"])
        assert_allclose(
            [native.significance, native.power],
            [c["significance"], c["power"]],
            rtol=2e-9,
            atol=1e-14,
        )
        assert r.trials <= c["trials"]
    else:
        # Original mode 3 reports success with size .51 for a requested .1.
        assert c["trials"] == 2 and c["alpha"] == 0.1
        assert r.trials > 2
    every = binomial_power(np.arange(2, int(r.trials) + 1), c["null"], c["alternative"], c["alpha"])
    assert np.all(every.power[:-1] < c["target_power"])
    assert every.power[-1] >= c["target_power"]


def test_independent_mass_sums_show_first_feasible_size():
    for p0, pa in [(0.3, 0.6), (0.7, 0.4)]:
        r = binomial_sample_size(p0, pa, 0.1, 0.8)
        for n in range(2, int(r.trials) + 1):
            m0 = [comb(n, k) * p0**k * (1 - p0) ** (n - k) for k in range(n + 1)]
            ma = [comb(n, k) * pa**k * (1 - pa) ** (n - k) for k in range(n + 1)]
            if pa > p0:
                m0, ma = m0[::-1], ma[::-1]
            size = power = 0.0
            for k in range(1, n + 2):
                if fsum(m0[:k]) <= 0.1:
                    size, power = fsum(m0[:k]), fsum(ma[:k])
            if n < int(r.trials):
                assert power < 0.8
            else:
                assert power >= 0.8
                assert_allclose([r.significance, r.power], [size, power], atol=1e-14)


@pytest.mark.parametrize("batch_size", [1, 7, 256])
def test_broadcasting_batch_invariance_and_inclusive_limits(batch_size):
    r = binomial_sample_size(
        [[0.2], [0.3]], [[0.06], [0.35]], 0.05, [0.5, 0.8], batch_size=batch_size
    )
    for i, j in np.ndindex(r.trials.shape):
        exact = binomial_sample_size(
            [0.2, 0.3][i],
            [0.06, 0.35][i],
            0.05,
            [0.5, 0.8][j],
            min_trials=int(r.trials[i, j]),
            max_trials=int(r.trials[i, j]),
        )
        assert exact.trials == r.trials[i, j]
    assert np.all(r.power >= np.array([0.5, 0.8]))


def test_power_dip_does_not_hide_smaller_design():
    r = binomial_sample_size(0.3, 0.35, 0.05, 0.5)
    assert r.trials == 236
    assert binomial_power(236, 0.3, 0.35, 0.05).power >= 0.5
    assert binomial_power(237, 0.3, 0.35, 0.05).power < 0.5


@pytest.mark.parametrize(
    "p0,pa,alpha,target,kwargs",
    [
        (0.5, 0.6, 0.05, 0.08, {"min_trials": 1, "max_trials": 1}),
        (0.5, 0.6, 0.05, 0.051, {"min_trials": 1, "max_trials": 1}),
        (0.3, 0.6, 0.1, 0.8, {"max_trials": 13}),
        (0.3, 0.6, 0.1, 0.8, {"min_trials": 3, "max_trials": 2}),
        (0.3, 0.6, 0.1, 0.8, {"max_trials": 10**10 + 1}),
        (0.3, 0.6, 0.1, 0.8, {"batch_size": 0}),
        (0.3, 0.6, 0.1, 0.8, {"min_trials": True}),
        (0.3, 0.6, 0.1, 0.8, {"max_trials": 2.5}),
        (0.3, 0.6, 0.1, 0.05, {}),
        (0.3, 0.6, 0.1, 1, {}),
        (0.3, 0.6, 0.1, np.nan, {}),
        (0.3, 0.3, 0.1, 0.8, {}),
    ],
)
def test_invalid_or_no_design(p0, pa, alpha, target, kwargs):
    with pytest.raises(ValueError):
        binomial_sample_size(p0, pa, alpha, target, **kwargs)


def test_one_trial_can_be_explicitly_allowed():
    r = binomial_sample_size(0.1, 0.8, 0.1, 0.8, min_trials=1)
    assert r.trials == 1 and r.significance <= 0.1 and r.power >= 0.8
