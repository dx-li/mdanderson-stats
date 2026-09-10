"""Published Simon designs and independent exact finite-sample enumeration."""

from fractions import Fraction
from itertools import product
from math import comb

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import SimonDesign, simon_two_stage


@pytest.mark.parametrize(
    "p0,p1,optimal,minimax",
    [
        (0.05, 0.25, (9, 17, 0, 2), (12, 16, 0, 2)),
        (0.1, 0.3, (10, 29, 1, 5), (15, 25, 1, 5)),
        (0.2, 0.4, (13, 43, 3, 12), (18, 33, 4, 10)),
        (0.3, 0.5, (15, 46, 5, 18), (19, 39, 6, 16)),
        (0.4, 0.6, (16, 46, 7, 23), (34, 39, 17, 20)),
        (0.6, 0.8, (11, 43, 7, 30), (13, 35, 8, 25)),
    ],
)
def test_simon_table_one(p0, p1, optimal, minimax):
    # Simon (1989), printed page 4, alpha=.05 and beta=.20 rows.
    result = simon_two_stage(p0, p1)
    assert result.optimal == SimonDesign(*optimal)
    assert result.minimax == SimonDesign(*minimax)
    for design in (result.optimal, result.minimax):
        oc = design.operating_characteristics([p0, p1])
        assert oc.reject_null[0] <= 0.05
        assert oc.reject_null[1] >= 0.8


def test_operating_characteristics_against_every_binary_path():
    design = SimonDesign(3, 7, 0, 2)
    p = np.array([[0, 0.01, 0.2], [0.5, 0.9, 1]])
    rejection = np.zeros_like(p)
    early = np.zeros_like(p)
    first_moment = np.zeros_like(p)
    second_moment = np.zeros_like(p)
    for path in product((0, 1), repeat=7):
        mass = p ** sum(path) * (1 - p) ** (7 - sum(path))
        stop = sum(path[:3]) == 0
        size = 3 if stop else 7
        early += mass * stop
        rejection += mass * (not stop and sum(path) > 2)
        first_moment += mass * size
        second_moment += mass * size**2
    oc = design.operating_characteristics(p)
    assert_allclose(oc.reject_null, rejection, atol=2e-15)
    assert_allclose(oc.stop_early, early, atol=2e-15)
    assert_allclose(oc.continue_probability + oc.stop_early, 1, atol=2e-15)
    assert_allclose(oc.expected_sample_size, first_moment, atol=2e-14)
    assert_allclose(oc.sample_size_sd**2, second_moment - first_moment**2, atol=1e-13)
    assert design.decision(0) == "stop_futility"
    assert design.decision(3) == "continue"  # no early efficacy stop
    assert design.decision(1, 2) == "do_not_reject"
    assert design.decision(1, 3) == "reject_null"
    with pytest.raises(ValueError, match="stopped"):
        design.decision(0, 3)


@pytest.mark.parametrize(
    "p0,p1", [(Fraction(1, 5), Fraction(3, 5)), (Fraction(2, 5), Fraction(4, 5))]
)
def test_search_against_unpruned_rational_enumeration(p0, p1):
    def pmf(x, n, p):
        return comb(n, x) * p**x * (1 - p) ** (n - x)

    candidates = []
    for n in range(2, 11):
        for n1 in range(1, n):
            for r1 in range(n1):
                for r in range(r1, n):
                    rejection = [
                        sum(
                            pmf(x, n1, p) * pmf(y, n - n1, p)
                            for x in range(r1 + 1, n1 + 1)
                            for y in range(n - n1 + 1)
                            if x + y > r
                        )
                        for p in (p0, p1)
                    ]
                    if rejection[0] <= Fraction(1, 5) and rejection[1] >= Fraction(7, 10):
                        continuation = sum(pmf(x, n1, p0) for x in range(r1 + 1, n1 + 1))
                        expected = n1 + (n - n1) * continuation
                        candidates.append((expected, n, n1, r1, -r))
    opt = min(candidates)
    mini = min(candidates, key=lambda c: (c[1], c[0], *c[2:]))
    result = simon_two_stage(float(p0), float(p1), alpha=0.2, power=0.7, max_n=10)
    assert result.optimal == SimonDesign(opt[2], opt[1], opt[3], -opt[4])
    assert result.minimax == SimonDesign(mini[2], mini[1], mini[3], -mini[4])


def test_tiny_rejection_tail_is_not_subtracted_from_one():
    design = SimonDesign(2, 5, 0, 3)
    # Enumerated degree-five polynomial: any path with >=4 responses continues.
    p = 1e-30
    expected = 5 * p**4 * (1 - p) + p**5
    assert_allclose(design.operating_characteristics(p).reject_null, expected, rtol=2e-14, atol=0)


def test_search_cap_and_default_protocol():
    with pytest.raises(ValueError, match="no feasible"):
        simon_two_stage(max_n=10)
    result = simon_two_stage()
    assert result.optimal == SimonDesign(23, 56, 1, 5)
    assert result.minimax == SimonDesign(30, 52, 1, 5)
    assert "23 evaluable participants" in result.protocol()
    assert "30 evaluable participants" in result.protocol("minimax")
    with pytest.raises(ValueError, match="null_rate"):
        simon_two_stage(0.3, 0.2)
    with pytest.raises(ValueError, match="power"):
        simon_two_stage(power=1)
    with pytest.raises(ValueError, match="r1"):
        SimonDesign(3, 7, 3, 5)
