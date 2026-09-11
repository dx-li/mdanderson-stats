"""Published designs and independent small-trial outcome enumeration."""

from itertools import product
from math import comb

import numpy as np
import pytest

from mdanderson_stats import (
    RoseDesign,
    rose_design,
    rose_operating_characteristics,
    rose_select,
    simulate_rose,
)


def test_published_designs():
    # Wang, Yuan & Liu (2025), Table 1 and exact-design numerical example.
    for ph, al, ah, expected in (
        (0.3, 0.6, 0.6, (11, 13)),
        (0.3, 0.6, 0.7, (24, 27)),
        (0.3, 0.7, 0.7, (44, 48)),
        (0.4, 0.65, 0.65, (28, 31)),
    ):
        for omega, n in zip((None, 0.5), expected, strict=True):
            design = rose_design(ph, 0.1, pcs_low=al, pcs_high=ah, interim_fraction=omega)
            assert design.n_low == design.n_high == n
    for omega, n in ((None, 23), (0.5, 19)):
        design = rose_design(
            0.3, 0.1, pcs_low=0.6, pcs_high=0.6, method="exact", interim_fraction=omega
        )
        assert design.n_low == design.n_high == n
        assert rose_operating_characteristics(design, 0.3, 0.3).select_low >= 0.6
        assert rose_operating_characteristics(design, 0.2, 0.3).select_high >= 0.6
    with pytest.raises(ValueError, match="no feasible"):
        rose_design(0.3, 0.1, method="exact", max_per_arm=2)


def test_exact_enumeration_and_simulation():
    # Direct joint binomial enumeration, unequal arms and unequal stage fractions.
    # Boundaries .25 and .5 are binary-exact so this oracle needs no library cut logic.
    for design in (RoseDesign(4, 7, 0.25), RoseDesign(4, 7, 0.25, 2, 3, 0.5)):
        pl, ph = 0.27, 0.61
        il, ih = design.interim_low, design.interim_high
        sizes = (il, ih, design.n_low - il, design.n_high - ih)
        low_probability = high_probability = early_probability = 0.0
        for counts in product(*(range(n + 1) for n in sizes)):
            mass = np.prod(
                [
                    comb(n, k) * p**k * (1 - p) ** (n - k)
                    for n, k, p in zip(sizes, counts, (pl, ph, pl, ph), strict=True)
                ]
            )
            l1, h1, l2, h2 = counts
            early = bool(il and h1 / ih - l1 / il > 0.5)
            high = early or (h1 + h2) / design.n_high - (l1 + l2) / design.n_low > 0.25
            early_probability += mass * early
            high_probability += mass * high
            low_probability += mass * (not high)
        exact = rose_operating_characteristics(design, pl, ph)
        np.testing.assert_allclose(
            [exact.select_low, exact.select_high, exact.early_high],
            [low_probability, high_probability, early_probability],
            atol=2e-14,
            rtol=0,
        )
        simulated = simulate_rose(design, pl, ph, trials=100000, rng=168)
        observed = simulated.operating_characteristics
        assert abs(observed.select_high - exact.select_high) < 6 * simulated.select_high_mcse
        assert abs(observed.early_high - exact.early_high) <= 6 * simulated.early_high_mcse
        assert observed.mean_low == il + (design.n_low - il) * (1 - observed.early_high)
    assert rose_select(RoseDesign(50, 50, 0.58), 0, 29) == "low"
    assert rose_select(RoseDesign(50, 50, 0.58), 0, 30) == "high"
    assert rose_select(RoseDesign(4, 7, 0.25, 2, 3, 0.5), 2, 0, interim=True) == "continue"
    for pl, ph, expected in ((0, 1, 1), (1, 0, 0), (1, 1, 0), (0, 0, 0)):
        assert rose_operating_characteristics(design, pl, ph).select_high == expected
