"""Published thresholds, independent likelihood crossings and usable table brackets."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import KeyboardDesign, tite_keyboard_boundaries, tite_keyboard_decision


def test_published_followup_thresholds_and_independent_fractional_beta_crossings():
    table = tite_keyboard_boundaries(KeyboardDesign(), 12)
    assert_allclose(table.stay_or_escalate[1:5, 1], [1.88, 3.75, 5.63, 7.50], atol=0.005, rtol=0)
    assert_allclose(table.escalate[1:3, 1], [3.07, 6.15], atol=0.005, rtol=0)
    # Beta(2,b) interval masses from its closed-form survival, independent of
    # incomplete beta evaluation and strongest-key search.
    for bracket, left, right in [
        (table.stay_or_escalate[1], (0.25, 0.35), (0.35, 0.45)),
        (table.escalate[1], (0.15, 0.25), (0.25, 0.35)),
    ]:
        b = bracket.mean() + 1

        def mass(interval):
            lo, hi = interval
            return (1 - lo) ** b * (1 + b * lo) - (1 - hi) ** b * (1 + b * hi)

        assert abs(mass(left) - mass(right)) < 5e-15


@pytest.mark.parametrize(
    "design",
    [
        KeyboardDesign(0.15),
        KeyboardDesign(0.3),
        KeyboardDesign(0.5),
        KeyboardDesign(0.3, lower=0.1, upper=0.4, edge_rule="rescale"),
        KeyboardDesign(0.3, lower=0, upper=0.5),
    ],
)
def test_boundary_lookup_matches_posterior_over_feasible_effective_counts(design):
    table = tite_keyboard_boundaries(design)
    rng = np.random.default_rng(135)
    y = rng.integers(0, 31, 2000)
    m = rng.random(2000) * (30 - y)
    assert_array_equal(table.moves(y, m), design._posterior_effective(y + m, y).move)
    for brackets, minimum_move in [(table.stay_or_escalate, 0), (table.escalate, 1)]:
        for toxicities, (low, high) in enumerate(brackets):
            if np.isnan(high) or high == 0:
                continue
            assert np.nextafter(low, np.inf) == high
            moves = design._posterior_effective(
                toxicities + np.array([low, high]), np.array([toxicities, toxicities])
            ).move
            assert moves[0] < minimum_move <= moves[1]
            assert_array_equal(table.moves(toxicities, [low, high]), moves)


def test_followup_table_and_interim_conduct_agree_with_suspension_checked_separately():
    design = KeyboardDesign()
    table = tite_keyboard_boundaries(design, 6)
    for m in np.linspace(2, 4.99, 40):
        result = tite_keyboard_decision(
            design, [0, 6, 0], [0, 1, 0], [[], [(m - 2) / 3] * 3, []], 2, 1
        )
        expected = {-1: "deescalate", 0: "stay", 1: "escalate"}[int(table.moves(1, m))]
        assert result.action == expected
    assert np.isnan(table.escalate[6]).all()
    assert table.moves(6, 0) == -1
