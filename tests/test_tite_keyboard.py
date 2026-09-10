"""Published ESS/example transitions and independent fractional-beta identities."""

from math import comb

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import (
    KeyboardDesign,
    tite_effective_sample_size,
    tite_keyboard_decision,
    toxicity_followup_weights,
)


def test_published_ess_and_piecewise_timing_weights():
    assert_allclose(tite_effective_sample_size(2, [30, 48, 75], 90).effective_sample_size, 3.7)
    prior = [0.1, 0.2, 0.7]
    weights = toxicity_followup_weights(
        [0, 15, 30, 45, 60, 75, 90], 90, trimester_probabilities=prior
    )
    assert_allclose(weights, [0, 0.05, 0.1, 0.2, 0.3, 0.65, 1])
    assert_allclose(
        tite_effective_sample_size(
            2, [30, 48, 75], 90, trimester_probabilities=prior
        ).effective_sample_size,
        2.97,
    )
    for scale in [1e-200, 1e200]:
        assert_allclose(
            toxicity_followup_weights(np.array([0, 0.25, 0.5, 1]) * scale, scale), [0, 0.25, 0.5, 1]
        )
    batched = tite_effective_sample_size([2, 3], [[0, 0.5], [0.5, 1]], 1)
    assert_allclose(batched.effective_sample_size, [2.5, 4.5])
    assert tite_effective_sample_size(3, [], 90).effective_sample_size == 3


def test_paper_delayed_example_and_fractional_beta_probability():
    design = KeyboardDesign()
    # Paper example: one DLT and two pending follow-ups of 1/3 and 1/6 windows.
    result = tite_keyboard_decision(
        design, [3, 3], [0, 1], [[], [1 / 3, 1 / 6]], 2, 1, pending_fraction_limit=None
    )
    assert result.action == "deescalate"
    assert result.next_dose == 1
    assert_allclose(result.effective_sample_size, [3, 1.5])
    lo, hi = result.posterior.intervals.T
    # Beta(2,b) has survival (1-x)^b*(1+b*x), here b=1.5.
    exact = (1 - lo) ** 1.5 * (1 + 1.5 * lo) - (1 - hi) ** 1.5 * (1 + 1.5 * hi)
    assert_allclose(result.posterior.probability, exact, atol=3e-16, rtol=1e-12)
    assert (
        tite_keyboard_decision(design, [3, 3], [0, 1], [[], [1 / 3, 1 / 6]], 2, 1).action
        == "suspend_pending"
    )


def test_published_rounded_effective_nontoxic_thresholds():
    design = KeyboardDesign()
    for effective_nontoxic, action in [(1.87, "deescalate"), (1.89, "stay")]:
        result = tite_keyboard_decision(
            design, [3, 3], [0, 1], [[], [effective_nontoxic - 1]], 2, 1
        )
        assert result.action == action
    for effective_nontoxic, action in [(3.06, "stay"), (3.08, "escalate")]:
        result = tite_keyboard_decision(
            design, [3, 6, 0], [0, 1, 0], [[], [(effective_nontoxic - 2) / 3] * 3, []], 2, 1
        )
        assert result.action == action


def test_complete_data_reduction_safety_and_escalation_wait():
    design = KeyboardDesign()
    n, y = [3, 6, 0], [0, 1, 0]
    full = tite_keyboard_decision(design, n, y, [[], [], []], 2, 90)
    assert full.action == design.next_dose(n, y, 2).action
    assert_allclose(full.posterior.probability, design.posterior_keys(6, 1).probability)
    wait = tite_keyboard_decision(
        design, [3, 0], [0, 0], [[0.8, 0.9], []], 1, 1, pending_fraction_limit=None
    )
    assert wait.action == "suspend_escalation"
    unsafe = tite_keyboard_decision(design, [6, 0], [4, 0], [[0.1, 0.2], []], 1, 1)
    assert unsafe.action == "stop_safety"
    assert_array_equal(unsafe.eliminated, [True, True])
    # Safety is based on enrolled n, not the smaller effective n used for keys.
    assert_allclose(
        unsafe.safety_overdose_probability[0],
        sum(comb(7, k) * 0.3**k * 0.7 ** (7 - k) for k in range(5)),
    )


def test_pending_records_must_be_consistent():
    with pytest.raises(ValueError, match="strictly shorter"):
        tite_keyboard_decision(KeyboardDesign(), [3, 0], [0, 0], [[90], []], 1, 90)
    with pytest.raises(ValueError, match="cannot exceed"):
        tite_keyboard_decision(KeyboardDesign(), [3, 0], [3, 0], [[1], []], 1, 90)
    with pytest.raises(ValueError, match="summing"):
        toxicity_followup_weights([1], 90, trimester_probabilities=[0.1, 0.2, 0.3])
