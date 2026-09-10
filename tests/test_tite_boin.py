"""Published STFT thresholds, independent imputation, and interim conduct."""

from fractions import Fraction

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import BOINDesign, tite_boin_decision, tite_boin_estimate


def test_published_table_s1_thresholds():
    design = BOINDesign(target=0.3)
    n = [3, 6, 6, 6, 6, 6, 9, 9, 9, 12, 12, 12]
    y = [1, 1, 1, 2, 2, 2, 2, 2, 3, 2, 2, 4]
    c = [1, 2, 3, 1, 2, 3, 1, 2, 1, 4, 5, 1]
    result = tite_boin_estimate(design, n, y, c, 0)
    expected = [0.88, 0.60, 1.96, 0.73, 1.80, 2.87, 0.59, 1.65, 0.58, 1.33, 2.72, 0.43]
    thresholds = np.where(np.array(y) / n < 0.3, result.escalate_stft, result.deescalate_stft)
    assert_allclose(thresholds, expected, atol=0.005, rtol=0)


def test_imputation_against_rational_arithmetic_and_complete_data():
    design = BOINDesign(target=0.3)
    for n in range(1, 13):
        for y in range(n + 1):
            for c in range(n - y + 1):
                # Distinct follow-up profiles with the same STFT share the imputation.
                follow = Fraction(2 * c, 3)
                odds = Fraction(20 * y + 3, 20 * (n - c - y) + 17)
                expected = (y + odds * (c - follow)) / n
                result = tite_boin_estimate(design, n, y, c, float(follow))
                assert_allclose(result.estimated_rate, float(expected), rtol=1e-14, atol=1e-15)
                if c == 0:
                    ordinary = design.next_dose([0, n, 0], [0, y, 0], 2)
                    actual = tite_boin_decision(design, [0, n, 0], [0, y, 0], [[], [], []], 2, 90)
                    assert actual.action == ordinary.action
                    assert actual.next_dose == ordinary.next_dose
    # Conservative imputation can exceed one, while coherence blocks de-escalation
    # when the actually observed rate is below the target.
    inflated = tite_boin_estimate(design, 10, 2, 8, 0)
    assert inflated.estimated_rate > 1
    assert inflated.move == 0


def test_current_suspension_rules_and_observed_toxicity_override():
    design = BOINDesign(target=0.3)
    assert (
        tite_boin_decision(design, [6, 0], [0, 0], [[45] * 3, []], 1, 90).action
        == "suspend_pending"
    )
    assert (
        tite_boin_decision(
            design, [6, 0], [0, 0], [[45] * 3, []], 1, 90, minimum_complete_fraction=0.5
        ).action
        == "escalate"
    )
    assert (
        tite_boin_decision(design, [3, 0], [0, 0], [[20], []], 1, 90).action == "suspend_followup"
    )
    assert tite_boin_decision(design, [3, 0], [0, 0], [[22.5], []], 1, 90).action == "escalate"
    # 2/5 observed DLTs already cross the de-escalation cutoff, despite 3 pending.
    unsafe = tite_boin_decision(design, [0, 5], [0, 2], [[], [0] * 3], 2, 90)
    assert unsafe.action == "deescalate"
    assert unsafe.next_dose == 1
    stopped = tite_boin_decision(design, [3, 0], [3, 0], [[], []], 1, 90)
    assert stopped.action == "stop_safety"
    precision = tite_boin_decision(
        BOINDesign(target=0.3, early_stop_patients=3), [3, 0], [1, 0], [[], []], 1, 90
    )
    assert precision.action == "stop_enrollment"


def test_weighted_stft_and_time_scale_invariance():
    settings = dict(trimester_probabilities=[0.1, 0.2, 0.7], minimum_complete_fraction=0.5)
    trial = tite_boin_decision(
        BOINDesign(target=0.3), [9, 0], [2, 0], [[30, 48, 75], []], 1, 90, **settings
    )
    assert_allclose(trial.standardized_followup, [0.97, 0])
    scaled = tite_boin_decision(
        BOINDesign(target=0.3),
        [9, 0],
        [2, 0],
        [np.array([30, 48, 75]) * 1e-200, []],
        1,
        90e-200,
        **settings,
    )
    assert_allclose(scaled.estimate.estimated_rate, trial.estimate.estimated_rate)
    assert_array_equal(scaled.pending_count, trial.pending_count)
    assert scaled.action == trial.action
