from math import comb

import numpy as np
from scipy.special import logsumexp

from mdanderson_stats import BOINDesign, IBOINDesign


def test_published_and_live_native_boundaries():
    design = IBOINDesign([0.1, 0.19, 0.3, 0.42, 0.54], [3] * 5, target=0.3)
    table = design.boundaries(np.arange(3, 31, 3))
    np.testing.assert_array_equal(
        table.escalate_max,
        [
            [1, 1, 2, 3, 4, 4, 5, 6, 6, 7],
            [0, 1, 2, 3, 3, 4, 5, 5, 6, 7],
            [0, 1, 2, 2, 3, 4, 4, 5, 6, 7],
            [0, 1, 1, 2, 3, 3, 4, 5, 6, 6],
            [0, 0, 1, 2, 2, 3, 4, 5, 5, 6],
        ],
    )
    np.testing.assert_array_equal(
        table.deescalate_min,
        [
            [2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
            [2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [1, 2, 3, 4, 6, 7, 8, 9, 10, 11],
            [1, 2, 3, 4, 5, 6, 7, 8, 10, 11],
        ],
    )
    # Native V1.6.3.0 default: target .25, five doses, ESS 2, robust prior off.
    native = IBOINDesign([0.06, 0.14, 0.25, 0.38, 0.5], [2] * 5).boundaries(np.arange(1, 13))
    np.testing.assert_array_equal(
        native.escalate_max,
        [
            [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2],
            [-1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2],
            [-1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        ],
    )
    np.testing.assert_array_equal(
        native.deescalate_min,
        [
            [1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5],
            [1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4],
            [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4],
            [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
            [0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4],
        ],
    )
    np.testing.assert_array_equal(native.eliminate_min, [2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6])


def test_prior_matches_direct_sum_and_retains_extreme_log_odds():
    phi = np.array([0.25, 0.15, 0.35])
    design = IBOINDesign([0.1, 0.4], [4, 7])
    for j, (q, n) in enumerate(zip([0.1, 0.4], [4, 7], strict=True)):
        expected = np.zeros(3)
        for x in range(n + 1):
            likelihood = phi**x * (1 - phi) ** (n - x)
            expected += likelihood / likelihood.sum() * comb(n, x) * q**x * (1 - q) ** (n - x)
        np.testing.assert_allclose(design.hypothesis_probability[j], expected, atol=2e-15)
    extreme = IBOINDesign([0, 1], [10000, 10000])
    expected_logs = np.array([10000 * np.log1p(-phi), 10000 * np.log(phi)])
    expected_logs -= logsumexp(expected_logs, axis=1, keepdims=True)
    np.testing.assert_allclose(extreme.log_hypothesis_probability, expected_logs, atol=2e-12)
    assert np.isfinite(extreme.log_hypothesis_probability).all()
    assert not extreme.log_hypothesis_probability.flags.writeable


def test_zero_information_reduces_to_boin_and_safety_ignores_history():
    design = IBOINDesign([0.01, 0.2, 0.9], [0, 0, 0])
    plain = BOINDesign()
    table = design.boundaries(np.arange(1, 31))
    np.testing.assert_array_equal(
        table.escalate_max, np.tile(plain.boundary_table().escalate_max, (3, 1))
    )
    np.testing.assert_array_equal(
        table.deescalate_min, np.tile(plain.boundary_table().deescalate_min, (3, 1))
    )
    for y in range(7):
        assert (
            design.next_dose([3, 6, 0], [0, y, 0], 2).next_dose
            == plain.next_dose([3, 6, 0], [0, y, 0], 2).next_dose
        )
    informed = IBOINDesign([0.01, 0.1, 0.5], [100, 3, 2])
    stop = informed.next_dose([3, 0, 0], [3, 0, 0], 1)
    assert stop.action == "stop_safety" and stop.eliminated.all()
    native = IBOINDesign([0.06, 0.14, 0.25, 0.38, 0.5], [2] * 5)
    assert native.next_dose([0, 0, 0, 1, 0], [0] * 5, 4).action == "stay"
    assert native.next_dose([0, 0, 0, 0, 1], [0] * 5, 5).next_dose == 4


def test_robust_prior_preserves_or_discards_history_as_native_guide_specifies():
    ess = [2, 3, 4, 2, 2]
    lower = IBOINDesign([0.1, 0.30, 0.42, 0.54, 0.6], ess, target=0.3, robust_prior=True)
    upper = IBOINDesign([0.1, 0.19, 0.30, 0.42, 0.54], ess, target=0.3, robust_prior=True)
    np.testing.assert_array_equal(lower.effective_prior_ess, ess)
    np.testing.assert_array_equal(upper.effective_prior_ess, [2, 3, 4, 0, 0])
    np.testing.assert_array_equal(upper.prior_ess, ess)
    # Dedicated help uses >= J/2, including the midpoint in an even dose range.
    middle = IBOINDesign([0.1, 0.3, 0.42, 0.54], [2] * 4, target=0.3, robust_prior=True)
    np.testing.assert_array_equal(middle.effective_prior_ess, [2, 2, 0, 0])
    ordinary = BOINDesign(target=0.3).boundary_table(12)
    table = upper.boundaries(np.arange(1, 13))
    np.testing.assert_array_equal(table.escalate_max[3:], np.tile(ordinary.escalate_max, (2, 1)))
    np.testing.assert_array_equal(
        table.deescalate_min[3:], np.tile(ordinary.deescalate_min, (2, 1))
    )
    assert not upper.effective_prior_ess.flags.writeable
    assert ess == [2, 3, 4, 2, 2]
    native = IBOINDesign([0.06, 0.14, 0.25, 0.38, 0.5], [2] * 5, robust_prior=True)
    native_table = native.boundaries(np.arange(1, 13))
    np.testing.assert_array_equal(
        native_table.escalate_max[3:], [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2]] * 2
    )
    np.testing.assert_array_equal(
        native_table.deescalate_min[3:], [[1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4]] * 2
    )
