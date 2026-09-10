import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import U2OETCriteria, u2oet_allocation, u2oet_posterior


def test_posterior_event_cutoffs_are_strict_and_utility_axes_are_explicit():
    p = np.zeros((10, 2, 2, 2, 2))
    p[:, :, :, 1, 0] = 1
    # Exactly 90% excessive toxicity: acceptable at cutoff .9.
    p[:9, 0, 0, 1, 0] = 0
    p[:9, 0, 0, 1, 1] = 1
    # 100% ineffective: unacceptable.
    p[:, 0, 1, 1, 0] = 0
    p[:, 0, 1, 0, 0] = 1
    # Every draw equals both event thresholds: neither bad event occurs.
    p[:, 1, 0] = [[0.25, 0.25], [0.25, 0.25]]
    utility = np.array([[10, 0], [100, 40]])
    result = u2oet_posterior(
        p, utility, criteria=U2OETCriteria(1, 1, min_efficacy=0.5, max_toxicity=0.5)
    )
    assert_allclose(result.inefficacy_probability, [[0, 1], [0, 0]])
    assert_allclose(result.toxicity_probability, [[0.9, 0], [0, 0]])
    assert_allclose(result.mean_utility, [[46, 10], [37.5, 100]])
    assert np.array_equal(result.acceptable, [[True, False], [True, True]])
    assert result.draws == 10 and not result.acceptable.flags.writeable
    with pytest.raises(ValueError, match="unit joint"):
        u2oet_posterior(p * 0.99, utility, criteria=U2OETCriteria(1, 1))
    with pytest.raises(ValueError, match="thresholds"):
        u2oet_posterior(p, utility)


def _posterior(utilities):
    utility = np.asarray(utilities, dtype=float)
    p = np.zeros((1, *utility.shape, 2, 2))
    p[0, :, :, 1, 0] = utility / 100
    p[0, :, :, 0, 0] = 1 - utility / 100
    return u2oet_posterior(
        p,
        [[0, 0], [100, 100]],
        criteria=U2OETCriteria(1, 1, min_efficacy=0, max_toxicity=1),
    )


def test_cohort_surplus_uses_all_candidates_and_prevents_skipping_agent_levels():
    posterior = _posterior([[60, 40, 95], [20, 10, 90], [80, 70, 100]])
    treated = np.zeros((3, 3))
    start = u2oet_allocation(posterior, treated, initial=(0, 0))
    assert start.best == (0, 0) and start.probabilities[0, 0] == 1
    treated[0, 0] = 3
    ar = u2oet_allocation(posterior, treated)
    assert ar.randomized and ar.best == (0, 0)
    assert_allclose(ar.probabilities, [[0.6, 0.4, 0], [0, 0, 0], [0, 0, 0]])
    assert np.array_equal(
        ar.candidate_mask, [[True, True, False], [True, True, False], [False] * 3]
    )
    # A lower-utility third pair has as many patients as the best: no AR.
    treated[1, 0] = 3
    # This also unlocks agent-1 level 2. Keep its utilities below the best.
    posterior = _posterior([[60, 40, 95], [20, 10, 90], [5, 4, 100]])
    greedy = u2oet_allocation(posterior, treated)
    assert not greedy.randomized and greedy.probabilities[0, 0] == 1
    treated[0, 0] = 6
    all_pairs = u2oet_allocation(posterior, treated, top=None)
    expected = np.array([[60, 40, 0], [20, 10, 0], [5, 4, 0]], dtype=float)
    assert_allclose(all_pairs.probabilities, expected / expected.sum())
    assert u2oet_allocation(posterior, treated, greedy=True).probabilities[0, 0] == 1
    # Once agent-2 level 1 has been tried, the high-utility (2,2) is reachable.
    treated[0, 1] = 3
    assert u2oet_allocation(posterior, treated).best == (2, 2)
    with pytest.raises(ValueError, match="all-zero"):
        u2oet_allocation(_posterior(np.zeros((3, 3))), treated, surplus=0)
    # A posterior with every dose ineffective stops allocation.
    p = np.zeros((1, 2, 2, 2, 2))
    p[..., 0, 0] = 1
    stopped = u2oet_posterior(p, [[0, 0], [100, 100]], criteria=U2OETCriteria(1, 1))
    decision = u2oet_allocation(stopped, [[3, 0], [0, 0]])
    assert decision.best is None and decision.probabilities.sum() == 0
