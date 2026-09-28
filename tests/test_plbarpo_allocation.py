import numpy as np
import pytest

from mdanderson_stats.plbarpo_allocation import plbarpo_active_allocation


def test_active_competition_and_barn2n_use_full_ledger_enrollment():
    result = plbarpo_active_allocation(
        successes=[1, 0, 1],
        failures=[0, 1, 0],
        assigned=[1, 1, 8],
        active=[True, True, False],
        prior=[[1, 1], [1, 1], [9, 1]],
        method="barn2n",
        max_n=20,
    )
    np.testing.assert_allclose(result.best_probability, [5 / 6, 1 / 6, 0.0], atol=2e-9)
    assert result.best_probability[2] == 0
    assert result.allocation_probability[2] == 0
    assert result.posterior_alpha[2] == 10
    assert result.posterior_beta[2] == 1
    assert result.global_enrolled == 10
    assert result.effective_exponent == pytest.approx(0.25)
    expected = np.array([5**0.25, 1.0])
    expected /= expected.sum()
    np.testing.assert_allclose(result.allocation_probability[:2], expected, rtol=2e-9)


def test_active_target_and_floor_use_full_ledger_alignment():
    result = plbarpo_active_allocation(
        successes=[1, 0, 0],
        failures=[0, 1, 0],
        assigned=[1, 1, 4],
        active=[True, True, False],
        prior=[[1, 1], [1, 1], [1, 1]],
        method="dbcd",
        tau=0.5,
        tau1=1.0,
        target_probability=[0.25, 0.75, 0.0],
        minimum_probability=[0.2, 0.1, 0.0],
    )
    assert result.allocation_probability[0] >= 0.2
    assert result.allocation_probability[1] >= 0.1
    assert result.allocation_probability[2] == 0
    assert result.allocation_probability.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="inactive"):
        plbarpo_active_allocation(
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 4],
            [True, True, False],
            prior=[[1, 1], [1, 1], [1, 1]],
            method="dbcd",
            target_probability=[0.25, 0.70, 0.05],
        )
