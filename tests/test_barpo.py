import numpy as np
import pytest

from mdanderson_stats.barpo import barpo_allocation, barpo_monitor, barpo_posterior


def posterior():
    return barpo_posterior([1, 4, 8], [4, 2, 1], prior=[[1, 1]] * 3)


def test_posterior_best_probabilities_and_immutability():
    result = posterior()
    np.testing.assert_allclose(result.alpha, [2, 5, 9])
    np.testing.assert_allclose(result.beta, [5, 3, 2])
    np.testing.assert_allclose(
        result.best_probability,
        [0.00421726835106833, 0.160641446093728, 0.835141285555204],
        atol=2e-9,
    )
    assert not result.alpha.flags.writeable


@pytest.mark.parametrize("method", ["barcp", "barn2n", "barmtv", "dbcd"])
def test_allocation_is_a_probability_vector(method):
    kwargs = {"method": method, "assigned": [5, 8, 12]}
    if method == "barn2n":
        kwargs["max_n"] = 50
    if method == "dbcd":
        kwargs.update(target_probability=[0.2, 0.3, 0.5], tau=2, tau1=0.5)
    result = barpo_allocation(posterior(), **kwargs)
    assert np.all(result >= 0)
    np.testing.assert_allclose(result.sum(), 1)
    assert not result.flags.writeable


def test_floors_and_stopped_arms_are_applied_by_water_filling():
    result = barpo_allocation(
        posterior(),
        [5, 8, 12],
        method="barcp",
        tau=0.7,
        stopped=[False, True, False],
        minimum_probability=[0.15, 0, 0.15],
    )
    assert result[1] == 0
    assert result[0] >= 0.15 and result[2] >= 0.15
    np.testing.assert_allclose(result.sum(), 1)


def test_dbcd_rejects_zero_assigned_proportion_without_smoothing():
    with pytest.raises(ValueError, match="zero assigned"):
        barpo_allocation(
            barpo_posterior([1, 0, 0], [1, 0, 0], prior=[[1, 1]] * 3),
            [5, 0, 12],
            method="dbcd",
            target_probability=[0.2, 0.3, 0.5],
        )


def test_monitoring_uses_strict_futility_and_inclusive_efficacy_cutoffs():
    result = barpo_monitor(
        [1, 4, 8],
        [4, 2, 1],
        prior=[[1, 1]] * 3,
        theta_fut=0.25,
        pfut=0.8,
        theta_eff=0.65,
        peff=0.8,
        theta_final=0.5,
        pfinal=0.8,
    )
    assert result.futile is not None and result.efficacious is not None
    np.testing.assert_array_equal(result.futile, result.futility_probability > 0.8)
    np.testing.assert_array_equal(result.efficacious, result.efficacy_probability >= 0.8)
    np.testing.assert_array_equal(
        result.final_efficacious, result.final_efficacy_probability >= 0.8
    )


def test_control_monitoring_uses_posterior_ordering():
    result = barpo_monitor(
        [2, 8],
        [8, 2],
        prior=[[1, 1]] * 2,
        control=True,
        pfut=0.8,
        peff=0.8,
        pfinal=0.8,
    )
    assert result.futility_probability[0] == 0
    assert result.efficacy_probability[0] == 0
    assert result.efficacy_probability[1] > 0.5


def test_dbcd_requires_explicit_target():
    with pytest.raises(ValueError, match="target_probability"):
        barpo_allocation(posterior(), [5, 8, 12], method="dbcd")


def test_extreme_power_stopped_arm_cannot_overflow_eligible_weights():
    result = barpo_allocation(
        posterior(),
        [5, 8, 12],
        method="barcp",
        tau=1e308,
        stopped=[True, False, False],
    )
    assert result[0] == 0
    np.testing.assert_allclose(result[1:].sum(), 1)


def test_symmetric_half_tail_is_exact():
    result = barpo_monitor(
        [0],
        [0],
        prior=[[7, 7]],
        theta_fut=0.5,
        pfut=0.5,
        theta_eff=0.5,
        peff=0.5,
        theta_final=0.5,
        pfinal=0.5,
    )
    assert result.futility_probability[0] == 0.5
    assert result.efficacy_probability[0] == 0.5
    assert result.final_efficacy_probability[0] == 0.5
