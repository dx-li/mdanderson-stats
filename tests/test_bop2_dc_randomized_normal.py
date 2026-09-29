import numpy as np
import pytest

from mdanderson_stats.bop2_dc_randomized_normal import bop2_dc_randomized_normal_design
from mdanderson_stats.bop2_dc_randomized_normal_simulation import (
    simulate_bop2_dc_randomized_normal,
)


def _design(*, shift=0.0, assignments=(0, 1, 0, 1), looks=(2, 4)):
    return bop2_dc_randomized_normal_design(
        4,
        -0.25,
        0.0,
        control_prior=(shift, 4.0, 2.0, 3.0),
        treatment_prior=(shift, 4.0, 2.0, 3.0),
        arm_assignments=assignments,
        looks=looks,
        lambda_lrv=0.01,
        lambda_cmv=0.5,
    )


def test_identical_independent_student_t_arms_have_symmetric_difference() -> None:
    design = _design()
    state = design.monitor([0.0, 1.0], [0.0, 1.0])

    assert state.control_df == state.treatment_df
    assert state.posterior_cmv == 0.5
    assert state.posterior_lrv > 0.5
    assert state.absolute_error_cmv == 0.0
    assert state.decision == "final_consider"


def test_randomized_normal_monitor_preserves_a_large_common_offset() -> None:
    shift = 1e15
    base = _design()
    shifted = _design(shift=shift)
    control = np.array([-0.25, 0.25])
    treatment = np.array([0.125, 0.5])
    reference = base.monitor(control, treatment)
    translated = shifted.monitor(control + shift, treatment + shift)

    np.testing.assert_allclose(
        translated.posterior_lrv, reference.posterior_lrv, rtol=0, atol=2e-12
    )
    np.testing.assert_allclose(
        translated.posterior_cmv, reference.posterior_cmv, rtol=0, atol=2e-12
    )
    np.testing.assert_allclose(
        translated.difference_location, reference.difference_location, rtol=0, atol=1e-12
    )


def test_replay_allows_a_prior_only_empty_arm_at_an_interim_look() -> None:
    design = _design(assignments=(0, 0, 1, 1))
    replay = design.replay([0.0, 1.0, 0.25, 0.75])

    assert [state.total_n for state in replay.states] == [2, 4]
    assert replay.states[0].control_n == 2
    assert replay.states[0].treatment_n == 0
    assert replay.terminal_decision == replay.states[-1].decision
    np.testing.assert_array_equal(replay.arm_assignments_observed, [0, 0, 1, 1])


def test_normal_simulation_is_replayable_and_common_shift_invariant() -> None:
    shift = 1e15
    kwargs = dict(
        max_subjects=4,
        theta_lrv=0.0,
        theta_cmv=0.5,
        control_prior=(0.25, 4.0, 2.0, 3.0),
        treatment_prior=(0.25, 4.0, 2.0, 3.0),
        arm_assignments=[0, 1, 0, 1],
        looks=[2, 4],
        lambda_lrv=0.5,
        lambda_cmv=0.5,
    )
    base_design = bop2_dc_randomized_normal_design(**kwargs)
    shifted_design = bop2_dc_randomized_normal_design(
        **{
            **kwargs,
            "control_prior": (shift + 0.25, 4.0, 2.0, 3.0),
            "treatment_prior": (shift + 0.25, 4.0, 2.0, 3.0),
        }
    )
    base = simulate_bop2_dc_randomized_normal(
        base_design, 0.25, 1.0, 0.5, 1.0, n_trials=12, rng=815
    )
    shifted = simulate_bop2_dc_randomized_normal(
        shifted_design,
        shift + 0.25,
        1.0,
        shift + 0.5,
        1.0,
        n_trials=12,
        rng=815,
    )
    replay = simulate_bop2_dc_randomized_normal(
        base_design, 0.25, 1.0, 0.5, 1.0, n_trials=12, rng=base.rng_seed
    )

    np.testing.assert_array_equal(shifted.decision_count, base.decision_count)
    np.testing.assert_array_equal(shifted.look_decision_probability, base.look_decision_probability)
    np.testing.assert_array_equal(replay.decision_probability, base.decision_probability)
    assert np.count_nonzero(base.decision_probability) > 1
    assert np.sum(base.decision_probability) == 1.0
    assert np.sum(base.sample_size_probability) == 1.0
    assert base.expected_sample_size == replay.expected_sample_size
    assert np.all(base.decision_mcse >= 0)


def test_normal_simulation_quadrature_work_preflight_does_not_consume_rng() -> None:
    design = _design()
    rng1, rng2 = np.random.default_rng(4), np.random.default_rng(4)
    with pytest.raises(ValueError, match="quadrature-work budget"):
        simulate_bop2_dc_randomized_normal(design, 0.0, 1.0, 0.5, 1.0, n_trials=100_000, rng=rng1)
    assert rng1.random() == rng2.random()
