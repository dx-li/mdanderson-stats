import numpy as np

from mdanderson_stats.bop2_dc_randomized_normal import bop2_dc_randomized_normal_design


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
