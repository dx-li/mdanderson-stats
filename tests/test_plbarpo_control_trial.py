import numpy as np

from mdanderson_stats.plbarpo_control_trial import run_plbarpo_control_trial


def _common(**updates):
    values = dict(
        true_response=[0.5, 0.0, 1.0],
        prior=np.ones((3, 2)),
        initial_active=[True, True, False],
        candidate_order=[2],
        min_n_per_arm=[1, 1, 1],
        max_n_per_arm=[8, 2, 2],
        max_total_n=8,
        look_sizes=[2, 4, 6, 8],
        burn_in_per_arm=1,
        control_mode="entire",
        method="barcp",
        tau=0.0,
        early_monitoring=False,
        pfinal=0.85,
        assignment_uniforms=[0.0, 0.9, 0.9, 0.0, 0.0, 0.9, 0.0, 0.0],
        outcome_uniforms=[0.2, 0.2, 0.2, 0.2, 0.8, 0.2, 0.2, 0.2],
    )
    values.update(updates)
    return values


def test_control_modes_use_real_control_windows_and_persistent_control():
    entire = run_plbarpo_control_trial(**_common())
    concurrent = run_plbarpo_control_trial(**_common(control_mode="concurrent"))
    expected_assignments = [0, 1, 1, 2, 0, 2]
    np.testing.assert_array_equal(entire.assignments, expected_assignments)
    np.testing.assert_array_equal(concurrent.assignments, expected_assignments)
    np.testing.assert_array_equal(entire.assigned, [2, 2, 2])
    np.testing.assert_array_equal(concurrent.assigned, [2, 2, 2])
    assert entire.entry_index.tolist() == [0, 0, 3]
    assert concurrent.entry_index.tolist() == [0, 0, 3]
    assert entire.stop_index.tolist() == [6, 3, 6]
    assert concurrent.stop_index.tolist() == [6, 3, 6]
    assert entire.final_assessed.tolist() == [False, True, True]
    assert concurrent.final_assessed.tolist() == [False, True, True]
    assert not entire.final_efficacy[2]
    assert concurrent.final_efficacy[2]
    assert not entire.active.any() and not concurrent.active.any()
    assert entire.stop_kind[0] == "trial_ended"
    assert np.isnan(entire.final_control_counts[0]).all()
    np.testing.assert_array_equal(entire.final_control_counts[2], [1.0, 1.0])
    np.testing.assert_array_equal(concurrent.final_control_counts[2], [0.0, 1.0])
    assert entire.looks[1].window_start[2] == 0
    assert concurrent.looks[1].window_start[2] == 3
    assert entire.looks[1].window_end[2] == 4
    assert concurrent.looks[1].window_end[2] == 4
    assert entire.looks[-1].window_start[2] == 0
    assert concurrent.looks[-1].window_start[2] == 3
    assert entire.looks[-1].window_end[2] == 6
    assert concurrent.looks[-1].window_end[2] == 6
    assert np.isfinite(entire.final_absolute_error[2])
    assert np.isfinite(concurrent.final_absolute_error[2])


def test_arm_cap_final_check_does_not_close_other_active_arms():
    result = run_plbarpo_control_trial(
        **_common(
            true_response=[0.0, 1.0, 1.0],
            initial_active=[True, True, True],
            candidate_order=[],
            max_n_per_arm=[6, 1, 3],
            max_total_n=6,
            look_sizes=[3, 6],
            pfinal=0.7,
            assignment_uniforms=[0.0, 0.9, 0.0, 0.0, 0.9, 0.9],
            outcome_uniforms=[0.2] * 6,
        )
    )
    assert result.assigned[1] == 1
    assert result.final_assessed[1]
    assert result.final_efficacy_probability[1] > 0.7
    assert result.stop_kind[1] == "final_efficacy_at_arm_cap"
    assert result.stop_index[1] == 3
    assert result.looks[0].active_after[2]
    assert result.stop_index[2] == 6
    assert result.final_assessed[2]
    assert result.stop_kind[0] == "trial_maximum"
    assert not result.active.any()


def test_cap_look_rejects_conflicting_futility_and_final_efficacy():
    with np.testing.assert_raises_regex(ValueError, "futile and efficacious"):
        run_plbarpo_control_trial(
            true_response=[1.0, 0.0],
            prior=np.ones((2, 2)),
            initial_active=[True, True],
            candidate_order=[],
            min_n_per_arm=[1, 1],
            max_n_per_arm=[4, 1],
            max_total_n=4,
            look_sizes=[2, 4],
            burn_in_per_arm=1,
            control_mode="entire",
            method="barcp",
            pfut=0.7,
            peff=None,
            pfinal=0.1,
            assignment_uniforms=[0.0, 0.9, 0.0, 0.0],
            outcome_uniforms=[0.2] * 4,
        )
