import numpy as np

from mdanderson_stats.plbarpo_trial import run_plbarpo_trial


def _common(**updates):
    result = dict(
        true_response=[1.0, 0.5, 0.0],
        prior=np.ones((3, 2)),
        initial_active=[True, True, False],
        candidate_order=[2],
        min_n_per_arm=[1, 1, 1],
        max_n_per_arm=[2, 2, 2],
        max_total_n=6,
        look_sizes=[2, 4, 6],
        burn_in_per_arm=1,
        method="barn2n",
        theta_fut=0.5,
        pfut=0.99,
        theta_eff=0.5,
        peff=0.99,
        theta_final=0.5,
        pfinal=0.8,
        assignment_uniforms=[0, 0.9, 0, 0.9, 0, 0.9],
        outcome_uniforms=[0.2, 0.8, 0.2, 0.2, 0.2, 0.2],
    )
    result.update(updates)
    return result


def test_cap_closure_assesses_final_rule_then_replaces_without_reopening():
    result = run_plbarpo_trial(**_common())
    assert result.enrolled == 6
    assert result.cap_closed[0]
    assert result.final_assessed[0]
    assert result.final_efficacy[0]
    assert result.stop_kind[0] == "final_efficacy_at_arm_cap"
    assert result.entered[2]
    assert result.assigned[0] == 2
    assert result.assignments[3] == 2
    assert result.stop_kind[2] == "trial_maximum"
    assert not result.active.any()
    assert np.all(result.stopped[result.entered])
    assert result.assigned.sum() == result.enrolled
    np.testing.assert_allclose(result.assignment_probability.sum(axis=1), 1.0)


def test_simultaneous_futility_closes_all_arms_when_no_replacements_remain():
    result = run_plbarpo_trial(
        **_common(
            true_response=[0.0, 0.0, 1.0],
            candidate_order=[],
            max_n_per_arm=[6, 6, 6],
            theta_fut=0.5,
            pfut=0.6,
            theta_eff=None,
            peff=None,
            theta_final=None,
            pfinal=None,
            assignment_uniforms=np.zeros(6),
            outcome_uniforms=np.zeros(6),
        )
    )
    assert result.enrolled == 2
    assert result.stop_reason == "all_arms_closed_no_candidates"
    np.testing.assert_array_equal(result.early_futility, [True, True, False])
    np.testing.assert_array_equal(result.stopped, [True, True, False])
    assert result.stop_kind[:2] == ("futility", "futility")
    np.testing.assert_array_equal(result.assigned, [1, 1, 0])
    assert result.looks[0].active_after.sum() == 0


def test_cap_at_interim_look_is_reflected_in_look_and_terminal_ledger():
    result = run_plbarpo_trial(
        true_response=[1.0, 0.0],
        prior=np.ones((2, 2)),
        initial_active=[True, True],
        candidate_order=[],
        min_n_per_arm=[1, 1],
        max_n_per_arm=[2, 4],
        max_total_n=4,
        look_sizes=[2, 4],
        burn_in_per_arm=0,
        method="barcp",
        theta_fut=None,
        pfut=None,
        theta_eff=None,
        peff=None,
        theta_final=0.5,
        pfinal=0.8,
        early_monitoring=False,
        assignment_uniforms=[0.0, 0.0, 0.5, 0.5],
        outcome_uniforms=[0.1, 0.1, 0.9, 0.9],
    )
    first_look = result.looks[0]
    assert result.assigned[0] == 2
    assert result.cap_closed[0]
    assert result.final_assessed[0]
    assert first_look.final_efficacious[0] == result.final_efficacy[0]
    assert result.stop_kind[0] == "final_efficacy_at_arm_cap"
    assert not first_look.active_after[0]
