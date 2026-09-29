from typing import Any, cast

import numpy as np
import pytest

from mdanderson_stats.beta_binomial import BetaBinomialPosterior
from mdanderson_stats.beta_comparison import compare_beta_difference
from mdanderson_stats.bop2_dc_randomized_paired import (
    _paired_decisions_from_tails,
    bop2_dc_randomized_paired_design,
)


def _design(**kwargs):
    defaults: dict[str, Any] = dict(
        max_subjects=4,
        endpoint="multiple_efficacy",
        lrv=[0.0, 0.0],
        cmv=[0.2, 0.2],
        control_prior=[1, 2, 3, 4],
        treatment_prior=[2, 1, 4, 3],
        arm_assignments=[0, 1, 0, 1],
        looks=[2, 4],
    )
    defaults.update(kwargs)
    return cast(Any, bop2_dc_randomized_paired_design)(**defaults)


def test_joint_dirichlet_marginals_match_beta_difference_reference():
    design = _design(
        endpoint="efficacy_toxicity",
        lrv=[-0.1, 0.2],
        cmv=[0.3, -0.1],
        comparison_tolerance=1e-9,
    )
    probability, error = design._posterior_tails_from_success_counts(
        2, 2, np.array([[2, 1]]), np.array([[1, 2]])
    )
    c1 = BetaBinomialPosterior(1 + 2 + 2, 3 + 4 + 0)
    t1 = BetaBinomialPosterior(2 + 1 + 1, 4 + 3 + 1)
    expected1_lrv = compare_beta_difference(c1, t1, -0.1)
    expected1_cmv = compare_beta_difference(c1, t1, 0.3)
    c2 = BetaBinomialPosterior(1 + 3 + 1, 2 + 4 + 1)
    t2 = BetaBinomialPosterior(2 + 4 + 2, 1 + 3 + 0)
    expected2_lrv = compare_beta_difference(t2, c2, -0.2)
    expected2_cmv = compare_beta_difference(t2, c2, 0.1)
    np.testing.assert_allclose(
        probability[0, 0], [expected1_lrv.above_margin, expected1_cmv.above_margin]
    )
    np.testing.assert_allclose(
        probability[0, 1], [expected2_lrv.above_margin, expected2_cmv.above_margin]
    )
    np.testing.assert_allclose(
        error[0, 0], [expected1_lrv.absolute_error, expected1_cmv.absolute_error]
    )
    np.testing.assert_allclose(
        error[0, 1], [expected2_lrv.absolute_error, expected2_cmv.absolute_error]
    )


def test_composite_rules_are_error_corner_stable_and_offlook_is_continue():
    p = np.array([[[0.9, 0.9], [0.5, 0.5]]])
    e = np.array([[[0.0, 0.0], [0.45, 0.45]]])
    common: dict[str, Any] = dict(
        total_n=4,
        max_subjects=4,
        looks=np.array([2, 4]),
        lambda_lrv=np.array([0.2, 0.2]),
        lambda_cmv=np.array([0.2, 0.2]),
        gamma_lrv=np.array([0.5, 0.5]),
        gamma_cmv=np.array([0.5, 0.5]),
        graduate_at_interim=False,
    )
    label = _paired_decisions_from_tails(p, e, endpoint="multiple_efficacy", **common)
    assert label.tolist() == ["final_go"]
    with pytest.raises(ArithmeticError, match="could change"):
        _paired_decisions_from_tails(p, e, endpoint="efficacy_toxicity", **common)

    design = _design()
    state = design.monitor([0, 0, 0, 0], [0, 0, 0, 0])
    assert state.total_n == 0
    assert state.decision == "continue"
    assert state.endpoint_decisions == ("continue", "continue")


def test_fixed_allocation_replay_and_prefix_validation():
    design = _design(lambda_lrv=[0.99, 0.99], lambda_cmv=[0.99, 0.99])
    replay = design.replay([3, 3, 3, 3])
    assert replay.terminal_decision == "final_no_go"
    assert replay.arm_assignments_observed.tolist() == [0, 1, 0, 1]
    assert replay.outcomes_observed.tolist() == [3, 3, 3, 3]
    assert replay.states[-1].control_counts.tolist() == [0, 0, 0, 2]
    assert not replay.outcomes_observed.flags.writeable
    with pytest.raises(ValueError, match="fixed allocation prefix"):
        design.monitor([0, 0, 0, 0], [0, 0, 0, 1])


def test_invalid_margin_direction_and_schedule_are_rejected():
    with pytest.raises(ValueError, match="toxicity cmv < lrv"):
        _design(endpoint="efficacy_toxicity", lrv=[0.0, 0.2], cmv=[0.3, 0.2])
    with pytest.raises(ValueError, match="increase strictly"):
        _design(looks=[2, 2, 4])
