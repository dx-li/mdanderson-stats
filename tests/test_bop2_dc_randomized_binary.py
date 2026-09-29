from itertools import product

import numpy as np
import pytest
from scipy.special import ndtr, ndtri

from mdanderson_stats.bop2_dc_randomized_binary import bop2_dc_randomized_binary_design


def _design(**kwargs):
    defaults = dict(
        max_subjects=4,
        theta_lrv=0.0,
        theta_cmv=0.5,
        control_prior=(1.0, 1.0),
        treatment_prior=(1.0, 1.0),
        arm_assignments=[0, 1, 0, 1],
        looks=[2, 4],
    )
    defaults.update(kwargs)
    return bop2_dc_randomized_binary_design(**defaults)


def test_fixed_allocation_replay_stops_at_first_no_go_look():
    design = _design(lambda_lrv=0.9, lambda_cmv=0.9)
    replay = design.replay([0, 0, 1, 1])
    assert replay.terminal_decision == "stop_no_go"
    assert replay.arm_assignments_observed.tolist() == [0, 1]
    assert replay.responses_observed.tolist() == [0, 0]
    assert len(replay.states) == 1
    assert replay.states[0].decision.item() == "stop_no_go"
    assert not replay.arm_assignments_observed.flags.writeable


def test_exact_operating_characteristics_match_all_binary_tapes():
    design = _design(lambda_lrv=0.6, lambda_cmv=0.7, gamma_lrv=0.3, gamma_cmv=0.4)
    p_control, p_treatment = 0.25, 0.7
    exact = design.operating_characteristics(p_control, p_treatment)

    stop = np.zeros(2)
    final = {"final_go": 0.0, "final_consider": 0.0, "final_no_go": 0.0}
    sample_size = np.zeros(2)
    expected_size = 0.0
    for responses in product((0, 1), repeat=4):
        probability = 1.0
        for arm, response in zip(design.arm_assignments, responses, strict=True):
            rate = p_control if arm == 0 else p_treatment
            probability *= rate if response else 1 - rate
        replay = design.replay(responses)
        n = int(replay.states[-1].total_n)
        expected_size += probability * n
        if replay.terminal_decision == "stop_no_go":
            look_index = list(design.looks).index(n)
            stop[look_index] += probability
            sample_size[look_index] += probability
        else:
            final[replay.terminal_decision] += probability
            sample_size[-1] += probability

    np.testing.assert_allclose(exact.stop_no_go, stop, atol=1e-12)
    np.testing.assert_allclose(exact.final_go, final["final_go"], atol=1e-12)
    np.testing.assert_allclose(exact.final_consider, final["final_consider"], atol=1e-12)
    np.testing.assert_allclose(exact.final_no_go, final["final_no_go"], atol=1e-12)
    np.testing.assert_allclose(exact.sample_size_probability, sample_size, atol=1e-12)
    np.testing.assert_allclose(exact.expected_sample_size, expected_size, atol=1e-12)
    np.testing.assert_allclose(exact.no_go_probability + exact.final_go + exact.final_consider, 1)


def test_strict_final_cutoffs_and_optional_obf_graduation():
    symmetric = _design(looks=[4], theta_cmv=0.4, lambda_lrv=0.5)
    equality = symmetric.monitor(1, 2, 1, 2)
    assert equality.absolute_error_lrv.item() == 0
    assert equality.decision.item() == "final_consider"

    graduated = _design(
        lambda_lrv=0.1,
        lambda_cmv=0.2,
        graduate_at_interim=True,
    )
    expected = 2 * ndtr(ndtri((1 + graduated.lambda_lrv) / 2) / np.sqrt(0.5)) - 1
    np.testing.assert_allclose(graduated._gradient_cutoffs(2)[0], expected, rtol=2e-15)
    tiny_gradient = _design(lambda_lrv=1e-20, lambda_cmv=0.2, graduate_at_interim=True)
    assert tiny_gradient._gradient_cutoffs(2)[0] > 0


def test_reported_quadrature_error_guards_only_decision_ambiguous_cutoffs():
    base = _design(
        max_subjects=2,
        looks=[2],
        arm_assignments=[0, 1],
        theta_cmv=0.5,
        lambda_lrv=0.1,
        lambda_cmv=0.1,
    )
    estimate = base.monitor(0, 1, 1, 1)
    exact_cutoff = float(estimate.posterior_lrv)
    ambiguous = _design(
        max_subjects=2,
        looks=[2],
        arm_assignments=[0, 1],
        theta_cmv=0.5,
        lambda_lrv=exact_cutoff,
        lambda_cmv=0.1,
    )
    with pytest.raises(ArithmeticError, match="quadrature error could change"):
        ambiguous.monitor(0, 1, 1, 1)
    with pytest.raises(ArithmeticError, match="quadrature error could change"):
        ambiguous.operating_characteristics(0.5, 0.8)

    interim = _design(
        max_subjects=4,
        looks=[2, 4],
        theta_cmv=0.5,
        lambda_lrv=exact_cutoff,
        lambda_cmv=0.1,
        gamma_lrv=0,
    )
    stable = interim.monitor(0, 1, 1, 1)
    assert stable.decision.item() == "continue"
