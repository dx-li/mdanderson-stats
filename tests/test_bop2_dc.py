import numpy as np
import pytest

from mdanderson_stats.bop2_dc import bop2_dc_design


def test_final_dual_criterion_decisions_and_equality_consider():
    design = bop2_dc_design(10, 0.2, 0.4, lambda_lrv=0.5, lambda_cmv=0.9, looks=[5, 10])
    go = design.monitor(10, 10)
    no_go = design.monitor(0, 10)
    consider = design.monitor(5, 10)
    assert go.decision == "final_go"
    assert no_go.decision == "final_no_go"
    assert consider.decision == "final_consider"


def test_interim_no_go_requires_both_criteria_to_fail():
    design = bop2_dc_design(10, 0.2, 0.4, lambda_lrv=0.9, lambda_cmv=0.9, looks=[5, 10])
    state = design.monitor(0, 5)
    assert state.decision == "stop_no_go"
    mixed = design.monitor(5, 5)
    assert mixed.decision == "continue"


def test_exact_operating_characteristics_conserve_probability():
    design = bop2_dc_design(6, 0.2, 0.4, looks=[3, 6])
    oc = design.operating_characteristics([0.2, 0.5])
    assert np.allclose(oc.sample_size_probability.sum(axis=-1), 1)
    assert np.all(oc.expected_sample_size >= 3)
    assert np.allclose(oc.no_go_probability + oc.final_go + oc.final_consider, 1)


def test_invalid_threshold_order_is_rejected():
    with pytest.raises(ValueError, match="lrv < cmv"):
        bop2_dc_design(10, 0.5, 0.4)
