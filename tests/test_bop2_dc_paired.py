import numpy as np
import pytest

from mdanderson_stats.bop2_dc_paired import bop2_dc_paired_design


def test_multiple_efficacy_or_combination():
    design = bop2_dc_paired_design(6, "multiple_efficacy", [0.2, 0.2], [0.4, 0.4], looks=[3, 6])
    assert design.monitor([3, 3, 0, 0]).decision == "final_go"
    assert design.monitor([0, 0, 0, 6]).decision == "final_no_go"
    assert design.monitor([0, 3, 0, 3]).decision == "final_go"


def test_coprimary_toxicity_uses_no_toxicity_success_tail():
    design = bop2_dc_paired_design(
        6,
        "efficacy_toxicity",
        [0.3, 0.2],
        [0.45, 0.15],
        lambda_lrv=[0.5, 0.5],
        lambda_cmv=[0.5, 0.5],
        looks=[3, 6],
    )
    safe_efficacious = design.monitor([0, 3, 0, 3])
    assert safe_efficacious.decision == "final_go"
    toxic = design.monitor([3, 0, 0, 3])
    assert toxic.decision == "final_no_go"


def test_joint_prior_and_counts_are_preserved():
    design = bop2_dc_paired_design(
        4, "multiple_efficacy", [0.2, 0.3], [0.4, 0.5], prior=[1, 2, 3, 4], looks=[2, 4]
    )
    state = design.monitor(np.array([[1, 1, 0, 0], [0, 0, 0, 4]]))
    assert state.posterior_shapes.shape == (2, 4)
    assert state.sample_size.tolist() == [2, 4]


def test_invalid_toxicity_reference_order_is_rejected():
    with pytest.raises(ValueError, match="toxicity lrv"):
        bop2_dc_paired_design(4, "efficacy_toxicity", [0.3, 0.1], [0.45, 0.2])


def test_tiny_shapes_and_toxicity_cutoffs_remain_finite():
    design = bop2_dc_paired_design(
        1,
        "efficacy_toxicity",
        [0.2, 2e-300],
        [0.4, 1e-300],
        prior=[1e-300] * 4,
        looks=[1],
    )
    state = design.monitor([1, 0, 0, 0])
    assert np.all(np.isfinite(state.marginal_posterior))
