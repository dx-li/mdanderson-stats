import numpy as np
import pytest

from mdanderson_stats.bop2_dc_randomized_paired_optimization import (
    bop2_dc_randomized_paired_operating_characteristics,
    optimize_bop2_dc_randomized_paired,
)


def _calibration(**kwargs):
    values = dict(
        max_subjects=4,
        endpoint="multiple_efficacy",
        lrv=(0.2, 0.2),
        cmv=(0.5, 0.5),
        futile_joint_probabilities=((0.05, 0.35, 0.35, 0.25),) * 2,
        effective_joint_probabilities=(
            (0.05, 0.15, 0.15, 0.65),
            (0.45, 0.25, 0.20, 0.10),
        ),
        arm_assignments=(0, 1, 0, 1),
        lambda_lrv_grid=(0.6,),
        lambda_cmv_grid=(0.6,),
        gamma_lrv_grid=(0.5,),
        gamma_cmv_grid=(0.5,),
        control_prior=(1.0, 1.0, 1.0, 1.0),
        treatment_prior=(1.0, 1.0, 1.0, 1.0),
        looks=(2, 4),
        false_go_limit=1.0,
        false_no_go_limit=1.0,
        false_consider_limit=1.0,
        graduate_at_interim=True,
    )
    values.update(kwargs)
    return optimize_bop2_dc_randomized_paired(**values)


def test_paired_randomized_exact_oc_mass_and_expected_n_are_conserved() -> None:
    result = _calibration()
    evidence = result.candidates
    assert evidence.parameters.shape == (1, 8)
    assert evidence.decision_probability.shape == (2, 1, 2, 5)
    np.testing.assert_allclose(evidence.decision_probability.sum(axis=(2, 3)), 1.0, atol=1e-12)
    np.testing.assert_allclose(evidence.sample_size_probability.sum(axis=2), 1.0, atol=1e-12)
    expected_n = evidence.sample_size_probability @ np.asarray((2, 4), dtype=float)
    np.testing.assert_allclose(evidence.expected_sample_size, expected_n, atol=1e-12)
    assert result.decision_labels == (
        "stop_no_go",
        "graduate",
        "final_go",
        "final_consider",
        "final_no_go",
    )
    assert not evidence.parameters.flags.writeable
    assert not evidence.decision_probability.flags.writeable
    assert result.candidates.correct_go_rate[0] == pytest.approx(
        evidence.decision_probability[1, 0, :, 1].sum() + evidence.decision_probability[1, 0, -1, 2]
    )
    standalone = bop2_dc_randomized_paired_operating_characteristics(
        result.design,
        np.stack((result.futile_joint_probabilities, result.effective_joint_probabilities)),
    )
    assert standalone.decision_probability.shape == (2, 2, 5)
    np.testing.assert_allclose(
        standalone.decision_probability,
        evidence.decision_probability[:, 0],
        atol=1e-12,
    )
    one_scenario_batch = bop2_dc_randomized_paired_operating_characteristics(
        result.design, result.futile_joint_probabilities[None, ...]
    )
    assert one_scenario_batch.category_probability.shape == (1, 2, 4)


def test_joint_association_is_retained_when_margins_match() -> None:
    low_association = _calibration(
        futile_joint_probabilities=(
            (0.05, 0.35, 0.35, 0.25),
            (0.05, 0.35, 0.35, 0.25),
        )
    )
    treatment_association = _calibration(
        futile_joint_probabilities=(
            (0.05, 0.35, 0.35, 0.25),
            (0.35, 0.05, 0.05, 0.55),
        )
    )
    np.testing.assert_allclose(
        low_association.futile_marginal_probabilities,
        treatment_association.futile_marginal_probabilities,
        atol=0,
    )
    assert not np.allclose(
        low_association.candidates.decision_probability[0],
        treatment_association.candidates.decision_probability[0],
        atol=1e-12,
        rtol=0,
    )


def test_invalid_effective_truth_and_work_limit_fail_before_oc_recursion() -> None:
    with pytest.raises(ValueError, match="clinical-go"):
        _calibration(
            effective_joint_probabilities=(
                (0.05, 0.35, 0.35, 0.25),
                (0.10, 0.40, 0.30, 0.20),
            )
        )
    with pytest.raises(ValueError, match="max_work"):
        _calibration(max_work=1)
