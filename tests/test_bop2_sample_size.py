"""Independent enumeration over size, cutoff and exponent; constraint propagation."""

from itertools import product

import pytest
from numpy.testing import assert_array_equal

from mdanderson_stats import (
    BOP2InfeasibleError,
    bop2_binary_design,
    optimize_bop2_binary,
    optimize_bop2_binary_sample_size,
)


@pytest.mark.parametrize("endpoint,p0,p1", [("efficacy", 0.2, 0.6), ("toxicity", 0.8, 0.4)])
def test_sample_size_objectives_against_full_grid(endpoint, p0, p1):
    sizes = [6, 9, 12]
    scales, gammas = [0.65, 0.8, 0.95, 0.99], [0, 0.5, 1]
    candidates = []
    for n, scale, gamma in product(sizes, scales, gammas):
        looks = [v for v in [3, 6, 9] if v < n] + [n]
        design = bop2_binary_design(
            n, p0, cutoff_scale=scale, gamma=gamma, looks=looks, endpoint=endpoint
        )
        oc = design.operating_characteristics([p0, p1])
        success = oc.positive_conclusion if endpoint == "efficacy" else oc.complete_negative
        if success[0] <= 0.1 and success[1] >= 0.75:
            candidates.append(
                (float(oc.expected_sample_size[0]), n, -float(success[1]), scale, gamma)
            )
    for objective in ["expected_sample_size", "minimax"]:
        result = optimize_bop2_binary_sample_size(
            sizes,
            p0,
            p1,
            minimum_power=0.75,
            endpoint=endpoint,
            objective=objective,
            interim_looks=[3, 6, 9],
            cutoff_scales=scales,
            gammas=gammas,
        )
        key = (
            (lambda c: (c[0], c[1], c[2]))
            if objective == "expected_sample_size"
            else (lambda c: (c[1], c[0], c[2]))
        )
        best = min(candidates, key=key)
        chosen = result.best
        assert (chosen.calibration_design.max_subjects, chosen.cutoff_scale, chosen.gamma) == (
            best[1],
            best[3],
            best[4],
        )
        assert_array_equal(result.feasible_sample_sizes, sorted({c[1] for c in candidates}))
        assert_array_equal(result.searched_sample_sizes, sizes)
        for fit in result.feasible_designs:
            per_n = min(
                [c for c in candidates if c[1] == fit.calibration_design.max_subjects],
                key=lambda c: (c[0], c[2]),
            )
            assert (fit.cutoff_scale, fit.gamma) == per_n[3:]
            assert fit.objective == "expected_sample_size"
            assert fit.minimum_power == 0.75


def test_expected_enrollment_objective_differs_from_power_and_keeps_calibration_prior():
    power = optimize_bop2_binary(40, 0.2, 0.4)
    enrollment = optimize_bop2_binary(
        40, 0.2, 0.4, objective="expected_sample_size", minimum_power=0.8
    )
    assert enrollment.calibration_success_probability[1] >= 0.8
    assert (
        enrollment.calibration_oc.expected_sample_size[0]
        < power.calibration_oc.expected_sample_size[0]
    )
    assert enrollment.calibration_success_probability[1] < power.calibration_success_probability[1]
    plain = optimize_bop2_binary_sample_size([25, 30, 35], 0.2, 0.4, minimum_power=0.8)
    changed = optimize_bop2_binary_sample_size(
        [25, 30, 35], 0.2, 0.4, minimum_power=0.8, analysis_prior=[20, 1]
    )
    assert (
        changed.best.calibration_design.max_subjects == plain.best.calibration_design.max_subjects
    )
    assert_array_equal(
        changed.best.calibration_design.futility_max, plain.best.calibration_design.futility_max
    )
    assert changed.best.analysis_success_probability[0] > 0.1


def test_infeasible_sizes_and_invalid_inputs_are_distinct():
    with pytest.raises(BOP2InfeasibleError, match="no sample size"):
        optimize_bop2_binary_sample_size(
            [1, 2, 3], 0.2, 0.4, minimum_power=0.99, type1_error=0.01, interim_looks=[]
        )
    with pytest.raises(ValueError, match="gammas") as error:
        optimize_bop2_binary_sample_size([10, 15], 0.2, 0.4, minimum_power=0.8, gammas=[2])
    assert not isinstance(error.value, BOP2InfeasibleError)
    with pytest.raises(ValueError, match="strict error control"):
        optimize_bop2_binary(
            20,
            0.2,
            0.4,
            objective="expected_sample_size",
            minimum_power=0.8,
            error_control="closest",
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        optimize_bop2_binary_sample_size([20, 10], 0.2, 0.4, minimum_power=0.8)
