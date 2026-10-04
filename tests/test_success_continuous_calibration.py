import numpy as np
import pytest

from mdanderson_stats.success_calibration import normal_success_oc, survival_success_oc
from mdanderson_stats.success_calibration_continuous import (
    calibrate_normal_success_cutoff,
    calibrate_survival_success_cutoff,
)


def test_normal_cutoff_bisection_returns_upper_feasible_bracket() -> None:
    result = calibrate_normal_success_cutoff(
        0.05,
        standard_error=1.0,
        design_mean=0.0,
        design_sd=1.0,
        analysis_mean=0.0,
        analysis_sd=1.0,
    )

    assert result.bracket[0] <= result.cutoff == result.bracket[1]
    assert result.bracket[1] - result.bracket[0] <= result.cutoff_tolerance
    assert result.candidates_evaluated <= 80
    assert result.operating_characteristics.incorrect_decision_probability <= result.target
    if result.bracket[0] != result.bracket[1]:
        below = normal_success_oc(
            result.bracket[0],
            standard_error=1.0,
            design_mean=0.0,
            design_sd=1.0,
            analysis_mean=0.0,
            analysis_sd=1.0,
        )
        assert below.incorrect_decision_probability > result.target


def test_survival_cutoff_uses_survival_oc_adapter() -> None:
    kwargs = {
        "events": 40,
        "treatment_allocation": 0.5,
        "design_mean": float(np.log(0.6)),
        "design_sd": 0.4,
        "analysis_mean": 0.0,
        "analysis_sd": 1.0,
    }
    result = calibrate_survival_success_cutoff(0.1, **kwargs)
    direct = survival_success_oc(result.cutoff, **kwargs)

    assert result.operating_characteristics == direct
    assert result.operating_characteristics.incorrect_decision_probability <= 0.1
    assert result.bracket[1] - result.bracket[0] <= result.cutoff_tolerance


def test_lower_endpoint_can_be_the_calibrated_cutoff() -> None:
    result = calibrate_normal_success_cutoff(
        0.99,
        cutoff_range=(0.6, 0.999),
        standard_error=1,
        design_mean=0,
    )

    assert result.cutoff == 0.6
    assert result.bracket == (0.6, 0.6)
    assert result.candidates_evaluated == 1


def test_unattainable_target_and_insufficient_evaluation_budget_raise() -> None:
    kwargs = {"standard_error": 1.0, "design_mean": 0.0}
    with pytest.raises(ValueError, match="upper cutoff"):
        calibrate_normal_success_cutoff(1e-12, cutoff_range=(0.8, 0.81), **kwargs)
    with pytest.raises(ArithmeticError, match="max_evaluations"):
        calibrate_normal_success_cutoff(
            0.05, cutoff_range=(0.6, 0.999), max_evaluations=2, **kwargs
        )


def test_invalid_range_fails_before_operating_characteristics() -> None:
    with pytest.raises(ValueError, match="cutoff_range"):
        calibrate_normal_success_cutoff(0.05, cutoff_range=(0.6, 1.0), standard_error=1.0)
