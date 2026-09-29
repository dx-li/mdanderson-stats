import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.rbop2_binary import rbop2_binary_design
from mdanderson_stats.rbop2_calibration import calibrate_rbop2_binary


def _candidates() -> np.ndarray:
    return np.array(
        [
            [[0.1, 0.9], [0.4, 0.4]],
            [[0.2, 0.8], [0.5, 0.5]],
            [[0.3, 0.7], [0.7, 0.7]],
        ]
    )


def test_single_candidate_matches_existing_exact_oc_engine() -> None:
    looks = [[1, 1], [2, 2]]
    prior = [[1, 1], [1, 1]]
    candidates = _candidates()[:1]
    result = calibrate_rbop2_binary(
        looks,
        calibration_prior=prior,
        endpoint="efficacy",
        margin=0,
        null_rates=(0.3, 0.3),
        alternative_rates=(0.6, 0.3),
        cutoff_candidates=candidates,
        alpha=0.99,
    )
    design = rbop2_binary_design(
        looks,
        prior=prior,
        endpoint="efficacy",
        margin=0,
        lower_cutoffs=candidates[0, :, 0],
        upper_cutoffs=candidates[0, :, 1],
    )
    null = design.operating_characteristics(0.3, 0.3)
    alternative = design.operating_characteristics(0.6, 0.3)
    assert result.selected_index == 0
    assert result.selected_design is not None
    assert result.feasible.dtype == np.bool_
    assert result.calibration_type_i_error[0] == null.overall_positive
    assert result.calibration_power[0] == alternative.overall_positive
    assert result.calibration_expected_null_sample_size[0] == null.expected_total_sample_size


def test_distinct_analysis_prior_reports_achieved_oc_and_selects_analysis_design() -> None:
    looks = [[1, 1], [2, 2]]
    analysis_prior = [[2, 1], [1, 2]]
    result = calibrate_rbop2_binary(
        looks,
        calibration_prior=[[1, 1], [1, 1]],
        analysis_prior=analysis_prior,
        endpoint="toxicity",
        margin=0,
        null_rates=(0.3, 0.3),
        alternative_rates=(0.2, 0.3),
        cutoff_candidates=_candidates(),
        alpha=0.99,
    )
    assert result.selected_index is not None
    assert result.selected_design is not None
    np.testing.assert_array_equal(result.selected_design.prior, analysis_prior)
    assert not np.array_equal(result.analysis_type_i_error, result.calibration_type_i_error)


@pytest.mark.parametrize("endpoint", ["efficacy", "toxicity"])
def test_independent_exact_two_look_calibration_reference(endpoint: str) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "rbop2-calibration-reference.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    case = fixture["endpoints"][endpoint]
    result = calibrate_rbop2_binary(
        fixture["looks"],
        calibration_prior=fixture["priors"]["calibration"],
        analysis_prior=fixture["priors"]["analysis"],
        endpoint=endpoint,
        margin=0,
        null_rates=tuple(case["null_rates"]),
        alternative_rates=tuple(case["alternative_rates"]),
        cutoff_candidates=fixture["cutoff_candidates"],
        alpha=fixture["alpha"],
    )
    assert result.selected_index == case["expected_selected_index"] == 3
    for index, (cal, ana) in enumerate(
        zip(case["candidates"]["calibration"], case["candidates"]["analysis"], strict=True)
    ):
        assert result.calibration_type_i_error[index] == pytest.approx(cal["null_positive"])
        assert result.calibration_expected_null_sample_size[index] == pytest.approx(
            cal["null_expected_total"]
        )
        assert result.calibration_power[index] == pytest.approx(cal["alternative_positive"])
        assert result.analysis_type_i_error[index] == pytest.approx(ana["null_positive"])
        assert result.analysis_expected_null_sample_size[index] == pytest.approx(
            ana["null_expected_total"]
        )
        assert result.analysis_power[index] == pytest.approx(ana["alternative_positive"])


def test_no_feasible_candidate_is_reported_without_selecting_a_design() -> None:
    result = calibrate_rbop2_binary(
        [[1, 1], [2, 2]],
        calibration_prior=[[1, 1], [1, 1]],
        endpoint="efficacy",
        margin=0,
        null_rates=(0.5, 0.5),
        alternative_rates=(0.8, 0.2),
        cutoff_candidates=_candidates(),
        alpha=1e-6,
    )
    assert not result.feasible.any()
    assert result.selected_index is None
    assert result.selected_design is None


def test_work_limit_is_checked_before_posterior_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_call(*args: object, **kwargs: object) -> tuple[float, float]:
        raise AssertionError("posterior evaluation must not precede budget rejection")

    monkeypatch.setattr("mdanderson_stats.rbop2_calibration._probability", unexpected_call)
    with pytest.raises(ValueError, match="max_work"):
        calibrate_rbop2_binary(
            [[1, 1], [2, 2]],
            calibration_prior=[[1, 1], [1, 1]],
            endpoint="efficacy",
            margin=0,
            null_rates=(0.3, 0.3),
            alternative_rates=(0.6, 0.3),
            cutoff_candidates=_candidates(),
            alpha=0.2,
            max_work=1,
        )
