from __future__ import annotations

import numpy as np
import pytest

from mdanderson_stats.interval_survival import (
    fit_interval_survival,
    predict_interval_survival,
)
from mdanderson_stats.interval_survival_stratified import (
    fit_stratified_interval_survival,
    predict_stratified_interval_survival,
)


def test_joint_shared_slope_and_distinct_baselines_match_independent_reference() -> None:
    # Each current-status cell has an explicit frequency weight.
    cell_x = np.array([0.0, 0.0, 1.0, 1.0] * 2)
    cell_lower = np.array([0.0, 1.0, 0.0, 1.0] * 2)
    cell_upper = np.array([1.0, np.inf, 1.0, np.inf] * 2)
    cell_strata = np.repeat(["A", "B"], 4)
    cell_weights = np.array([2, 8, 5, 5, 4, 6, 7, 3], dtype=float)
    fit = fit_stratified_interval_survival(
        cell_lower, cell_upper, cell_x, strata=cell_strata, weights=cell_weights
    )
    assert fit.stratum_labels == ("A", "B")
    assert fit.log_likelihood == pytest.approx(-24.8084555704438, abs=2e-7)
    assert fit.coefficients[0] == pytest.approx(0.962469326124804, abs=2e-7)
    assert fit.strata[0].log_cumulative_hazard[0] - fit.scaled_coefficients[0] == pytest.approx(
        np.log(0.251054260277), abs=3e-7
    )
    assert fit.strata[1].log_cumulative_hazard[0] - fit.scaled_coefficients[0] == pytest.approx(
        np.log(0.478374591831488), abs=3e-7
    )
    for baseline in fit.strata:
        assert not baseline.support_mass.flags.writeable
        assert baseline.support_mass.sum() == pytest.approx(1.0)
    pa = predict_stratified_interval_survival(fit, "A", [1.0], [[0.0]])
    pb = predict_stratified_interval_survival(fit, "B", [1.0], [[0.0]])
    assert pa.survival_lower[0, 0] == pytest.approx(0.777980156994862, abs=3e-7)
    assert pb.survival_lower[0, 0] == pytest.approx(0.619789985226692, abs=3e-7)


def test_single_stratum_view_matches_existing_interval_fit() -> None:
    lower = np.array([0, 0, 1, 1, 0, 1], dtype=float)
    upper = np.array([1, 1, 2, np.inf, 2, np.inf])
    x = np.array([0, 1, 0, 1, 1, 0], dtype=float)
    ordinary = fit_interval_survival(lower, upper, x)
    grouped = fit_stratified_interval_survival(
        lower, upper, x, strata=np.repeat("only", lower.size)
    )
    np.testing.assert_array_equal(grouped.coefficients, ordinary.coefficients)
    np.testing.assert_array_equal(grouped.strata[0].support_mass, ordinary.support_mass)
    prediction = predict_stratified_interval_survival(grouped, "only", [0, 1, 2], [[0], [1]])
    expected = predict_interval_survival(ordinary, [0, 1, 2], [[0], [1]])
    np.testing.assert_array_equal(prediction.survival_lower, expected.survival_lower)
    np.testing.assert_array_equal(prediction.survival_upper, expected.survival_upper)


def test_covariate_constant_inside_each_stratum_is_not_identified() -> None:
    lower = [0, 1, 0, 1]
    upper = [1, np.inf, 1, np.inf]
    x = [0, 0, 1, 1]
    with pytest.raises(ValueError, match="not identified within strata"):
        fit_stratified_interval_survival(lower, upper, x, strata=["A", "A", "B", "B"])


def test_baseline_only_strata_have_distinct_normalized_predictions() -> None:
    lower = [0, 0, 1, 0, 1, 1]
    upper = [1, 1, np.inf, 1, np.inf, np.inf]
    fit = fit_stratified_interval_survival(
        lower, upper, strata=["early", "early", "early", "late", "late", "late"]
    )
    assert fit.coefficients.size == 0
    assert fit.strata[0].support_mass.sum() == pytest.approx(1.0)
    assert fit.strata[1].support_mass.sum() == pytest.approx(1.0)
    early = predict_stratified_interval_survival(fit, "early", [1.0])
    late = predict_stratified_interval_survival(fit, "late", [1.0])
    assert early.survival_lower[0, 0] < late.survival_lower[0, 0]
