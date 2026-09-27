import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.survan_cox import survan_cox
from mdanderson_stats.survival_contour import survival_stratified_cox_contour


def test_shared_stratified_fit_handles_opposite_within_stratum_separation():
    time = np.array([1.0, 2.0, 1.0, 2.0])
    event = np.array([1, 0, 1, 0])
    design = np.array([[0.0], [1.0], [1.0], [0.0]])
    strata = np.array(["A", "A", "B", "B"], dtype=object)

    fit = survan_cox(time, event, design, ties="breslow", strata=strata)

    assert fit.strata_labels == ("A", "B")
    assert_allclose(fit.coefficients, [0.0], atol=1e-12)
    assert_allclose(fit.covariance, [[2.0]], atol=1e-12)
    assert_allclose(fit.negative_log_likelihood, 2 * np.log(2), atol=1e-12)
    with pytest.raises(ValueError, match="monotone"):
        survan_cox(time, event, [0.0, 1.0, 0.0, 1.0], strata=strata)


def test_stratified_contours_share_coefficients_but_keep_no_event_baseline():
    time = np.array([0, 1, 2, 3, 1, 2, 4, 5], dtype=float)
    event = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    x = np.array([[0], [1], [0], [1], [0], [1], [0], [1]])
    strata = np.array(["event", "event", "event", "event", "quiet", "quiet", "quiet", "quiet"])

    result = survival_stratified_cox_contour(
        time, event, x, 0, strata=strata, n_grid=3, quantile_probabilities=[0.25, 0.75]
    )

    assert result.stratum_labels == ("event", "quiet")
    assert result.fit.strata_labels == result.stratum_labels
    assert result.contours[0].times[0] == 0
    assert result.contours[0].survival[0, 0] < 1  # event at zero is post-event
    assert_allclose(result.contours[1].survival, 1)
    assert_allclose(result.contours[1].lower, 1)
    assert result.for_stratum("quiet") is result.contours[1]


def test_stratified_cox_is_stable_to_large_stratum_specific_offsets():
    time = np.array([1.0, 2.0, 1.0, 2.0])
    event = np.array([1, 0, 1, 0])
    design = np.array([[0.0], [1.0], [1.0], [0.0]])
    strata = np.array([0, 0, 1, 1])
    shifted = design + np.array([[1e8], [1e8], [-1e8], [-1e8]])

    fit = survan_cox(time, event, shifted, strata=strata)

    assert_allclose(fit.coefficients, [0.0], atol=1e-8)
    assert_allclose(fit.covariance, [[2.0]], atol=1e-8)
    with pytest.raises(ValueError, match="labels"):
        survan_cox(time, event, design, strata=["A", np.nan, "B", "B"])
    with pytest.raises(ValueError, match="sequence"):
        survan_cox(time, event, design, strata="AABB")
