import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.wfmm_variance_init import initialize_wfmm_variances

FIXTURES = Path(__file__).parent / "fixtures" / "wfmm-variance-init.csv"


def test_residual_only_reml_matches_closed_form_and_independent_reference():
    with FIXTURES.open(newline="") as stream:
        reference = list(csv.DictReader(stream))
    ref = next(row for row in reference if row["kind"] == "residual_only")
    y = np.asarray([1.0, 2.0, 3.0, 4.0])[:, None]
    result = initialize_wfmm_variances(y, np.ones((4, 1)))
    assert result.status == ("analytic_residual_only_reml",)
    assert result.optimizer_success[0]
    np.testing.assert_allclose(result.fixed_effect_estimates[0, 0], float(ref["beta"]))
    np.testing.assert_allclose(result.raw_residual_variance[0, 0], float(ref["variance"]))
    np.testing.assert_allclose(result.restricted_log_likelihood[0], float(ref["log_reml"]))
    assert result.starts_usable


def test_balanced_random_intercept_reml_matches_independent_reference():
    with FIXTURES.open(newline="") as stream:
        reference = list(csv.DictReader(stream))
    rows = [row for row in reference if row["kind"] == "mixed_reml"]
    y = np.asarray([-1.0, -0.8, 0.2, 0.4, 1.1, 0.9, 2.4, 2.2])[:, None]
    x = np.ones((8, 1))
    z = np.zeros((8, 4))
    for group in range(4):
        z[2 * group : 2 * group + 2, group] = 1.0
    result = initialize_wfmm_variances(y, x, z, max_evaluations=2_000)
    assert result.status == ("optimized",)
    assert result.optimizer_success[0]
    np.testing.assert_allclose(
        result.raw_random_variance[0, 0], float(rows[0]["variance"]), rtol=2e-5
    )
    np.testing.assert_allclose(
        result.raw_residual_variance[0, 0], float(rows[1]["variance"]), rtol=2e-5
    )
    np.testing.assert_allclose(
        result.fixed_effect_estimates[0, 0], float(rows[0]["beta"]), rtol=2e-6
    )
    np.testing.assert_allclose(
        result.restricted_log_likelihood[0], float(rows[0]["log_reml"]), rtol=2e-6
    )
    assert result.evaluations[0] <= 2_000


def test_perfect_fit_is_reported_as_numerical_boundary_and_floored_for_sampler():
    result = initialize_wfmm_variances(3.0 * np.ones((4, 1)), np.ones((4, 1)))
    assert result.status == ("numerical_zero_residual_boundary",)
    assert result.normalized_residual_norm[0] <= 8 * np.finfo(float).eps * 4
    np.testing.assert_array_equal(result.raw_residual_variance, 0.0)
    np.testing.assert_allclose(result.residual_variance[0, 0], 9.0e-8)
    assert result.starts_usable
    assert np.isnan(result.restricted_log_likelihood[0])


def test_zero_data_requires_explicit_absolute_scale_for_usable_starts():
    y = np.zeros((4, 1))
    x = np.ones((4, 1))
    no_scale = initialize_wfmm_variances(y, x)
    assert no_scale.status == ("zero_data_no_scale",)
    assert not no_scale.starts_usable
    absolute = initialize_wfmm_variances(y, x, zero_data_floor=0.25)
    assert absolute.status == ("zero_data_absolute_floor",)
    assert absolute.starts_usable
    np.testing.assert_array_equal(absolute.residual_variance, 0.25)


def test_rejects_unidentified_designs_and_caps_actual_likelihood_evaluations():
    x = np.ones((4, 1))
    y = np.asarray([0.0, 1.0, 2.0, 3.0])[:, None]
    with pytest.raises(ValueError, match="positive residual degrees"):
        initialize_wfmm_variances(y[:1], np.ones((1, 1)))
    with pytest.raises(ValueError, match="full column rank"):
        initialize_wfmm_variances(y, np.column_stack((x, x)))
    with pytest.raises(ValueError, match="not identifiable"):
        initialize_wfmm_variances(y, x, x.copy())

    z = np.zeros((4, 2))
    z[:2, 0] = 1.0
    z[2:, 1] = 1.0
    limited = initialize_wfmm_variances(y, x, z, max_evaluations=1)
    assert limited.evaluations[0] <= 1
    assert not limited.optimizer_success[0]
