import numpy as np

from mdanderson_stats import condis_impute, condis_linear_refine


def sample():
    fit = condis_impute([8, 1, 2, 2, 4, 6, 10, 10, 12, 15], [0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    x = np.column_stack(([1, 4, 2, 3, 7, 5, 9, 8, 6, 10], [2, 1, 4, 1, 5, 9, 2, 6, 5, 3]))
    return fit, x


def test_native_caret_gaussian_model_fixture_and_event_restoration():
    fit, x = sample()
    result = condis_linear_refine(fit, x)
    # Executed caret modelInfo$glm$fit on the exact segment-integral targets.
    expected = [
        9.478586289821392,
        4.383744494582396,
        10.437055851279418,
        3.768596827798641,
        6.915831284282199,
        13.140803588316354,
        7.631143775838308,
        14.471263746656218,
        6.300683617498445,
        15.186576238212323,
    ]
    np.testing.assert_allclose(result.fitted_time, expected, atol=5e-14, rtol=0)
    event = fit.status == 1
    np.testing.assert_array_equal(result.refined_time[event], fit.observed_time[event])
    np.testing.assert_array_equal(result.refined_time[~event], result.fitted_time[~event])
    assert result.rank == 4
    assert result.residual_degrees_of_freedom == 6
    assert result.above_horizon[-1]


def test_censoring_violation_is_reported_and_clipping_is_explicit():
    fit, _ = sample()
    x = np.zeros((10, 1))
    raw = condis_linear_refine(fit, x)
    censored = fit.status == 0
    # Intercept plus status fits separate observed/censored group means.
    np.testing.assert_allclose(raw.refined_time[censored], fit.imputed_time[censored].mean())
    assert raw.below_censoring[-1]
    assert raw.refined_time[-1] < fit.observed_time[-1]
    clipped = condis_linear_refine(fit, x, enforce_censoring=True)
    assert clipped.refined_time[-1] == fit.observed_time[-1]
    np.testing.assert_array_equal(clipped.fitted_time, raw.fitted_time)
    np.testing.assert_array_equal(clipped.below_censoring, raw.below_censoring)
    assert not clipped.refined_time.flags.writeable


def test_scaled_and_dependent_covariates_preserve_fitted_values():
    fit, x = sample()
    baseline = condis_linear_refine(fit, x)
    expanded = np.column_stack((x[:, 0] * 1e200, x[:, 1] * 1e-200, x[:, 0], np.ones(10)))
    for scale in [1e-200, 1e200]:
        scaled = condis_impute(fit.observed_time * scale, fit.status)
        result = condis_linear_refine(scaled, expanded)
        np.testing.assert_allclose(result.fitted_time / scale, baseline.fitted_time, rtol=2e-14)
        np.testing.assert_allclose(result.residual_rmse / scale, baseline.residual_rmse, rtol=2e-14)
        assert result.rank == baseline.rank
    zero = condis_linear_refine(condis_impute([0, 0], [1, 0]), [[1], [1]])
    np.testing.assert_array_equal(zero.refined_time, [0, 0])
