import numpy as np
import pytest

from mdanderson_stats import condis_impute


def test_hand_calculated_conditional_means_and_partial_horizon():
    # KM levels at 1,2,4,6 are .75,.75,.375,.375.
    time, status = [1, 2, 4, 6], [1, 0, 1, 0]
    linear = condis_impute(time, status)
    step = condis_impute(time, status, interpolation="step")
    np.testing.assert_allclose(linear.imputed_time, [1, 4.5, 4, 6])
    np.testing.assert_allclose(step.imputed_time, [1, 5, 4, 6])
    np.testing.assert_allclose(
        condis_impute(time, status, horizon=3).imputed_time, [1, 2.875, 4, 6]
    )
    np.testing.assert_allclose(
        condis_impute(time, status, horizon=3, interpolation="step").imputed_time, [1, 3, 4, 6]
    )
    assert not linear.imputed_time.flags.writeable


def test_native_condis_012_fixture_and_input_order():
    t = np.array([8, 1, 2, 2, 4, 6, 10, 10, 12, 15])
    d = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    # Unmodified R source with survival::survfit and default integrate tolerances.
    for horizon, expected in [
        (
            None,
            [
                11.99999882086889,
                1,
                10.71417176476433,
                2,
                4,
                12.00000540715811,
                10,
                13.00000044896761,
                12,
                15,
            ],
        ),
        (9, [8.9375, 1, 8.089287738050544, 2, 4, 8.937499670897255, 10, 10, 12, 15]),
    ]:
        result = condis_impute(t, d, horizon=horizon)
        np.testing.assert_allclose(result.imputed_time, expected, atol=1.2e-4, rtol=0)
        permutation = np.array([5, 3, 8, 9, 1, 0, 7, 4, 6, 2])
        shuffled = condis_impute(t[permutation], d[permutation], horizon=horizon)
        np.testing.assert_array_equal(shuffled.imputed_time, result.imputed_time[permutation])
        np.testing.assert_array_equal(result.imputed_time[d == 1], t[d == 1])


def test_restriction_degenerate_samples_and_extreme_time_units():
    t = np.array([1.0, 2.0, 4.0, 6.0])
    d = [1, 0, 1, 0]
    for method in ["linear", "step"]:
        baseline = condis_impute(t, d, interpolation=method).imputed_time
        for scale in [1e-200, 1e200]:
            fit = condis_impute(t * scale, d, interpolation=method)
            np.testing.assert_allclose(fit.imputed_time / scale, baseline, rtol=2e-15)
        np.testing.assert_array_equal(
            condis_impute([2, 2], [0, 0], interpolation=method).imputed_time, [2, 2]
        )
        np.testing.assert_array_equal(
            condis_impute(t, [0, 0, 0, 0], interpolation=method).imputed_time, [6] * 4
        )
        np.testing.assert_array_equal(
            condis_impute([0], [0], interpolation=method).imputed_time, [0]
        )
    # Scaling must use the chosen horizon, not a remote maximum observation.
    restricted = condis_impute([1e-200, 1e200], [0, 1], horizon=2e-200, interpolation="step")
    assert restricted.imputed_time[0] == 2e-200
    with pytest.raises(ValueError, match="maximum observed"):
        condis_impute(t, d, horizon=7)
