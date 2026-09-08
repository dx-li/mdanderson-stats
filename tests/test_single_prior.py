"""Raw-moment conversion and its composition with prior quadrature."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import single_normal_criterion, single_prior_parameters


def test_exact_lognormal_moments_and_latent_correlation():
    mean, covariance = single_prior_parameters(
        [0.2, 1.2], [0.02, 0.03], [[1, 0.4], [0.4, 1]], lognormal=[False, True]
    )
    result = single_normal_criterion(
        [-1, 1], [50, 50], mean, covariance, lognormal=[False, True], order=12
    )
    observed_mean = result.weights @ result.parameters
    assert_allclose(observed_mean, [0.2, 1.2], rtol=1e-13)
    assert_allclose(result.weights @ ((result.parameters - observed_mean) ** 2), [0.02, 0.03])
    assert_allclose(covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1]), 0.4)


def test_original_alnton_formula():
    mean, covariance = single_prior_parameters(
        [2, 3, -1],
        [1, 9, 4],
        [[1, 0.2, 0], [0.2, 1, -0.3], [0, -0.3, 1]],
        lognormal=[True, True, False],
        conversion="legacy",
    )
    assert_allclose(mean, [np.log(2), np.log(3), -1])
    assert_allclose(covariance, [[0.25, 0.1, 0], [0.1, 1, -0.6], [0, -0.6, 4]])


@pytest.mark.parametrize("conversion", ["exact", "legacy"])
def test_fixed_parameters_and_no_input_mutation(conversion):
    raw = np.array([2.0, -1.0])
    mean, cov = single_prior_parameters(
        raw, [0, 0], np.eye(2), lognormal=[True, False], conversion=conversion
    )
    assert_allclose(mean, [np.log(2), -1])
    assert_allclose(cov, 0)
    assert_allclose(raw, [2, -1])


def test_extreme_moment_ratio_uses_log_space():
    mean, cov = single_prior_parameters([1e-200, 1], [1e200, 1], np.eye(2), lognormal=[True, False])
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(cov))
    assert_allclose(cov[0, 0], 600 * np.log(10))
    with pytest.raises(ValueError, match="overflowed"):
        single_prior_parameters(
            [1e-200, 1], [1e200, 1], np.eye(2), lognormal=[True, False], conversion="legacy"
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"variance": [-1, 1]},
        {"mean": [1]},
        {"mean": [np.nan, 1]},
        {"mean": [0, 1], "lognormal": [True, False]},
        {"lognormal": [0, 1]},
        {"correlation": [[1, 0.1], [0, 1]]},
        {"correlation": [[0, 0], [0, 1]]},
        {"correlation": [[1, 2], [2, 1]]},
        {"conversion": "approximate"},
    ],
)
def test_invalid_inputs(changes):
    args = dict(mean=[1, 2], variance=[1, 1], correlation=np.eye(2))
    args.update(changes)
    with pytest.raises(ValueError):
        single_prior_parameters(**args)


def test_inconsistent_three_way_correlations():
    with pytest.raises(ValueError, match="positive semidefinite"):
        single_prior_parameters(
            [1, 2, 3], [1, 1, 1], [[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]]
        )
