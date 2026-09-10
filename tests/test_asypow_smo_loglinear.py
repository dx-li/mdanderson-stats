import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import asypow_smo_binomial, asypow_smo_design


def test_original_multiplicative_binomial_expected_likelihood():
    x = asypow_smo_design(
        [0.2, 0.8, 1.1],
        [[1, -1, 0], [1, 0, 1], [1, 1, 0], [1, 2, 1]],
        family="loglinear",
        constraints=[1, 3, 1],
        lower=[0.05, 0.6, 0.8],
        upper=[0.3, 1.2, 1.3],
        observations=[1, 2, 3, 4],
    )
    assert_allclose(x.null_parameters, [0.20835227932660202, 0.81372863369567638, 1], atol=1e-8)
    assert_allclose(x.divergence_per_observation, 0.00040554979088569532, rtol=1e-10)
    assert_allclose(x.sample_size(), 21819.418251950872, rtol=1e-10)
    assert_allclose(x.power(x.sample_size()), 0.8, atol=1e-13)


def test_identity_design_constraints_and_rare_probabilities():
    for scale in [1, 1e-250]:
        p = np.array([0.2, 0.4]) * scale
        x = asypow_smo_design(
            p,
            np.eye(2),
            family="loglinear",
            constraints=[2, 1, 2],
            lower=0.01 * scale,
            upper=0.99 * scale,
            observations=[1e300, 3e300],
        )
        binary = asypow_smo_binomial(p, group_size=[1, 3])
        assert_allclose(x.null_parameters, binary.null_parameters, rtol=1e-8, atol=0)
        assert_allclose(
            x.divergence_per_observation, binary.divergence_per_observation, rtol=1e-10, atol=0
        )
        assert x.degrees_of_freedom == 1
        assert not x.null_parameters.flags.writeable
    fixed = asypow_smo_design(
        [0.2, 0.4],
        np.eye(2),
        family="loglinear",
        constraints=[[1, 1, 0.3], [1, 2, 0.3]],
        lower=0.01,
        upper=0.99,
    )
    assert_allclose(fixed.null_parameters, 0.3, rtol=1e-14)
    assert fixed.degrees_of_freedom == 2
    with pytest.raises(ValueError, match="positive"):
        asypow_smo_design(
            [0.2], [[1]], family="loglinear", constraints=[1, 1, 0], lower=0.01, upper=0.99
        )
    with pytest.raises(ValueError, match="predictors must be negative"):
        asypow_smo_design(
            [2], [[1]], family="loglinear", constraints=[1, 1, 1.5], lower=0.5, upper=3
        )
