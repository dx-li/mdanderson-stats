import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import asypow_smo_binomial


def test_original_example_smo_and_df_correction():
    for correction, power, sample_size in (
        (False, 0.18209832212104851, 715.21739140064585),
        (True, 0.06123192415320388, 806.3411144622047),
    ):
        x = asypow_smo_binomial([0.4, 0.3], group_size=[10, 9], subtract_df=correction)
        assert_allclose(x.null_parameters, 0.35263157894736841, atol=1e-15)
        assert_allclose(x.divergence_per_observation, 0.010974090680255110, rtol=1e-13)
        assert_allclose(x.power(100), power, atol=1e-13)
        assert_allclose(x.sample_size(), sample_size, rtol=1e-12)
        assert_allclose(x.power(x.sample_size()), 0.8, atol=1e-13)
        assert x.power(np.floor(x.sample_size())) < 0.8 <= x.power(np.ceil(x.sample_size()))


def test_fixed_null_multiple_df_and_invalid_low_sample_size():
    x = asypow_smo_binomial([0.2, 0.4, 0.7], null_probabilities=0.5)
    assert x.degrees_of_freedom == 3
    alpha = x.significance([100, 200], [0.8, 0.9])
    assert_allclose(x.power([100, 200], alpha), [0.8, 0.9], atol=1e-13)
    with pytest.raises(ValueError, match="positive noncentrality"):
        x.power(0)
    same = asypow_smo_binomial([0.5, 0.5])
    assert same.divergence_per_observation == 0
    with pytest.raises(ValueError, match="non-null"):
        same.sample_size()


def test_close_alternatives_and_weight_scaling():
    delta = 1e-10
    p = 0.5 + delta
    x = asypow_smo_binomial([p], null_probabilities=0.5, subtract_df=False)
    # Twice Bernoulli KL around 1/2 is 4*delta^2 + O(delta^4).
    assert_allclose(x.divergence_per_observation, 4 * (p - 0.5) ** 2, rtol=1e-12)
    a = asypow_smo_binomial([0.4, 0.3], group_size=[10, 9])
    b = asypow_smo_binomial([0.4, 0.3], group_size=[1e301, 9e300])
    assert_allclose(a.divergence_per_observation, b.divergence_per_observation, rtol=1e-13)
    rare = asypow_smo_binomial([1e-300], null_probabilities=2e-300, subtract_df=False)
    assert_allclose(rare.divergence_per_observation, 2e-300 * (1 - np.log(2)), rtol=1e-12)
    assert not a.null_parameters.flags.writeable
