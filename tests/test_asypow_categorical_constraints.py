import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import asypow_smo_binomial, asypow_smo_multinomial, asypow_smo_ordinal


def test_partial_multinomial_null_maximizes_expected_likelihood():
    x = asypow_smo_multinomial([0.2, 0.3], constraints=[1, 1, 0.4])
    assert x.degrees_of_freedom == 1
    assert_allclose(x.null_parameters, [[0.4, 0.225]], atol=1e-15)
    assert_allclose(x.divergence_per_observation, 0.18303244369887126, rtol=1e-13)
    # Original source leaves q2=.3, giving a nonmaximal likelihood and larger w.
    assert x.divergence_per_observation < 0.23356675154201234
    binary = asypow_smo_binomial([0.2], null_probabilities=0.4)
    assert_allclose(x.power(100), binary.power(100), atol=1e-13)
    assert_allclose(x.power(x.sample_size()), 0.8, atol=1e-13)
    q = np.r_[x.null_parameters[0], 1 - x.null_parameters.sum()]
    assert_allclose(0.3 / q[1] - 0.5 / q[2], 0, atol=1e-14)


def test_fixed_ordinal_threshold_preserves_conditional_masses():
    x = asypow_smo_ordinal([0.2, 0.5, 0.8], constraints=[1, 2, 0.6])
    assert_allclose(x.null_parameters, [[0.24, 0.6, 0.84]], atol=1e-15)
    assert_allclose(x.divergence_per_observation, 0.040821994520255173, rtol=1e-13)
    binary = asypow_smo_binomial([0.5], null_probabilities=0.6)
    assert_allclose(x.power(300), binary.power(300), atol=1e-13)
    multiple = asypow_smo_ordinal([0.1, 0.3, 0.5, 0.8], constraints=[[1, 2, 0.4], [1, 4, 0.9]])
    assert multiple.degrees_of_freedom == 2
    assert_allclose(multiple.null_parameters, [[0.4 / 3, 0.4, 0.6, 0.9]], atol=1e-15)


def test_fixed_components_multiple_groups_and_invalid_constraints():
    p = [[0.2, 0.3], [0.1, 0.4]]
    # Anchor a component spanning groups at a fixed probability.
    x = asypow_smo_multinomial(p, constraints=[[2, 1, 3], [1, 1, 0.4]], group_size=[1, 3])
    assert x.degrees_of_freedom == 2
    assert_allclose(x.null_parameters, [[0.4, 0.225], [0.4, 0.6 * 0.4 / 0.9]], atol=1e-15)
    unchanged = asypow_smo_multinomial(p, constraints=[1, 1, 0.2])
    assert unchanged.divergence_per_observation == 0
    assert not x.null_parameters.flags.writeable
    with pytest.raises(NotImplementedError, match="unfixed"):
        asypow_smo_multinomial(p, constraints=[2, 1, 3])
    with pytest.raises(ValueError, match="sum to less"):
        asypow_smo_multinomial([0.2, 0.3], constraints=[[1, 1, 0.6], [1, 2, 0.5]])
    with pytest.raises(ValueError, match="increase strictly"):
        asypow_smo_ordinal([0.2, 0.5], constraints=[[1, 1, 0.6], [1, 2, 0.5]])
    with pytest.raises(ValueError, match="either"):
        asypow_smo_ordinal([0.2, 0.5], constraints=[1, 1, 0.3], null_cumulative=[0.3, 0.6])
