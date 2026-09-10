import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import asypow_smo_binomial, asypow_smo_multinomial, asypow_smo_ordinal


def test_partial_cross_group_equality_and_binary_reduction():
    p = np.array([[0.2, 0.3], [0.4, 0.1]])
    x = asypow_smo_multinomial(p, constraints=[2, 1, 3], group_size=[1, 3])
    y = asypow_smo_ordinal(p.cumsum(axis=1), constraints=[2, 1, 3], group_size=[1, 3])
    assert_allclose(x.null_parameters, [[0.35, 0.24375], [0.35, 0.65 / 6]], atol=1e-11)
    assert_allclose(y.null_parameters, x.null_parameters.cumsum(axis=1), atol=1e-11)
    binary = asypow_smo_binomial([0.2, 0.4], group_size=[1, 3])
    for design in [x, y]:
        assert design.degrees_of_freedom == 1
        assert_allclose(
            design.divergence_per_observation, binary.divergence_per_observation, rtol=1e-12
        )
        assert_allclose(design.power(design.sample_size()), 0.8, atol=1e-13)
    # Complete equality agrees with the original native example and shortcut.
    complete = asypow_smo_multinomial(p, constraints=[[2, 1, 3], [2, 2, 4]], group_size=[1, 3])
    assert_allclose(complete.divergence_per_observation, 0.06730956764893814, rtol=1e-12)


def test_coupled_components_have_zero_likelihood_score():
    p = [[0.1, 0.2], [0.3, 0.4], [0.25, 0.15]]
    constraints = [[2, 1, 3], [2, 3, 6], [2, 2, 5]]
    x = asypow_smo_multinomial(p, constraints=constraints, group_size=[1, 2, 4])
    # Profile the two conditional probabilities analytically, then optimize a.
    a = 1.3 / 7
    b = (1 - a) * 1.2 / 4.3
    c = (1 - a) * 0.8 / 1.4
    assert_allclose(x.null_parameters, [[a, b], [a, c], [b, a]], atol=1e-11)
    assert x.degrees_of_freedom == 3
    a, b = x.null_parameters[0]
    c = x.null_parameters[1, 1]
    score = [
        1.3 / a - 3.1 / (1 - a - b) - 0.6 / (1 - a - c),
        1.2 / b - 3.1 / (1 - a - b),
        0.8 / c - 0.6 / (1 - a - c),
    ]
    assert_allclose(score, 0, atol=1e-9)
    redundant = asypow_smo_multinomial(
        p, constraints=constraints[::-1] + [[2, 6, 1]], group_size=[1e300, 2e300, 4e300]
    )
    assert redundant.degrees_of_freedom == x.degrees_of_freedom
    assert_allclose(redundant.null_parameters, x.null_parameters, atol=1e-11)
    within = asypow_smo_multinomial([0.2, 0.3], constraints=[2, 1, 2])
    assert_allclose(within.null_parameters, [[0.25, 0.25]], atol=1e-11)


def test_mixed_fixed_equality_and_infeasible_ordinal_null():
    x = asypow_smo_multinomial([[0.2, 0.3], [0.4, 0.1]], constraints=[[2, 1, 3], [1, 2, 0.2]])
    assert x.degrees_of_freedom == 2
    assert_allclose(x.null_parameters[0, 1], 0.2, atol=0)
    assert_allclose(x.null_parameters[0, 0], x.null_parameters[1, 0], atol=0)
    assert_allclose(x.power(300, x.significance(300)), 0.8, atol=1e-13)
    same = asypow_smo_ordinal([[0.2, 0.5], [0.2, 0.7]], constraints=[2, 1, 3])
    assert same.divergence_per_observation == 0
    with pytest.raises(ValueError, match="positive interior"):
        asypow_smo_ordinal([0.2, 0.5], constraints=[2, 1, 2])
    with pytest.raises(ValueError, match="feasibility|positive interior"):
        asypow_smo_multinomial(
            [[0.2, 0.3], [0.4, 0.1]], constraints=[[2, 1, 2], [1, 1, 0.6], [2, 3, 4]]
        )


def test_near_null_equality_retains_small_divergence():
    p = np.array([[0.2 + 1e-9, 0.3], [0.2, 0.3]])
    x = asypow_smo_multinomial(p, constraints=[2, 1, 3])
    binary = asypow_smo_binomial(p[:, 0])
    assert_allclose(
        x.divergence_per_observation, binary.divergence_per_observation, rtol=1e-5, atol=0
    )
