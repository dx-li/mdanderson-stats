import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import asypow_smo_binomial, asypow_smo_exponential, asypow_smo_poisson


def test_native_mixed_constraints_and_free_groups():
    constraints = [[2, 1, 2], [1, 3, 0.3]]
    binary = asypow_smo_binomial(
        [0.1, 0.2, 0.4, 0.6], constraints=constraints, group_size=[1, 2, 3, 4]
    )
    poisson = asypow_smo_poisson(
        [1, 2, 4, 6], constraints=[[2, 1, 2], [1, 3, 3]], group_size=[1, 2, 3, 4]
    )
    survival = asypow_smo_exponential(
        [0.1, 0.2, 0.4, 0.6], [5, 8, 4, 2], constraints=constraints, group_size=[1, 2, 3, 4]
    )
    for design, w, null in [
        (binary, 0.018708613876832647, [1 / 6, 1 / 6, 0.3, 0.6]),
        (poisson, 0.13412909456623945, [5 / 3, 5 / 3, 3, 6]),
        (survival, 0.02105833508977506, [0.17016978358077059] * 2 + [0.3, 0.6]),
    ]:
        assert design.degrees_of_freedom == 2
        assert_allclose(design.null_parameters, null, rtol=1e-14)
        assert_allclose(design.divergence_per_observation, w, rtol=1e-12)
        assert_allclose(design.power(design.sample_size()), 0.8, atol=1e-13)
        assert_allclose(design.power(1000, design.significance(1000)), 0.8, atol=1e-13)


def test_joined_chains_redundancy_and_fixed_components():
    p = [0.1, 0.2, 0.4, 0.8]
    bridge = np.array([[2, 1, 4], [2, 2, 3], [2, 3, 4]])
    expected = asypow_smo_binomial(p, group_size=[1, 2, 3, 4])
    for constraints in [bridge, bridge[::-1], np.vstack([bridge, [2, 4, 1], [2, 1, 1]])]:
        actual = asypow_smo_binomial(p, group_size=[1, 2, 3, 4], constraints=constraints)
        assert actual.degrees_of_freedom == 3
        assert_allclose(actual.null_parameters, 0.49, atol=1e-15)
        assert_allclose(actual.power(100), expected.power(100), atol=1e-14)
    fixed = asypow_smo_binomial(p, constraints=np.vstack([bridge, [1, 2, 0.3], [1, 4, 0.3]]))
    assert fixed.degrees_of_freedom == 4
    assert_allclose(fixed.null_parameters, 0.3, atol=0)
    assert_allclose(
        fixed.power(100), asypow_smo_binomial(p, null_probabilities=0.3).power(100), atol=1e-14
    )
    # An equality within a tiny-allocation component still uses its local weights.
    scaled = asypow_smo_poisson(
        [1, 3, 1e300], constraints=[2, 1, 2], group_size=[1e-300, 3e-300, 1e300]
    )
    assert_allclose(scaled.null_parameters, [2.5, 2.5, 1e300], rtol=1e-12)


def test_inconsistent_and_invalid_hypotheses():
    for constraints, message in [
        ([[2, 1, 2], [1, 1, 0.2], [1, 2, 0.3]], "conflicting"),
        ([2, 0, 1], "indices"),
        ([2, 1, 2.5], "indices"),
        ([2, 1, 3], "indices"),
        ([3, 1, 2], "type"),
        ([2, 1, 1], "independent restriction"),
        ([1, 1, 1], "strictly"),
    ]:
        with pytest.raises(ValueError, match=message):
            asypow_smo_binomial([0.2, 0.4], constraints=constraints)
    with pytest.raises(ValueError, match="either"):
        asypow_smo_poisson([1, 2], null_means=1, constraints=[2, 1, 2])
