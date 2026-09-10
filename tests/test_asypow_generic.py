import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import asypow_smo_binomial, asypow_smo_generic


@pytest.mark.parametrize("analytic", [False, True])
def test_correlated_normal_expected_likelihood(analytic):
    information = np.array([[2.0, 1.0], [1.0, 2.0]])

    def ell(p, q):
        delta = q - p
        return -0.5 * float(delta @ information @ delta)

    def grad(p, q):
        return -information @ (q - p)

    x = asypow_smo_generic(
        [1, 2], ell, gradient=grad if analytic else None, lower=-10, upper=10, constraints=[1, 1, 0]
    )
    assert_allclose(x.null_parameters, [0, 2.5], atol=1e-9)
    assert_allclose(x.divergence_per_observation, 1.5, rtol=1e-13)
    assert_allclose(x.power(x.sample_size()), 0.8, atol=1e-13)
    assert not x.null_parameters.flags.writeable


def test_generic_binomial_matches_original_model():
    weights = np.array([10, 9]) / 19

    def ell(p, q):
        return float(weights @ (p * np.log(q) + (1 - p) * np.log1p(-q)))

    def grad(p, q):
        return weights * (p / q - (1 - p) / (1 - q))

    x = asypow_smo_generic(
        [0.4, 0.3],
        ell,
        gradient=grad,
        lower=0.01,
        upper=0.99,
        constraints=[2, 1, 2],
        initial=[0.35, 0.35],
    )
    original = asypow_smo_binomial([0.4, 0.3], group_size=[10, 9])
    assert_allclose(x.null_parameters, original.null_parameters, atol=1e-9)
    assert_allclose(x.divergence_per_observation, original.divergence_per_observation, rtol=1e-12)
    assert_allclose(x.sample_size(), original.sample_size(), rtol=1e-12)


def test_scaled_parameters_fixed_null_and_invalid_callbacks():
    for scale in [1e-200, 1e200]:

        def ell(p, q):
            delta = q / scale - p / scale
            return -0.5 * float(delta @ delta)

        def grad(p, q):
            return -(q / scale - p / scale) / scale

        x = asypow_smo_generic(
            np.array([1, 2, 3]) * scale,
            ell,
            gradient=grad,
            lower=-10 * scale,
            upper=10 * scale,
            constraints=[[2, 1, 2], [1, 3, 0]],
        )
        assert x.degrees_of_freedom == 2
        assert_allclose(x.null_parameters / scale, [1.5, 1.5, 0], atol=1e-9)
        assert_allclose(x.divergence_per_observation, 9.5, rtol=1e-13)
    fixed = asypow_smo_generic(
        [1], lambda p, q: -0.5 * float((q - p) @ (q - p)), lower=-2, upper=2, constraints=[1, 1, 0]
    )
    assert fixed.divergence_per_observation == 1
    with pytest.raises(ValueError, match="finite"):
        asypow_smo_generic([1], lambda p, q: np.nan, lower=-2, upper=2, constraints=[1, 1, 0])
    with pytest.raises(ArithmeticError, match="centered"):
        asypow_smo_generic(
            [1],
            lambda p, q: 1e20 - 0.5 * float((q - p) @ (q - p)),
            lower=-2,
            upper=2,
            constraints=[1, 1, 0],
        )
    with pytest.raises(ValueError, match="fixed values"):
        asypow_smo_generic([1], lambda p, q: 0.0, lower=0, upper=2, constraints=[1, 1, 0])
