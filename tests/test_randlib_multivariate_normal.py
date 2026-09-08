import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import RandlibGenerator, RandlibMultivariateNormal

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/randlib_multivariate_normal.json").read_text()
)


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_factors_vectors_and_states(case):
    source = {"f77": "fortran", "f95": "fortran95", "c": "c"}[case["language"]]
    params = RandlibMultivariateNormal(case["mean"], case["covariance"], legacy=True, source=source)
    packed = np.asarray(case["packed_parameters"], dtype=np.float32).astype(float)
    d = len(params.mean)
    assert_array_equal(params.mean, packed[1 : d + 1])
    assert_array_equal(params.factor.T[np.triu_indices(d)], packed[d + 1 :])
    bank = RandlibGenerator(case["seed"], stream=case["stream"])
    bank.set_antithetic(case["antithetic"])
    values, states = [], []
    for _ in case["values"]:
        values.append(bank.multivariate_normal(params)[0])
        states.append(bank.get_seeds())
    assert_array_equal(values, np.asarray(case["values"], dtype=np.float32).astype(float))
    assert_array_equal(states, case["states"])
    bank.reinitialize()
    assert_array_equal(bank.multivariate_normal(params, len(values)), values)
    assert bank.get_seeds() == tuple(case["states"][-1])


@pytest.mark.parametrize(
    "legacy,source", [(False, "fortran"), (True, "fortran"), (True, "fortran95"), (True, "c")]
)
def test_joint_distribution_and_factor(legacy, source):
    covariance = np.array([[4, -1, 0.2], [-1, 2, 0.5], [0.2, 0.5, 1]])
    params = RandlibMultivariateNormal([1, -2, 3], covariance, legacy=legacy, source=source)
    assert_allclose(params.factor @ params.factor.T, covariance, rtol=2e-6, atol=2e-7)
    values = RandlibGenerator().multivariate_normal(params, 20000)
    assert np.all(
        abs(values.mean(axis=0) - params.mean) < 6 * np.sqrt(np.diag(covariance) / len(values))
    )
    assert_allclose(np.cov(values, rowvar=False, bias=True), covariance, rtol=0.04, atol=0.035)
    assert values.shape == (20000, 3)
    assert not values.flags.writeable


def test_default_normal_identity_and_scalar_batch():
    params = RandlibMultivariateNormal([0, 0, 0], np.eye(3))
    bank, reference = RandlibGenerator(), RandlibGenerator()
    values = bank.multivariate_normal(params, 100)
    assert_array_equal(values, reference.normal(300).reshape(100, 3))
    assert bank.get_seeds() == reference.get_seeds()
    correlated = RandlibMultivariateNormal([1, -2], [[3, -0.5], [-0.5, 1]])
    bank.reinitialize()
    values = bank.multivariate_normal(correlated, 100)
    before = bank.get_seeds()
    bank.reinitialize()
    assert_allclose(
        values, [bank.multivariate_normal(correlated)[0] for _ in values], rtol=2e-15, atol=1e-15
    )
    assert bank.get_seeds() == before


def test_prepared_parameters_are_independent_and_immutable():
    mean = np.array([1.0, 2.0])
    covariance = np.eye(2)
    params = RandlibMultivariateNormal(mean, covariance)
    expected = RandlibGenerator().multivariate_normal(params, 10)
    mean[:] = 999
    covariance[:] = 0
    assert_array_equal(RandlibGenerator().multivariate_normal(params, 10), expected)
    with pytest.raises(ValueError):
        params.mean[0] = 0
    with pytest.raises(ValueError):
        params.factor.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        params.source = "c"


def test_legacy_uses_only_upper_triangle():
    params = RandlibMultivariateNormal([0, 0], [[2, 0.5], [np.nan, 1]], legacy=True)
    other = RandlibMultivariateNormal([0, 0], [[2, 0.5], [0.5, 1]], legacy=True)
    assert_array_equal(params.factor, other.factor)
    assert_array_equal(
        RandlibGenerator().multivariate_normal(params, 10),
        RandlibGenerator().multivariate_normal(other, 10),
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_empty_resource_and_budget_rollback(legacy):
    bank = RandlibGenerator(max_draws=10)
    params = RandlibMultivariateNormal([0, 0], np.eye(2), legacy=legacy)
    before = bank.get_seeds()
    assert bank.multivariate_normal(params, 0).shape == (0, 2)
    assert bank.get_seeds() == before
    with pytest.raises(ArithmeticError, match="max_attempts"):
        bank.multivariate_normal(params, 5, max_attempts=1)
    with pytest.raises(ValueError, match="dimension"):
        bank.multivariate_normal(params, 6)
    for size in [-1, True, 1.5]:
        with pytest.raises(ValueError):
            bank.multivariate_normal(params, size)
    with pytest.raises(ValueError):
        bank.multivariate_normal(params, max_attempts=0)
    with pytest.raises(ValueError):
        bank.multivariate_normal(None)
    assert bank.get_seeds() == before


@pytest.mark.parametrize(
    "mean,covariance,kwargs",
    [
        ([], [], {}),
        ([0], [[0]], {}),
        ([0], [[-1]], {}),
        ([0, 0], [[1, 2], [2, 1]], {}),
        ([0, 0], [[1, 1], [1, 1]], {}),
        ([0, 0], [[1, 0], [0.1, 1]], {}),
        ([0], [[np.inf]], {}),
        ([np.nan], [[1]], {}),
        ([0], [1], {}),
        ([0, 0], [[1, 0], [0, 1]], {"max_dimension": 1}),
        ([0], [[1]], {"max_dimension": True}),
        ([0], [[1]], {"legacy": 1}),
        ([0], [[1]], {"source": "c"}),
        ([0], [[1]], {"source": "bad", "legacy": True}),
        ([0], [[1e-50]], {"legacy": True}),
        ([1e50], [[1]], {"legacy": True}),
        ([0, 0], [[1, 1], [1, 1 + 1e-8]], {"legacy": True}),
    ],
)
def test_invalid_covariance_and_parameters(mean, covariance, kwargs):
    with pytest.raises(ValueError):
        RandlibMultivariateNormal(mean, covariance, **kwargs)


def test_default_retains_nearly_singular_positive_definite_covariance():
    cov = np.array([[1, 1], [1, 1 + 1e-8]])
    params = RandlibMultivariateNormal([0, 0], cov)
    assert_allclose(params.factor @ params.factor.T, cov, rtol=1e-15, atol=0)


def test_prepared_models_can_be_interleaved():
    first = RandlibMultivariateNormal([1, 2], [[2, 0.3], [0.3, 1]], legacy=True, source="fortran95")
    expected = RandlibGenerator().multivariate_normal(first, 10)
    second = RandlibMultivariateNormal([-1, 0, 1], np.eye(3), legacy=True, source="fortran95")
    RandlibGenerator().multivariate_normal(second, 10)
    assert_array_equal(RandlibGenerator().multivariate_normal(first, 10), expected)
