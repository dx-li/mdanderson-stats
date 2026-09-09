"""Independent delta-method, native, tiny-group and boundary-uncertainty checks."""

import math
from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext

import numpy as np
import pytest
from test_sppcr_reference import FIXTURE, rows

from mdanderson_stats import sppcr_fit_means, sppcr_frequencies


@pytest.mark.parametrize("means", [[1, 2], [1, 2, 3], [0.01, 0.3, 2, 7]])
@pytest.mark.parametrize("parents", [(0, 0), (0, 1), (1, 1)])
@pytest.mark.parametrize("scale", [1e-150, 1, 1e150])
def test_delta_method_matches_independent_jacobian(means, parents, scale):
    mu = np.array(means)
    variance = np.arange(1, len(mu) + 1) * 0.03
    result = sppcr_frequencies(mu * scale, variance * scale**2, progenitor=parents)
    total = mu.sum()
    gradient = (np.eye(len(mu)) * total - mu[:, None]) / total**2
    expected = (gradient * variance) @ gradient.T
    np.testing.assert_allclose(result.frequency.value, mu / total, rtol=2e-13)
    np.testing.assert_allclose(result.frequency.complement, (total - mu) / total, rtol=2e-13)
    np.testing.assert_allclose(result.frequency.variance, np.diag(expected), rtol=4e-13)
    np.testing.assert_allclose(
        result.frequency.standard_error, np.sqrt(np.diag(expected)), rtol=3e-13
    )
    mask = np.array([i not in parents for i in range(len(mu))])
    mutant = mu[mask].sum() / total
    mutant_gradient = (mask - mutant) / total
    expected_var = np.sum(mutant_gradient**2 * variance)
    assert result.mutant.value == pytest.approx(mutant, rel=2e-13)
    assert result.mutant.variance == pytest.approx(expected_var, rel=4e-13)
    assert result.calibration.value == pytest.approx(total * scale, rel=3e-15)
    assert result.calibration.variance == pytest.approx(variance.sum() * scale**2, rel=2e-13, abs=0)
    p = mu / total
    np.testing.assert_allclose(
        result.frequency.transformed.value, 2 * np.arcsin(np.sqrt(p)), rtol=2e-13
    )
    np.testing.assert_allclose(
        result.frequency.transformed.variance, np.diag(expected) / (p * (1 - p)), rtol=4e-13
    )


@pytest.mark.parametrize(
    "name", ["single_level", "scaled_dna", "heterozygous", "two_levels_exact_model"]
)
def test_mean_fit_to_frequency_summary_matches_valid_native_results(name):
    case = FIXTURE["cases"][name]
    fit = sppcr_fit_means(case["dna"], case["seen"], case["wells"])
    result = sppcr_frequencies(
        fit.mu, fit.variance, progenitor=tuple(i - 1 for i in case["progenitor"])
    )
    for actual, tag in [
        (result.frequency, "frequency"),
        (result.mutant, "mutant"),
        (result.calibration, "calibration"),
    ]:
        native = rows(name, tag)
        if tag != "frequency":
            native = native[0]
        np.testing.assert_allclose(actual.value, native[..., 0], rtol=5e-14)
        np.testing.assert_allclose(actual.variance, native[..., 1], rtol=5e-14)
        np.testing.assert_allclose(actual.standard_error, native[..., 2], rtol=5e-14)


def test_tiny_mutant_group_and_complement_are_not_lost_to_subtraction():
    mu = [1e100, 1e80, 1e80]
    var = [1e180, 1e160, 2e160]
    result = sppcr_frequencies(mu, var, progenitor=(0, 0))
    with localcontext() as ctx:
        ctx.prec = 100
        m = list(map(Decimal, map(str, mu)))
        v = list(map(Decimal, map(str, var)))
        total = sum(m)
        x = m[1] + m[2]
        y = m[0]
        mutant = x / total
        expected_var = (y * y * (v[1] + v[2]) + x * x * v[0]) / total**4
    assert result.frequency.value[0] == 1
    assert result.frequency.complement[0] == pytest.approx(float(mutant), rel=5e-14)
    assert result.mutant.value == pytest.approx(float(mutant), rel=5e-14)
    assert result.mutant.variance == pytest.approx(float(expected_var), rel=1e-13)
    assert result.frequency.variance[0] == pytest.approx(float(expected_var), rel=1e-13)
    assert result.frequency.transformed.value[0] < math.pi


def test_transform_retains_information_when_raw_variance_underflows():
    result = sppcr_frequencies([1, 1e-300], [0, 1e-320], progenitor=(0, 0))
    assert result.mutant.variance > 0
    assert result.mutant.transformed.variance == pytest.approx(1e-320 / 1e-300, rel=2e-13)
    assert result.mutant.transformed.value == pytest.approx(2e-150, rel=1e-13)
    # A standard error can remain representable when variance does not.
    scaled = sppcr_frequencies([1e100, 1e100], [1e-200, 1e-200], progenitor=(0, 0))
    assert scaled.frequency.variance[0] == 0
    assert scaled.frequency.standard_error[0] > 0


def test_boundary_fit_uncertainty_remains_unavailable():
    fit = sppcr_fit_means([1], [[0, 5]], 10)
    result = sppcr_frequencies(fit.mu, fit.variance, progenitor=(0, 0))
    np.testing.assert_array_equal(result.frequency.value, [0, 1])
    assert np.all(np.isnan(result.frequency.variance))
    assert np.all(~result.frequency.variance_available)
    assert np.isnan(result.calibration.variance)
    assert np.isnan(result.mutant.transformed.variance)


def test_estimates_without_variances_and_structural_constants():
    result = sppcr_frequencies([1, 2], progenitor=(0, 1))
    assert np.all(np.isnan(result.frequency.variance))
    assert result.mutant.value == 0 and result.mutant.variance == 0
    assert result.mutant.transformed.variance == 0
    single = sppcr_frequencies([2], progenitor=(0, 0))
    assert single.frequency.value[0] == 1 and single.frequency.variance[0] == 0
    assert single.frequency.transformed.value[0] == math.pi
    assert single.frequency.transformed.variance[0] == 0
    assert np.isnan(single.calibration.variance)


def test_nonstructural_transform_at_boundary_is_not_clipped():
    result = sppcr_frequencies([0, 1], [0, 1], progenitor=(0, 0))
    np.testing.assert_array_equal(result.frequency.variance, [0, 0])
    assert np.all(np.isnan(result.frequency.transformed.variance))


def test_batches_empty_batches_and_immutable_storage():
    mu = np.array([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], [[3.0, 4.0, 5.0], [4.0, 5.0, 6.0]]])
    variance = np.array([0.01, 0.02, 0.03])
    result = sppcr_frequencies(mu, variance, progenitor=(0, 1))
    for index in np.ndindex(mu.shape[:-1]):
        scalar = sppcr_frequencies(mu[index], variance, progenitor=(0, 1))
        np.testing.assert_array_equal(result.frequency.variance[index], scalar.frequency.variance)
    mu[:] = 9
    variance[:] = 9
    assert result.mu[0, 0, 0] == 1 and result.mu_variance[0, 0, 0] == 0.01
    with pytest.raises(ValueError):
        result.frequency.value.setflags(write=True)
    with pytest.raises(ValueError):
        result.frequency.variance_available.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        result.progenitor = (1, 1)
    empty = sppcr_frequencies(np.empty((0, 3)), 0.1, progenitor=(0, 1))
    assert empty.frequency.value.shape == (0, 3) and empty.mutant.value.shape == (0,)


@pytest.mark.parametrize(
    "mu,var,parents",
    [
        ([], None, (0, 0)),
        (1, None, (0, 0)),
        ([-1, 2], None, (0, 0)),
        ([0, 0], None, (0, 0)),
        ([math.inf, 1], None, (0, 0)),
        ([math.nan, 1], None, (0, 0)),
        ([1, 2], [-1, 1], (0, 0)),
        ([1, 2], [math.inf, 1], (0, 0)),
        ([1, 2], [1, 2, 3], (0, 0)),
        ([1, 2], None, (-1, 0)),
        ([1, 2], None, (0, 2)),
        ([1, 2], None, (0,)),
        ([1, 2], None, (0, 0.5)),
    ],
)
def test_invalid_inputs(mu, var, parents):
    with pytest.raises(ValueError):
        sppcr_frequencies(mu, var, progenitor=parents)


def test_unrepresentable_summary_raises():
    with pytest.raises(ArithmeticError):
        sppcr_frequencies([1e308, 1e308], progenitor=(0, 0))
    with pytest.raises(ArithmeticError):
        sppcr_frequencies([1, 1], [1e308, 1e308], progenitor=(0, 0))
