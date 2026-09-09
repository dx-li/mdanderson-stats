"""Closed-form, independent high-precision, native and boundary checks of PCR fits."""

import math
from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext

import numpy as np
import pytest
from test_sppcr_reference import FIXTURE, rows

from mdanderson_stats import sppcr_fit_means


@pytest.mark.parametrize("n", [2, 5, 20, 100, 1000])
@pytest.mark.parametrize("fraction", [0.1, 0.3, 0.5, 0.9])
@pytest.mark.parametrize("dna", [0.01, 1, 10, 1e-150, 1e150])
def test_single_level_matches_closed_form(n, fraction, dna):
    s = max(1, min(n - 1, round(n * fraction)))
    result = sppcr_fit_means([dna], [[s]], n)
    p = s / n
    expected = -math.log1p(-p) / dna
    assert result.mu[0] == pytest.approx(expected, rel=3e-14)
    # Log form prevents overflowing dna**2 in the independent variance expression.
    variance = math.exp(math.log(p / (n * (1 - p))) - 2 * math.log(dna))
    assert result.variance[0] == pytest.approx(variance, rel=4e-13, abs=0)
    ll = s * math.log(p) + (n - s) * math.log1p(-p)
    assert result.log_likelihood[0] == pytest.approx(ll, rel=2e-14)
    assert result.interior[0] and not result.adjusted[0]
    assert result.mu_lower[0] <= result.mu[0] <= result.mu_upper[0]


def decimal_mean(dna, seen, wells):
    # Solve for q=exp(-mu), independently of the production tau-space algorithm.
    with localcontext() as context:
        context.prec = 70
        lo, hi = Decimal(0), Decimal(1)
        for _ in range(220):
            q = (lo + hi) / 2
            derivative = sum(
                Decimal(s * d) * q**d / (1 - q**d) - Decimal((n - s) * d)
                for d, s, n in zip(dna, seen, wells, strict=True)
            )
            if derivative > 0:
                hi = q
            else:
                lo = q
        return float(-((lo + hi) / 2).ln())


@pytest.mark.parametrize("seen", [[0, 20, 100], [1, 10, 50], [19, 30, 1], [0, 0, 1], [19, 39, 99]])
def test_unequal_levels_match_independent_decimal_likelihood_root(seen):
    d, n = [1, 2, 3], [20, 40, 100]
    result = sppcr_fit_means(d, np.array(seen)[:, None], n)
    expected = decimal_mean(d, seen, n)
    assert result.mu[0] == pytest.approx(expected, rel=3e-14)
    assert not result.adjusted[0]
    info = sum(
        s * di**2 * math.exp(-di * expected) / (-math.expm1(-di * expected)) ** 2
        for s, di in zip(seen, d, strict=True)
    )
    assert result.variance[0] == pytest.approx(1 / info, rel=4e-14)


@pytest.mark.parametrize(
    "name", ["single_level", "scaled_dna", "heterozygous", "two_levels_exact_model"]
)
def test_agrees_with_valid_native_interior_baseline(name):
    case = FIXTURE["cases"][name]
    result = sppcr_fit_means(case["dna"], case["seen"], case["wells"])
    native = rows(name, "mu")
    np.testing.assert_allclose(result.mu, native[:, 1], rtol=3e-14)
    np.testing.assert_allclose(result.variance, native[:, 2], rtol=4e-14)


def test_batch_and_allele_axes_agree_with_independent_fits():
    seen = np.array([[[2, 5], [5, 8]], [[1, 0], [4, 0]], [[8, 3], [9, 6]]])
    wells = np.array([[10, 20], [8, 12], [12, 20]])
    result = sppcr_fit_means([1, 2], seen, wells)
    assert result.mu.shape == (3, 2)
    for i in range(3):
        for j in range(2):
            scalar = sppcr_fit_means([1, 2], seen[i, :, j : j + 1], wells[i])
            assert result.mu[i, j] == scalar.mu[0]
            np.testing.assert_equal(result.variance[i, j], scalar.variance[0])
    assert np.isnan(result.variance[1, 1]) and not result.interior[1, 1]


def test_zero_boundary_is_not_a_pseudo_detection():
    result = sppcr_fit_means([1, 2], [[0, 5], [0, 10]], [20, 20])
    assert result.mu[0] == result.mu_lower[0] == result.mu_upper[0] == 0
    assert result.log_likelihood[0] == 0
    assert np.isnan(result.variance[0]) and not result.interior[0]
    assert not np.any(result.adjusted)
    np.testing.assert_array_equal(result.seen, result.original_seen)


def test_partial_saturation_never_changes_original_counts():
    for first in [0, 10]:
        result = sppcr_fit_means([1, 2], [[first, 5], [20, 10]], 20, saturation="half")
        assert not np.any(result.adjusted)
        np.testing.assert_array_equal(result.seen, [[first, 5], [20, 10]])
        expected = decimal_mean([1, 2], [first, 20], [20, 20])
        assert result.mu[0] == pytest.approx(expected, rel=3e-14)


def test_full_saturation_requires_explicit_policy_and_discloses_adjusted_data():
    with pytest.raises(ValueError, match="no finite"):
        sppcr_fit_means([1, 2], [[20, 5], [20, 10]], 20)
    result = sppcr_fit_means([2, 1], [[20, 10], [20, 5]], 20, saturation="half")
    np.testing.assert_array_equal(result.seen, [[20, 10], [19.5, 5]])
    np.testing.assert_array_equal(result.original_seen, [[20, 10], [20, 5]])
    np.testing.assert_array_equal(result.adjusted, [True, False])
    np.testing.assert_array_equal(result.seen + result.unseen, np.full((2, 2), 20))
    scalar = sppcr_fit_means([1], [[20]], 20, saturation="half")
    assert scalar.mu[0] == pytest.approx(math.log(40), rel=3e-14)


def test_permutations_and_dna_unit_change_preserve_solution():
    d, s, n = np.array([1, 2, 3]), np.array([[2, 5], [4, 9], [8, 10]]), np.array([10, 20, 30])
    original = sppcr_fit_means(d, s, n)
    permuted = sppcr_fit_means(d[::-1], s[::-1, ::-1], n[::-1])
    np.testing.assert_allclose(original.mu, permuted.mu[::-1], rtol=3e-14)
    scaled = sppcr_fit_means(d * 100, s, n)
    np.testing.assert_allclose(original.mu / 100, scaled.mu, rtol=3e-14)
    np.testing.assert_allclose(original.variance / 10000, scaled.variance, rtol=5e-14)


def test_empty_batch_ownership_and_immutable_results():
    empty = sppcr_fit_means([1, 2], np.empty((0, 2, 3)), 10)
    assert empty.mu.shape == (0, 3)
    d, s, n = np.array([1.0, 2.0]), np.array([[2.0, 3.0], [4.0, 6.0]]), np.array([10.0, 10.0])
    result = sppcr_fit_means(d, s, n)
    d[:] = 9
    s[:] = 9
    n[:] = 9
    assert result.dna[0] == 1 and result.original_seen[0, 0] == 2 and result.wells[0] == 10
    with pytest.raises(ValueError):
        result.mu.setflags(write=True)
    with pytest.raises(ValueError):
        result.interior.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        result.mu = np.array([1])


@pytest.mark.parametrize(
    "dna,seen,wells",
    [
        ([], [[1]], 2),
        ([0], [[1]], 2),
        ([-1], [[1]], 2),
        ([math.inf], [[1]], 2),
        ([[1]], [[1]], 2),
        ([1], [1], 2),
        ([1], [[1], [1]], 2),
        ([1], np.empty((1, 0)), 2),
        ([1], [[1.5]], 2),
        ([1], [[-1]], 2),
        ([1], [[math.nan]], 2),
        ([1], [[3]], 2),
        ([1], [[0]], 0),
        ([1], [[1]], 2.5),
        ([1], [[1]], 2**53),
    ],
)
def test_invalid_input_rejected(dna, seen, wells):
    with pytest.raises(ValueError):
        sppcr_fit_means(dna, seen, wells)


@pytest.mark.parametrize("iterations", [0, -1, 1.5, True])
def test_invalid_iteration_limits(iterations):
    with pytest.raises(ValueError):
        sppcr_fit_means([1], [[1]], 2, max_iterations=iterations)


def test_unconverged_or_unrepresentable_results_fail_explicitly():
    with pytest.raises(ArithmeticError, match="max_iterations"):
        sppcr_fit_means([1], [[1]], 2, max_iterations=1)
    with pytest.raises(ArithmeticError, match="variance"):
        sppcr_fit_means([1e-200], [[1]], 2)
    with pytest.raises(ArithmeticError, match="half-count"):
        sppcr_fit_means([1], [[2**53 - 1]], 2**53 - 1, saturation="half")
    with pytest.raises(ValueError, match="saturation"):
        sppcr_fit_means([1], [[1]], 2, saturation="implicit")
