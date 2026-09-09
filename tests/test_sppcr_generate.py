"""Sampling contracts, independent moments and end-to-end simulation checks."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext

import numpy as np
import pytest

from mdanderson_stats import (
    sppcr_detection_probabilities,
    sppcr_fit_means,
    sppcr_generate,
    sppcr_observed_probabilities,
)


@pytest.mark.parametrize("dna", [0.01, 1, 10, 1e-100, 1e100])
@pytest.mark.parametrize("mu", [0, 0.3, 3, 1e-100, 1e100])
def test_detection_probability_against_decimal(dna, mu):
    with localcontext() as ctx:
        ctx.prec = 230
        expected = float(1 - (-Decimal(str(dna)) * Decimal(str(mu))).exp())
    actual = sppcr_detection_probabilities([dna], [mu])[0, 0]
    assert actual == pytest.approx(expected, rel=3e-15, abs=0)


def test_detection_extremes_and_batch_shapes():
    np.testing.assert_array_equal(sppcr_detection_probabilities([1e300], [0, 1e300]), [[0, 1]])
    assert sppcr_detection_probabilities([1e-200], [1e-100])[0, 0] == 1e-300
    assert sppcr_detection_probabilities([1e-200], [1e-200])[0, 0] == 0
    m = np.arange(12).reshape(2, 2, 3) / 10
    p = sppcr_detection_probabilities([1, 2], m)
    assert p.shape == (2, 2, 2, 3)
    for i in range(2):
        for j in range(2):
            np.testing.assert_array_equal(p[i, j], sppcr_detection_probabilities([1, 2], m[i, j]))


def test_observed_model_uses_raw_counts_including_endpoints():
    seen = [[[0, 5], [20, 10]], [[2, 10], [30, 15]]]
    p = sppcr_observed_probabilities(seen, [[10, 20], [10, 30]])
    np.testing.assert_array_equal(p, [[[0, 0.5], [1, 0.5]], [[0.2, 1], [1, 0.5]]])
    # A different model is obtained by fitting the means; observed fractions
    # intentionally do not enforce the cross-DNA Poisson relationship.
    fit = sppcr_fit_means([1, 2], seen, [[10, 20], [10, 30]])
    assert not np.allclose(p, sppcr_detection_probabilities([1, 2], fit.mu))


def test_reproducibility_stream_continuation_and_replicate_order():
    p = np.array([[[0.1, 0.4], [0.7, 0.9]], [[0.2, 0.5], [0.8, 0.6]]])
    a, b = np.random.default_rng(857), np.random.default_rng(857)
    batch = sppcr_generate(p, [40, 100], rng=a, replicates=12)
    parts = [sppcr_generate(p, [40, 100], rng=b, replicates=3).seen for _ in range(4)]
    np.testing.assert_array_equal(batch.seen, np.concatenate(parts))
    assert a.bit_generator.state == b.bit_generator.state
    np.testing.assert_array_equal(
        batch.seen + batch.unseen, np.broadcast_to(np.array([40, 100])[:, None], batch.seen.shape)
    )
    assert batch.seen.shape == (12, 2, 2, 2)


def test_binomial_distribution_and_independent_cells():
    p = np.array([[0.03, 0.3, 0.8], [0.1, 0.6, 0.97]])
    n = np.array([10, 80])[:, None]
    draws = sppcr_generate(p, [10, 80], rng=np.random.default_rng(912), replicates=100_000).seen
    mean, variance = n * p, n * p * (1 - p)
    # Six standard errors; variance bound uses exact binomial fourth moments.
    np.testing.assert_array_less(abs(draws.mean(axis=0) - mean), 6 * np.sqrt(variance / len(draws)))
    fourth = 3 * variance**2 + variance * (1 - 6 * p * (1 - p))
    np.testing.assert_array_less(
        abs(draws.var(axis=0) - variance), 6 * np.sqrt((fourth - variance**2) / len(draws))
    )
    standardized = ((draws - mean) / np.sqrt(variance)).reshape(len(draws), -1)
    covariance = standardized.T @ standardized / len(draws)
    off_diagonal = covariance[~np.eye(6, dtype=bool)]
    assert np.max(abs(off_diagonal)) < 6 / np.sqrt(len(draws))


def test_simulated_data_fits_truth_with_many_wells():
    truth = np.array([0.2, 0.5, 0.8])
    p = sppcr_detection_probabilities([0.5, 1, 2], truth)
    samples = sppcr_generate(p, 100_000, rng=np.random.default_rng(942), replicates=8)
    fit = sppcr_fit_means([0.5, 1, 2], samples.seen, samples.wells)
    np.testing.assert_array_less(abs(fit.mu - truth), 6 * np.sqrt(fit.variance))


def test_endpoints_large_counts_and_owned_immutable_results():
    p = np.array([[0.0, 1.0]])
    n = np.array([2**53 - 1])
    sample = sppcr_generate(p, n, rng=np.random.default_rng(1), replicates=2)
    np.testing.assert_array_equal(sample.seen, [[[0, 2**53 - 1]], [[0, 2**53 - 1]]])
    p[:] = 0.5
    n[:] = 1
    assert sample.probability[0, 1] == 1
    assert sample.wells[0] == 2**53 - 1
    for a in [sample.probability, sample.wells, sample.seen, sample.unseen]:
        with pytest.raises(ValueError):
            a.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        sample.wells = n


@pytest.mark.parametrize("shape,replicates", [((2, 3), 0), ((0, 2, 3), 4)])
def test_empty_samples_consume_no_randomness(shape, replicates):
    rng = np.random.default_rng(7)
    before = deepcopy(rng.bit_generator.state)
    sample = sppcr_generate(np.full(shape, 0.5), 10, rng=rng, replicates=replicates)
    assert sample.seen.shape == (replicates,) + shape
    assert rng.bit_generator.state == before


@pytest.mark.parametrize(
    "p,n,replicates",
    [
        ([0.5], 10, 1),
        (np.empty((0, 2)), 10, 1),
        ([[float("nan")]], 10, 1),
        ([[-0.1]], 10, 1),
        ([[1.1]], 10, 1),
        ([[0.5]], 0, 1),
        ([[0.5]], 1.5, 1),
        ([[0.5]], 2**53, 1),
        ([[0.5]], [1, 2], 1),
        ([[0.5]], 10, -1),
        ([[0.5]], 10, True),
        ([[0.5]], 10, 1.5),
    ],
)
def test_invalid_generation_preserves_state(p, n, replicates):
    rng = np.random.default_rng(7)
    before = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError):
        sppcr_generate(p, n, rng=rng, replicates=replicates)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("seen,n", [([[3]], 2), ([[1.5]], 2), ([[-1]], 2), ([[0]], 0)])
def test_invalid_observed_counts(seen, n):
    with pytest.raises(ValueError):
        sppcr_observed_probabilities(seen, n)


@pytest.mark.parametrize(
    "dna,mu",
    [([], [1]), ([0], [1]), ([1], []), ([1], [-1]), ([1], [float("inf")]), ([[1]], [1]), ([1], 1)],
)
def test_invalid_model(dna, mu):
    with pytest.raises(ValueError):
        sppcr_detection_probabilities(dna, mu)


def test_explicit_generator_required():
    with pytest.raises(TypeError, match="Generator"):
        sppcr_generate([[0.5]], 10, rng=7)
