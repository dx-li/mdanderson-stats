from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import (
    catbub_analysis,
    catbub_binary_compare,
    catbub_compare,
    catbub_design,
    catbub_operating_characteristics,
    catbub_simulate_counts,
    catbub_thresholds,
)


def test_posterior_comparison_against_original_r_and_binary_mc():
    cases = [
        ([[25, 15, 10], [30, 15, 5]], [100, 50, 0], 0.913734016480359679),
        ([[0, 0, 0], [0, 0, 0]], [100, 20, 0], 0.5),
        ([[3, 7], [8, 2]], [0, 100], 0.01109582194542293),
        ([[0, 1, 0, 0], [0, 0, 1, 0]], [0, 35, 70, 100], 0.77570102422467802),
        ([[1000, 300, 50], [900, 400, 50]], [100, 50, 0], 0.00020735499268089599),
    ]
    reference = np.loadtxt(
        Path(__file__).parent / "fixtures/catbub-reference.csv", delimiter=",", skiprows=1
    )
    for index, (counts, utility, expected) in enumerate(cases):
        result = catbub_compare(counts, utility)
        assert_allclose(result.probability, [expected, 1 - expected], atol=3e-8, rtol=0)
        assert_allclose(result.probability, reference[index], atol=2e-9, rtol=0)
        assert np.max(result.error) < 1e-8
        assert_allclose(
            catbub_compare(counts, np.asarray(utility) * 1e300).probability,
            result.probability,
            atol=1e-10,
        )
    comparator = catbub_binary_compare(cases[0][0])
    assert_allclose(comparator.treatment_greater, 0.84254744240164514, atol=1e-8)
    binary = catbub_compare([[3, 7], [8, 2]], [0, 100])
    mc = catbub_compare(
        [[3, 7], [8, 2]], [0, 100], method="mc", draws=100_000, rng=np.random.default_rng(9701)
    )
    assert np.all(np.abs(mc.probability - binary.probability) < 5 * mc.error)
    assert_allclose(catbub_compare([[0, 1], [2, 0]], [-1e308, 1e308]).scaled_mean, [0.75, 1 / 6])
    assert_allclose(catbub_compare([[1, 2], [3, 4]], [10, 10]).probability, [0, 0])
    with pytest.raises(ValueError, match="strictly positive"):
        catbub_compare([[1, 2], [3, 4]], [0, 1], prior_probability=[[0, 1], [1, 0]])


def test_boundary_spending_and_first_stopping_not_later_reversal():
    p = np.array(
        [
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.7, 0.3], [0.85, 0.15]],
            [[0.6, 0.4], [0.6, 0.4]],
            [[0.55, 0.45], [0.55, 0.45]],
        ]
    )
    cuts = catbub_thresholds(p, [0.5, 1], alpha=0.2, rho=1)
    # R type-7 quantile at .9 is .84; among survivors at look 2,
    # quantile (.8/.9) interpolates .6 and .85 to .794444...
    assert_allclose(cuts, [0.84, 0.7945])
    oc = catbub_operating_characteristics(p, [10, 20], [0.8, 0.8])
    assert_allclose(oc.stopping_probability, [[0.25, 0], [0.25, 0]])
    assert_allclose(oc.superiority_probability, [0.5, 0])
    assert_allclose(oc.mean_sample_size, [17.5, 17.5])
    assert oc.no_selection_probability == 0.5
    assert_allclose(oc.sample_size_mcse, [2.5, 2.5])


def test_sequential_simulation_design_and_actual_data_analysis():
    rng = np.random.default_rng(9702)
    counts = catbub_simulate_counts(
        [[5, 10], [4, 12]], [[0.3, 0.7], [0.8, 0.2]], trials=2000, rng=rng
    )
    assert_allclose(counts.sum(axis=-1), np.broadcast_to([[5, 4], [10, 12]], (2000, 2, 2)))
    assert np.all(np.diff(counts, axis=1) >= 0)
    assert_allclose(counts[:, -1].mean(axis=0), [[3, 7], [9.6, 2.4]], atol=0.12)
    result = catbub_design(
        [[[0.4, 0.6], [0.7, 0.3]], [[0.4, 0.6], [0.4, 0.6]]],
        [0.5, 1],
        [100, 0],
        null_trials=200,
        alternative_trials=200,
        epsilon=0.1,
        max_iterations=10,
        rng=rng,
    )
    assert 0.7 < result.alternatives[0].superiority_probability[0] < 0.9
    assert len(result.alternatives) == 2
    assert result.thresholds.shape == (2,)
    assert result.null.mean_sample_size[0] <= result.sample_sizes[-1]
    analysis = catbub_analysis([[20, 5], [8, 17]], [100, 0], [10], 50, null_trials=100, rng=rng)
    assert_allclose(
        [analysis.incremental_alpha, analysis.cumulative_alpha],
        [0.05 * (0.5**3 - 0.2**3), 0.05 * 0.5**3],
    )
    assert analysis.decision in ("A>B", "Continue Enrollment")
    first = catbub_analysis([[5, 0], [0, 5]], [100, 0], [], 5, null_trials=100, rng=rng)
    assert first.decision == "A>B"
