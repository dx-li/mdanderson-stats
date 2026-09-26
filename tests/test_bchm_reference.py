"""Independent native-formula, exact partition and quadrature BCHM references."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.bchm import _native_probability, bchm_borrow, bchm_cluster, bchm_fit
from mdanderson_stats.bchm_clustering import _assignment_probabilities


def _rows(name):
    with (Path(__file__).parent / "fixtures" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_weighted_assignments_against_native_r_integrated_likelihood():
    rows = _rows("bchm-assignment.csv")
    for case in dict.fromkeys(r["case"] for r in rows):
        records = [r for r in rows if r["case"] == case]
        r = records[0]
        probability = _assignment_probabilities(
            float(r["y"]) / float(r["n"]),
            float(r["n"]),
            np.array([float(v) for v in r["W"].split(";")]),
            np.array([float(v) for v in r["S"].split(";")]),
            float(r["mu"]),
            float(r["prior_variance"]),
            float(r["data_variance"]),
            float(r["concentration"]),
        )
        np.testing.assert_allclose(
            probability, [float(r["probability"]) for r in records], rtol=3e-11, atol=0
        )


def test_native_decision_probability_rounding_against_r():
    for row in _rows("bchm-rounding.csv"):
        assert _native_probability(float(row["probability"])) == float(row["rounded"])


def test_allocation_sweep_against_exact_three_subgroup_partition_law():
    result = bchm_cluster(
        [1, 2, 7],
        [10, 10, 10],
        mu=0.2,
        sigma02=2,
        sigmaD2=0.2,
        alpha=4,
        burn_in=300,
        iterations=2000,
        seed=158,
    )
    expected = np.array([float(r["similarity"]) for r in _rows("bchm-partition.csv")])
    np.testing.assert_allclose(result.result.raw_similarity, expected.reshape(3, 3), atol=0.06)


def test_similarity_weighted_borrowing_against_random_mean_quadrature():
    for row in _rows("bchm-borrowing-limit.csv"):
        target = int(row["group"]) - 1
        result = bchm_borrow(
            [0, 6],
            [8, 10],
            [[1, 0.2], [0.2, 1]],
            target=target,
            alpha1=3e10,
            beta1=1e10,
            tau2=0.7,
            draws=2000,
            warmup=500,
            seed=158 + target,
        )
        error = abs(result.posterior_mean - float(row["mean"]))
        assert error < max(0.008, 6 * result.summary.batch_mean_mcse[0])
        assert abs(result.probability - float(row["efficacy"])) < 0.035
        assert result.summary.split_rhat[0] < 1.08


def test_two_stage_fit_against_official_help_example():
    # BCHM_HelpFile.pdf, slides 8-9: reported JAGS estimates rounded to 3 places.
    # Tolerances allow Monte Carlo variation in both the native and Python fits.
    result = bchm_fit(
        [1, 2, 3, 7, 8],
        [15, 18, 10, 15, 20],
        iterations=500,
        burn_in=200,
        draws=2000,
        warmup=500,
        seed=1234,
    )
    np.testing.assert_allclose(
        result.posterior_mean, [0.092, 0.099, 0.361, 0.424, 0.394], atol=0.018, rtol=0
    )
    np.testing.assert_allclose(
        result.raw_probability, [0.001, 0.002, 0.517, 0.772, 0.681], atol=0.06, rtol=0
    )
    for summary in result.summaries:
        assert summary.split_rhat[0] < 1.08
