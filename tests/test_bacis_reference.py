"""BaCIS mathematical references evaluated independently in base R."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.bacis import bacis_classify, bacis_fit


def _rows(name):
    with (Path(__file__).parent / "fixtures" / name).open(newline="") as f:
        return list(csv.DictReader(f))


def test_classification_evidence_and_both_adaptive_cutoffs_against_r():
    rows = _rows("bacis-classification.csv")
    for case in dict.fromkeys(row["case"] for row in rows):
        group = [r for r in rows if r["case"] == case]
        for weighting in ("subgroup", "patient"):
            result = bacis_classify(
                [int(r["y"]) for r in group],
                [int(r["n"]) for r in group],
                phi_low=float(group[0]["phi_low"]),
                phi_high=float(group[0]["phi_high"]),
                classification_precision=float(group[0]["precision"]),
                adaptive_weighting=weighting,
            )
            np.testing.assert_allclose(
                result.log_evidence,
                [[float(r["log_low"]), float(r["log_high"])] for r in group],
                rtol=1e-10,
                atol=1e-10,
            )
            for field, column in (("high_probability", "p_high"), ("low_probability", "p_low")):
                np.testing.assert_allclose(
                    getattr(result, field),
                    [float(r[column]) for r in group],
                    rtol=2e-9,
                    atol=0,
                )
            cutoff = float(group[0][f"cutoff_{weighting}"])
            np.testing.assert_allclose(result.cutoff, cutoff, atol=2e-15, rtol=0)
            np.testing.assert_array_equal(
                result.cluster, [2 if float(r["p_high"]) > cutoff else 1 for r in group]
            )


def test_native_singleton_posterior_summaries_are_exact():
    for row in _rows("bacis-singleton.csv"):
        result = bacis_fit([int(row["y"])], [int(row["n"])], draws=8, warmup=0, seed=153)
        for field, column in (
            ("posterior_mean", "mean"),
            ("posterior_sd", "sd"),
            ("efficacy_probability", "efficacy"),
            ("high_response_probability", "high"),
        ):
            np.testing.assert_allclose(getattr(result, field)[0], float(row[column]), rtol=2e-13)
        assert result.cluster_fits == (None, None)


def test_borrowing_uses_each_clusters_center_in_fixed_hyperprior_limit():
    rows = _rows("bacis-borrowing-limit.csv")
    result = bacis_fit(
        [int(r["y"]) for r in rows],
        [int(r["n"]) for r in rows],
        mean_precision=1e10,
        precision_shape=1e10,
        precision_rate=1e10,
        draws=1500,
        warmup=300,
        chains=2,
        seed=153,
    )
    np.testing.assert_array_equal(result.classification.cluster, [1, 1, 2, 2])
    expected = np.array([float(r["mean"]) for r in rows])
    assert np.all(
        np.abs(result.posterior_mean - expected)
        < np.maximum(0.006, 6 * result.summary.batch_mean_mcse)
    )
    np.testing.assert_allclose(
        result.efficacy_probability, [float(r["efficacy"]) for r in rows], atol=0.035, rtol=0
    )
    assert np.all(result.summary.split_rhat < 1.08)
