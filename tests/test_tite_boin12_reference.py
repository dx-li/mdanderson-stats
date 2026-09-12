"""Independent base-R reference for the declared TITE-BOIN12 AL model."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.boin12 import BOIN12Design
from mdanderson_stats.tite_boin12 import tite_boin12_posterior


def _rows(name):
    with (Path(__file__).parent / "fixtures" / name).open(newline="") as f:
        return list(csv.DictReader(f))


def test_all_ascertainment_patterns_against_base_r():
    data = _rows("tite-boin12-patients.csv")
    reference = _rows("tite-boin12-al.csv")
    design = BOIN12Design(0.35, 0.25, utilities=(100, 30, 65, 0))
    args = [[float(r[key]) for r in data] for key in ("dose", "tox", "eff", "tf", "ef")]
    result = tite_boin12_posterior(design, *args, toxicity_window=1, efficacy_window=2, n_doses=3)
    for actual, keys in (
        (result.ESS[:2], ("ess_t", "ess_e")),
        (result.MLE[:2], ("pi_t", "pi_e")),
        (result.expected_joint_counts[:2], ("j01", "j00", "j11", "j10")),
    ):
        np.testing.assert_allclose(
            actual, [[float(r[k]) for k in keys] for r in reference], atol=2e-14, rtol=2e-14
        )
    np.testing.assert_allclose(
        result.patient_conditional_means,
        [[float(r[k]) for k in ("expected_t", "expected_e")] for r in data],
        atol=2e-14,
        rtol=2e-14,
    )
    for actual, key in (
        (result.posterior.utility_events, "utility_events"),
        (result.posterior.utility_mean, "utility_mean"),
        (result.posterior.utility_probability, "utility_probability"),
        (result.posterior.toxicity_overdose_probability, "overdose"),
        (result.posterior.efficacy_futility_probability, "futility"),
    ):
        np.testing.assert_allclose(
            actual[:2], [float(r[key]) for r in reference], atol=2e-13, rtol=2e-13
        )
    np.testing.assert_allclose(result.expected_joint_counts.sum(axis=1), result.N)
    assert np.isnan(result.MLE[2]).all()
    prior = design.posterior([0], [0], [0], efficacy_without_toxicity=[0])
    assert result.posterior.utility_probability[2] == prior.utility_probability[0]


def test_complete_joint_data_reduce_to_boin12():
    design = BOIN12Design(0.35, 0.25, utilities=(100, 30, 65, 0))
    # Every joint cell appears, with unequal counts to detect a swapped cell order.
    t = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    e = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0])
    result = tite_boin12_posterior(
        design,
        np.ones(10),
        t,
        e,
        np.ones(10),
        np.ones(10) * 2,
        toxicity_window=1,
        efficacy_window=2,
        n_doses=2,
    )
    expected = design.posterior([10, 0], [5, 0], [4, 0], efficacy_without_toxicity=[3, 0])
    for key in vars(expected):
        np.testing.assert_allclose(
            getattr(result.posterior, key), getattr(expected, key), atol=2e-14, rtol=2e-14
        )
    np.testing.assert_array_equal(result.expected_joint_counts[0], [3, 2, 1, 4])


def test_tiny_pending_complement_is_retained():
    # ESS rounds to 1, but the effective non-event contribution is nonzero.
    result = tite_boin12_posterior(
        BOIN12Design(0.35, 0.25, utilities=(100, 30, 65, 0)),
        [1, 1],
        [1, -1],
        [1, 1],
        [0, 1e-20],
        [0, 0],
        toxicity_window=1,
        efficacy_window=1,
        n_doses=1,
    )
    np.testing.assert_allclose(result.expected_joint_counts[0, 0], 1e-20, atol=0, rtol=1e-12)
    assert np.all(np.isfinite(result.posterior.utility_probability))
