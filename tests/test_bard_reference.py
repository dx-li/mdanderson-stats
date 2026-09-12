"""Independent base-R checks for BARD's declared stage-two model."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.bard import bard_minimization, bard_select_obd


def _rows(name):
    with (Path(__file__).parent / "fixtures" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_dirichlet_utility_and_beta_tails_against_base_r():
    rows = _rows("bard-posterior.csv")
    for case in dict.fromkeys(r["case"] for r in rows):
        pair = [r for r in rows if r["case"] == case]
        result = bard_select_obd(
            [[int(r[f"n{k}"]) for k in range(1, 5)] for r in pair],
            prior=[[float(r[f"a{k}"]) for k in range(1, 5)] for r in pair],
            safety_weights=[float(r["weight"]) for r in pair],
        )
        for actual, key in (
            (result.mean_utility, "utility"),
            (result.raw_overdose, "overdose"),
            (result.adjusted_overdose, "adjusted"),
            (result.low_efficacy, "loweff"),
        ):
            np.testing.assert_allclose(
                actual, [float(r[key]) for r in pair], atol=2e-13, rtol=2e-13
            )


def test_all_profiles_in_official_enrollment_example():
    arms = [1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1]
    factors = [
        [1, 1, 2],
        [1, 2, 2],
        [2, 2, 1],
        [2, 1, 1],
        [3, 1, 1],
        [1, 1, 2],
        [2, 2, 2],
        [3, 2, 1],
        [1, 1, 2],
        [3, 2, 1],
        [2, 2, 2],
        [2, 1, 2],
        [1, 1, 1],
        [2, 2, 2],
        [1, 2, 1],
    ]
    for row in _rows("bard-minimization.csv"):
        result = bard_minimization(
            arms, factors, [int(row[f"Factor{k}"]) for k in range(1, 4)], seed=165
        )
        np.testing.assert_array_equal(result.scores, [int(row["score1"]), int(row["score2"])])
        np.testing.assert_allclose(result.probabilities[0], float(row["probability1"]), atol=1e-15)


def test_inclusive_admissibility_and_explicit_ties():
    kwargs = dict(prior=[0.5] * 4, safety_weights=[1, 1], toxicity_limit=0.5, efficacy_limit=0.5)
    counts = np.zeros((2, 4), dtype=int)
    result = bard_select_obd(counts, safety_cutoff=0.5, efficacy_cutoff=0.5, tie_arm=2, **kwargs)
    np.testing.assert_array_equal(result.admissible, [True, True])
    assert result.selected_arm == 2
    assert (
        bard_select_obd(
            counts, safety_cutoff=np.nextafter(0.5, 0), efficacy_cutoff=0.5, **kwargs
        ).selected_arm
        is None
    )
    assert (
        bard_select_obd(
            counts, safety_cutoff=0.5, efficacy_cutoff=np.nextafter(0.5, 0), **kwargs
        ).selected_arm
        is None
    )
    only_lower = bard_select_obd(
        [[2, 2, 0, 4], [0, 2, 4, 1]], prior=[0.25] * 4, safety_weights=[8, 7], safety_cutoff=0.5
    )
    np.testing.assert_array_equal(only_lower.admissible, [True, False])
    assert only_lower.selected_arm == 1
    only_upper = bard_select_obd(
        [[7, 1, 2, 0], [0, 4, 1, 5]],
        prior=[[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
        safety_weights=[1, 3],
        efficacy_cutoff=0.01,
    )
    np.testing.assert_array_equal(only_upper.admissible, [False, True])
    assert only_upper.selected_arm == 2


def test_large_prior_normalization_and_resource_guard():
    import pytest

    result = bard_select_obd(np.zeros((2, 4)), prior=[1e307] * 4, safety_weights=[1, 1])
    np.testing.assert_allclose(result.mean_utility, [45, 45], atol=1e-13, rtol=0)
    assert np.all(np.isfinite(result.raw_overdose))
    for field in vars(result).values():
        if isinstance(field, np.ndarray):
            assert not field.flags.writeable
    with pytest.raises(ValueError, match="100000"):
        bard_minimization([], np.broadcast_to(1, (100001, 1)), [1])
    with pytest.raises(ValueError, match="at least one observed"):
        bard_select_obd(
            np.zeros((2, 4)),
            prior=[0.5] * 4,
            safety_weights=[1, 1],
            method="noninferiority",
            safety_cutoff=0,
            efficacy_cutoff=0,
        )
    for tie, arm in ((0, 2), (1, 1)):
        allocation = bard_minimization([], np.empty((0, 1)), [1], tie_probability=tie, seed=165)
        assert allocation.assigned_arm == arm
