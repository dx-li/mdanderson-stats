import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.survan_tests import survan_group_test


def test_original_fortran_risk_scores_and_covariance():
    cases = json.loads((Path(__file__).parent / "fixtures/survan-native.json").read_text())
    for case in cases:
        data = np.asarray(case["records"])
        native = case["native"]
        for method in ["logrank", "gehan_breslow"]:
            result = survan_group_test(
                data[:, 0], data[:, 1], data[:, 2].astype(int), strata=case["strata"], method=method
            )
            for j, ref in enumerate(native):
                score = np.subtract(ref[0], ref[1]) if method == "logrank" else ref[3]
                cov = ref[2] if method == "logrank" else ref[4]
                assert_allclose(result.stratum_scores[j], score, rtol=2e-5, atol=3e-5)
                assert_allclose(
                    result.stratum_covariances[j][np.tril_indices(len(result.groups))],
                    cov,
                    rtol=2e-6,
                    atol=3e-6,
                )
            if case["name"] == "manual":
                target = 3.123 if method == "logrank" else 2.651
                assert abs(result.statistic - target) < 0.0005


def test_ties_stratum_addition_and_label_invariance():
    # At time 1 all four remain at risk, including its tied censor.
    r = survan_group_test([1, 2, 1, 2], [1, 1, 0, 1], ["a", "a", "b", "b"])
    assert_allclose(r.score, [0.5, -0.5])
    assert_allclose(r.covariance, [[0.25, -0.25], [-0.25, 0.25]])
    assert_allclose(r.statistic, 1)
    both = survan_group_test(
        [1, 2, 1, 2] * 2, [1, 1, 0, 1] * 2, ["a", "a", "b", "b"] * 2, strata=[0] * 4 + [1] * 4
    )
    assert_allclose(both.statistic, 2 * r.statistic)
    reversed_labels = survan_group_test([1, 2, 1, 2], [1, 1, 0, 1], ["b", "b", "a", "a"])
    assert_allclose(reversed_labels.statistic, r.statistic)


def test_uninformative_and_disconnected_groups():
    with pytest.raises(ValueError, match="no information"):
        survan_group_test([1, 2], [0, 0], [0, 1])
    with pytest.raises(ValueError, match="zero or one"):
        survan_group_test([1, 2], [2, 1], [0, 1])
    # Group 2 exists only in its own stratum and adds no comparison dimension.
    r = survan_group_test(
        [1, 2, 1, 2, 1, 2], [1, 1, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2], strata=[0, 0, 0, 0, 1, 1]
    )
    assert r.degrees_of_freedom == 1
    assert_allclose(r.statistic, 1)
