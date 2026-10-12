"""Original WFMM EB traces, plus independent boundary and stopping identities."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.wfmm_native_eb import wfmm_native_empirical_bayes

CASES = json.loads((Path(__file__).parent / "fixtures/wfmm-native-eb.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case"])
def test_native_eb_input_output(case):
    s = case["settings"]
    settings = {
        b: s[a]
        for a, b in [
            ("minp", "min_inclusion"),
            ("minT", "min_ratio"),
            ("bigT", "vague_ratio"),
            ("maxO", "max_odds"),
        ]
        if a in s
    }
    labels = np.repeat(np.arange(len(case["partition_sizes"])), case["partition_sizes"])
    for row in case["traces"]:
        result = wfmm_native_empirical_bayes(
            row["Z"],
            profile="legacy_3_1",
            coefficient_partition=labels,
            unsmoothed_partitions=int(s.get("nj_nosmooth", 0)),
            **settings,
        )
        np.testing.assert_allclose(
            result.group_inclusion_probability, row["pout"], atol=2e-15, rtol=2e-14
        )
        np.testing.assert_allclose(
            result.group_slab_to_sampling_variance, row["Tout"], atol=2e-12, rtol=2e-14
        )
        np.testing.assert_allclose(result.inclusion_probability, row["gam"], atol=2e-15, rtol=2e-14)


def test_zero_signal_signed_stop_is_not_convergence():
    result = wfmm_native_empirical_bayes(np.zeros((1, 4)), profile="legacy_3_1")
    # First update: odds=1/sqrt(2), gamma=1/(1+sqrt(2)), T floored to10.
    first = 1 / (1 + np.sqrt(2))
    second_odds = first / (1 - first) / np.sqrt(11)
    second = second_odds / (1 + second_odds)
    assert result.iterations == 2
    assert result.group_inclusion_probability[0, 0] == pytest.approx(second)
    assert result.group_slab_to_sampling_variance[0, 0] == 10
    assert result.maximum_change > 0.23
    assert result.maximum_signed_change == 0
    assert result.legacy_stopping_satisfied and not result.converged
    # Native gamma belongs to the preceding parameters, not the returned ones.
    final_odds = second / (1 - second) / np.sqrt(11)
    assert abs(result.inclusion_probability[0, 0] - final_odds / (1 + final_odds)) > 0.1


def test_all_partitions_share_stopping_and_unsmoothed_overrides():
    z = np.array([[0, 0, 20, 20.0]])
    result = wfmm_native_empirical_bayes(
        z, profile="legacy_3_1", coefficient_partition=[3, 3, 9, 9], unsmoothed_partitions=1
    )
    np.testing.assert_array_equal(result.partition_labels, [3, 9])
    assert result.group_inclusion_probability[0, 0] == 1 - 1e-14
    assert result.group_slab_to_sampling_variance[0, 0] == 1000
    assert result.group_slab_to_sampling_variance[0, 1] == pytest.approx(399)


def test_iteration_budget_and_immutable_outputs():
    z = np.zeros((2, 4))
    result = wfmm_native_empirical_bayes(z, profile="legacy_3_1", max_iterations=1)
    assert not result.legacy_stopping_satisfied and not result.converged
    for value in vars(result).values():
        if isinstance(value, np.ndarray):
            assert value.flags.owndata and not value.flags.writeable
            assert not np.shares_memory(value, z)


@pytest.mark.parametrize(
    "key,value",
    [
        ("profile", "stable"),
        ("min_inclusion", 0),
        ("min_inclusion", 0.5),
        ("min_inclusion", True),
        ("min_ratio", 0),
        ("vague_ratio", np.inf),
        ("max_odds", -1),
        ("max_odds", True),
        ("initial_inclusion_probability", 0),
        ("initial_inclusion_probability", 1),
        ("initial_slab_to_sampling_variance", 0),
        ("initial_slab_to_sampling_variance", True),
        ("max_iterations", 0),
        ("max_iterations", True),
        ("max_iterations", 1.5),
        ("unsmoothed_partitions", 2),
        ("unsmoothed_partitions", True),
        ("coefficient_partition", [0, 1]),
        ("standardized_estimates", [[np.nan]]),
        ("standardized_estimates", [[1e101]]),
        ("standardized_estimates", [1, 2]),
    ],
)
def test_reject_invalid_inputs(key, value):
    kwargs = dict(standardized_estimates=np.zeros((1, 4)), profile="legacy_3_1") | {key: value}
    with pytest.raises(ValueError):
        wfmm_native_empirical_bayes(**kwargs)
