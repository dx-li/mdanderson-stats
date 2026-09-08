"""Native Gray scores/covariance and independent test-distribution contracts."""

import json
from math import erfc, exp, sqrt
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import gray_test

CASES = json.loads((Path(__file__).parent / "fixtures/gray.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_original_crstm(case):
    result = gray_test(
        case["time"], case["event"], case["group"], strata=case["strata"], rho=case["rho"]
    )
    assert_allclose(result.score, case["score"], rtol=1e-10, atol=2e-13)
    assert_allclose(result.covariance, case["covariance"], rtol=1e-10, atol=2e-13)


def test_single_event_time_hypergeometric_variance_and_chi_square_tail():
    # Four of eight subjects are in group a; both failures are in a.
    # Score=2-E[X]=1, Var[X]=2*(4/8)*(4/8)*(6/7)=3/7.
    result = gray_test([1, 1, 2, 2, 2, 2, 2, 2], [1, 1, 0, 0, 0, 0, 0, 0], ["a"] * 4 + ["b"] * 4)
    assert_allclose(result.score, [1])
    assert_allclose(result.covariance, [[3 / 7]])
    assert_allclose(result.statistic, 7 / 3)
    assert_allclose(result.pvalue, erfc(sqrt(7 / 6)))
    # The original wrapper's normal tail of the *squared* statistic is wrong.
    assert abs(result.pvalue - erfc((7 / 3) / sqrt(2))) > 0.1


def test_three_group_chi_square_has_two_degrees_of_freedom():
    result = gray_test([1, 1, 2, 2, 2, 2], [1, 1, 0, 0, 0, 0], ["a", "b", "a", "b", "c", "c"])
    assert result.degrees_freedom == 2
    assert result.rank == 2
    assert_allclose(result.pvalue, exp(-result.statistic / 2))


def test_stratum_scores_and_covariances_add_before_testing():
    case = CASES[4]
    time, event, group, strata = [np.array(case[k]) for k in ["time", "event", "group", "strata"]]
    combined = gray_test(time, event, group, strata=strata, rho=case["rho"])
    parts = [
        gray_test(time[strata == s], event[strata == s], group[strata == s], rho=case["rho"])
        for s in np.unique(strata)
    ]
    assert_allclose(combined.score, sum(r.score for r in parts))
    assert_allclose(combined.covariance, sum(r.covariance for r in parts))
    assert_allclose(
        combined.statistic, combined.score @ np.linalg.solve(combined.covariance, combined.score)
    )


def test_reference_group_change_permutation_and_time_scaling_preserve_test():
    case = CASES[36]
    time, event, group, strata = [np.array(case[k]) for k in ["time", "event", "group", "strata"]]
    first = gray_test(time, event, group, strata=strata, rho=0.5)
    # Reverse labels, moving the reference group; reverse row order too.
    second = gray_test(
        time[::-1] * 3,
        event[::-1] + 7,
        -group[::-1],
        strata=strata[::-1] * 11,
        rho=0.5,
        event_of_interest=8,
        censor=7,
    )
    assert_allclose(first.statistic, second.statistic, rtol=1e-12)
    assert_allclose(first.pvalue, second.pvalue, rtol=1e-12)
    with pytest.raises(ValueError):
        first.score[0] = 999
    with pytest.raises(ValueError):
        first.covariance[0, 0] = 999


@pytest.mark.parametrize("event", [[0] * 6, [2] * 6])
def test_no_target_events_give_unavailable_test(event):
    result = gray_test(range(6), event, [1, 2, 3, 1, 2, 3])
    assert result.statistic is None and result.pvalue is None
    assert result.rank == 0
    assert_allclose(result.score, 0)
    assert_allclose(result.covariance, 0)


def test_groups_separated_by_stratum_have_no_comparison_information():
    result = gray_test(
        [1, 2, 3, 4], [1, 0, 1, 0], ["a", "a", "b", "b"], strata=["x", "x", "y", "y"]
    )
    assert result.statistic is None and result.pvalue is None
    assert result.rank == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"time": []},
        {"time": [-1, 2, 3, 4]},
        {"time": [1, np.nan, 3, 4]},
        {"event": [0, 1, 1]},
        {"event": [0, 1, 1.5, 2]},
        {"group": [1, 1, 1, 1]},
        {"group": [1, 2]},
        {"group": [1, None, 1, 2]},
        {"group": [1, "a", 1, 2]},
        {"group": [1, np.inf, 1, 2]},
        {"strata": [1, 2]},
        {"rho": np.inf},
        {"rho": [0]},
        {"event_of_interest": 0},
        {"event_of_interest": 1.5},
        {"censor": True},
    ],
)
def test_invalid_input(overrides):
    arguments = dict(time=[1, 2, 3, 4], event=[0, 1, 1, 2], group=[1, 1, 2, 2])
    arguments.update(overrides)
    with pytest.raises(ValueError):
        gray_test(**arguments)


def test_partially_estimable_contrast_does_not_silently_drop_a_group():
    result = gray_test(
        [1, 2, 3, 4, 1, 2],
        [1, 1, 0, 0, 1, 0],
        [1, 2, 1, 2, 3, 3],
        strata=[0, 0, 0, 0, 1, 1],
    )
    assert result.rank == 1
    assert result.degrees_freedom == 2
    assert result.statistic is None and result.pvalue is None
