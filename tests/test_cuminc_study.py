"""Combined competing-risk workflows and jointly applied missing-data policy."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import cuminc

GRAY = json.loads((Path(__file__).parent / "fixtures/gray.json").read_text())["cases"]
CURVES = json.loads((Path(__file__).parent / "fixtures/cuminc.json").read_text())["cases"]


@pytest.mark.parametrize("case", GRAY[:48])
def test_combined_test_matches_native_oracle_with_string_labels(case):
    labels = np.array(["censored", "target", "other"])
    groups = np.array([f"group {x}" for x in case["group"]])
    result = cuminc(
        case["time"],
        labels[case["event"]],
        groups,
        strata=case["strata"],
        rho=case["rho"],
        censor="censored",
    )
    assert_allclose(result.tests["target"].score, case["score"], rtol=1e-10, atol=2e-13)
    assert_allclose(result.tests["target"].covariance, case["covariance"], rtol=1e-10, atol=2e-13)
    for g in result.groups:
        assert result.group_counts[g] == np.count_nonzero(groups == g)
        for c in result.causes:
            assert result.curves[c, g].n_events == np.count_nonzero(
                (groups == g) & (labels[case["event"]] == c)
            )


@pytest.mark.parametrize("case", CURVES)
def test_single_group_target_curve_matches_native_oracle(case):
    result = cuminc(case["time"], case["event"])
    assert result.tests == {}
    if 1 in result.causes:
        curve = result.curves[1, 1]
        assert_allclose(
            np.column_stack([curve.time, curve.estimate, curve.variance]),
            case["corners"],
            rtol=2e-11,
            atol=2e-15,
        )
    else:
        assert all(c != 1 for c, _ in result.curves)


def test_joint_complete_case_deletion_records_count_and_retains_labels():
    result = cuminc(
        [0, 1, None, 3, 4, 5],
        ["a", "b", "a", None, "a", "c"],
        ["x", "y", "x", "x", None, "x"],
        strata=[1, 1, 1, 1, 1, np.nan],
        censor="c",
        missing="drop",
    )
    assert result.n_dropped == 4
    assert result.group_counts == {"x": 1, "y": 1}
    assert result.causes == ("a", "b")
    assert result.curves["a", "x"].n_events == 1
    assert result.curves["b", "x"].n_events == 0


def test_summaries_select_exact_labels_preserve_corners_and_write_reports(tmp_path):
    result = cuminc(
        [0, 1, 2, 3], ["a", "a", "b", "c"], ["x", "x", "y", "y"], censor="c", confidence=0.8
    )
    selected = result.summaries([0, 1, 4], causes="a", groups="x")
    assert list(selected) == [("a", "x")]
    assert_allclose(selected["a", "x"].rows[:, 1], [0.5, 1, 1])
    assert "80% CI" in result.report()
    assert result.summaries()["a", "x"].rows[0, 1] == 0
    destination = tmp_path / "study.txt"
    result.write_report(destination, times=[0, 1], digits=17)
    assert destination.read_text() == result.report(times=[0, 1], digits=17)
    assert "Group counts" in destination.read_text()
    assert "Gray tests" in destination.read_text()
    with pytest.raises(TypeError):
        result.curves["new", "new"] = result.curves["a", "x"]
    for kwargs in [{"causes": "absent"}, {"groups": ["x", "x"]}, {"groups": [["x"]]}]:
        with pytest.raises(ValueError):
            result.summaries(**kwargs)


def test_strata_change_tests_but_not_group_curves():
    case = GRAY[4]
    a = cuminc(case["time"], case["event"], case["group"])
    b = cuminc(case["time"], case["event"], case["group"], strata=case["strata"])
    for key in a.curves:
        assert_array_equal(a.curves[key].estimate, b.curves[key].estimate)
    assert not np.allclose(a.tests[1].score, b.tests[1].score)


def test_all_censored_has_group_counts_and_no_invented_events_or_tests():
    result = cuminc([1, 2], [0, 0], ["a", "b"])
    assert result.causes == ()
    assert result.curves == result.tests == result.summaries() == {}
    assert result.group_counts == {"a": 1, "b": 1}
    assert "a\t1" in result.report()


@pytest.mark.parametrize(
    "overrides",
    [
        {"time": []},
        {"time": [-1, 1]},
        {"cause": [1]},
        {"cause": [1, "x"]},
        {"group": ["a"]},
        {"strata": [1]},
        {"confidence": 1},
        {"rho": np.nan},
        {"missing": "ignore"},
        {"time": [None, 1]},
        {"cause": [None, 1]},
        {"time": [None, None], "missing": "drop"},
        {"time": [np.inf, 1], "missing": "drop"},
        {"censor": None},
        {"censor": True},
    ],
)
def test_invalid_inputs(overrides):
    arguments = dict(time=[0, 1], cause=[1, 2])
    arguments.update(overrides)
    with pytest.raises(ValueError):
        cuminc(**arguments)


def test_report_retains_analysis_settings_even_without_group_tests():
    result = cuminc([1, 2], ["a", "c"], censor="c", rho=0.5, strata=["x", "x"])
    assert result.censor == "c"
    assert result.rho == 0.5
    assert "Censor label: c; rho: 0.5; confidence: 0.95" in result.report()
    assert "Strata: ('x',)" in result.report()
