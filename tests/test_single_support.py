"""Original support rejection preserves the previous design and reason."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import SingleOptimizedDesign, SingleStudySpecification, single_search_design
from mdanderson_stats.single_search import _original_support_stop

CASES = json.loads((Path(__file__).parent / "fixtures/single_support.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_original_native_support_predicates(case):
    design = SingleOptimizedDesign(np.array(case["doses"]), np.array(case["subjects"]), 1, 1, 0, 0)
    assert _original_support_stop(design) == case["stop_reason"]


def test_original_checks_first_group_only():
    design = SingleOptimizedDesign(
        (np.array([-1.0, 1.0]), np.array([0.0, 0.0])),
        (np.array([50.0, 50.0]), np.array([100.0, 0.0])),
        1,
        1,
        0,
        0,
    )
    assert _original_support_stop(design) is None
    swapped = SingleOptimizedDesign(design.doses[::-1], design.subjects[::-1], 1, 1, 0, 0)
    assert _original_support_stop(swapped) == "zero_subjects"


def test_minimum_two_dose_design_is_retained_on_overlap():
    result = single_search_design(
        [0, 1],
        [-0.0004, 0.0004],
        criterion="slope",
        support_stopping="original",
        max_doses=4,
        scan_points=4,
    )
    assert result.stop_reason == "dose_overlap"
    assert len(result.steps) == 1 and result.steps[0].accepted
    assert result.best is result.steps[0].design
    assert_allclose(result.best.doses, [-0.0004, 0.0004])
    assert result.scan_evaluations == 6


def test_unused_added_dose_reverts_to_previous_design_and_replays():
    study = SingleStudySpecification(
        [0, 1],
        [-5, 5],
        criterion="slope",
        support_stopping="original",
        relative_improvement=0,
        max_doses=4,
        scan_points=4,
    ).run()
    result = study.search
    assert result.stop_reason == "zero_subjects"
    assert result.best is result.steps[0].design
    assert len(result.steps) == 2 and not result.steps[1].accepted
    assert result.steps[1].relative_improvement >= 0
    assert np.min(result.steps[1].design.subjects) < 1e-5
    assert "support_stopping\toriginal" in study.report()
    assert "Stop reason\tzero_subjects" in study.report()
    replay = SingleStudySpecification.from_json(study.specification.to_json()).run()
    assert replay.search.stop_reason == "zero_subjects"
    assert_allclose(replay.search.best.value, result.best.value)
    # The original checks criterion improvement before checking support.
    revised = study.revise(relative_improvement=0.01)
    assert revised.search.stop_reason == "relative_improvement"


def test_invalid_support_policy():
    with pytest.raises(ValueError, match="support_stopping"):
        single_search_design([0, 1], [-5, 5], support_stopping="unknown")
