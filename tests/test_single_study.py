"""Study revision, reproducible inputs and model/prior/search reports."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import SingleStudySpecification


def test_json_replay_and_complete_numerical_report(tmp_path):
    study = SingleStudySpecification(
        [0, 1], [-5, 5], criterion="slope", max_doses=3, scan_points=4
    ).run()
    path = study.write_specification(tmp_path / "study.json")
    restored = SingleStudySpecification.from_json(path.read_text()).run()
    assert_allclose(restored.search.best.value, study.search.best.value)
    assert_allclose(restored.search.best.doses, study.search.best.doses)
    assert restored.search.stop_reason == study.search.stop_reason
    report = study.report(digits=17)
    for name in (
        "model",
        "form",
        "criterion",
        "comparison",
        "measure",
        "aggregation",
        "quantile",
        "total_subjects",
        "max_doses",
        "scan_points",
        "relative_improvement",
        "tolerance",
        "max_iterations",
    ):
        assert f"{name}\t" in report
    assert "Node\tWeight\tParameter 1\tParameter 2" in report
    assert "1\t1\t0\t1" in report
    assert "Stop reason\trelative_improvement" in report
    assert "Stage 2 design" in report and "\tfalse\n" in report
    assert "Selected design\n" in report
    destination = tmp_path / "report.tsv"
    destination.write_text("old")
    assert study.write_report(destination, digits=17) == destination
    assert destination.read_text() == report


def test_input_snapshot_and_revision_are_independent():
    parameters = np.array([0.0, 1.0])
    bounds = np.array([-5.0, 5.0])
    original = SingleStudySpecification(parameters, bounds, criterion="slope", max_doses=2).run()
    parameters[0] = 99
    bounds[0] = -999
    assert_allclose(original.specification.parameters, [[0, 1]])
    assert_allclose(original.specification.dose_bounds, [-5, 5])
    revised = original.revise(criterion="quantile", quantile=0.05)
    assert original.specification.criterion == "slope"
    assert revised.specification.criterion == "quantile"
    assert revised.search.best.value > original.search.best.value
    assert json.loads(original.specification.to_json())["parameters"] == [[0, 1]]
    with pytest.raises(ValueError):
        original.specification.parameters[0, 0] = 1


def test_two_sample_weighted_prior_report_and_replay():
    study = SingleStudySpecification(
        [[0, 0.8, 0.8], [0, 1.2, 1.2]],
        [-5, 5],
        prior_weights=[0.3, 0.7],
        comparison="slope",
        aggregation="harmonic",
        max_doses=2,
        scan_points=4,
    ).run()
    report = study.report(digits=17)
    assert "Parameter 3" in report and "comparison\tslope" in report
    assert "aggregation\tharmonic" in report
    restored = SingleStudySpecification.from_json(study.specification.to_json()).run()
    assert_allclose(restored.search.best.value, study.search.best.value)
    for old, new in zip(study.search.best.subjects, restored.search.best.subjects, strict=True):
        assert_allclose(old, new)


def test_invalid_revisions_and_serialization():
    study = SingleStudySpecification([0, 1], [-5, 5], max_doses=2, scan_points=3).run()
    with pytest.raises(TypeError):
        study.revise(unknown_option=1)
    with pytest.raises(ValueError):
        study.revise(prior_weights=[0.5])
    with pytest.raises(ValueError):
        SingleStudySpecification.from_json("[]")
    with pytest.raises(TypeError):
        SingleStudySpecification.from_json('{"parameters":[0,1],"dose_bounds":[-5,5],"extra":1}')
    with pytest.raises(ValueError):
        SingleStudySpecification([np.nan, 1], [-5, 5]).to_json()
    with pytest.raises(ValueError):
        study.report(digits=0)
