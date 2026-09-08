"""SEQBIN reports, display compaction, revision and reproducible input snapshots."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import SeqBinDesign, SeqBinStudySpecification, seqbin_boundary_table

FIXTURE = json.loads((Path(__file__).parent / "fixtures/seqbin_table.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_compact_table_without_corrupting_null_properties(case):
    study = SeqBinStudySpecification(
        50,
        prior=[0.5, 0.5],
        null_probability=0.2,
        tail_probability=0.005,
        alternative=case["alternative"],
        looks=case["looks"],
        legacy_bounds=True,
    ).run()
    table = study.boundary_table(compact=True)
    assert_allclose(table.rows, case["rows"], rtol=2e-13, atol=1e-15)
    assert_allclose(
        study.properties.rejection_probability[0], case["correct_significance"], atol=1e-15
    )
    assert not np.isclose(case["correct_significance"], case["mutated_significance"])
    before = study.properties.rejection_probability.copy()
    study.report(compact=True)
    assert_array_equal(study.properties.rejection_probability, before)
    assert_allclose(
        study.design.operating_characteristics(0.2).rejection_probability,
        case["correct_significance"],
        atol=1e-15,
    )


@pytest.mark.parametrize("mode", ["posterior", "common", "separate"])
def test_full_precision_report_json_replay_and_revision(mode, tmp_path):
    options = (
        {"tail_probability": [0.02, 0.03]}
        if mode == "posterior"
        else {"significance": 0.05 if mode == "common" else [0.02, 0.03], "selection": "nearest"}
    )
    prior = np.array([0.5, 0.5])
    probabilities = np.array([0.0, 0.4, 1.0])
    study = SeqBinStudySpecification(
        20, prior=prior, alternative="two-sided", probabilities=probabilities, **options
    ).run()
    prior[:] = 100
    probabilities[:] = 0.9
    assert study.specification.prior == (0.5, 0.5)
    assert study.specification.probabilities == (0.0, 0.4, 1.0)
    specification = study.write_specification(tmp_path / "study.json")
    replay = SeqBinStudySpecification.from_json(specification.read_text()).run()
    assert_array_equal(replay.design.continue_low, study.design.continue_low)
    assert_array_equal(replay.design.continue_high, study.design.continue_high)
    assert_allclose(replay.properties.expected_subjects, study.properties.expected_subjects)
    report = study.report(digits=17)
    rows = report.split("Operating characteristics\n", 1)[1].splitlines()
    parsed = np.array(
        [[float(v) if v != "NA" else np.nan for v in line.split("\t")[1:]] for line in rows[1:]]
    )
    assert_array_equal(parsed[:, 0], study.properties.probability)
    assert_array_equal(parsed[:, 1], study.properties.rejection_probability)
    assert_array_equal(parsed[:, 2], study.properties.complete)
    assert_array_equal(parsed[:, 3], study.properties.expected_subjects)
    assert_array_equal(parsed[:, 5], study.properties.expected_subjects_quit_low)
    assert_array_equal(parsed[:, 7], study.properties.expected_subjects_quit_high)
    assert '"two-sided"' in report and "Actual posterior tail cutoffs" in report
    if mode != "posterior":
        assert "Calibration\n" in report and "\tlower\t" in report and "\tupper\t" in report
    destination = tmp_path / "report.tsv"
    destination.write_text("old")
    assert study.write_report(destination, digits=17) == destination
    assert destination.read_text() == report
    assert study.revise(max_subjects=25).design.looks[-1] == 25
    assert study.design.looks[-1] == 20


def test_empty_compact_table_retains_full_design_and_null_only_report():
    design = SeqBinDesign(10, prior=[1, 100], alternative="greater", tail_probability=1e-6)
    table = seqbin_boundary_table(design, compact=True)
    assert table.rows.shape == (0, 9)
    assert seqbin_boundary_table(design).rows.shape == (10, 9)
    study = SeqBinStudySpecification(
        10, prior=[1, 100], tail_probability=1e-6, probabilities=[]
    ).run()
    assert study.properties.probability.shape == (1,)
    assert "\tNA" in study.report(compact=True)
    assert_allclose(study.properties.expected_subjects[0], 10, rtol=2e-15)


def test_table_totals_include_first_look_and_remain_read_only():
    design = SeqBinDesign(10, prior=[30, 1], alternative="two-sided")
    table = seqbin_boundary_table(design)
    assert table.rows[0, 7] == 1 and table.rows[0, 8] == 1
    assert_array_equal(table.rows[:, 8], np.cumsum(table.rows[:, 7]))
    with pytest.raises(ValueError):
        table.rows[0, 0] = 100


@pytest.mark.parametrize(
    "options",
    [
        {"tail_probability": 0.05, "significance": 0.05},
        {"significance": [0.01, 0.02]},
        {"probabilities": [-1]},
        {"probabilities": [[0.2]]},
        {"probabilities": [0.2] * 102},
        {"tail_bounds": [0.2, 0.1]},
        {"selection": "unknown"},
    ],
)
def test_invalid_specification(options):
    with pytest.raises(ValueError):
        SeqBinStudySpecification(10, **options).run()


def test_report_and_serialization_errors(tmp_path):
    study = SeqBinStudySpecification(10).run()
    with pytest.raises(ValueError):
        study.report(digits=0)
    with pytest.raises(ValueError):
        study.report(compact=1)
    with pytest.raises(OSError):
        study.write_report(tmp_path / "missing" / "report.tsv")
    with pytest.raises(TypeError):
        study.revise(unknown=1)
    with pytest.raises(ValueError):
        SeqBinStudySpecification.from_json("[]")
    with pytest.raises(TypeError):
        SeqBinStudySpecification.from_json('{"max_subjects":10,"unknown":1}')
    with pytest.raises(ValueError):
        SeqBinStudySpecification(10, probabilities=[np.nan]).to_json()


def test_verbose_group_table_includes_unscheduled_subject_counts():
    design = SeqBinDesign(20, looks=[5, 10, 20])
    table = seqbin_boundary_table(design).rows
    assert_array_equal(table[:, 0], np.arange(1, 21))
    nonlooks = np.ones(20, dtype=bool)
    nonlooks[design.looks - 1] = False
    assert_array_equal(table[nonlooks, 1], 0)
    assert_array_equal(table[nonlooks, 4], table[nonlooks, 0])
    assert_array_equal(table[nonlooks, 7], 0)
    assert_allclose(table[-1, 8], design.operating_characteristics(0.2).rejection_probability)
