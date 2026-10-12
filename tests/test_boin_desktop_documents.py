"""Standalone reports and captured-input protocol documents preserve analysis scope."""

import pytest
from test_boin_desktop_comparison import comparison_study
from test_boin_desktop_patients import study as combination_study

from mdanderson_stats.boin_desktop import read_boin_desktop_project
from mdanderson_stats.boin_desktop_comparison import simulate_boin_desktop_workflow
from mdanderson_stats.boin_desktop_documents import (
    boin_desktop_protocol_html,
    boin_desktop_single_selection_html,
    boin_desktop_stft_html,
)
from mdanderson_stats.boin_desktop_single import set_boin_desktop_single_counts


@pytest.mark.parametrize("tite", [False, True])
def test_standalone_final_report(tite):
    study = set_boin_desktop_single_counts(comparison_study(tite), [3, 6], [0, 2])
    report = boin_desktop_single_selection_html(study, outcomes_complete=True)
    assert read_boin_desktop_project(report) == study
    assert "Safety overdose probability" in report
    assert "not simultaneous credible bands" in report
    with pytest.raises(ValueError, match="complete-outcome"):
        boin_desktop_single_selection_html(study, outcomes_complete=False)


def test_stft_report_preserves_days_and_weight_total():
    study = comparison_study(True)
    report = boin_desktop_stft_html(study)
    assert read_boin_desktop_project(report) == study
    assert "Follow-up (days)" in report and "does not determine the next dose" in report


@pytest.mark.parametrize("mode", ["single", "tite", "combination", "contour"])
@pytest.mark.parametrize("language", ["en", "zh"])
def test_protocol_binds_methods_table_comparison_and_saved_inputs(mode, language):
    study = (
        combination_study()
        if mode in ("combination", "contour")
        else comparison_study(mode == "tite")
    )
    if mode == "contour":
        study = read_boin_desktop_project(
            study.xml.replace("<FindContours>false", "<FindContours>true").replace(
                "</TrialDesign>",
                "<MaximumSampleSizeForSubTrials><int>12</int><int>6</int></MaximumSampleSizeForSubTrials></TrialDesign>",
            )
        )
    workflow = simulate_boin_desktop_workflow(study, trials=2, seed=613)
    report = boin_desktop_protocol_html(workflow, language=language)
    assert read_boin_desktop_project(report) == study
    assert report.count("<!doctype html>") == 1
    assert "Scenario summary" in report and "seed" in report.lower()
    if mode == "tite":
        assert "<svg " in report
    if language == "zh":
        assert "统计方法" in report


def test_safe_methods_cannot_be_paired_with_legacy_simulation():
    workflow = simulate_boin_desktop_workflow(
        comparison_study(), trials=1, profile="desktop_legacy"
    )
    with pytest.raises(ValueError, match="require safe simulation"):
        boin_desktop_protocol_html(workflow)
