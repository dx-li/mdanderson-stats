"""Installed-study-compatible reports, uncertainty labels and stale-result guards."""

import pytest
from test_boin_desktop_patients import study

from mdanderson_stats.boin_desktop import read_boin_desktop_project
from mdanderson_stats.boin_desktop_controller import recommend_boin_desktop
from mdanderson_stats.boin_desktop_patients import add_boin_desktop_patient
from mdanderson_stats.boin_desktop_report import boin_desktop_conduct_html


def test_pending_report_is_reopenable_and_reports_actual_settings():
    saved = add_boin_desktop_patient(study(), patient_id=1, date_entered="2026-10-12", seed=13)
    recommendation = recommend_boin_desktop(saved, seed=13)
    report = boin_desktop_conduct_html(saved, recommendation)
    assert read_boin_desktop_project(report) == saved
    assert "Completed DLT fraction" in report and "Pending" in report
    assert recommendation.seed == 13 and "Complete saved inputs" in report
    with pytest.raises(ValueError, match="does not match"):
        boin_desktop_conduct_html(study(), recommendation)


def test_final_report_contains_selection_and_native_display_qualifications():
    saved = read_boin_desktop_project(
        study().xml.replace("<MaximumSampleSize>12", "<MaximumSampleSize>3")
    )
    for i in range(3):
        saved = add_boin_desktop_patient(
            saved, patient_id=i + 1, date_entered="2026-10-12", seed=7, outcome=0
        )
    result = recommend_boin_desktop(saved, seed=7)
    assert result.status == "complete" and result.selection_details is not None
    report = boin_desktop_conduct_html(saved, result)
    assert "Isotonic mean" in report and "Display lower" in report
    assert "not simultaneous credible bands" in report and "1e-10" in report
    assert read_boin_desktop_project(report) == saved


def test_report_escapes_retained_data_and_rejects_invalid_seed():
    saved = read_boin_desktop_project(
        study().xml.replace("Independent -- example", "&lt;script&gt;x&lt;/script&gt;")
    )
    report = boin_desktop_conduct_html(saved, recommend_boin_desktop(saved, seed=1))
    assert "<script>" not in report
    for bad in (True, -1, 2**64, 1.2):
        with pytest.raises(ValueError, match="seed"):
            recommend_boin_desktop(saved, seed=bad)
